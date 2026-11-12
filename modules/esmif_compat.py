"""Runtime compatibility for ESM-IF when torch-scatter wheels are unavailable."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

import torch


def install_biotite_compat():
    """Alias the renamed backbone filter that fair-esm imports at load time."""
    # 新版 biotite 把 filter_backbone 改名为 filter_peptide_backbone，
    # 而 fair-esm 仍按旧名导入；补一个别名避免加载即报 AttributeError
    import biotite.structure

    if not hasattr(biotite.structure, "filter_backbone"):
        biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone


def install_torch_scatter_fallback():
    """Install the two scatter operations used by fair-esm via native PyTorch.

    幂等：三种状态都要能安全重复调用——
    1. 真的装了 torch_scatter：直接返回 False；
    2. 本函数之前注入过兼容模块：返回 False；
    3. sys.modules 里有个别处塞进来的残缺 stub（__spec__ 为 None 且没有
       scatter_add）：覆盖成我们的实现，而不是让 find_spec 抛 ValueError。
    """
    existing = sys.modules.get("torch_scatter")
    if existing is not None and hasattr(existing, "scatter_add"):
        return False

    try:
        if importlib.util.find_spec("torch_scatter") is not None:
            return False
    except (ImportError, ValueError):
        # ValueError: 模块在 sys.modules 里但 __spec__ 为 None（残缺 stub），
        # 继续往下覆盖即可
        pass

    # Import first so torch-geometric records torch-scatter as unavailable and
    # continues to use its own native-PyTorch fallback paths.
    import torch_geometric  # noqa: F401

    install_biotite_compat()

    module = types.ModuleType("torch_scatter")
    module.scatter_add = scatter_add
    module.scatter = scatter
    # 给假模块补一个合法 __spec__，否则重复调用时 find_spec("torch_scatter")
    # 会因 __spec__ 为 None 抛 ValueError（run_esmif 与 load_esmif 各调用一次）。
    # 补上后第二次 find_spec 返回非空 -> 幂等地提前 return False。
    module.__spec__ = importlib.machinery.ModuleSpec("torch_scatter", loader=None)
    sys.modules["torch_scatter"] = module
    return True


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    """Subset of torch_scatter.scatter implemented with scatter_reduce_."""
    # 用 PyTorch 原生 scatter_reduce_ 复刻 torch_scatter.scatter 的常用子集：
    # 1) 负的 dim 按 Python 惯例从尾部折算
    dim = dim if dim >= 0 else src.dim() + dim
    # 2) 一维 index 广播成与 src 同形；min/max 归约时填充值取 +inf/-inf 保证正确性
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = index.numel()
        index = index.view(shape).expand_as(src)
    elif index.shape != src.shape:
        index = index.expand_as(src)
    if dim_size is None:
        # 未指定输出大小时按 index 最大值推断；空 index 产生 0 大小输出
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    output_shape = list(src.shape)
    output_shape[dim] = int(dim_size)
    if out is None:
        fill = 0.0
        if reduce in {"min", "amin"}:
            fill = float("inf")
        elif reduce in {"max", "amax"}:
            fill = float("-inf")
        out = torch.full(output_shape, fill, dtype=src.dtype, device=src.device)
    reduction = {"add": "sum", "mean": "mean", "min": "amin", "max": "amax"}.get(
        reduce, reduce)
    # include_self=False：输出张量仅作初始填充容器，不参与归约
    out.scatter_reduce_(dim, index, src, reduce=reduction, include_self=False)
    return out
