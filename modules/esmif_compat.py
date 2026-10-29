"""Runtime compatibility for ESM-IF when torch-scatter wheels are unavailable."""

from __future__ import annotations

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
    """Install the two scatter operations used by fair-esm via native PyTorch."""
    # 已装了真正的 torch_scatter 就什么都不做（返回 False 表示未安装兼容层）
    if importlib.util.find_spec("torch_scatter") is not None:
        return False

    # Import first so torch-geometric records torch-scatter as unavailable and
    # continues to use its own native-PyTorch fallback paths.
    import torch_geometric  # noqa: F401

    install_biotite_compat()

    module = types.ModuleType("torch_scatter")
    module.scatter_add = scatter_add
    module.scatter = scatter
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
