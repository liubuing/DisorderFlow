"""Runtime compatibility for ESM-IF when torch-scatter wheels are unavailable."""

from __future__ import annotations

import importlib.util
import sys
import types

import torch


def install_torch_scatter_fallback():
    """Install the two scatter operations used by fair-esm via native PyTorch."""
    if importlib.util.find_spec("torch_scatter") is not None:
        return False

    # Import first so torch-geometric records torch-scatter as unavailable and
    # continues to use its own native-PyTorch fallback paths.
    import torch_geometric  # noqa: F401
    import biotite.structure

    if not hasattr(biotite.structure, "filter_backbone"):
        biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone

    module = types.ModuleType("torch_scatter")
    module.scatter_add = scatter_add
    module.scatter = scatter
    sys.modules["torch_scatter"] = module
    return True


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    """Subset of torch_scatter.scatter implemented with scatter_reduce_."""
    dim = dim if dim >= 0 else src.dim() + dim
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = index.numel()
        index = index.view(shape).expand_as(src)
    elif index.shape != src.shape:
        index = index.expand_as(src)
    if dim_size is None:
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
    out.scatter_reduce_(dim, index, src, reduce=reduction, include_self=False)
    return out
