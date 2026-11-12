"""ESMIF 兼容层测试：验证 torch_scatter 缺失时的 scatter/scatter_add 纯 PyTorch 回退。

守护 modules.esmif_compat 的依赖降级不变量：
scatter_add 必须与按组求和逐位一致、scatter 的 mean 归约必须与分组均值一致——
防止回退实现与 torch_scatter 语义不一致而静默改变模型数值。
"""
import torch

from modules.esmif_compat import scatter, scatter_add


# scatter_add 回退必须与按索引分组的求和完全一致——防止梯度聚合错位。
def test_scatter_add_matches_grouped_sum():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])
    observed = scatter_add(source, index, dim=0, dim_size=2)
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
    assert torch.equal(observed, expected)


# scatter mean 归约必须与分组均值完全一致——防止池化统计错误传播到下游打分。
def test_scatter_mean_matches_grouped_mean():
    source = torch.tensor([1.0, 3.0, 5.0])
    index = torch.tensor([0, 1, 0])
    assert torch.equal(scatter(source, index, dim=0, reduce="mean"), torch.tensor([3.0, 3.0]))


# ── 安装函数的幂等性 ────────────────────────────────────────────────────────
# 该垫片在 run_esmif 与 load_esmif 里各调用一次；第二次曾因 sys.modules 中
# 假模块的 __spec__ 为 None 抛 ValueError: torch_scatter.__spec__ is None。

import sys
import types

import pytest

from modules.esmif_compat import install_torch_scatter_fallback


@pytest.fixture()
def isolated_modules(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_scatter", raising=False)
    monkeypatch.setitem(sys.modules, "torch_geometric", types.ModuleType("torch_geometric"))
    yield


def test_install_is_idempotent(isolated_modules, monkeypatch):
    monkeypatch.setattr("modules.esmif_compat.importlib.util.find_spec", lambda name: None)

    assert install_torch_scatter_fallback() is True
    assert hasattr(sys.modules["torch_scatter"], "scatter_add")
    assert install_torch_scatter_fallback() is False


def test_install_replaces_foreign_stub(isolated_modules):
    """别处塞进来的残缺 stub（没有 __spec__ 与 scatter_add）要被覆盖，而不是报错。"""
    sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")

    assert install_torch_scatter_fallback() is True
    module = sys.modules["torch_scatter"]
    assert hasattr(module, "scatter_add")
    assert module.__spec__ is not None
