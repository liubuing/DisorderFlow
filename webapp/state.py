"""state — 配置加载、checkpoint 解析、模型单例（FastAPI 后端共用；2026-11-11 去 Gradio 化）。"""
import os, json, subprocess, tempfile, re, sys, yaml, time, threading
from pathlib import Path
import torch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))
import target_design_helpers as tdh
import cascade_filter as cf
import misfolding_knowledge_base as mkb
import misfolding_pipeline as mfp
import closed_loop_scorer as cls
import closed_loop_orchestrator as clo
from runtime_environment import (
    build_colabfold_command, colabfold_environment, resolve_colabfold_executable,
)



PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_DIR / 'app_config.yaml'
DEFAULT_MODEL_CONFIG = PROJECT_DIR / 'configs' / 'demo_design.yml'
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
def _detect_device():
    """Auto-detect available device: CUDA > XPU > CPU."""
    # 设备优先级：NVIDIA CUDA > Intel XPU > CPU；结果缓存在模块级 DEVICE
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'
DEVICE = _detect_device()
# 模块级单例缓存：模型/配置只加载一次，跨所有 Gradio 请求复用
_bfn_model = None
_bfn_config = None
_esmif_model = None
_app_config = None
def load_app_config():
    # 读取并缓存 app_config.yaml；返回值同时写入 _app_config 供其它模块查询
    global _app_config
    with open(CONFIG_FILE, encoding='utf-8') as f:
        _app_config = yaml.safe_load(f)
    return _app_config
def get_bfn_ckpt():
    # 检查点解析顺序：环境变量 DISORDERFLOW_CHECKPOINT 覆盖 > 配置文件；
    # 相对路径按项目根目录解析
    override = os.environ.get('DISORDERFLOW_CHECKPOINT')
    if override:
        return str(Path(override).expanduser().resolve())
    cfg = load_app_config()
    path = Path(cfg['models']['bfn']['checkpoint']).expanduser()
    return str(path if path.is_absolute() else PROJECT_DIR / path)
# 检查点允许目录：torch.load(weights_only=False) 只能作用于这些根目录下的文件，
# 避免"配置被改写 -> 指向上传的临时文件 -> 反序列化执行"这条链（审计 S0-2）
_CKPT_ROOTS = (PROJECT_DIR / 'logs',)
_load_lock = threading.Lock()


def load_checkpoint(ckpt_path, map_location='cpu'):
    """带路径白名单的检查点加载。

    优先用 weights_only=True（只反序列化张量）。现有检查点内嵌了
    easydict.EasyDict 配置对象，PyTorch 的受限 unpickler 不允许对 dict 子类做
    SETITEMS，因此会回退到 weights_only=False —— 但**仅当文件位于白名单目录**。
    上传目录（Gradio temp）不在白名单内，攻击者无法把恶意 pickle 送进来。

    TODO: 提供一个一次性迁移脚本，把 config 拆成旁挂 JSON，之后即可彻底只用
    weights_only=True。
    """
    p = Path(ckpt_path).expanduser()
    if not p.is_absolute():
        p = PROJECT_DIR / p
    p = p.resolve()

    override = os.environ.get('DISORDERFLOW_CHECKPOINT')
    trusted = bool(override) and Path(override).expanduser().resolve() == p
    if not trusted and not any(p.is_relative_to(root.resolve()) for root in _CKPT_ROOTS):
        raise PermissionError(
            f'检查点不在允许目录内: {p}（允许: '
            + ', '.join(str(r) for r in _CKPT_ROOTS) + '）'
        )

    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(p, map_location=map_location, weights_only=False)


def _build_bfn():
    from disorderflow.models import get_model
    from disorderflow.utils.misc import load_config as _lc
    ckpt_path = get_bfn_ckpt()
    config, _ = _lc(DEFAULT_MODEL_CONFIG)
    ckpt = load_checkpoint(ckpt_path, map_location=DEVICE)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to(DEVICE)
    ckpt_state = ckpt['model']
    # Skip ipTM head keys if architecture mismatches (e.g. contrastive 512→256 vs old 256→256)
    if any('head_iptm' in k for k in ckpt_state):
        new_iptm_keys = [k for k in model.state_dict() if 'head_iptm' in k]
        old_iptm_keys = [k for k in ckpt_state if 'head_iptm' in k]
        # Check shape match
        shape_mismatch = False
        for k in new_iptm_keys:
            if k in ckpt_state and ckpt_state[k].shape != model.state_dict()[k].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            for k in old_iptm_keys:
                ckpt_state.pop(k)
    model.load_state_dict(ckpt_state, strict=False)
    return model.eval(), config


def load_bfn():
    # 加载 BFN 模型（带进程内缓存）：超参与损失权重以检查点内保存的训练配置为准。
    # 双重检查 + 锁：两个并发请求不会同时构建两份模型把显存翻倍（审计 S1-4）
    global _bfn_model, _bfn_config
    if _bfn_model is not None:
        return _bfn_model, _bfn_config
    with _load_lock:
        if _bfn_model is None:
            _bfn_model, _bfn_config = _build_bfn()
    return _bfn_model, _bfn_config


def load_esmif():
    # ESM-IF 按需加载；先安装 torch_scatter/biotite 兼容层，
    # 否则 fair-esm 在缺轮子的 Windows 环境上会导入失败
    global _esmif_model
    if _esmif_model is not None:
        return _esmif_model
    with _load_lock:
        if _esmif_model is None:
            from esmif_compat import install_biotite_compat, install_torch_scatter_fallback
            install_torch_scatter_fallback()
            install_biotite_compat()
            from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
            model, _ = esm_if1_gvp4_t16_142M_UR50()
            # fair-esm 的 GVPTransformerModel.sample() 内部用 torch.full 无条件在
            # CPU 上建 token 张量，模型放到 cuda 会与 CPU 索引冲突而报错。
            # 与冻结基准 scripts/benchmark_h3_candidate_reranking.py 一致：ESM-IF 采样跑 CPU。
            _esmif_model = model.cpu().eval()
    return _esmif_model
# 置信度分档阈值（用于把 pLDDT/ipTM/PAE 数值翻译成 高/中/低 标签）
CONFIDENCE_DEFAULTS = {
    'plddt_high': 0.80, 'plddt_medium': 0.60,
    'iptm_high': 0.70, 'iptm_medium': 0.40,
    'pae_low': 4.0, 'pae_medium': 8.0,
}
