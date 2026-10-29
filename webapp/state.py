"""app.py split: state (auto-generated 2026-09-27, bodies verbatim)."""
#!/usr/bin/env python
"""
蛋白质序列设计平台 — Gradio Web 界面（管理增强版）

访问: http://127.0.0.1:7860  （仅本机访问，外部无法连接）
管理: python manage.py start|stop|restart|status|batch|config|test
"""
import os, json, subprocess, tempfile, re, sys, yaml, time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import gradio as gr
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
def load_bfn():
    # 加载 BFN 模型（带进程内缓存）：超参与损失权重以检查点内保存的训练配置为准
    global _bfn_model, _bfn_config
    if _bfn_model is not None:
        return _bfn_model, _bfn_config
    from disorderflow.models import get_model
    from disorderflow.utils.misc import load_config as _lc
    ckpt_path = get_bfn_ckpt()
    config, _ = _lc(DEFAULT_MODEL_CONFIG)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    model.eval()
    _bfn_model = model
    _bfn_config = config
    return model, config
def load_esmif():
    # ESM-IF 按需加载；先安装 torch_scatter/biotite 兼容层，
    # 否则 fair-esm 在缺轮子的 Windows 环境上会导入失败
    global _esmif_model
    if _esmif_model is not None:
        return _esmif_model
    from esmif_compat import install_biotite_compat, install_torch_scatter_fallback
    install_torch_scatter_fallback()
    install_biotite_compat()
    from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
    model, _ = esm_if1_gvp4_t16_142M_UR50()
    model = model.to(DEVICE).eval()
    _esmif_model = model
    return model
# 置信度分档阈值（用于把 pLDDT/ipTM/PAE 数值翻译成 高/中/低 标签）
CONFIDENCE_DEFAULTS = {
    'plddt_high': 0.80, 'plddt_medium': 0.60,
    'iptm_high': 0.70, 'iptm_medium': 0.40,
    'pae_low': 4.0, 'pae_medium': 8.0,
}
def get_config_display():
    try:
        cfg = load_app_config()
        return yaml.dump(cfg, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        return f"配置读取错误: {e}"
def save_config_from_text(config_text):
    # 保存配置后把 _app_config 置 None 强制下次重新加载，
    # 已加载的模型不受影响（检查点路径在加载时一次性解析）
    try:
        new_cfg = yaml.safe_load(config_text)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(new_cfg, f, allow_unicode=True, default_flow_style=False)
        # Force reload
        global _app_config
        _app_config = None
        return "✅ 配置已保存，下次设计任务生效"
    except Exception as e:
        return f"❌ 配置格式错误: {e}"
def get_system_status():
    # 汇总服务/模型/依赖的就绪状态，输出为状态面板文本（供 UI 的"系统状态"页）
    import socket, threading
    cfg = load_app_config()
    host = cfg['server']['host']
    port = cfg['server']['port']

    # Check port
    port_open = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        port_open = s.connect_ex((host, port)) == 0
        s.close()
    except:
        pass

    bfn_ok = Path(get_bfn_ckpt()).exists()
    mpnn_dir = Path(cfg['models']['proteinmpnn']['weights_dir'])
    mpnn_ok = (mpnn_dir if mpnn_dir.is_absolute() else PROJECT_DIR / mpnn_dir).exists()
    esmif_loaded = _esmif_model is not None
    bfn_loaded = _bfn_model is not None
    af_cfg = cfg.get('alphafold', {})
    af2_cfg = af_cfg.get('af2', {}) if isinstance(af_cfg, dict) else {}
    af2_backend = str(af2_cfg.get('backend', 'native'))
    try:
        af2_ok = resolve_colabfold_executable(af2_cfg, PROJECT_DIR) is not None
    except Exception:
        af2_ok = False

    status = f"""服务状态面板
{'='*50}
监听地址:     {host}:{port}
端口状态:     {'✓ 运行中' if port_open else '✗ 未启动'}
外部访问:     {'禁止 (仅本机)' if host == '127.0.0.1' else '允许'}
{'='*50}
BFN 模型:      {'✓ 加载就绪' if bfn_loaded else ('✓ 文件存在' if bfn_ok else '✗ 未找到')}
ProteinMPNN:   {'✓ 就绪' if mpnn_ok else '✗ 未找到'}
ESM-IF:        {'✓ 已加载' if esmif_loaded else '○ 按需加载'}
AlphaFold2:    {'✓ 就绪 (' + af2_backend + ')' if af2_ok else '✗ 未找到'}
{'='*50}
配置文件:      {CONFIG_FILE}
Python:        {sys.version.split()[0]}
PyTorch:       {torch.__version__}
CUDA:          {torch.cuda.is_available()}
"""
    return status
