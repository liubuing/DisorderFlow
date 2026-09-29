"""app.py split: af2 (auto-generated 2026-09-27, bodies verbatim)."""
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
from runtime_environment import build_colabfold_command, colabfold_environment

from webapp.sequtils import parse_fasta
from webapp.state import PROJECT_DIR, load_app_config


# ── Global caches ──
_predicted_pdb = None  # path to most recent AlphaFold prediction
def run_alphafold_prediction(fasta_text, fasta_file,
                              num_models, num_recycle, use_dropout,
                              progress=gr.Progress()):
    global _predicted_pdb
    if fasta_file is not None:
        try:
            with open(fasta_file.name, encoding='utf-8') as f:
                fasta_text = f.read()
        except Exception:
            pass

    if not fasta_text or not fasta_text.strip():
        return "请提供 FASTA 序列", None, None

    seqs = parse_fasta(fasta_text)
    if not seqs:
        return "无效的 FASTA 格式", None, None

    cfg = load_app_config()
    af_cfg = cfg.get('alphafold', {})
    default_out = af_cfg.get('output_dir', 'alphafold_results')
    af2_runtime = af_cfg.get('af2', {})

    out_dir = Path(default_out)
    out_dir.mkdir(exist_ok=True)
    pid = seqs[0][0].split()[0] if seqs else 'query'
    pid = re.sub(r'[^A-Za-z0-9_-]', '_', pid)
    fasta_path = out_dir / f'{pid}.fasta'

    # Sanitize to ASCII: colabfold uses system default encoding (GBK on Chinese Windows)
    safe_text = fasta_text.encode('ascii', errors='replace').decode('ascii')
    with open(fasta_path, 'w', encoding='ascii') as f:
        f.write(safe_text)

    stop_at = af_cfg.get('defaults', {}).get('stop_at_score', 85)
    model_type = af_cfg.get('defaults', {}).get('model_type', 'auto')
    rank_mode = af_cfg.get('defaults', {}).get('rank', 'auto')

    arguments = [
        '--num-models', str(int(num_models)),
        '--num-recycle', str(int(num_recycle)),
        '--stop-at-score', str(int(stop_at)),
        '--model-type', model_type,
        '--rank', rank_mode,
    ]
    if af_cfg.get('defaults', {}).get('calc_extra_ptm', False):
        arguments.append('--calc-extra-ptm')
    if use_dropout:
        arguments.append('--use-dropout')

    try:
        cmd = build_colabfold_command(
            af2_runtime, PROJECT_DIR, fasta_path, out_dir / pid, arguments)
    except FileNotFoundError as exc:
        return f"AlphaFold2 环境错误: {exc}", None, None

    progress(0.05, desc="启动 AlphaFold2 (ColabFold)...")
    log_lines = [f"🔮 AlphaFold2 (ColabFold) 结构预测",
                 f"{'='*50}",
                 f"序列: {seqs[0][0]} ({len(seqs[0][1])} aa)",
                 f"模型: {model_type}  |  回收: {int(num_recycle)}  |  达标分: {int(stop_at)}",
                 f"输出: {out_dir / pid}/", f"",
                 f"⏱ 当前使用 WSL2 GPU，首次运行会包含 JAX 编译和模型加载时间"]

    try:
        env = colabfold_environment(af2_runtime, PROJECT_DIR)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          cwd=str(PROJECT_DIR), env=env)
        out_text = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
        if out_text:
            log_lines.append(out_text)
        if r.stderr:
            err_text = r.stderr[-800:]
            if err_text.strip():
                log_lines.append(f"[stderr]\n{err_text}")

        if r.returncode != 0:
            log_lines.append(f"\n❌ 预测失败 (exit code {r.returncode})")
            return '\n'.join(log_lines), None, None
    except subprocess.TimeoutExpired:
        log_lines.append("\n❌ 超时 (1小时)")
        return '\n'.join(log_lines), None, None

    progress(0.8, desc="解析置信度分数...")
    result_dir = out_dir / pid
    pdb_files = []
    if result_dir.exists():
        pdb_files = sorted(result_dir.glob('*_rank_001_*.pdb'))
        if not pdb_files:
            pdb_files = sorted(result_dir.glob('*_relaxed_rank_*.pdb'))
        if not pdb_files:
            pdb_files = sorted(result_dir.glob('*.pdb'))

    if pdb_files:
        _predicted_pdb = pdb_files[0]
        best_name = os.path.basename(_predicted_pdb)

        # Parse AF2 confidence scores from JSON
        scores_json = sorted(result_dir.glob('*_scores_rank_001_*.json'))
        if scores_json:
            try:
                with open(scores_json[0]) as sf:
                    scores = json.load(sf)
                mean_plddt = sum(scores['plddt']) / len(scores['plddt'])
                ptm = scores.get('ptm', float('nan'))
                iptm = scores.get('iptm', float('nan'))
                max_pae = scores.get('max_pae', float('nan'))
                log_lines.append(f"\n{'='*50}")
                log_lines.append(f"📊 AF2 置信度评估")
                log_lines.append(f"  平均 pLDDT: {mean_plddt:.1f}/100")
                log_lines.append(f"  pTM:        {ptm:.3f}  (全局折叠置信度)")
                if isinstance(iptm, float) and iptm == iptm:  # not NaN
                    log_lines.append(f"  ipTM:       {iptm:.3f}  (界面置信度)")
                log_lines.append(f"  max PAE:    {max_pae:.1f}  (最大预测误差)")
                # Qualitative assessment
                if mean_plddt >= 80:
                    log_lines.append(f"  ✓ pLDDT ≥ 80: 高置信度，适合下游分析")
                elif mean_plddt >= 60:
                    log_lines.append(f"  △ pLDDT 60-80: 中等置信度，折叠大体可信")
                else:
                    log_lines.append(f"  ✗ pLDDT < 60: 低置信度，需谨慎使用")
                log_lines.append(f"{'='*50}")
            except Exception as e:
                log_lines.append(f"\n⚠ 解析置信度分数失败: {e}")

        log_lines.append(f"\n✅ 完成! 最佳结构: {best_name}")
        progress(1.0, desc="完成!")
        return '\n'.join(log_lines), str(_predicted_pdb), str(result_dir)
    else:
        log_lines.append(f"\n⚠️ 预测完成但未找到 PDB 输出. 检查: {result_dir}")
        return '\n'.join(log_lines), None, None
