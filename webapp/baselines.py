"""baselines — ProteinMPNN / ESM-IF 基线打分与生成（FastAPI 后端调用；2026-11-11 去 Gradio 化）。"""
import os, json, subprocess, tempfile, re, sys, yaml, time
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
from runtime_environment import build_colabfold_command, colabfold_environment

from webapp.design import diagnose_design_results
from webapp.state import PROJECT_DIR, load_esmif


def run_mpnn(pdb_path, chains, num_samples, temperature, seed, omit_aas):
    # ProteinMPNN 基线设计：以子进程调用官方脚本，解析其 FASTA 输出并计算诊断
    out_dir = tempfile.mkdtemp(prefix='mpnn_')
    # protein_mpnn_run.py derives its output file names with rfind("/"); a
    # Windows backslash path defeats that and the name becomes the whole path.
    # Hand the script the POSIX form of the same absolute path.
    cmd = [
        sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
        '--pdb_path', str(pdb_path).replace('\\', '/'), '--pdb_path_chains', chains,
        '--num_seq_per_target', str(num_samples),
        '--sampling_temp', str(temperature),
        '--seed', str(seed),
        '--out_folder', out_dir, '--save_score', '1',
        '--path_to_model_weights', str(PROJECT_DIR / 'ProteinMPNN' / 'vanilla_model_weights'),
    ]
    if omit_aas and omit_aas.strip():
        cmd.extend(['--omit_AAs', omit_aas.strip()])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_DIR))
    if r.returncode != 0:
        return f"ProteinMPNN 错误:\n{r.stderr[:2000]}", "", []

    pid = os.path.basename(pdb_path).replace('.pdb', '')
    fa = os.path.join(out_dir, 'seqs', f'{pid}.fa')
    if not os.path.exists(fa):
        return f"未找到输出: {fa}", "", []

    lines = [f"ProteinMPNN | 链 {chains} | 温度 {temperature} | {num_samples}序列"]
    fasta_lines = []
    results_list = []
    with open(fa) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>T='):
                parts = {p.split('=')[0].strip(): p.split('=')[1].strip()
                         for p in line.split(',') if '=' in p}
                seq = next(f).strip()
                score_str = parts.get('score', '?')
                rec_str = parts.get('seq_recovery', '?')
                lines.append(f"#{parts.get('sample','?')}: {seq[:80]}{'...' if len(seq)>80 else ''} | score={score_str} | recovery={rec_str}")
                sample_id = parts.get('sample', '?')
                fasta_lines.append(f">MPNN_{pid}_sample_{sample_id} score={score_str}\n{seq}")
                try:
                    results_list.append({'sequence': seq, 'ppl': float(score_str), 'recovery': float(rec_str) if rec_str != '?' else None})
                except ValueError:
                    results_list.append({'sequence': seq})
    fasta_str = '\n'.join(fasta_lines)
    design_length = len(results_list[0]['sequence'].replace('/', '')) if results_list else 0
    diag = diagnose_design_results(results_list, design_length)
    if diag:
        lines.append(diag)
    return '\n'.join(lines), fasta_str, results_list
def run_esmif(pdb_path, chain, temperature, num_samples):
    # ESM-IF 逆折叠基线：按温度采样并统计与天然序列的恢复率
    # The compat shim must land before fair-esm's import-time biotite lookup;
    # load_esmif() also patches, but it is called after this import.
    from esmif_compat import install_biotite_compat, install_torch_scatter_fallback
    install_torch_scatter_fallback()
    install_biotite_compat()
    from esm.inverse_folding import util as if_util
    model = load_esmif()
    try:
        coords, native = if_util.load_coords(pdb_path, chain)
    except Exception as e:
        return f"ESM-IF 加载错误: {e}", "", []

    lines = [f"ESM-IF | 链 {chain} | {len(native)}残基 | 温度 {temperature}"]
    lines.append(f"原始: {native}")
    fasta_lines = []
    results_list = []
    pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
    for i in range(num_samples):
        s = model.sample(coords, temperature=temperature)
        rec = sum(1 for a,b in zip(s, native) if a==b)/len(native)
        lines.append(f"#{i+1}: {s} | 恢复率{rec*100:.1f}%")
        fasta_lines.append(f">ESMIF_{pdb_name}_sample_{i+1} T={temperature:.2f} recovery={rec*100:.1f}%\n{s}")
        results_list.append({'sequence': s, 'recovery': rec})
    fasta_str = '\n'.join(fasta_lines)
    return '\n'.join(lines), fasta_str, results_list
# 单字母 -> 三字母氨基酸码（生成 PDB 时写残基名用）
AA_NAMES = {
    'A': 'ALA','C': 'CYS','D': 'ASP','E': 'GLU','F': 'PHE','G': 'GLY',
    'H': 'HIS','I': 'ILE','K': 'LYS','L': 'LEU','M': 'MET','N': 'ASN',
    'P': 'PRO','Q': 'GLN','R': 'ARG','S': 'SER','T': 'THR','V': 'VAL',
    'W': 'TRP','Y': 'TYR',
}
