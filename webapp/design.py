"""app.py split: design (auto-generated 2026-09-27, bodies verbatim)."""
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

from webapp.state import AA_LETTERS, CONFIDENCE_DEFAULTS, DEVICE, load_app_config, load_bfn


def confidence_quality_label(value, thresholds, metric='plddt'):
    if metric in ('plddt', 'iptm'):
        if value >= thresholds[f'{metric}_high']: return 'HIGH'
        if value >= thresholds[f'{metric}_medium']: return 'MEDIUM'
        return 'LOW'
    elif metric == 'pae':
        if value <= thresholds['pae_low']: return 'HIGH'
        if value <= thresholds['pae_medium']: return 'MEDIUM'
        return 'LOW'
    return 'N/A'
def diagnose_design_results(results_list, design_length):
    """Generate runtime diagnostic warnings for BFN design/evaluation results.

    Checks: length bounds, IDP-like pLDDT/ipTM patterns, PAE outliers.
    Returns a warning message string (empty if all checks pass).
    """
    from disorderflow.utils.runtime_checks import diagnose_protein, Severity as S

    if not results_list:
        return ""

    plddt_values = [r['plddt'] for r in results_list if r.get('plddt') is not None]
    iptm_values = [r['iptm'] for r in results_list if r.get('iptm') is not None]
    pae_values = [r['pae'] for r in results_list if r.get('pae') is not None]
    mean_plddt = float(np.mean(plddt_values)) if plddt_values else None
    mean_iptm = float(np.mean(iptm_values)) if iptm_values else None
    mean_pae = float(np.mean(pae_values)) if pae_values else None

    report = diagnose_protein(
        length=design_length,
        mean_plddt=mean_plddt,
        iptm=mean_iptm,
        mean_pae=mean_pae,
    )

    warnings = []
    for f in report.flags:
        tag = {S.ERROR: "!!", S.WARNING: "!!", S.INFO: "i"}.get(f.severity, "i")
        warnings.append(f"  [{tag}] {f.message}")

    if not warnings and report.idp_likelihood == "low":
        return ""

    header = "\n── 运行时诊断 ──"
    if report.idp_likelihood in ("possible", "high"):
        warnings.append(f"  [IDP] 无序蛋白可能性: {report.idp_likelihood}. BFN置信度可能不可靠.")
    if not report.is_in_distribution:
        warnings.append(f"  [DIST] 蛋白不在训练分布内 (L={report.length}, 训练范围50-250)")

    return header + "\n" + "\n".join(warnings)
def build_results_dataframe(results_list):
    if not results_list:
        return pd.DataFrame()
    rows = []
    for i, r in enumerate(results_list):
        seq = r.get('sequence', '')
        if len(seq) > 40:
            seq_display = seq[:18] + '…' + seq[-18:]
        else:
            seq_display = seq
        plddt = r.get('plddt')
        iptm = r.get('iptm')
        ppl = r.get('ppl')
        entropy = r.get('entropy')
        composite = r.get('composite_score')
        quality = 'N/A'
        plddt_normalized = plddt / 100.0 if plddt is not None and plddt > 1 else plddt
        if plddt_normalized is not None and iptm is not None:
            if plddt_normalized >= 0.80 and iptm >= 0.70:
                quality = 'HIGH'
            elif plddt_normalized >= 0.60 or iptm >= 0.40:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'
        rows.append({
            'Rank': i + 1,
            'Sequence': seq_display,
            'Full Sequence': seq,
            'PPL': f'{ppl:.2f}' if ppl else 'N/A',
            'Entropy': f'{entropy:.3f}' if entropy is not None else 'N/A',
            'pLDDT': (f'{plddt:.1f}' if plddt and plddt > 1 else
                      f'{plddt:.3f}' if plddt else 'N/A'),
            'ipTM': f'{iptm:.3f}' if iptm else 'N/A',
            'PAE': f'{r.get("pae", 0):.1f}' if r.get('pae') else 'N/A',
            'Composite': f'{composite:.3f}' if composite else 'N/A',
            'Quality': quality,
        })
    return pd.DataFrame(rows)
def run_bfn_confidence_evaluation(pdb_path, region_spec):
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform

    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_spec):
        indices = []
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg: continue
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip()) + 1))
            else:
                indices.append(int(seg))
        regions[cid] = sorted(set(indices))

    if not regions:
        return "Error: invalid region format", pd.DataFrame()

    model, config = load_bfn()
    seed_all(getattr(config.sampling, 'seed', 42))
    structure = preprocess_protein_structure(pdb_path, chain_ids=list(regions.keys()))
    if structure is None:
        return "Error: cannot parse structure", pd.DataFrame()

    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
    gen_mask = batch['generate_flag'][0].bool()
    if gen_mask.sum() == 0:
        return "Error: no residues selected for evaluation", pd.DataFrame()

    sample_opt = {'deterministic': True, 'num_recycles': 3}
    with torch.no_grad():
        traj = model.sample(batch, sample_opt=sample_opt)

    plddt_full = traj['plddt'][0]
    plddt_region = plddt_full[gen_mask]
    iptm_val = traj['iptm'][0].item()
    pae_full = traj['pae'][0]
    pae_region = pae_full[gen_mask][:, gen_mask]

    n_res_all = len(plddt_full)
    n_res_region = gen_mask.sum().item()

    cfg = load_app_config()
    thresholds = cfg.get('confidence', CONFIDENCE_DEFAULTS)

    lines = [
        f"Confidence Evaluation | {region_spec} | {n_res_region}/{n_res_all} residues",
        "",
        f"  pLDDT (region mean):  {plddt_region.mean().item():.4f}  [{confidence_quality_label(plddt_region.mean().item(), thresholds, 'plddt')}]",
        f"  pLDDT (all residues):  {plddt_full.mean().item():.4f}",
        f"  ipTM (global):         {iptm_val:.4f}  [{confidence_quality_label(iptm_val, thresholds, 'iptm')}]",
        f"  PAE (region self):     {pae_region.mean().item():.2f}  [{confidence_quality_label(pae_region.mean().item(), thresholds, 'pae')}]",
        f"  PAE (full matrix avg): {pae_full.mean().item():.2f}",
        "",
        "  Recycles: 3 | Mode: deterministic | Model: V6 Phase 2",
    ]

    per_residue = []
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    native_aa = batch['aa'][0][gen_mask]
    native_seq = ''.join(AA[a] if a < 20 else 'X' for a in native_aa.cpu())
    gen_indices = torch.where(gen_mask)[0].cpu().numpy()
    for j, (idx, aa) in enumerate(zip(gen_indices, native_seq)):
        res_plddt = plddt_full[idx].item()
        per_residue.append({
            'Position': int(idx) + 1,
            'Residue': aa,
            'pLDDT': f'{res_plddt:.4f}',
            'Quality': confidence_quality_label(res_plddt, thresholds, 'plddt'),
        })

    return '\n'.join(lines), pd.DataFrame(per_residue)
def run_bfn_antibody(pdb_path, heavy, light, num_samples, stochastic, eval_mode):
    from disorderflow.datasets.custom import preprocess_antibody_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform

    model, config = load_bfn()
    cfg = load_app_config()
    seed_all(getattr(config.sampling, 'seed', 42))
    structure = preprocess_antibody_structure(pdb_path, heavy_chain=heavy, light_chain=light)
    if structure is None:
        return "错误：无法解析抗体结构", "", []

    cdrs = cfg['bfn_defaults']['antibody'].get('cdrs',
              ['H_CDR1','H_CDR2','H_CDR3','L_CDR1','L_CDR2','L_CDR3'])
    transform = get_transform([
        {'type': 'mask_cdr', 'sample_cdr': cdrs, 'mode': 'all'},
        {'type': 'merge_antibody'},
        {'type': 'patch_around_anchor'},
    ])
    data = transform(structure)
    batch = recursive_to(PaddingCollate()([data]), 'cpu')
    gen_mask = batch['generate_flag'][0].bool()
    if gen_mask.sum() == 0:
        return "错误：未找到设计区域", "", []

    native_seq = None
    if eval_mode:
        native_aa = batch['aa'][0][gen_mask]
        native_seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in native_aa.cpu())

    sample_opt = {'deterministic': not stochastic, 'num_recycles': 3}
    lines = [f"BFN抗体CDR设计 | {gen_mask.sum().item()}残基 | {'随机' if stochastic else '确定性'} | 回收×3" +
             f" | 重链{heavy} 轻链{light}"]
    if native_seq:
        lines.append(f"原始: {native_seq}")
    lines.append("")

    best_ppl, best_seq = float('inf'), ""
    fasta_entries = []
    results_list = []
    for i in range(num_samples):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)
        pred_aa = traj[0][2][0][gen_mask]
        seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())
        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()
        # Per-residue entropy (model certainty): H = -sum(p*log(p))
        entropy = -(torch.exp(lp) * lp).sum(dim=-1).mean().item()
        # BFN内置置信度 (receiver.py 已做 sigmoid → [0,1])
        plddt_val = traj['plddt'][0][gen_mask].mean().item()
        iptm_val = traj['iptm'][0].item()
        pae_val = traj['pae'][0][gen_mask][:, gen_mask].mean().item()
        recovery = None
        rec_str = ""
        if native_seq:
            recovery = sum(1 for a,b in zip(seq, native_seq) if a==b)/len(native_seq)
            rec_str = f" | 恢复率{recovery*100:.1f}%"
        lines.append(f"#{i+1}: {seq} | PPL={ppl:.2f} | ent={entropy:.3f} | pLDDT={plddt_val:.2f} | ipTM={iptm_val:.2f} | PAE={pae_val:.1f}{rec_str}")
        fasta_entries.append((i+1, seq, ppl, plddt_val, iptm_val, pae_val, rec_str.strip()))
        results_list.append({'sequence': seq, 'ppl': ppl, 'entropy': entropy,
                             'plddt': plddt_val, 'iptm': iptm_val,
                             'pae': pae_val, 'recovery': recovery})
        if ppl < best_ppl:
            best_ppl, best_seq = ppl, seq

    # Cascade filter
    pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
    filtered, filter_report = cf.apply_cascade(results_list)
    if filtered:
        fasta_str, best_filtered = cf.format_filtered_fasta(filtered, f"BFN_Ab_{pdb_name}")
        lines.append("")
        lines.append(filter_report)
        if best_filtered and best_filtered != best_seq:
            lines.append(f"\n  综合最优: {best_filtered} (综合评分={filtered[0]['composite_score']:.3f})")
            best_seq = best_filtered
    else:
        # Fallback: use old FASTA format
        fasta_lines = []
        for idx, seq, ppl_val, plddt_v, iptm_v, pae_v, rec in fasta_entries:
            tag = f"BFN_Ab_{pdb_name}_sample_{idx} PPL={ppl_val:.2f} pLDDT={plddt_v:.2f} ipTM={iptm_v:.2f} PAE={pae_v:.1f}"
            if rec: tag += f" {rec}"
            fasta_lines.append(f">{tag}\n{seq}")
        fasta_lines.append(f">BFN_Ab_{pdb_name}_best PPL={best_ppl:.2f}\n{best_seq}")
        fasta_str = '\n'.join(fasta_lines)
        lines.append("")
        lines.append(filter_report)

    diag = diagnose_design_results(results_list, gen_mask.sum().item())
    if diag:
        lines.append(diag)
    return '\n'.join(lines), fasta_str, results_list
def run_bfn_protein(pdb_path, region_spec, num_samples, stochastic, eval_mode):
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.misc import seed_all
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform

    from modules.bfn_loader import parse_region_spec

    try:
        regions = parse_region_spec(region_spec)
    except ValueError:
        return f"区域格式错误: {region_spec}", "", []

    model, config = load_bfn()
    seed_all(getattr(config.sampling, 'seed', 42))
    structure = preprocess_protein_structure(pdb_path)
    if structure is None:
        return "错误：无法解析结构", "", []

    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
    gen_mask = batch['generate_flag'][0].bool()
    if gen_mask.sum() == 0:
        return "错误：未选中设计残基", "", []

    native_seq = None
    if eval_mode:
        native_aa = batch['aa'][0][gen_mask]
        native_seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in native_aa.cpu())

    sample_opt = {'deterministic': not stochastic, 'num_recycles': 3}
    lines = [f"BFN通用设计 | {region_spec} | {gen_mask.sum().item()}残基 | {'随机' if stochastic else '确定性'} | 回收×3"]
    if native_seq:
        lines.append(f"原始: {native_seq}")
    lines.append("")

    best_ppl, best_seq = float('inf'), ""
    fasta_entries = []
    results_list = []
    for i in range(num_samples):
        with torch.no_grad():
            traj = model.sample(batch, sample_opt=sample_opt)
        pred_aa = traj[0][2][0][gen_mask]
        seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())
        logits = traj['pred_logits'][0][gen_mask]
        lp = torch.log_softmax(logits[..., :20], dim=-1)
        nll = -lp[range(len(pred_aa)), pred_aa].mean()
        ppl = torch.exp(nll).item()
        # Per-residue entropy (model certainty): H = -sum(p*log(p))
        entropy = -(torch.exp(lp) * lp).sum(dim=-1).mean().item()
        # BFN内置置信度 (receiver.py 已做 sigmoid → [0,1])
        plddt_val = traj['plddt'][0][gen_mask].mean().item()
        iptm_val = traj['iptm'][0].item()
        pae_val = traj['pae'][0][gen_mask][:, gen_mask].mean().item()
        recovery = None
        rec_str = ""
        if native_seq:
            recovery = sum(1 for a,b in zip(seq, native_seq) if a==b)/len(native_seq)
            rec_str = f" | 恢复率{recovery*100:.1f}%"
        lines.append(f"#{i+1}: {seq} | PPL={ppl:.2f} | ent={entropy:.3f} | pLDDT={plddt_val:.2f} | ipTM={iptm_val:.2f} | PAE={pae_val:.1f}{rec_str}")
        fasta_entries.append((i+1, seq, ppl, plddt_val, iptm_val, pae_val, rec_str.strip()))
        results_list.append({'sequence': seq, 'ppl': ppl, 'entropy': entropy,
                             'plddt': plddt_val, 'iptm': iptm_val,
                             'pae': pae_val, 'recovery': recovery})
        if ppl < best_ppl:
            best_ppl, best_seq = ppl, seq

    # Cascade filter
    pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
    filtered, filter_report = cf.apply_cascade(results_list)
    if filtered:
        fasta_str, best_filtered = cf.format_filtered_fasta(filtered, f"BFN_protein_{pdb_name}")
        lines.append("")
        lines.append(filter_report)
        if best_filtered and best_filtered != best_seq:
            lines.append(f"\n  综合最优: {best_filtered} (综合评分={filtered[0]['composite_score']:.3f})")
            best_seq = best_filtered
    else:
        # Fallback: use old FASTA format
        fasta_lines = []
        for idx, seq, ppl_val, plddt_v, iptm_v, pae_v, rec in fasta_entries:
            tag = f"BFN_protein_{pdb_name}_sample_{idx} PPL={ppl_val:.2f} pLDDT={plddt_v:.2f} ipTM={iptm_v:.2f} PAE={pae_v:.1f} region={region_spec}"
            if rec: tag += f" {rec}"
            fasta_lines.append(f">{tag}\n{seq}")
        fasta_lines.append(f">BFN_protein_{pdb_name}_best PPL={best_ppl:.2f}\n{best_seq}")
        fasta_str = '\n'.join(fasta_lines)
        lines.append("")
        lines.append(filter_report)

    return '\n'.join(lines), fasta_str, results_list
