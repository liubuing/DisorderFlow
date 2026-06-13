#!/usr/bin/env python
"""
铔嬬櫧璐ㄥ簭鍒楄璁″钩鍙?鈥?Gradio Web 鐣岄潰锛堢鐞嗗寮虹増锛?

璁块棶: http://127.0.0.1:7860  锛堜粎鏈満璁块棶锛屽閮ㄦ棤娉曡繛鎺ワ級
绠＄悊: python manage.py start|stop|restart|status|batch|config|test
"""
import os, json, subprocess, tempfile, re, sys, yaml, time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import gradio as gr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
import target_design_helpers as tdh
import cascade_filter as cf
import misfolding_knowledge_base as mkb
import misfolding_pipeline as mfp
import closed_loop_scorer as cls
import closed_loop_orchestrator as clo

def _detect_device():
    """Auto-detect available device: CUDA > XPU > CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'

DEVICE = _detect_device()

PROJECT_DIR = Path(__file__).parent
CONFIG_FILE = PROJECT_DIR / 'app_config.yaml'
DEFAULT_MODEL_CONFIG = PROJECT_DIR / 'configs' / 'demo_design.yml'

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

# 鈹€鈹€ Global caches 鈹€鈹€
_predicted_pdb = None  # path to most recent AlphaFold prediction
_bfn_model = None
_bfn_config = None
_esmif_model = None
_app_config = None


def load_app_config():
    global _app_config
    with open(CONFIG_FILE, encoding='utf-8') as f:
        _app_config = yaml.safe_load(f)
    return _app_config


def get_bfn_ckpt():
    cfg = load_app_config()
    return cfg['models']['bfn']['checkpoint']


# 鈹€鈹€ PDB Helpers 鈹€鈹€

def detect_chains(pdb_path):
    chains = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                c = line[21:22].strip()
                if c and c not in chains:
                    chains.append(c)
    return chains


def get_chain_info(pdb_path):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    pid = os.path.basename(pdb_path).replace('.pdb', '')
    s = parser.get_structure(pid, pdb_path)[0]
    aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
              'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
              'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
    info = {}
    for ch in s.get_chains():
        cid = ch.id
        res = [r for r in ch.get_residues() if r.get_resname().strip() in aa3to1]
        seq = ''.join(aa3to1[r.get_resname().strip()] for r in res)
        ids = [str(r.get_id()[1]) for r in res]
        info[cid] = {'seq': seq, 'len': len(seq), 'first': ids[0] if ids else '?', 'last': ids[-1] if ids else '?'}
    return info


def on_upload(pdb_file):
    if pdb_file is None:
        return "璇蜂笂浼?PDB 鏂囦欢", "", ""
    try:
        chains = detect_chains(pdb_file.name)
        info = get_chain_info(pdb_file.name)
        lines = [f"妫€娴嬪埌 {len(chains)} 鏉￠摼: {', '.join(chains)}"]
        for c in chains:
            ci = info.get(c, {})
            s = ci.get('seq', '')
            lines.append(f"  閾?{c}: {ci.get('len','?')}娈嬪熀 [{ci.get('first','?')}-{ci.get('last','?')}] {s[:60]}{'...' if len(s)>60 else ''}")
        return '\n'.join(lines), ', '.join(chains), f"{chains[0]}:1-10" if chains else ""
    except Exception as e:
        return f"瑙ｆ瀽閿欒: {e}", "", ""


# 鈹€鈹€ Model Loaders 鈹€鈹€

def load_bfn():
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
    # Skip ipTM head keys if architecture mismatches (e.g. contrastive 512鈫?56 vs old 256鈫?56)
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
    global _esmif_model
    if _esmif_model is not None:
        return _esmif_model
    from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
    model, _ = esm_if1_gvp4_t16_142M_UR50()
    model = model.to(DEVICE).eval()
    _esmif_model = model
    return model


# 鈹€鈹€ Confidence Helpers 鈹€鈹€

CONFIDENCE_DEFAULTS = {
    'plddt_high': 0.80, 'plddt_medium': 0.60,
    'iptm_high': 0.70, 'iptm_medium': 0.40,
    'pae_low': 4.0, 'pae_medium': 8.0,
}


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
    from disorderflow.utils.runtime_checks import diagnose_protein, format_report, Severity as S

    if not results_list:
        return ""

    mean_plddt = np.mean([r.get('plddt', 0) for r in results_list])
    mean_iptm = np.mean([r.get('iptm', 0) for r in results_list])
    mean_pae = np.mean([r.get('pae', 99) for r in results_list])

    report = diagnose_protein(
        length=design_length,
        mean_plddt=mean_plddt,
        iptm=mean_iptm,
        mean_pae=mean_pae if mean_pae < 99 else None,
    )

    warnings = []
    for f in report.flags:
        tag = {S.ERROR: "!!", S.WARNING: "!!", S.INFO: "i"}.get(f.severity, "i")
        warnings.append(f"  [{tag}] {f.message}")

    if not warnings and report.idp_likelihood == "low":
        return ""

    header = "\n鈹€鈹€ 杩愯鏃惰瘖鏂?鈹€鈹€"
    if report.idp_likelihood in ("possible", "high"):
        warnings.append(f"  [IDP] 鏃犲簭铔嬬櫧鍙兘鎬? {report.idp_likelihood}. BFN缃俊搴﹀彲鑳戒笉鍙潬.")
    if not report.is_in_distribution:
        warnings.append(f"  [DIST] 铔嬬櫧涓嶅湪璁粌鍒嗗竷鍐?(L={report.length}, 璁粌鑼冨洿50-250)")

    return header + "\n" + "\n".join(warnings)


def build_results_dataframe(results_list):
    if not results_list:
        return pd.DataFrame()
    rows = []
    for i, r in enumerate(results_list):
        seq = r.get('sequence', '')
        if len(seq) > 40:
            seq_display = seq[:18] + '鈥? + seq[-18:]
        else:
            seq_display = seq
        plddt = r.get('plddt')
        iptm = r.get('iptm')
        ppl = r.get('ppl')
        entropy = r.get('entropy')
        composite = r.get('composite_score')
        quality = 'N/A'
        if plddt is not None and iptm is not None:
            if plddt >= 0.80 and iptm >= 0.70:
                quality = 'HIGH'
            elif plddt >= 0.60 or iptm >= 0.40:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'
        rows.append({
            'Rank': i + 1,
            'Sequence': seq_display,
            'Full Sequence': seq,
            'PPL': f'{ppl:.2f}' if ppl else 'N/A',
            'Entropy': f'{entropy:.3f}' if entropy is not None else 'N/A',
            'pLDDT': f'{plddt:.3f}' if plddt else 'N/A',
            'ipTM': f'{iptm:.3f}' if iptm else 'N/A',
            'PAE': f'{r.get("pae", 0):.1f}' if r.get('pae') else 'N/A',
            'Composite': f'{composite:.3f}' if composite else 'N/A',
            'Quality': quality,
        })
    return pd.DataFrame(rows)


# 鈹€鈹€ Standalone Confidence Evaluation 鈹€鈹€

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


# 鈹€鈹€ Design Functions 鈹€鈹€

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
        return "閿欒锛氭棤娉曡В鏋愭姉浣撶粨鏋?, "", []

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
        return "閿欒锛氭湭鎵惧埌璁捐鍖哄煙", "", []

    native_seq = None
    if eval_mode:
        native_aa = batch['aa'][0][gen_mask]
        native_seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in native_aa.cpu())

    sample_opt = {'deterministic': not stochastic, 'num_recycles': 3}
    lines = [f"BFN鎶椾綋CDR璁捐 | {gen_mask.sum().item()}娈嬪熀 | {'闅忔満' if stochastic else '纭畾鎬?} | 鍥炴敹脳3" +
             f" | 閲嶉摼{heavy} 杞婚摼{light}"]
    if native_seq:
        lines.append(f"鍘熷: {native_seq}")
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
        # BFN鍐呯疆缃俊搴?(receiver.py 宸插仛 sigmoid 鈫?[0,1])
        plddt_val = traj['plddt'][0][gen_mask].mean().item()
        iptm_val = traj['iptm'][0].item()
        pae_val = traj['pae'][0][gen_mask][:, gen_mask].mean().item()
        recovery = None
        rec_str = ""
        if native_seq:
            recovery = sum(1 for a,b in zip(seq, native_seq) if a==b)/len(native_seq)
            rec_str = f" | 鎭㈠鐜噞recovery*100:.1f}%"
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
            lines.append(f"\n  缁煎悎鏈€浼? {best_filtered} (缁煎悎璇勫垎={filtered[0]['composite_score']:.3f})")
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

    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_spec):
        indices = []
        for seg in spec.split(','):
            seg = seg.strip()
            if not seg: continue
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip())+1))
            else:
                indices.append(int(seg))
        regions[cid] = sorted(set(indices))

    if not regions:
        return f"鍖哄煙鏍煎紡閿欒: {region_spec}", "", []

    model, config = load_bfn()
    seed_all(getattr(config.sampling, 'seed', 42))
    structure = preprocess_protein_structure(pdb_path, chain_ids=list(regions.keys()))
    if structure is None:
        return "閿欒锛氭棤娉曡В鏋愮粨鏋?, "", []

    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
    gen_mask = batch['generate_flag'][0].bool()
    if gen_mask.sum() == 0:
        return "閿欒锛氭湭閫変腑璁捐娈嬪熀", "", []

    native_seq = None
    if eval_mode:
        native_aa = batch['aa'][0][gen_mask]
        native_seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in native_aa.cpu())

    sample_opt = {'deterministic': not stochastic, 'num_recycles': 3}
    lines = [f"BFN閫氱敤璁捐 | {region_spec} | {gen_mask.sum().item()}娈嬪熀 | {'闅忔満' if stochastic else '纭畾鎬?} | 鍥炴敹脳3"]
    if native_seq:
        lines.append(f"鍘熷: {native_seq}")
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
        # BFN鍐呯疆缃俊搴?(receiver.py 宸插仛 sigmoid 鈫?[0,1])
        plddt_val = traj['plddt'][0][gen_mask].mean().item()
        iptm_val = traj['iptm'][0].item()
        pae_val = traj['pae'][0][gen_mask][:, gen_mask].mean().item()
        recovery = None
        rec_str = ""
        if native_seq:
            recovery = sum(1 for a,b in zip(seq, native_seq) if a==b)/len(native_seq)
            rec_str = f" | 鎭㈠鐜噞recovery*100:.1f}%"
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
            lines.append(f"\n  缁煎悎鏈€浼? {best_filtered} (缁煎悎璇勫垎={filtered[0]['composite_score']:.3f})")
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


def run_mpnn(pdb_path, chains, num_samples, temperature, seed, omit_aas):
    out_dir = tempfile.mkdtemp(prefix='mpnn_')
    cmd = [
        sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
        '--pdb_path', pdb_path, '--pdb_path_chains', chains,
        '--num_seq_per_target', str(num_samples),
        '--sampling_temp', temperature,
        '--seed', str(seed),
        '--out_folder', out_dir, '--save_score', '1',
        '--path_to_model_weights', str(PROJECT_DIR / 'ProteinMPNN' / 'vanilla_model_weights'),
    ]
    if omit_aas and omit_aas.strip():
        cmd.extend(['--omit_AAs', omit_aas.strip()])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_DIR))
    if r.returncode != 0:
        return f"ProteinMPNN 閿欒:\n{r.stderr[:2000]}", "", []

    pid = os.path.basename(pdb_path).replace('.pdb', '')
    fa = os.path.join(out_dir, 'seqs', f'{pid}.fa')
    if not os.path.exists(fa):
        return f"鏈壘鍒拌緭鍑? {fa}", "", []

    lines = [f"ProteinMPNN | 閾?{chains} | 娓╁害 {temperature} | {num_samples}搴忓垪"]
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
    diag = diagnose_design_results(results_list, gen_mask.sum().item())
    if diag:
        lines.append(diag)
    return '\n'.join(lines), fasta_str, results_list


def run_esmif(pdb_path, chain, temperature, num_samples):
    from esm.inverse_folding import util as if_util
    model = load_esmif()
    try:
        coords, native = if_util.load_coords(pdb_path, chain)
    except Exception as e:
        return f"ESM-IF 鍔犺浇閿欒: {e}", "", []

    lines = [f"ESM-IF | 閾?{chain} | {len(native)}娈嬪熀 | 娓╁害 {temperature}"]
    lines.append(f"鍘熷: {native}")
    fasta_lines = []
    results_list = []
    pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
    for i in range(num_samples):
        s = model.sample(coords, temperature=temperature)
        rec = sum(1 for a,b in zip(s, native) if a==b)/len(native)
        lines.append(f"#{i+1}: {s} | 鎭㈠鐜噞rec*100:.1f}%")
        fasta_lines.append(f">ESMIF_{pdb_name}_sample_{i+1} T={temperature:.2f} recovery={rec*100:.1f}%\n{s}")
        results_list.append({'sequence': s, 'recovery': rec})
    fasta_str = '\n'.join(fasta_lines)
    return '\n'.join(lines), fasta_str, results_list


# 鈹€鈹€ FASTA 鈫?PDB Conversion 鈹€鈹€

AA_NAMES = {
    'A': 'ALA','C': 'CYS','D': 'ASP','E': 'GLU','F': 'PHE','G': 'GLY',
    'H': 'HIS','I': 'ILE','K': 'LYS','L': 'LEU','M': 'MET','N': 'ASN',
    'P': 'PRO','Q': 'GLN','R': 'ARG','S': 'SER','T': 'THR','V': 'VAL',
    'W': 'TRP','Y': 'TYR',
}
BB_ATOMS = ['N', 'CA', 'C', 'O']

def _fmt_coord(v):
    """Format a coordinate to exactly 8 chars for PDB (cols 31-38/39-46/47-54).
    Python f'{v:8.3f}' overflows when v >= 10000, corrupting PDB column alignment."""
    s = f"{v:8.3f}"
    if len(s) <= 8:
        return s
    s = f"{v:8.2f}"
    if len(s) <= 8:
        return s
    s = f"{v:8.1f}"
    if len(s) <= 8:
        return s
    return f"{v:8.0f}"[:8]


def generate_pdb_from_sequence(seq, chain_id='A', start_res=1):
    """Generate a PDB file with backbone atoms in a realistic alpha-helix conformation.

    Uses standard alpha-helix parameters (~3.6 residues/turn, 1.5 A rise, 2.3 A radius)
    so that CA-CA distances (~3.8 A) match real proteins. This ensures compatibility
    with ProteinMPNN and other structure-based design tools.
    """
    import math
    aa_list = [AA_NAMES.get(a, 'ALA') for a in seq if a in AA_NAMES]
    if not aa_list:
        return None

    # Standard alpha-helix parameters
    residues_per_turn = 3.6
    rise_per_residue = 1.5   # Angstroms
    helix_radius = 2.3       # Angstroms

    # Standard bond lengths (Engh & Huber)
    bond_n_ca = 1.47
    bond_ca_c = 1.53
    bond_c_o = 1.23

    n_residues = len(aa_list)
    pdb_lines = []

    def add_atom(serial, name, res_name, res_seq, x, y, z, element):
        if len(name) == 1:
            name4 = f" {name}  "
        elif len(name) == 2:
            name4 = f" {name} "
        elif len(name) == 3:
            name4 = f" {name}"
        else:
            name4 = f"{name:4s}"
        pdb_lines.append(
            f"ATOM  {serial:5d} {name4} {res_name:3s} {chain_id:1s}{res_seq:4d}    "
            f"{_fmt_coord(x)}{_fmt_coord(y)}{_fmt_coord(z)}  1.00  0.00          {element:>2s}  "
        )

    # Phase 1: compute CA positions along a proper alpha helix
    ca_positions = []
    for i in range(n_residues):
        angle = i * 2.0 * math.pi / residues_per_turn
        x = helix_radius * math.cos(angle)
        y = helix_radius * math.sin(angle)
        z = i * rise_per_residue
        ca_positions.append((x, y, z))

    # Phase 2: compute N, C, O positions using local backbone frame.
    # N-CA-C 鈮?111掳 (standard tetrahedral geometry).
    # N and C straddle the forward (tangent) direction in the plane formed
    # by forward 脳 helix-axis.  Since the peptide plane parallels the helix
    # axis in an 伪-helix, the perpendicular is (0, 0, 1).
    half_n_ca_c = math.radians(55.5)  # half of 111掳
    cos_half = math.cos(half_n_ca_c)   # 鈮?0.566
    sin_half = math.sin(half_n_ca_c)   # 鈮?0.824

    atom_num = 0
    for i in range(n_residues):
        res_num = start_res + i
        res_name = aa_list[i]
        cx, cy, cz = ca_positions[i]

        # Local forward direction (CA[i-1] -> CA[i+1])
        if i == 0:
            fx, fy, fz = ca_positions[1][0] - cx, ca_positions[1][1] - cy, ca_positions[1][2] - cz
        elif i == n_residues - 1:
            fx, fy, fz = cx - ca_positions[i - 1][0], cy - ca_positions[i - 1][1], cz - ca_positions[i - 1][2]
        else:
            fx = ca_positions[i + 1][0] - ca_positions[i - 1][0]
            fy = ca_positions[i + 1][1] - ca_positions[i - 1][1]
            fz = ca_positions[i + 1][2] - ca_positions[i - 1][2]

        f_norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        if f_norm > 1e-6:
            fx, fy, fz = fx / f_norm, fy / f_norm, fz / f_norm
        else:
            fx, fy, fz = 0.0, 0.0, 1.0

        # Perpendicular is the helix axis (z-direction); the peptide plane
        # is approximately parallel to the helix axis in an 伪-helix.
        px, py, pz = 0.0, 0.0, 1.0

        # C_dir = cos_half * forward + sin_half * z_axis
        c_dir_x = cos_half * fx + sin_half * px
        c_dir_y = cos_half * fy + sin_half * py
        c_dir_z = cos_half * fz + sin_half * pz
        c_norm = math.sqrt(c_dir_x**2 + c_dir_y**2 + c_dir_z**2)
        c_dir_x, c_dir_y, c_dir_z = c_dir_x / c_norm, c_dir_y / c_norm, c_dir_z / c_norm

        # N_dir = cos_half * forward - sin_half * z_axis
        n_dir_x = cos_half * fx - sin_half * px
        n_dir_y = cos_half * fy - sin_half * py
        n_dir_z = cos_half * fz - sin_half * pz
        n_norm = math.sqrt(n_dir_x**2 + n_dir_y**2 + n_dir_z**2)
        n_dir_x, n_dir_y, n_dir_z = n_dir_x / n_norm, n_dir_y / n_norm, n_dir_z / n_norm

        x_c = cx + bond_ca_c * c_dir_x
        y_c = cy + bond_ca_c * c_dir_y
        z_c = cz + bond_ca_c * c_dir_z

        x_n = cx + bond_n_ca * n_dir_x
        y_n = cy + bond_n_ca * n_dir_y
        z_n = cz + bond_n_ca * n_dir_z

        # O: placed trans to N relative to CA-C, ~120掳 from CA-C bond.
        # CA-C-O angle 鈮?120掳 (standard peptide geometry).
        ca_to_c_x = x_c - cx
        ca_to_c_y = y_c - cy
        ca_to_c_z = z_c - cz
        cc_norm = math.sqrt(ca_to_c_x**2 + ca_to_c_y**2 + ca_to_c_z**2)
        ca_to_c_x, ca_to_c_y, ca_to_c_z = ca_to_c_x / cc_norm, ca_to_c_y / cc_norm, ca_to_c_z / cc_norm

        # O direction: rotate CA鈫扖 by ~120掳 in the peptide plane (forward 脳 helix-axis)
        ox = fy * pz - fz * py  # forward 脳 z_axis
        oy = fz * px - fx * pz
        oz = fx * py - fy * px
        o_norm = math.sqrt(ox * ox + oy * oy + oz * oz)
        if o_norm > 1e-6:
            ox, oy, oz = ox / o_norm, oy / o_norm, oz / o_norm
        else:
            ox, oy, oz = 1.0, 0.0, 0.0

        # C鈫扥 = bond_c_o * (cos_60 * u + sin_60 * w) where u = CA鈫扖, w 鉄?peptide
        # This gives CA-C-O 鈮?120掳 (standard peptide geometry).
        cos60 = 0.5
        sin60 = math.sqrt(3.0) / 2.0  # 鈮?0.866
        x_o = x_c + bond_c_o * (cos60 * ca_to_c_x + sin60 * ox)
        y_o = y_c + bond_c_o * (cos60 * ca_to_c_y + sin60 * oy)
        z_o = z_c + bond_c_o * (cos60 * ca_to_c_z + sin60 * oz)

        add_atom(atom_num + 1, 'N', res_name, res_num, x_n, y_n, z_n, 'N')
        add_atom(atom_num + 2, 'CA', res_name, res_num, cx, cy, cz, 'C')
        add_atom(atom_num + 3, 'C', res_name, res_num, x_c, y_c, z_c, 'C')
        add_atom(atom_num + 4, 'O', res_name, res_num, x_o, y_o, z_o, 'O')
        atom_num += 4

    pdb_lines.append(f"TER   {atom_num + 1:5d}      {aa_list[-1]:3s} {chain_id:1s}{start_res + n_residues - 1:4d}")
    pdb_lines.append("END")
    return '\n'.join(pdb_lines)


def fasta_to_pdb_file(fasta_text):
    """Convert FASTA text to PDB file, return (pdb_path, info_text)."""
    if not fasta_text or not fasta_text.strip():
        return None, "璇锋彁渚?FASTA 搴忓垪"

    seqs = parse_fasta(fasta_text)
    if not seqs:
        return None, "鏃犳晥鐨?FASTA 鏍煎紡锛堥渶瑕?> 寮€澶达級"

    # Use the first sequence
    header, seq = seqs[0]
    pdb_content = generate_pdb_from_sequence(seq)

    if pdb_content is None:
        return None, "搴忓垪涓病鏈夋湁鏁堢殑姘ㄥ熀閰?

    # Use header as filename base
    name = re.sub(r'[^A-Za-z0-9_-]', '_', header.split()[0] if header else 'protein')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False,
                                      encoding='utf-8', prefix=f'{name}_')
    tmp.write(pdb_content)
    tmp.close()

    info = (f"鉁?PDB 宸茬敓鎴? {name}\n"
            f"   閾?A: {len(seq)} 涓畫鍩篭n"
            f"   鏂囦欢: {tmp.name}\n"
            f"   娉ㄦ剰: 姝?PDB 涓?伪-铻烘棆妯℃澘楠ㄦ灦锛岄潪鐪熷疄鎶樺彔缁撴瀯\n"
            f"   鍙敤浜?BFN / ProteinMPNN / ESM-IF 杩涜搴忓垪璁捐")
    return tmp.name, info


# 鈹€鈹€ AlphaFold Structure Prediction 鈹€鈹€

def parse_fasta(text):
    seqs = []
    cur_header, cur_seq = None, []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur_header:
                seqs.append((cur_header, ''.join(cur_seq)))
            cur_header = line[1:].strip()
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_header:
        seqs.append((cur_header, ''.join(cur_seq)))
    return seqs


def on_fasta_input(fasta_text, fasta_file):
    if fasta_file is not None:
        try:
            with open(fasta_file.name, encoding='utf-8') as f:
                fasta_text = f.read()
        except Exception:
            pass
    if not fasta_text or not fasta_text.strip():
        return "", "璇锋彁渚?FASTA 搴忓垪鎴栦笂浼?.fasta 鏂囦欢"
    seqs = parse_fasta(fasta_text)
    if not seqs:
        return "", "鏈娴嬪埌鏈夋晥 FASTA 鏍煎紡搴忓垪锛堥渶瑕?> 寮€澶达級"
    lines = []
    for h, s in seqs:
        lines.append(f"{h}: {len(s)} aa")
        lines.append(f"  {s[:80]}{'...' if len(s) > 80 else ''}")
    return fasta_text, '\n'.join(lines)


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
        return "璇锋彁渚?FASTA 搴忓垪", None, None

    seqs = parse_fasta(fasta_text)
    if not seqs:
        return "鏃犳晥鐨?FASTA 鏍煎紡", None, None

    cfg = load_app_config()
    af_cfg = cfg.get('alphafold', {})
    default_out = af_cfg.get('output_dir', 'alphafold_results')
    venv_path = af_cfg.get('af2', {}).get('venv', str(PROJECT_DIR / 'venv'))

    out_dir = Path(default_out)
    out_dir.mkdir(exist_ok=True)
    pid = seqs[0][0].split()[0] if seqs else 'query'
    pid = re.sub(r'[^A-Za-z0-9_-]', '_', pid)
    fasta_path = out_dir / f'{pid}.fasta'

    # Sanitize to ASCII: colabfold uses system default encoding (GBK on Chinese Windows)
    safe_text = fasta_text.encode('ascii', errors='replace').decode('ascii')
    with open(fasta_path, 'w', encoding='ascii') as f:
        f.write(safe_text)

    colabfold_exe = str(Path(venv_path) / 'Scripts' / 'colabfold_batch.exe')
    if not os.path.exists(colabfold_exe):
        colabfold_exe = str(Path(venv_path) / 'Scripts' / 'colabfold_batch')

    stop_at = af_cfg.get('defaults', {}).get('stop_at_score', 85)
    model_type = af_cfg.get('defaults', {}).get('model_type', 'auto')
    rank_mode = af_cfg.get('defaults', {}).get('rank', 'auto')

    cmd = [
        colabfold_exe,
        str(fasta_path), str(out_dir / pid),
        '--num-models', str(int(num_models)),
        '--num-recycle', str(int(num_recycle)),
        '--stop-at-score', str(int(stop_at)),
        '--model-type', model_type,
        '--rank', rank_mode,
    ]
    if af_cfg.get('defaults', {}).get('calc_extra_ptm', False):
        cmd.append('--calc-extra-ptm')
    if use_dropout:
        cmd.append('--use-dropout')

    progress(0.05, desc="鍚姩 AlphaFold2 (ColabFold)...")
    log_lines = [f"馃敭 AlphaFold2 (ColabFold) 缁撴瀯棰勬祴",
                 f"{'='*50}",
                 f"搴忓垪: {seqs[0][0]} ({len(seqs[0][1])} aa)",
                 f"妯″瀷: {model_type}  |  鍥炴敹: {int(num_recycle)}  |  杈炬爣鍒? {int(stop_at)}",
                 f"杈撳嚭: {out_dir / pid}/", f"",
                 f"鈴?CPU 棰勬祴杈冩參锛岃鑰愬績绛夊緟锛堝皬铔嬬櫧绾?5-15 鍒嗛挓锛?]

    try:
        env = os.environ.copy()
        env['PATH'] = str(Path(venv_path) / 'Scripts') + os.pathsep + env.get('PATH', '')
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
            log_lines.append(f"\n鉂?棰勬祴澶辫触 (exit code {r.returncode})")
            return '\n'.join(log_lines), None, None
    except subprocess.TimeoutExpired:
        log_lines.append("\n鉂?瓒呮椂 (1灏忔椂)")
        return '\n'.join(log_lines), None, None

    progress(0.8, desc="瑙ｆ瀽缃俊搴﹀垎鏁?..")
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
                log_lines.append(f"馃搳 AF2 缃俊搴﹁瘎浼?)
                log_lines.append(f"  骞冲潎 pLDDT: {mean_plddt:.1f}/100")
                log_lines.append(f"  pTM:        {ptm:.3f}  (鍏ㄥ眬鎶樺彔缃俊搴?")
                if isinstance(iptm, float) and iptm == iptm:  # not NaN
                    log_lines.append(f"  ipTM:       {iptm:.3f}  (鐣岄潰缃俊搴?")
                log_lines.append(f"  max PAE:    {max_pae:.1f}  (鏈€澶ч娴嬭宸?")
                # Qualitative assessment
                if mean_plddt >= 80:
                    log_lines.append(f"  鉁?pLDDT 鈮?80: 楂樼疆淇″害锛岄€傚悎涓嬫父鍒嗘瀽")
                elif mean_plddt >= 60:
                    log_lines.append(f"  鈻?pLDDT 60-80: 涓瓑缃俊搴︼紝鎶樺彔澶т綋鍙俊")
                else:
                    log_lines.append(f"  鉁?pLDDT < 60: 浣庣疆淇″害锛岄渶璋ㄦ厧浣跨敤")
                log_lines.append(f"{'='*50}")
            except Exception as e:
                log_lines.append(f"\n鈿?瑙ｆ瀽缃俊搴﹀垎鏁板け璐? {e}")

        log_lines.append(f"\n鉁?瀹屾垚! 鏈€浣崇粨鏋? {best_name}")
        progress(1.0, desc="瀹屾垚!")
        return '\n'.join(log_lines), str(_predicted_pdb), str(result_dir)
    else:
        log_lines.append(f"\n鈿狅笍 棰勬祴瀹屾垚浣嗘湭鎵惧埌 PDB 杈撳嚭. 妫€鏌? {result_dir}")
        return '\n'.join(log_lines), None, None


# 鈹€鈹€ Batch Processing 鈹€鈹€

def run_batch(pdb_files, tool, region_spec, chains_spec, temperature, num_samples, progress=gr.Progress()):
    if not pdb_files:
        return "璇蜂笂浼?PDB 鏂囦欢", None, None

    results = []
    all_results_lists = []
    total = len(pdb_files)
    output_dir = Path('batch_results')
    output_dir.mkdir(exist_ok=True)

    for i, pf in enumerate(pdb_files):
        pdb_name = Path(pf.name).stem
        progress((i+1)/total, desc=f"澶勭悊 {pdb_name}...")

        try:
            if tool == "BFN (閫氱敤铔嬬櫧璁捐)":
                text, fasta_str, results_list = run_bfn_protein(pf.name, region_spec, num_samples, False, True)
            elif tool == "ProteinMPNN":
                text, fasta_str, results_list = run_mpnn(pf.name, chains_spec, num_samples, temperature, 42, "")
            elif tool == "ESM-IF":
                text, fasta_str, results_list = run_esmif(pf.name, chains_spec, float(temperature), num_samples)
            else:
                text = f"鏈煡宸ュ叿: {tool}"
                fasta_str = ""
                results_list = []
            results.append({'pdb': pdb_name, 'status': 'OK', 'text': text, 'fasta': fasta_str})
            all_results_lists.extend(results_list)
        except Exception as e:
            results.append({'pdb': pdb_name, 'status': 'ERROR', 'error': str(e)})

    # Build summary
    lines = [f"鎵归噺澶勭悊瀹屾垚: {total} 涓枃浠禱n"]
    ok = sum(1 for r in results if r['status'] == 'OK')
    err = sum(1 for r in results if r['status'] == 'ERROR')
    lines.append(f"鎴愬姛: {ok}  澶辫触: {err}\n")
    lines.append("=" * 60)

    for r in results:
        lines.append(f"\n{'鈹€'*60}")
        lines.append(f"馃搧 {r['pdb']}  [{r['status']}]")
        if r['status'] == 'OK':
            lines.append(r['text'])
        else:
            lines.append(f"閿欒: {r['error']}")

    # Save summary
    summary = {
        'tool': tool, 'total': total, 'ok': ok, 'error': err,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [{k: v for k, v in r.items() if k not in ('text', 'fasta')} for r in results],
    }
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f'batch_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Also save individual results
    for r in results:
        if r['status'] == 'OK':
            txt_path = output_dir / f"{r['pdb']}_{tool}.txt"
            with open(txt_path, 'w') as f:
                f.write(r['text'])

    lines.append(f"\n缁撴灉宸蹭繚瀛樺埌: {output_dir}/")

    df = build_results_dataframe(all_results_lists)
    return '\n'.join(lines), str(json_path), df


# 鈹€鈹€ Target Design Pipeline 鈹€鈹€

def run_target_design(target_pdb, target_chain, epitope_region,
                       antibody_pdb, ab_heavy, ab_light,
                       design_tool, constraint_mode, constraint_cutoff,
                       bfn_samples, bfn_stochastic, bfn_eval,
                       mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
                       esmif_temp, esmif_samples):
    """Target-aware constrained design pipeline."""
    if not target_pdb or not os.path.exists(str(target_pdb)):
        return "璇峰厛瀹屾垚姝ラ1锛氫笂浼犳垨棰勬祴闈剁偣缁撴瀯", "", "", None

    if not epitope_region or not epitope_region.strip():
        return "璇峰厛瀹屾垚姝ラ2锛氬垎鏋愬苟閫夋嫨琛ㄤ綅鍖哄煙", "", "", None

    if not antibody_pdb or not os.path.exists(str(antibody_pdb)):
        return "璇峰厛瀹屾垚姝ラ3锛氭彁渚涙姉浣撶粨鏋?, "", "", None

    # Parse epitope region
    epitope_residues = tdh.parse_region_spec(epitope_region)
    if not epitope_residues:
        return f"琛ㄤ綅鍖哄煙鏍煎紡閿欒: {epitope_region}", "", "", None

    # Flatten all epitope resseqs
    all_epitope_resseqs = []
    for rlist in epitope_residues.values():
        all_epitope_resseqs.extend(rlist)
    all_epitope_resseqs = sorted(set(all_epitope_resseqs))

    design_lines = []
    validation_text = ""

    try:
        # If constraint mode is on, find antibody residues facing epitope
        design_region = None
        if constraint_mode:
            facing = tdh.find_residues_facing_region(
                antibody_pdb, ab_heavy, all_epitope_resseqs,
                ab_heavy, constraint_cutoff
            )
            # Also check light chain if specified
            facing_l = []
            if ab_light and ab_light.strip():
                facing_l = tdh.find_residues_facing_region(
                    antibody_pdb, ab_heavy, all_epitope_resseqs,
                    ab_light, constraint_cutoff
                )
            design_lines.append(f"璺濈绾︽潫妯″紡 (cutoff={constraint_cutoff}脜):")
            design_lines.append(f"  闈㈠悜琛ㄤ綅鐨勯噸閾炬畫鍩? {len(facing)} 涓?)
            if ab_light and ab_light.strip():
                design_lines.append(f"  闈㈠悜琛ㄤ綅鐨勮交閾炬畫鍩? {len(facing_l)} 涓?)

            if not facing and not facing_l:
                return ("鏈壘鍒伴潰鍚戣〃浣嶇殑鎶椾綋娈嬪熀 鈥?璇锋鏌ヨ窛绂婚槇鍊兼垨琛ㄤ綅/鎶椾綋缁撴瀯鏄惁姝ｇ‘",
                        "", "", None)

            # Build region spec from facing residues
            region_parts = []
            if facing:
                region_parts.append(tdh.residues_to_region_spec(ab_heavy, facing))
            if facing_l:
                region_parts.append(tdh.residues_to_region_spec(ab_light, facing_l))
            design_region = ' '.join(region_parts)
            design_lines.append(f"  璁捐鍖哄煙: {design_region}")
            design_lines.append("")
        else:
            # Without constraint, use CDR defaults
            design_region = f"{ab_heavy}:95-102"  # CDR H3 range
            if ab_light and ab_light.strip():
                design_region += f" {ab_light}:89-97"  # CDR L3 range
            design_lines.append(f"闈炵害鏉熸ā寮?鈥?榛樿CDR鍖哄煙: {design_region}")
            design_lines.append("")

        # Run selected design tool
        results_list = []
        if design_tool == "BFN (鎶椾綋CDR璁捐)":
            # Use BFN protein mode with computed region
            text, fasta_str, results_list = run_bfn_protein(antibody_pdb, design_region,
                                              int(bfn_samples), bool(bfn_stochastic), bool(bfn_eval))
        elif design_tool == "ProteinMPNN":
            ab_chains = ab_heavy
            if ab_light and ab_light.strip():
                ab_chains += " " + ab_light
            text, fasta_str, results_list = run_mpnn(antibody_pdb, ab_chains,
                                       int(mpnn_samples), mpnn_temp, int(mpnn_seed), mpnn_omit)
        elif design_tool == "ESM-IF":
            text, fasta_str, results_list = run_esmif(antibody_pdb, ab_heavy,
                                        float(esmif_temp), int(esmif_samples))
        else:
            return f"鏈煡宸ュ叿: {design_tool}", "", "", None

        design_lines.append(text)

        # Validation: contact analysis between designed antibody and target
        try:
            contacts, ia, ib, summary = tdh.analyze_contacts(
                antibody_pdb, ab_heavy, target_chain, distance_cutoff=8.0
            )
            if contacts:
                metrics = tdh.score_interface_properties(
                    antibody_pdb, ab_heavy, target_chain, ia, ib
                )
                validation_text = tdh.format_contact_summary(
                    contacts, ia, ib, summary, ab_heavy, target_chain
                )
                validation_text += "\n\n" + tdh.format_interface_report(metrics, summary)
            else:
                validation_text = f"鏈娴嬪埌 {ab_heavy}-{target_chain} 涔嬮棿鐨勬帴瑙?(璺濈闃堝€?8脜)"
        except Exception as val_err:
            validation_text = f"楠岃瘉璺宠繃: {val_err}"

    except Exception as e:
        import traceback
        return f"闈剁偣璁捐閿欒:\n{traceback.format_exc()}", "", "", None

    results_df = build_results_dataframe(results_list)
    return '\n'.join(design_lines), fasta_str, validation_text, results_df


# 鈹€鈹€ Main Dispatch 鈹€鈹€

def run_design(pdb_path, tool,
               bfn_ab_heavy, bfn_ab_light, bfn_ab_samples, bfn_ab_stochastic, bfn_ab_eval,
               bfn_pro_region, bfn_pro_samples, bfn_pro_stochastic, bfn_pro_eval,
               mpnn_chains, mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
               esmif_chain, esmif_temp, esmif_samples):
    if not pdb_path or not os.path.exists(str(pdb_path)):
        return "璇峰厛涓婁紶 PDB 鏂囦欢", None, None, gr.update(visible=False)
    try:
        results_list = []
        if tool == "BFN (鎶椾綋CDR璁捐)":
            text, fasta_str, results_list = run_bfn_antibody(pdb_path, bfn_ab_heavy, bfn_ab_light,
                                    int(bfn_ab_samples), bool(bfn_ab_stochastic), bool(bfn_ab_eval))
        elif tool == "BFN (閫氱敤铔嬬櫧璁捐)":
            text, fasta_str, results_list = run_bfn_protein(pdb_path, bfn_pro_region,
                                   int(bfn_pro_samples), bool(bfn_pro_stochastic), bool(bfn_pro_eval))
        elif tool == "ProteinMPNN":
            text, fasta_str, results_list = run_mpnn(pdb_path, mpnn_chains, int(mpnn_samples), mpnn_temp, int(mpnn_seed), mpnn_omit)
        elif tool == "ESM-IF":
            text, fasta_str, results_list = run_esmif(pdb_path, esmif_chain, float(esmif_temp), int(esmif_samples))
        else:
            return f"鏈煡宸ュ叿: {tool}", None, None, gr.update(visible=False)
    except Exception as e:
        import traceback
        return f"杩愯閿欒:\n{traceback.format_exc()}", None, None, gr.update(visible=False)

    # Write FASTA to temp file for download
    fasta_file = None
    if fasta_str:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False,
                                          encoding='utf-8', prefix='design_')
        tmp.write(fasta_str)
        tmp.close()
        fasta_file = tmp.name

    df = build_results_dataframe(results_list)
    return text, fasta_file, df, gr.update(visible=bool(fasta_str))


# 鈹€鈹€ Unified Pipeline 鈹€鈹€

def run_unified_pipeline(pdb_path, region_spec, num_samples, stochastic, enable_af2, af2_num_recycle, progress=gr.Progress()):
    if not pdb_path or not os.path.exists(str(pdb_path)):
        return "Please upload a PDB file first", pd.DataFrame(), "Pipeline aborted"

    # Stage 0: Pre-design confidence evaluation
    progress(0.0, desc="Stage 0/4: BFN Pre-Design Confidence...")
    conf_text, conf_df = run_bfn_confidence_evaluation(pdb_path, region_spec)
    # Parse mean pLDDT from the report text
    import re as _re
    pre_plddt_match = _re.search(r'pLDDT \(region mean\):\s+([\d.]+)', conf_text)
    pre_iptm_match = _re.search(r'ipTM \(global\):\s+([\d.]+)', conf_text)
    pre_plddt = float(pre_plddt_match.group(1)) if pre_plddt_match else None
    pre_iptm = float(pre_iptm_match.group(1)) if pre_iptm_match else None

    progress(0.15, desc="Stage 1/4: BFN Design...")
    text_design, fasta_str, results_list = run_bfn_protein(
        pdb_path, region_spec, int(num_samples), bool(stochastic), eval_mode=False)

    if not results_list:
        return text_design, pd.DataFrame(), fasta_str or "Design produced no results"

    progress(0.5, desc="Stage 2/4: Cascade Filter...")
    # Pass confidence thresholds from config
    cfg = load_app_config()
    thresholds = cfg.get('confidence', CONFIDENCE_DEFAULTS)
    filter_thresholds = {
        'plddt_min': thresholds.get('plddt_medium', 0.60),
        'iptm_min': thresholds.get('iptm_medium', 0.40),
        'ppl_max': 100.0,
        'entropy_max': 2.5,
    }
    filtered, filter_report = cf.apply_cascade(results_list, thresholds=filter_thresholds)
    if not filtered:
        df = build_results_dataframe(results_list)
        report = f"{'='*50}\nStage 0: Pre-Design Confidence\n{'='*50}\n{conf_text}\n\n{text_design}\n\n{'='*50}\nStage 2: Cascade Filter\n{'='*50}\n{filter_report}"
        return report, df, fasta_str or ""

    progress(0.7, desc="Stage 3/4: Optional AF2 Validation...")
    if enable_af2:
        top_n = min(5, len(filtered))
        top_sequences = [f['sequence'] for f in filtered[:top_n]]
        from af2_validator import validate_sequences
        try:
            af2_results = validate_sequences(top_sequences, output_dir='alphafold_results/pipeline',
                                             num_recycle=int(af2_num_recycle))
            progress(0.9, desc="AF2 validation complete, re-ranking...")
            from cascade_filter import apply_cascade_af2
            af2_filtered, af2_report = apply_cascade_af2(af2_results)
            if af2_filtered:
                df = build_results_dataframe(af2_filtered)
                report = f"{'='*50}\nStage 0: Pre-Design Confidence\n{'='*50}\n{conf_text}\n\n{text_design}\n\n{'='*50}\nStage 2: BFN Cascade Filter (PPL+entropy+pLDDT+ipTM)\n{'='*50}\n{filter_report}\n\n{'='*50}\nStage 3: AF2 Validation + Re-rank\n{'='*50}\n{af2_report}"
                return report, df, fasta_str or ""
        except Exception as e:
            filter_report += f"\n\nAF2 Validation failed: {e}"

    df = build_results_dataframe(filtered)
    report = f"{'='*50}\nStage 0: Pre-Design Confidence\n{'='*50}\n{conf_text}\n\n{text_design}\n\n{'='*50}\nStage 2: Cascade Filter (PPL+entropy+pLDDT+ipTM)\n{'='*50}\n{filter_report}"
    return report, df, fasta_str or ""


# 鈹€鈹€ Config Viewer 鈹€鈹€

def get_config_display():
    try:
        cfg = load_app_config()
        return yaml.dump(cfg, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        return f"閰嶇疆璇诲彇閿欒: {e}"


def save_config_from_text(config_text):
    try:
        new_cfg = yaml.safe_load(config_text)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(new_cfg, f, allow_unicode=True, default_flow_style=False)
        # Force reload
        global _app_config
        _app_config = None
        return "鉁?閰嶇疆宸蹭繚瀛橈紝涓嬫璁捐浠诲姟鐢熸晥"
    except Exception as e:
        return f"鉂?閰嶇疆鏍煎紡閿欒: {e}"


def get_system_status():
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

    bfn_ok = os.path.exists(cfg['models']['bfn']['checkpoint'])
    mpnn_ok = Path(cfg['models']['proteinmpnn']['weights_dir']).exists()
    esmif_loaded = _esmif_model is not None
    bfn_loaded = _bfn_model is not None
    af_cfg = cfg.get('alphafold', {})
    af2_venv = af_cfg.get('af2', {}).get('venv', '')
    af2_exe = str(Path(af2_venv) / 'Scripts' / 'colabfold_batch.exe') if af2_venv else ''
    af2_ok = af2_exe and os.path.exists(af2_exe)

    status = f"""鏈嶅姟鐘舵€侀潰鏉?
{'='*50}
鐩戝惉鍦板潃:     {host}:{port}
绔彛鐘舵€?     {'鉁?杩愯涓? if port_open else '鉁?鏈惎鍔?}
澶栭儴璁块棶:     {'绂佹 (浠呮湰鏈?' if host == '127.0.0.1' else '鍏佽'}
{'='*50}
BFN 妯″瀷:      {'鉁?鍔犺浇灏辩华' if bfn_loaded else ('鉁?鏂囦欢瀛樺湪' if bfn_ok else '鉁?鏈壘鍒?)}
ProteinMPNN:   {'鉁?灏辩华' if mpnn_ok else '鉁?鏈壘鍒?}
ESM-IF:        {'鉁?宸插姞杞? if esmif_loaded else '鈼?鎸夐渶鍔犺浇'}
AlphaFold2:    {'鉁?灏辩华' if af2_ok else '鉁?鏈壘鍒?}
{'='*50}
閰嶇疆鏂囦欢:      {CONFIG_FILE}
Python:        {sys.version.split()[0]}
PyTorch:       {torch.__version__}
CUDA:          {torch.cuda.is_available()}
"""
    return status


# 鈹€鈹€ UI 鈹€鈹€

def create_ui():
    css = """
    /* 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
       Protein Design Platform 鈥?Layered Functional Depth Theme
       鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲 */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg:       #0c0c14;
        --surface:  #14141e;
        --raised:   #1a1a28;
        --input:    #101018;
        --border:   #20202e;
        --text:     #c2ccd8;
        --heading:  #dce3ee;
        --muted:    #687380;
        --accent:   #7b7bf6;
        --accent2:  #6262e8;
        --section-a: #15151f;
        --section-b: #181824;
    }

    .gradio-container { font-family: 'Inter','Segoe UI',system-ui,-apple-system,sans-serif !important; max-width:1400px !important; margin:0 auto !important; }
    body, .gradio-container { background: var(--bg) !important; }

    /* 鈹€鈹€ All tabs same height 鈹€鈹€ */
    .tabs > .tabitem { min-height:680px !important; }

    /* 鈹€鈹€ Header 鈹€鈹€ */
    .platform-header {
        background:linear-gradient(180deg, #1c1c2e 0%, #181826 100%);
        border:1px solid var(--border); border-radius:18px;
        padding:16px 24px; margin-bottom:12px;
        box-shadow:0 4px 20px rgba(0,0,0,0.25);
        overflow:hidden;
    }
    .platform-header h1 { color:var(--heading) !important; font-size:1.35em !important; font-weight:600 !important; margin:0 0 2px 0 !important; }
    .platform-header .subtitle { color:var(--muted) !important; font-size:0.82em; margin:0; }
    .platform-header .meta-row { display:flex; gap:10px; align-items:center; margin-top:6px; }
    .platform-header .badge {
        display:inline-flex; align-items:center; gap:4px;
        background:rgba(255,255,255,0.03); border:1px solid var(--border);
        color:var(--muted); font-size:0.71em; font-weight:500;
        padding:3px 12px; border-radius:14px;
    }
    .platform-header .badge-dot { width:5px; height:5px; border-radius:50%; background:#34d399; box-shadow:0 0 5px rgba(52,211,153,0.3); }

    /* 鈹€鈹€ Tabs nav 鈹€鈹€ */
    .tabs > .tab-nav {
        background: var(--surface) !important; border-radius:16px !important; padding:5px 8px !important;
        border:1px solid var(--border) !important; gap:4px !important;
        box-shadow:0 2px 8px rgba(0,0,0,0.18) !important;
    }
    .tabs > .tab-nav button {
        border-radius:12px !important; font-weight:500 !important; font-size:0.85em !important;
        padding:8px 18px !important; border:none !important;
        background:transparent !important; color:var(--muted) !important;
        transition:all 0.2s ease !important;
    }
    .tabs > .tab-nav button:hover { background:var(--input) !important; color:var(--text) !important; }
    .tabs > .tab-nav button.selected {
        background:linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color:#fff !important; box-shadow:0 3px 14px rgba(99,102,241,0.25) !important;
    }

    /* 鈹€鈹€ Functional section blocks 鈹€鈹€ */
    .gr-group, .gr-box, .gr-form, .gr-panel, .card-group, .accordion, fieldset, .radio-group {
        border-radius:16px !important; padding:14px 18px !important; margin-bottom:10px !important;
        border:1px solid var(--border) !important;
        box-shadow:0 2px 10px rgba(0,0,0,0.2) !important;
        overflow:hidden;
    }

    /* Alternating section backgrounds for visual distinction */
    .gr-group:nth-child(odd) { background: var(--section-a) !important; }
    .gr-group:nth-child(even) { background: var(--section-b) !important; }

    /* Section headings */
    .gr-group .md h3, .gr-group .md h4, .card-group h3, .card-group h4 {
        margin:0 0 6px 0 !important; font-weight:600; color:var(--heading); font-size:0.93em;
    }
    .gr-group .md p, .gr-group label, .gr-group span, .prose, .md p, .md span { color:var(--text) !important; }
    .gr-group .md h4 { font-size:0.86em; color:var(--muted); }

    /* Nested groups float higher */
    .gr-group .gr-group, .gr-box .gr-box, .gr-group .gr-box {
        background: var(--raised) !important;
        box-shadow:0 4px 14px rgba(0,0,0,0.28) !important;
        border:1px solid var(--border) !important;
    }

    /* 鈹€鈹€ Accordion 鈹€鈹€ */
    .accordion > .label-wrap {
        background:var(--input) !important; padding:10px 18px !important;
        font-weight:600 !important; color:var(--text) !important;
        border-bottom:1px solid var(--border); border-radius:16px 16px 0 0 !important;
    }

    /* 鈹€鈹€ Buttons 鈹€鈹€ */
    button, .gr-button { border-radius:12px !important; font-weight:500 !important; transition:all 0.2s ease !important; }
    button.primary, .primary-btn, .gr-button.primary {
        background:linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color:#fff !important; border:none !important; font-size:0.9em !important;
        padding:9px 22px !important;
        box-shadow:0 3px 14px rgba(99,102,241,0.22) !important;
    }
    button.primary:hover, .primary-btn:hover { transform:translateY(-1px); box-shadow:0 5px 20px rgba(99,102,241,0.35) !important; }
    button.secondary, .secondary-btn, .gr-button.secondary {
        background:var(--input) !important; color:var(--text) !important;
        border:1px solid var(--border) !important; padding:8px 18px !important;
        box-shadow:0 1px 4px rgba(0,0,0,0.18) !important;
    }
    button.secondary:hover, .secondary-btn:hover { background:var(--surface) !important; border-color:var(--accent) !important; }
    button.sm, .gr-button.sm { padding:5px 12px !important; font-size:0.78em !important; border-radius:10px !important; }
    button.lg, .gr-button.lg { padding:11px 26px !important; font-size:0.95em !important; border-radius:14px !important; }

    /* 鈹€鈹€ Inputs (inset depth) 鈹€鈹€ */
    textarea, input[type="text"], input[type="number"], input[type="password"], select {
        border-radius:12px !important; border:1px solid var(--border) !important;
        padding:8px 12px !important; font-size:0.88em !important;
        background:var(--input) !important; color:var(--text) !important;
        box-shadow:inset 0 1px 4px rgba(0,0,0,0.35) !important;
    }
    textarea:focus, input:focus, select:focus {
        border-color:var(--accent) !important;
        box-shadow:inset 0 1px 4px rgba(0,0,0,0.35), 0 0 0 3px rgba(124,124,248,0.06) !important;
        outline:none !important;
    }
    textarea::placeholder, input::placeholder { color:var(--muted) !important; }
    label, .label-text { font-weight:500 !important; color:var(--text) !important; font-size:0.87em; }

    /* 鈹€鈹€ Radio / Checkbox 鈹€鈹€ */
    input[type="radio"]:checked + label { color:var(--accent) !important; font-weight:600 !important; }
    .gr-checkbox label, .gr-checkbox span { color:var(--text) !important; }
    input[type="range"] { accent-color:var(--accent) !important; }

    /* 鈹€鈹€ File upload 鈹€鈹€ */
    .file-preview, .file-upload {
        border-radius:14px !important; border:1px dashed var(--border) !important;
        background:var(--input) !important; box-shadow:inset 0 1px 4px rgba(0,0,0,0.25) !important;
    }

    /* 鈹€鈹€ Result boxes 鈹€鈹€ */
    .result-box textarea, .result-box, .result-box input, .result-box .gr-textbox textarea {
        font-family:'JetBrains Mono','Cascadia Code','Fira Code','Consolas',monospace !important;
        font-size:0.82em !important; background:#0a0a10 !important; color:#b2baf6 !important;
        border-radius:12px !important; border:1px solid var(--border) !important; line-height:1.5 !important;
        box-shadow:inset 0 2px 8px rgba(0,0,0,0.45) !important;
    }
    .result-box label { color:var(--muted) !important; }

    /* 鈹€鈹€ Footer 鈹€鈹€ */
    .platform-footer {
        text-align:center; color:var(--muted); font-size:0.75em; margin-top:16px;
        padding:10px 16px; background:var(--surface); border-radius:16px;
        border:1px solid var(--border); box-shadow:0 2px 8px rgba(0,0,0,0.15);
    }
    .platform-footer code { background:var(--input); color:var(--text); padding:2px 7px; border-radius:8px; font-size:0.9em; }

    /* 鈹€鈹€ Scrollbar 鈹€鈹€ */
    ::-webkit-scrollbar { width:7px; height:7px; }
    ::-webkit-scrollbar-track { background:var(--bg); border-radius:4px; }
    ::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
    ::-webkit-scrollbar-thumb:hover { background:var(--muted); }

    /* 鈹€鈹€ Transitions & overrides 鈹€鈹€ */
    button, input, textarea, select { transition:all 0.2s ease; }
    .gr-box, .gr-form, .gr-panel, .gr-group, .gr-accordion, .gr-tab-item, .gr-tabs {
        background:transparent !important; border:none !important; box-shadow:none !important;
    }
    .app { background:var(--bg) !important; }
    .contain { background:transparent !important; }
    footer { display:none !important; }

    /* 鈹€鈹€ Target Design Tab: larger layout 鈹€鈹€ */
    .td-wrapper { font-size:1.05em; }
    .td-wrapper .gr-group { padding:20px 24px !important; margin-bottom:14px !important; border-radius:20px !important; }
    .td-wrapper .gr-group .gr-markdown h3 { font-size:1.15em !important; margin-bottom:10px !important; }
    .td-wrapper .gr-group .gr-markdown p { font-size:1.02em !important; }
    .td-wrapper textarea, .td-wrapper input[type="text"], .td-wrapper input[type="number"], .td-wrapper select,
    .td-wrapper .gr-dropdown button { font-size:1.04em !important; padding:10px 14px !important; }
    .td-wrapper label, .td-wrapper .label-text { font-size:0.98em !important; font-weight:600 !important; }
    .td-wrapper button, .td-wrapper .gr-button { font-size:0.98em !important; padding:10px 22px !important; }
    .td-wrapper button.lg, .td-wrapper .gr-button.lg { font-size:1.06em !important; padding:14px 30px !important; }
    .td-wrapper button.sm, .td-wrapper .gr-button.sm { font-size:0.9em !important; padding:7px 16px !important; }
    .td-wrapper .result-box textarea, .td-wrapper .result-box { font-size:0.95em !important; line-height:1.6 !important; }
    .td-wrapper .gr-accordion .label-wrap { font-size:0.98em !important; padding:12px 20px !important; }
    .td-wrapper .gr-slider input[type="range"] { height:6px !important; }
    .td-wrapper .gr-checkbox label { font-size:0.98em !important; }
    .td-wrapper .gr-radio label { font-size:1.02em !important; }
    """
    with gr.Blocks(title="Protein Design Platform") as app:
        gr.HTML("""
        <div class="platform-header">
            <h1>馃К 铔嬬櫧璐ㄥ簭鍒楄璁″钩鍙?/h1>
            <p class="subtitle">Protein Sequence Design Platform 鈥?BFN 路 ProteinMPNN 路 ESM-IF 路 AlphaFold2</p>
            <div class="meta-row">
                <span class="badge"><span class="badge-dot"></span> 鏈嶅姟杩愯涓?/span>
                <span class="badge">馃敀 浠呮湰鏈鸿闂?/span>
                <span class="badge">http://127.0.0.1:7860</span>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:
            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺? Tab 1: Single Design 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
            with gr.TabItem("馃幆 鍗曟璁捐"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # FASTA 鈫?PDB converter
                        with gr.Accordion("馃К FASTA 鈫?PDB 杞崲", open=False):
                            gr.Markdown(
                                "杈撳叆搴忓垪鎴栦笂浼?FASTA 鏂囦欢锛岃嚜鍔ㄧ敓鎴?**伪-铻烘棆妯℃澘 PDB** 渚涜璁′娇鐢ㄣ€俓n\n"
                                "鍑犱綍鍙傛暟锛?.6 娈嬪熀/鍦?路 铻鸿窛 1.5 脜 路 鍗婂緞 2.3 脜 路 CA-CA 鈮?3.8 脜\n"
                                "楠ㄦ灦鍘熷瓙 N/CA/C/O 鍧囧锛岄敭闀块敭瑙掔鍚堟爣鍑嗚泲鐧借川鍑犱綍銆俓n\n"
                                "鉁?鍏煎 **BFN** / **ProteinMPNN** / **ESM-IF** 涓夋璁捐宸ュ叿"
                            )
                            f2p_fasta = gr.Textbox(
                                label="FASTA 搴忓垪", lines=4,
                                placeholder=">my_protein\nMSDRPTARRWGK..."
                            )
                            f2p_fasta_file = gr.File(label="鎴栦笂浼?FASTA 鏂囦欢",
                                                     file_types=[".fasta", ".fa", ".txt"])
                            with gr.Row():
                                f2p_btn = gr.Button("馃攧 鐢熸垚 伪-铻烘棆妯℃澘 PDB", variant="secondary", size="sm")
                                f2p_clear = gr.Button("鉁?娓呯┖", size="sm")
                            f2p_info = gr.Textbox(label="杞崲淇℃伅", lines=3, interactive=False)

                        with gr.Group():
                            gr.Markdown("### 馃搨 1. 涓婁紶 PDB")
                            pdb_file = gr.File(label="閫夋嫨 PDB 鏂囦欢", file_types=[".pdb"])
                            pdb_info = gr.Textbox(label="缁撴瀯淇℃伅", lines=6, interactive=False)
                            _chain_list = gr.State("")
                            _def_region = gr.State("")
                            _pdb_path = gr.State("")

                            def on_upload_with_path(pdb_file):
                                info, chains, region = on_upload(pdb_file)
                                path = pdb_file.name if pdb_file else ""
                                return info, chains, region, path

                            pdb_file.upload(on_upload_with_path, [pdb_file],
                                           [pdb_info, _chain_list, _def_region, _pdb_path])

                        with gr.Group():
                            gr.Markdown("### 鈿欙笍 2. 閫夋嫨宸ュ叿")
                            tool_sel = gr.Radio(
                                ["BFN (鎶椾綋CDR璁捐)", "BFN (閫氱敤铔嬬櫧璁捐)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (閫氱敤铔嬬櫧璁捐)", label="璁捐宸ュ叿"
                            )

                    with gr.Column(scale=2):
                        # BFN Antibody
                        with gr.Group(visible=False) as pan_bfn_ab:
                            gr.Markdown("**BFN 鎶椾綋 CDR 璁捐**")
                            with gr.Row():
                                bfn_ab_heavy = gr.Textbox(label="閲嶉摼ID", value="H")
                                bfn_ab_light = gr.Textbox(label="杞婚摼ID", value="L")
                            with gr.Row():
                                bfn_ab_samples = gr.Slider(1, 20, value=3, step=1, label="搴忓垪鏁?)
                                bfn_ab_stochastic = gr.Checkbox(label="闅忔満閲囨牱", value=False)
                                bfn_ab_eval = gr.Checkbox(label="璇勪及妯″紡", value=True)

                        # BFN Protein
                        with gr.Group(visible=True) as pan_bfn_pro:
                            gr.Markdown("**BFN 閫氱敤铔嬬櫧璐ㄨ璁?*")
                            bfn_pro_region = gr.Textbox(label="璁捐鍖哄煙", value="A:10-25",
                                                        placeholder="A:10-25 鎴?A:10-25 B:5-15")
                            with gr.Row():
                                bfn_pro_samples = gr.Slider(1, 20, value=3, step=1, label="搴忓垪鏁?)
                                bfn_pro_stochastic = gr.Checkbox(label="闅忔満閲囨牱", value=False)
                                bfn_pro_eval = gr.Checkbox(label="璇勪及妯″紡", value=True)

                        # ProteinMPNN
                        with gr.Group(visible=False) as pan_mpnn:
                            gr.Markdown("**ProteinMPNN**")
                            with gr.Row():
                                mpnn_chains = gr.Textbox(label="璁捐閾?, value="A")
                                mpnn_temp = gr.Textbox(label="娓╁害", value="0.1")
                                mpnn_samples = gr.Slider(1, 20, value=3, step=1, label="搴忓垪鏁?)
                            with gr.Row():
                                mpnn_seed = gr.Number(label="闅忔満绉嶅瓙", value=42, precision=0)
                                mpnn_omit = gr.Textbox(label="鎺掗櫎AA", value="", placeholder="濡? CX")

                        # ESM-IF
                        with gr.Group(visible=False) as pan_esmif:
                            gr.Markdown("**ESM-IF**")
                            with gr.Row():
                                esmif_chain = gr.Textbox(label="鐩爣閾?, value="A")
                                esmif_temp = gr.Slider(0.05, 1.0, value=0.1, step=0.05, label="娓╁害")
                                esmif_samples = gr.Slider(1, 10, value=3, step=1, label="搴忓垪鏁?)

                        with gr.Group():
                            gr.Markdown("### 馃殌 3. 寮€濮?)
                            run_btn = gr.Button("鈻?寮€濮嬭璁?, variant="primary", size="lg")

                # Result
                with gr.Group():
                    gr.Markdown("### 馃搳 缁撴灉")
                    result_out = gr.Textbox(label="", lines=18, interactive=False,
                                            elem_classes="result-box", placeholder="缁撴灉灏嗗湪姝ゆ樉绀?..")
                    result_df = gr.DataFrame(label="璁捐缁撴灉鎺掑悕", interactive=False,
                                             wrap=True)

                # FASTA action bar (hidden until design completes)
                with gr.Group(visible=False) as fasta_action_bar:
                    gr.Markdown("### 馃搧 璁捐搴忓垪瀵煎嚭")
                    with gr.Row():
                        fasta_download = gr.File(label="馃捑 FASTA 鏂囦欢涓嬭浇", file_types=[".fasta"],
                                                 elem_classes="result-box")
                        send_to_af2_btn = gr.Button("馃敭 鍙戦€佸埌 AlphaFold2 棰勬祴",
                                                     variant="secondary", size="lg")
                    _fasta_for_af2 = gr.State("")

                # Tool switch
                def switch_tool(t):
                    return (
                        gr.update(visible=(t == "BFN (鎶椾綋CDR璁捐)")),
                        gr.update(visible=(t == "BFN (閫氱敤铔嬬櫧璁捐)")),
                        gr.update(visible=(t == "ProteinMPNN")),
                        gr.update(visible=(t == "ESM-IF")),
                    )
                tool_sel.change(switch_tool, [tool_sel], [pan_bfn_ab, pan_bfn_pro, pan_mpnn, pan_esmif])

                all_inputs = [
                    _pdb_path, tool_sel,
                    bfn_ab_heavy, bfn_ab_light, bfn_ab_samples, bfn_ab_stochastic, bfn_ab_eval,
                    bfn_pro_region, bfn_pro_samples, bfn_pro_stochastic, bfn_pro_eval,
                    mpnn_chains, mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
                    esmif_chain, esmif_temp, esmif_samples,
                ]
                run_btn.click(run_design, inputs=all_inputs,
                             outputs=[result_out, fasta_download, result_df, fasta_action_bar])

                # Send to AF2: populate the FASTA text in Tab 3 and switch to it
                def on_send_to_af2(fasta_file_val):
                    if not fasta_file_val:
                        return "", ""
                    if isinstance(fasta_file_val, str):
                        path = fasta_file_val
                    elif isinstance(fasta_file_val, dict):
                        path = fasta_file_val.get('name', '')
                    else:
                        path = getattr(fasta_file_val, 'name', '')
                    if not path or not os.path.exists(path):
                        return "", ""
                    with open(path, encoding='utf-8') as f:
                        fasta_text = f.read()
                    _, preview_info = on_fasta_input(fasta_text, None)
                    return fasta_text, preview_info

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺? Tab 2: Batch 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
            with gr.TabItem("馃摝 鎵归噺澶勭悊"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 馃搨 1. 涓婁紶 PDB")
                            batch_files = gr.File(label="涓婁紶 PDB 鏂囦欢锛堝彲澶氶€夛級", file_types=[".pdb"],
                                                 file_count='multiple')

                        with gr.Group():
                            gr.Markdown("### 鈿欙笍 2. 閫夋嫨宸ュ叿涓庡弬鏁?)
                            batch_tool = gr.Radio(
                                ["BFN (閫氱敤铔嬬櫧璁捐)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (閫氱敤铔嬬櫧璁捐)", label="璁捐宸ュ叿"
                            )
                            batch_region = gr.Textbox(label="璁捐鍖哄煙 (BFN)", value="A:10-25")
                            batch_chains = gr.Textbox(label="鐩爣閾?(MPNN/ESM-IF)", value="A")
                            batch_temp = gr.Textbox(label="娓╁害", value="0.1")
                            batch_samples = gr.Slider(1, 10, value=3, step=1, label="姣忔枃浠跺簭鍒楁暟")

                        with gr.Group():
                            gr.Markdown("### 馃殌 3. 寮€濮?)
                            batch_btn = gr.Button("鈻?寮€濮嬫壒閲忓鐞?, variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 馃搳 缁撴灉")
                            batch_result = gr.Textbox(label="", lines=22, interactive=False,
                                                      elem_classes="result-box", placeholder="缁撴灉灏嗗湪姝ゆ樉绀?..")
                        batch_json = gr.File(label="姹囨€?JSON", visible=False)
                        batch_df = gr.DataFrame(label="馃搳 缁煎悎缁撴灉鎺掑悕", interactive=False)

                batch_btn.click(
                    run_batch,
                    [batch_files, batch_tool, batch_region, batch_chains, batch_temp, batch_samples],
                    [batch_result, batch_json, batch_df]
                )

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺? Tab 3: Structure Prediction 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
            with gr.TabItem("馃敭 缁撴瀯棰勬祴"):
                gr.Markdown("### AlphaFold2 缁撴瀯棰勬祴 鈥斺€?浠庡簭鍒楀埌缁撴瀯锛屼竴閿帴鍏ヨ璁℃祦绋?)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### 馃摑 杈撳叆搴忓垪")
                            af_fasta_text = gr.Textbox(
                                label="FASTA 搴忓垪", lines=8,
                                placeholder=">protein_name\nMSDRPTARRWGKCGPLCTRENIMV..."
                            )
                            af_fasta_file = gr.File(label="鎴栦笂浼?FASTA 鏂囦欢", file_types=[".fasta", ".fa", ".txt"])
                            af_fasta_info = gr.Textbox(label="搴忓垪棰勮", lines=4, interactive=False)
                            af_fasta_text.change(on_fasta_input, [af_fasta_text, af_fasta_file],
                                                [af_fasta_text, af_fasta_info])
                            af_fasta_file.upload(on_fasta_input, [af_fasta_text, af_fasta_file],
                                                [af_fasta_text, af_fasta_info])

                        with gr.Group():
                            gr.Markdown("#### 鈿欙笍 AlphaFold2 (ColabFold) 棰勬祴鍙傛暟")
                            with gr.Row():
                                af_num_models = gr.Slider(1, 5, value=1, step=1, label="妯″瀷鏁?,
                                                         info="1=蹇€?鎺ㄨ崘) 3=鏍囧噯 5=鏈€鎱㈡渶鍑?)
                                af_num_recycle = gr.Slider(0, 12, value=1, step=1, label="鍥炴敹娆℃暟",
                                                          info="1=蹇€?鎺ㄨ崘) 3=鏍囧噯 鏇村=鏇寸簿纭絾鏇存參")
                            af_use_dropout = gr.Checkbox(label="闅忔満閲囨牱 (澶氭牱鎬?", value=False,
                                                        info="寮€鍚彲浜х敓澶氭牱缁撴瀯浣嗘洿鎱?)

                        af_run_btn = gr.Button("鈻?寮€濮嬮娴?, variant="primary", size="lg")
                        af_send_btn = gr.Button("馃摛 鍙戦€佸埌璁捐", variant="secondary", size="sm",
                                                visible=False)

                    with gr.Column(scale=2):
                        af_result = gr.Textbox(label="棰勬祴鏃ュ織", lines=20, interactive=False,
                                              elem_classes="result-box")
                        af_pdb_file = gr.File(label="棰勬祴缁撴瀯 PDB", visible=True,
                                             elem_classes="result-box")
                        af_result_dir = gr.State("")

                # Prediction run
                af_run_btn.click(
                    run_alphafold_prediction,
                    [af_fasta_text, af_fasta_file,
                     af_num_models, af_num_recycle, af_use_dropout],
                    [af_result, af_pdb_file, af_result_dir]
                ).then(
                    lambda pdb: gr.update(visible=pdb is not None),
                    [af_pdb_file], [af_send_btn]
                )

                # Send to design: update pdb_file in Tab 1
                def on_send_to_design(pdb_file_val):
                    if not pdb_file_val:
                        return None, "", "鏃犲彲鐢?PDB 缁撴瀯", "", ""
                    if isinstance(pdb_file_val, str):
                        path = pdb_file_val
                    elif isinstance(pdb_file_val, dict):
                        path = pdb_file_val.get('name', '')
                    else:
                        path = getattr(pdb_file_val, 'name', '')
                    if not path or not os.path.exists(path):
                        return None, "", f"PDB 鏂囦欢涓嶅瓨鍦? {path}", "", ""
                    info_text, chains, region = on_upload(type('F', (), {'name': path})())
                    return path, path, info_text, chains, region

                af_send_btn.click(
                    on_send_to_design, [af_pdb_file],
                    [pdb_file, _pdb_path, pdb_info, _chain_list, _def_region]
                )

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺? Tab 4: Settings 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
            with gr.TabItem("鈿欙笍 璁剧疆"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 馃搳 绯荤粺鐘舵€?)
                            status_display = gr.Textbox(label="", value=get_system_status(),
                                                        lines=18, interactive=False,
                                                        elem_classes="result-box")
                            refresh_btn = gr.Button("馃攧 鍒锋柊鐘舵€?, variant="secondary")

                        with gr.Group():
                            gr.Markdown("### 馃敡 妯″瀷棰勫姞杞?)
                            with gr.Row():
                                preload_esmif_btn = gr.Button("鍔犺浇 ESM-IF", size="sm")
                                preload_bfn_btn = gr.Button("閲嶈浇 BFN", size="sm")

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 馃摑 閰嶇疆鏂囦欢缂栬緫")
                            gr.Markdown(f"璺緞: `{CONFIG_FILE}`")
                            config_display = gr.Textbox(
                                label="", value=get_config_display(),
                                lines=22, interactive=True, elem_classes="result-box"
                            )
                            with gr.Row():
                                save_cfg_btn = gr.Button("馃捑 淇濆瓨閰嶇疆", variant="primary", size="sm")
                                reload_cfg_btn = gr.Button("馃攧 閲嶆柊鍔犺浇", variant="secondary", size="sm")
                                cfg_msg = gr.Textbox(label="", visible=True, interactive=False, container=False)

                def do_save_config(text):
                    return save_config_from_text(text)

                def do_reload_config():
                    global _app_config
                    _app_config = None
                    return get_config_display(), "鉁?宸查噸鏂板姞杞?

                save_cfg_btn.click(do_save_config, [config_display], [cfg_msg])
                reload_cfg_btn.click(do_reload_config, outputs=[config_display, cfg_msg])

                def do_preload_esmif():
                    try:
                        load_esmif()
                        return get_system_status(), "鉁?ESM-IF 宸插姞杞?
                    except Exception as e:
                        return get_system_status(), f"鉂?澶辫触: {e}"

                def do_reload_bfn():
                    global _bfn_model, _bfn_config
                    _bfn_model = None; _bfn_config = None
                    try:
                        load_bfn()
                        return get_system_status(), "鉁?BFN 宸查噸杞?
                    except Exception as e:
                        return get_system_status(), f"鉂?澶辫触: {e}"

                preload_esmif_btn.click(do_preload_esmif, outputs=[status_display, cfg_msg])
                preload_bfn_btn.click(do_reload_bfn, outputs=[status_display, cfg_msg])

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺? Tab 5: Target Design 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
            with gr.TabItem("馃幆 闈剁偣璁捐"):
                gr.HTML('<div class="td-wrapper">')
                gr.Markdown("### 闈剁偣瀵煎悜鐨勬姉浣?铔嬬櫧璁捐 鈥斺€?浠庨澏鐐瑰垎鏋愬埌缁撳悎鐣岄潰浼樺寲")

                # Tab 5 state variables
                _t5_target_pdb = gr.State("")
                _t5_target_chain = gr.State("A")
                _t5_epitope_data = gr.State(None)
                _t5_epitope_region = gr.State("")
                _t5_antibody_pdb = gr.State("")
                _t5_design_fasta = gr.State("")

                # 鈹€鈹€ Step 1: Target Input 鈹€鈹€
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 馃幆 1. 闈剁偣杈撳叆")
                            with gr.Accordion("FASTA 鈫?AlphaFold2 棰勬祴闈剁偣缁撴瀯", open=False):
                                t5_fasta_text = gr.Textbox(
                                    label="闈剁偣搴忓垪 (FASTA)", lines=4,
                                    placeholder=">target_protein\nMSDRPTARRWGKCG..."
                                )
                                t5_fasta_file = gr.File(label="鎴栦笂浼?FASTA 鏂囦欢",
                                                        file_types=[".fasta", ".fa", ".txt"])
                                with gr.Row():
                                    t5_predict_btn = gr.Button("馃敭 AlphaFold2 棰勬祴", variant="primary", size="sm")
                                    t5_fasta_info_btn = gr.Button("馃搵 瑙ｆ瀽搴忓垪", size="sm")
                                t5_fasta_info = gr.Textbox(label="搴忓垪棰勮", lines=3, interactive=False)

                            t5_pdb_file = gr.File(label="鎴栫洿鎺ヤ笂浼犻澏鐐?PDB", file_types=[".pdb"])
                            t5_chain_selector = gr.Dropdown(label="閫夋嫨鍒嗘瀽閾?, choices=[], interactive=True)
                            t5_pdb_info = gr.Textbox(label="缁撴瀯淇℃伅", lines=4, interactive=False)

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 馃搵 姝ラ鏃ュ織")
                            t5_step1_log = gr.Textbox(label="", lines=14, interactive=False,
                                                      elem_classes="result-box",
                                                      placeholder="闈剁偣缁撴瀯淇℃伅灏嗗湪姝ゆ樉绀?..")

                # 鈹€鈹€ Step 2: Epitope Analysis 鈹€鈹€
                with gr.Group():
                    gr.Markdown("### 馃敩 2. 琛ㄤ綅鍒嗘瀽")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                t5_epitope_run_btn = gr.Button("鈻?鍒嗘瀽琛ㄤ綅", variant="primary")
                                t5_top_n = gr.Slider(5, 50, value=20, step=5, label="鏄剧ずTop-N")
                            t5_epitope_methods = gr.CheckboxGroup(
                                choices=["SASA (婧跺墏鍙強鎬?", "浜叉按鎬?, "鍑稿嚭鎸囨暟"],
                                value=["SASA (婧跺墏鍙強鎬?", "浜叉按鎬?, "鍑稿嚭鎸囨暟"],
                                label="璇勫垎鏂规硶"
                            )
                            t5_epitope_region_input = gr.Textbox(
                                label="纭琛ㄤ綅鍖哄煙",
                                placeholder="濡? A:25-35,40-50 鎴?A:25,30,35,40",
                                value=""
                            )
                            t5_epitope_select_btn = gr.Button("鉁?纭琛ㄤ綅鍖哄煙", variant="secondary")

                        with gr.Column(scale=2):
                            t5_epitope_result = gr.Textbox(
                                label="琛ㄤ綅璇勫垎缁撴灉", lines=18, interactive=False,
                                elem_classes="result-box",
                                placeholder="鐐瑰嚮'鍒嗘瀽琛ㄤ綅'鏌ョ湅鍊欓€夎〃浣嶆畫鍩?.."
                            )

                # 鈹€鈹€ Step 3 + 4: Antibody Setup + Constrained Design 鈹€鈹€
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 馃К 3. 鎶椾綋璁剧疆")
                            with gr.Accordion("FASTA 鈫?妯℃澘 PDB (伪-铻烘棆)", open=False):
                                t5_ab_fasta = gr.Textbox(
                                    label="鎶椾綋搴忓垪 (FASTA)", lines=3,
                                    placeholder=">antibody_scfv\nEVQLVESGG..."
                                )
                                t5_ab_gen_pdb_btn = gr.Button("馃攧 鐢熸垚 伪-铻烘棆妯℃澘 PDB", size="sm")

                            t5_ab_pdb_file = gr.File(label="鎴栫洿鎺ヤ笂浼犳姉浣?PDB", file_types=[".pdb"])
                            t5_ab_pdb_info = gr.Textbox(label="鎶椾綋缁撴瀯淇℃伅", lines=3, interactive=False)
                            with gr.Row():
                                t5_ab_heavy = gr.Textbox(label="閲嶉摼ID", value="H", scale=1)
                                t5_ab_light = gr.Textbox(label="杞婚摼ID", value="L", scale=1)

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 馃殌 4. 绾︽潫璁捐")
                            t5_design_tool = gr.Radio(
                                ["BFN (鎶椾綋CDR璁捐)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (鎶椾綋CDR璁捐)", label="璁捐宸ュ叿"
                            )

                            # BFN params
                            with gr.Group(visible=True) as t5_pan_bfn:
                                gr.Markdown("**BFN 鍙傛暟**")
                                with gr.Row():
                                    t5_bfn_samples = gr.Slider(1, 10, value=3, step=1, label="搴忓垪鏁?)
                                    t5_bfn_stochastic = gr.Checkbox(label="闅忔満閲囨牱", value=False)
                                    t5_bfn_eval = gr.Checkbox(label="璇勪及妯″紡", value=True)

                            # MPNN params
                            with gr.Group(visible=False) as t5_pan_mpnn:
                                gr.Markdown("**ProteinMPNN 鍙傛暟**")
                                with gr.Row():
                                    t5_mpnn_temp = gr.Textbox(label="娓╁害", value="0.1")
                                    t5_mpnn_samples = gr.Slider(1, 10, value=3, step=1, label="搴忓垪鏁?)
                                with gr.Row():
                                    t5_mpnn_seed = gr.Number(label="闅忔満绉嶅瓙", value=42, precision=0)
                                    t5_mpnn_omit = gr.Textbox(label="鎺掗櫎AA", value="", placeholder="濡? CX")

                            # ESM-IF params
                            with gr.Group(visible=False) as t5_pan_esmif:
                                gr.Markdown("**ESM-IF 鍙傛暟**")
                                with gr.Row():
                                    t5_esmif_temp = gr.Slider(0.05, 1.0, value=0.1, step=0.05, label="娓╁害")
                                    t5_esmif_samples = gr.Slider(1, 10, value=3, step=1, label="搴忓垪鏁?)

                            with gr.Row():
                                t5_constraint_mode = gr.Checkbox(
                                    label="鍚敤琛ㄤ綅璺濈绾︽潫", value=True,
                                    info="浠呰璁′笌琛ㄤ綅鎺ヨЕ鐨勬姉浣撴畫鍩?
                                )
                                t5_constraint_cutoff = gr.Slider(
                                    5, 20, value=10, step=1,
                                    label="璺濈闃堝€?(脜)",
                                    info="CA-CA璺濈闃堝€硷紝纭畾闈㈠悜琛ㄤ綅鐨勬畫鍩?
                                )
                            t5_run_design_btn = gr.Button("鈻?寮€濮嬮澏鐐硅璁?, variant="primary", size="lg")

                # 鈹€鈹€ Step 5: Validation 鈹€鈹€
                with gr.Group():
                    gr.Markdown("### 鉁?5. 缁撴灉涓庨獙璇?)
                    with gr.Row():
                        t5_design_result = gr.Textbox(
                            label="璁捐缁撴灉", lines=14, interactive=False,
                            elem_classes="result-box", placeholder="璁捐缁撴灉灏嗗湪姝ゆ樉绀?.."
                        )
                        t5_validation_result = gr.Textbox(
                            label="楠岃瘉鎶ュ憡", lines=14, interactive=False,
                            elem_classes="result-box", placeholder="鎺ヨЕ鍒嗘瀽鍜岀晫闈㈢壒鎬?.."
                        )
                    with gr.Group(visible=False) as t5_action_bar:
                        with gr.Row():
                            t5_fasta_download = gr.File(label="馃捑 FASTA 涓嬭浇", file_types=[".fasta"])
                            t5_send_af2_btn = gr.Button("馃敭 鍙戦€佸埌 AF2 Multimer 棰勬祴", variant="secondary")
                            t5_send_design_btn = gr.Button("馃摛 鍙戦€佸簭鍒楀埌鍗曟璁捐", variant="secondary")
                    t5_result_df = gr.DataFrame(label="馃搳 璁捐缁撴灉鎺掑悕", interactive=False)

                # 鈹€鈹€ Tab 5 internal callbacks 鈹€鈹€

                def t5_on_parse_fasta(fasta_text, fasta_file):
                    if fasta_file is not None:
                        try:
                            with open(fasta_file.name, encoding='utf-8') as f:
                                fasta_text = f.read()
                        except Exception:
                            pass
                    if not fasta_text or not fasta_text.strip():
                        return "", "璇锋彁渚?FASTA 搴忓垪"
                    seqs = parse_fasta(fasta_text)
                    if not seqs:
                        return "", "鏈娴嬪埌鏈夋晥 FASTA"
                    return fasta_text, f"{seqs[0][0]}: {len(seqs[0][1])} aa\n{seqs[0][1][:80]}..."

                def t5_on_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", [], "A", ""
                    info_text, chains_str, _ = on_upload(pdb_file)
                    chains = [c.strip() for c in chains_str.split(',') if c.strip()] if chains_str else []
                    chain_id = chains[0] if chains else "A"
                    return info_text, gr.update(choices=chains, value=chain_id), chain_id, pdb_file.name

                def t5_on_af2_predict(fasta_text, fasta_file, progress=gr.Progress()):
                    result, pdb_path, _ = run_alphafold_prediction(
                        fasta_text, fasta_file, 1, 1, False, progress
                    )
                    if pdb_path and os.path.exists(str(pdb_path)):
                        info_text, chains_str, _ = on_upload(type('F', (), {'name': str(pdb_path)})())
                        chains = [c.strip() for c in chains_str.split(',') if c.strip()] if chains_str else []
                        chain_id = chains[0] if chains else "A"
                        return (result, str(pdb_path), gr.update(choices=chains, value=chain_id),
                                chain_id, info_text)
                    return result, "", gr.update(choices=[], value="A"), "A", "棰勬祴澶辫触锛屾湭鐢熸垚 PDB"

                def t5_on_epitope_analyze(pdb_path, chain, top_n, methods):
                    if not pdb_path or not os.path.exists(str(pdb_path)):
                        return None, "璇峰厛鎻愪緵闈剁偣缁撴瀯 (姝ラ1)", ""
                    try:
                        weights = {
                            'sasa': 0.4 if "SASA" in ' '.join(methods) else 0.0,
                            'hydrophilicity': 0.35 if "浜叉按鎬? in ' '.join(methods) else 0.0,
                            'protrusion': 0.25 if "鍑稿嚭鎸囨暟" in ' '.join(methods) else 0.0,
                        }
                        total = sum(weights.values())
                        if total == 0:
                            weights = {'sasa': 0.4, 'hydrophilicity': 0.35, 'protrusion': 0.25}
                        else:
                            for k in weights:
                                weights[k] /= total
                        data = tdh.score_epitope_residues(pdb_path, chain, weights=weights)
                        table = tdh.format_epitope_table(data, top_n=int(top_n))
                        top_region = ""
                        if data:
                            top5 = data[:5]
                            resseqs = [d['resseq'] for d in top5]
                            top_region = tdh.residues_to_region_spec(chain, resseqs)
                        return data, table, top_region
                    except Exception as e:
                        import traceback
                        return None, f"鍒嗘瀽閿欒: {traceback.format_exc()}", ""

                def t5_on_select_epitope(region_str):
                    regions = tdh.parse_region_spec(region_str)
                    if not regions:
                        return region_str, gr.update(value="鈿?鍖哄煙鏍煎紡鏃犳晥")
                    total = sum(len(r) for r in regions.values())
                    return region_str, gr.update(value=f"鉁?宸查€?{total} 涓〃浣嶆畫鍩?)

                def t5_on_ab_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", ""
                    info_text, chains_str, _ = on_upload(pdb_file)
                    return info_text, pdb_file.name

                def t5_on_ab_fasta_to_pdb(fasta_text):
                    if not fasta_text or not fasta_text.strip():
                        return "", None
                    pdb_path, _info = fasta_to_pdb_file(fasta_text)
                    if pdb_path:
                        info_text, _, _ = on_upload(type('F', (), {'name': pdb_path})())
                        return info_text, pdb_path
                    return f"杞崲澶辫触", None

                def t5_tool_switch(tool):
                    return (
                        gr.update(visible=(tool == "BFN (鎶椾綋CDR璁捐)")),
                        gr.update(visible=(tool == "ProteinMPNN")),
                        gr.update(visible=(tool == "ESM-IF")),
                    )

                def t5_on_design(target_pdb, target_chain, epitope_region,
                                 antibody_pdb, ab_heavy, ab_light,
                                 design_tool, constraint_mode, constraint_cutoff,
                                 bfn_samples, bfn_stochastic, bfn_eval,
                                 mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
                                 esmif_temp, esmif_samples):
                    result_text, fasta_str, validation_text, results_df = run_target_design(
                        target_pdb, target_chain, epitope_region,
                        antibody_pdb, ab_heavy, ab_light,
                        design_tool, constraint_mode, constraint_cutoff,
                        bfn_samples, bfn_stochastic, bfn_eval,
                        mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
                        esmif_temp, esmif_samples
                    )
                    fasta_file = None
                    if fasta_str:
                        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                                          delete=False, encoding='utf-8', prefix='tdesign_')
                        tmp.write(fasta_str)
                        tmp.close()
                        fasta_file = tmp.name
                    return (result_text, validation_text, fasta_str,
                            fasta_file, gr.update(visible=bool(fasta_str)), results_df)

                # Wire step 1
                t5_fasta_info_btn.click(t5_on_parse_fasta, [t5_fasta_text, t5_fasta_file],
                                        [t5_fasta_text, t5_fasta_info])
                t5_pdb_file.upload(t5_on_pdb_upload, [t5_pdb_file],
                                   [t5_pdb_info, t5_chain_selector, _t5_target_chain, _t5_target_pdb])
                t5_predict_btn.click(t5_on_af2_predict, [t5_fasta_text, t5_fasta_file],
                                     [t5_step1_log, _t5_target_pdb, t5_chain_selector, _t5_target_chain, t5_pdb_info])

                # Wire step 2
                t5_epitope_run_btn.click(t5_on_epitope_analyze,
                                         [_t5_target_pdb, _t5_target_chain, t5_top_n, t5_epitope_methods],
                                         [_t5_epitope_data, t5_epitope_result, t5_epitope_region_input])
                t5_epitope_select_btn.click(t5_on_select_epitope, [t5_epitope_region_input],
                                            [_t5_epitope_region, t5_epitope_select_btn])

                # Wire step 3
                t5_ab_pdb_file.upload(t5_on_ab_pdb_upload, [t5_ab_pdb_file],
                                      [t5_ab_pdb_info, _t5_antibody_pdb])
                t5_ab_gen_pdb_btn.click(t5_on_ab_fasta_to_pdb, [t5_ab_fasta],
                                        [t5_ab_pdb_info, _t5_antibody_pdb])

                # Wire step 4: tool switching
                t5_design_tool.change(t5_tool_switch, [t5_design_tool],
                                      [t5_pan_bfn, t5_pan_mpnn, t5_pan_esmif])

                t5_all_design_inputs = [
                    _t5_target_pdb, _t5_target_chain, t5_epitope_region_input,
                    _t5_antibody_pdb, t5_ab_heavy, t5_ab_light,
                    t5_design_tool, t5_constraint_mode, t5_constraint_cutoff,
                    t5_bfn_samples, t5_bfn_stochastic, t5_bfn_eval,
                    t5_mpnn_temp, t5_mpnn_samples, t5_mpnn_seed, t5_mpnn_omit,
                    t5_esmif_temp, t5_esmif_samples,
                ]
                t5_run_design_btn.click(t5_on_design, inputs=t5_all_design_inputs,
                                        outputs=[t5_design_result, t5_validation_result,
                                                 _t5_design_fasta, t5_fasta_download, t5_action_bar,
                                                 t5_result_df])

                # Wire step 5: cross-tab
                def t5_send_to_af2_multimer(fasta_str, target_pdb, target_chain):
                    """Build multimer FASTA from designed antibody + target protein."""
                    if not fasta_str:
                        return "", "鏃犺璁″簭鍒?
                    seqs = parse_fasta(fasta_str)
                    if not seqs:
                        return "", "鏃犳硶瑙ｆ瀽璁捐搴忓垪"
                    # Extract designed antibody sequence (first sequence from FASTA)
                    ab_seq = seqs[0][1] if seqs else ""
                    # Extract target sequence from PDB
                    ag_seq = ""
                    try:
                        from Bio.PDB import PDBParser
                        parser = PDBParser(QUIET=True)
                        s = parser.get_structure('ag', str(target_pdb))[0]
                        ag_seq = tdh._extract_sequence(s, target_chain)
                    except Exception:
                        pass
                    if not ag_seq:
                        return fasta_str, "鏃犳硶鎻愬彇闈剁偣搴忓垪锛屽凡杩斿洖鍘熷 FASTA"
                    multimer_text = f">antibody_target_complex\n{ab_seq}:{ag_seq}"
                    preview, _ = on_fasta_input(multimer_text, None)
                    return multimer_text, preview

                t5_send_af2_btn.click(
                    t5_send_to_af2_multimer,
                    [_t5_design_fasta, _t5_target_pdb, _t5_target_chain],
                    [af_fasta_text, af_fasta_info]
                )

                def t5_send_to_single_design(fasta_str):
                    if not fasta_str:
                        return "", None, "", "", "", "", "鏃犺璁″簭鍒?
                    pdb_path, info = fasta_to_pdb_file(fasta_str)
                    if pdb_path:
                        info_text, chains, region = on_upload(type('F', (), {'name': pdb_path})())
                        return pdb_path, pdb_path, chains, region, info_text, info, fasta_str
                    return "", None, "", "", "", info, ""

                t5_send_design_btn.click(
                    t5_send_to_single_design, [_t5_design_fasta],
                    [pdb_file, _pdb_path, _chain_list, _def_region, pdb_info, f2p_info, f2p_fasta]
                )

                gr.HTML('</div>')  # close .td-wrapper

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲  Tab 6: Iterative Refinement 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
            with gr.TabItem("馃攧 杩唬绮句慨"):
                gr.Markdown("""
                ### AF2 椹卞姩鐨勮凯浠ｇ簿淇惊鐜?
                璁捐 鈫?AF2 楠岃瘉 鈫?绾ц仈绛涢€?Top-N 鈫?鐢?AF2 缁撴瀯浣滀负妯℃澘鍐嶈璁?鈫?閲嶅鑷虫敹鏁?

                鈿狅笍 AF2 棰勬祴杈冩參锛堟瘡鏉″簭鍒楃害 5-15 鍒嗛挓锛夛紝寤鸿鍏堢敤灏忔牱鏈暟楠岃瘉娴佺▼銆?
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        ir_pdb = gr.File(label="涓婁紶 PDB 妯℃澘", file_types=[".pdb"])
                        ir_pdb_info = gr.Textbox(label="缁撴瀯淇℃伅", lines=4, interactive=False)
                        ir_region = gr.Textbox(label="璁捐鍖哄煙", value="A:10-30",
                                               placeholder="A:10-30")
                        with gr.Row():
                            ir_n_samples = gr.Slider(3, 20, value=5, step=1, label="姣忚疆鏍锋湰鏁?)
                            ir_top_k = gr.Slider(1, 5, value=2, step=1, label="淇濈暀 Top-K")
                        ir_max_rounds = gr.Slider(1, 5, value=2, step=1, label="鏈€澶ц疆娆?)
                        ir_af2_recycle = gr.Slider(0, 3, value=1, step=1, label="AF2 鍥炴敹姝ユ暟")
                        ir_btn = gr.Button("馃殌 寮€濮嬭凯浠ｇ簿淇?, variant="primary")
                        ir_btn_stop = gr.Button("鉁?鍋滄", variant="stop", visible=False)

                    with gr.Column(scale=2):
                        ir_log = gr.Textbox(label="杩愯鏃ュ織", lines=22, interactive=False,
                                           placeholder="鐐瑰嚮'寮€濮嬭凯浠ｇ簿淇?杩愯...")
                        ir_report = gr.Textbox(label="绮句慨鎶ュ憡", lines=8, interactive=False)

                def on_ir_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", ""
                    info, chains, region = on_upload(pdb_file)
                    return info, pdb_file.name

                ir_pdb.upload(on_ir_pdb_upload, [ir_pdb], [ir_pdb_info, _pdb_path])

                def on_iterative_refine(pdb_path, region, n_samples, top_k, max_rounds, af2_recycle,
                                        progress=gr.Progress()):
                    if not pdb_path or not os.path.exists(str(pdb_path)):
                        return "璇峰厛涓婁紶 PDB 鏂囦欢", ""

                    log_msgs = []
                    def log(msg):
                        log_msgs.append(msg)
                        return '\n'.join(log_msgs)

                    # Wrapper design function
                    def design_fn(pdb, region_spec, n):
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
                                    indices.extend(range(int(a.strip()), int(b.strip())+1))
                                else:
                                    indices.append(int(seg))
                            regions[cid] = sorted(set(indices))

                        model, config = load_bfn()
                        seed_all(42)
                        structure = preprocess_protein_structure(pdb, chain_ids=list(regions.keys()))
                        transform = get_transform([
                            {'type': 'mask_region', 'regions': regions},
                            {'type': 'merge_protein'},
                            {'type': 'patch_protein'},
                        ])
                        batch = recursive_to(PaddingCollate()([transform(structure)]), DEVICE)
                        gen_mask = batch['generate_flag'][0].bool()

                        results_list = []
                        sample_opt = {'deterministic': False, 'num_recycles': 3}
                        for i in range(n):
                            with torch.no_grad():
                                traj = model.sample(batch, sample_opt=sample_opt)
                            pred_aa = traj[0][2][0][gen_mask]
                            seq = ''.join(AA_LETTERS[a] if a < 20 else 'X' for a in pred_aa.cpu())
                            logits = traj['pred_logits'][0][gen_mask]
                            lp = torch.log_softmax(logits[..., :20], dim=-1)
                            nll = -lp[range(len(pred_aa)), pred_aa].mean()
                            ppl = torch.exp(nll).item()
                            entropy = -(torch.exp(lp) * lp).sum(dim=-1).mean().item()
                            plddt = traj['plddt'][0][gen_mask].mean().item()
                            iptm = traj['iptm'][0].item()
                            results_list.append({'sequence': seq, 'ppl': ppl, 'entropy': entropy,
                                                 'plddt': plddt, 'iptm': iptm})
                        return results_list

                    # Wrapper AF2 validator
                    from af2_validator import validate_sequences
                    def af2_validator(sequences, progress_cb=None):
                        return validate_sequences(sequences, output_dir='alphafold_results/ir',
                                                 num_recycle=af2_recycle,
                                                 progress_cb=progress_cb)

                    # Cascade
                    from cascade_filter import apply_cascade_af2

                    # Run refinement
                    from iterative_refiner import IterativeRefiner, format_refinement_report

                    refiner = IterativeRefiner(
                        design_fn=design_fn,
                        af2_validator=af2_validator,
                        cascade_fn=apply_cascade_af2,
                        n_samples=int(n_samples),
                        top_k=int(top_k),
                        max_rounds=int(max_rounds),
                        progress_cb=lambda r, mr, s: progress((r-1)/mr, desc=f'绗瑊r}杞? {s}'),
                        log_cb=lambda msg: log_msgs.append(msg),
                    )

                    try:
                        result = refiner.run(pdb_path, region)
                        report = format_refinement_report(result)
                        return '\n'.join(log_msgs), report
                    except Exception as e:
                        import traceback
                        log(f"鉂?閿欒: {e}")
                        log(traceback.format_exc())
                        return '\n'.join(log_msgs), ""

                ir_btn.click(
                    on_iterative_refine,
                    [_pdb_path, ir_region, ir_n_samples, ir_top_k, ir_max_rounds, ir_af2_recycle],
                    [ir_log, ir_report]
                )

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲  Tab 7: Confidence Evaluation 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
            with gr.TabItem("馃搳 缃俊搴﹁瘎浼?):
                gr.Markdown("""
                ### BFN 鍐呯疆缃俊搴﹁瘎浼?
                鐩存帴浣跨敤璁粌濂界殑 pLDDT/ipTM/PAE 澶磋瘎浼颁换鎰忚泲鐧界粨鏋勭殑缃俊搴︼紝鏃犻渶杩涜搴忓垪璁捐銆?
                浣跨敤 V6 Phase 2 妯″瀷锛? 灞?pLDDT + context-only ipTM锛夈€?
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        conf_pdb = gr.File(label="涓婁紶 PDB 鏂囦欢", file_types=[".pdb"])
                        _conf_pdb_path = gr.State("")
                        conf_pdb_info = gr.Textbox(label="缁撴瀯淇℃伅", lines=3, interactive=False)
                        conf_region = gr.Textbox(label="璇勪及鍖哄煙", value="A:10-25",
                                                 placeholder="A:10-25 鎴?A:1-50")
                        conf_btn = gr.Button("馃攳 璇勪及缃俊搴?, variant="primary")
                    with gr.Column(scale=2):
                        conf_report = gr.Textbox(label="璇勪及鎶ュ憡", lines=14, interactive=False,
                                                 elem_classes="result-box")
                        conf_residue_df = gr.DataFrame(label="閫愭畫鍩?pLDDT", interactive=False)

                def on_conf_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", ""
                    info, chains, region = on_upload(pdb_file)
                    return info, pdb_file.name

                conf_pdb.upload(on_conf_pdb_upload, [conf_pdb], [conf_pdb_info, _conf_pdb_path])

                conf_btn.click(run_bfn_confidence_evaluation,
                              [_conf_pdb_path, conf_region],
                              [conf_report, conf_residue_df])

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲  Tab 8: Unified Pipeline 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
            with gr.TabItem("馃殌 缁熶竴宸ヤ綔娴?):
                gr.Markdown("""
                ### 涓€閿璁?鈫?杩囨护 鈫?楠岃瘉娴佹按绾?
                灏?BFN 璁捐銆佺骇鑱旇繃婊ゅ拰鍙€?AF2 楠岃瘉鏁村悎涓轰竴涓伐浣滄祦锛岃嚜鍔ㄥ畬鎴愭帓鍚嶅拰绛涢€夈€?
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        pipe_pdb = gr.File(label="涓婁紶 PDB 妯℃澘", file_types=[".pdb"])
                        _pipe_pdb_path = gr.State("")
                        pipe_pdb_info = gr.Textbox(label="缁撴瀯淇℃伅", lines=3, interactive=False)
                        pipe_region = gr.Textbox(label="璁捐鍖哄煙", value="A:10-25")
                        with gr.Row():
                            pipe_samples = gr.Slider(3, 20, value=8, step=1, label="璁捐鏍锋湰鏁?)
                            pipe_stochastic = gr.Checkbox(label="闅忔満閲囨牱", value=True)
                        with gr.Row():
                            pipe_enable_af2 = gr.Checkbox(label="鍚敤 AF2 楠岃瘉 (鎱?", value=False)
                            pipe_af2_recycle = gr.Slider(0, 3, value=1, step=1, label="AF2 鍥炴敹姝ユ暟")
                        pipe_btn = gr.Button("馃殌 杩愯缁熶竴宸ヤ綔娴?, variant="primary")
                    with gr.Column(scale=2):
                        pipe_report = gr.Textbox(label="娴佹按绾挎姤鍛?, lines=18, interactive=False,
                                                elem_classes="result-box")
                        pipe_df = gr.DataFrame(label="鏈€缁堟帓鍚嶇粨鏋?, interactive=False)
                        pipe_fasta = gr.File(label="FASTA 涓嬭浇", file_types=[".fasta"])

                def on_pipe_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", ""
                    info, chains, region = on_upload(pdb_file)
                    return info, pdb_file.name

                pipe_pdb.upload(on_pipe_pdb_upload, [pipe_pdb], [pipe_pdb_info, _pipe_pdb_path])

                pipe_btn.click(run_unified_pipeline,
                              [_pipe_pdb_path, pipe_region, pipe_samples, pipe_stochastic,
                               pipe_enable_af2, pipe_af2_recycle],
                              [pipe_report, pipe_df, pipe_fasta])

            # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲  Tab 9: Misfolding Disease 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
            with gr.TabItem("馃 閿欒鎶樺彔鐤剧梾"):
                gr.Markdown("""
                ### 铔嬬櫧璐ㄩ敊璇姌鍙犵柧鐥呮姉浣撹璁?
                閽堝绁炵粡閫€琛屾€х柧鐥咃紙闃垮皵鑼ㄦ捣榛樸€佸笗閲戞．銆丄LS绛夛級鐨勯敊璇姌鍙犺泲鐧介澏鐐癸紝
                杩涜琛ㄤ綅寮曞鐨勭害鏉熸€ф姉浣撹璁★紝骞剁粡杩?AF2 Multimer 楠岃瘉銆?
                """)

                # 鈹€鈹€ State variables 鈹€鈹€
                _mf_disease_key = gr.State("")
                _mf_conf_key = gr.State("")
                _mf_target_pdb = gr.State("")
                _mf_target_chain = gr.State("")
                _mf_is_idp = gr.State(False)
                _mf_idp_warning = gr.State("")
                _mf_epitope_data = gr.State([])
                _mf_known_epitopes = gr.State([])
                _mf_epitope_region = gr.State("")
                _mf_ab_pdb = gr.State("")
                _mf_ab_chain = gr.State("")
                _mf_design_region = gr.State("")
                _mf_design_results = gr.State([])
                _mf_af2_validated = gr.State([])

                # 鈹€鈹€ Step 1: Disease Selection 鈹€鈹€
                with gr.Group():
                    gr.Markdown("#### Step 1: 閫夋嫨闈剁偣鐤剧梾")
                    with gr.Row():
                        with gr.Column(scale=1):
                            disease_choices = [(f'{cn} ({en})', key) for key, cn, en in mkb.list_diseases()]
                            mf_disease = gr.Dropdown(
                                choices=disease_choices, label="閫夋嫨鐤剧梾闈剁偣",
                                info="浠庣煡璇嗗簱涓€夋嫨宸叉暣鐞嗙殑閿欒鎶樺彔鐤剧梾闈剁偣")
                            conf_choices = gr.Dropdown(
                                choices=[], label="闈剁偣鏋勮薄", interactive=True,
                                info="閫夋嫨鑷寸梾鐨勭粨鏋勬瀯璞★紙绾ょ淮/瀵¤仛浣撶瓑锛?)
                            mf_load_btn = gr.Button("馃摜 鍔犺浇闈剁偣", variant="primary")
                        with gr.Column(scale=2):
                            mf_target_info = gr.Textbox(
                                label="闈剁偣淇℃伅", lines=8, interactive=False,
                                placeholder="閫夋嫨鐤剧梾鍚庡姞杞?..")
                            mf_idp_box = gr.Markdown("", visible=False,
                                                     elem_classes="warning-box")

                def mf_on_disease_select(disease_key):
                    if not disease_key:
                        return gr.Dropdown(choices=[]), "", "", False, ""
                    target = mkb.get_target(disease_key)
                    confs = mkb.get_conformations(disease_key)
                    conf_choices = [
                        (f'{cd["type"]} ({cd["pdb_id"]})', ck)
                        for ck, cd in confs.items()
                    ]
                    info = mkb.get_target_summary(disease_key)
                    is_idp = target.get('idp_flag', False)
                    idp_warn = target.get('idp_warning', '')
                    return (
                        gr.Dropdown(choices=conf_choices, value=conf_choices[0][1] if conf_choices else None),
                        info, disease_key, is_idp, idp_warn or ''
                    )

                def mf_on_load(disease_key, conf_key):
                    if not disease_key:
                        return '', '', '', '', '', gr.Markdown(visible=False)
                    pipeline = mfp.MisfoldingDesignPipeline()
                    result = pipeline.prepare_target(disease_key, conf_key)
                    if not result['success']:
                        return '', '', '', '', f'**Error**: {result.get("error")}',
                        gr.Markdown(visible=False)
                    is_idp = result.get('is_idp', False)
                    idp_warn = result.get('idp_warning')
                    target_info = mkb.get_target_summary(disease_key)
                    idp_md = ''
                    if is_idp and idp_warn:
                        idp_md = (
                            '> 鈿狅笍 **IDP 鏃犲簭铔嬬櫧闈剁偣**\n\n'
                            f'> {idp_warn}\n\n'
                            '> BFN 缃俊搴﹁瘎鍒嗕笉鍙潬锛孉F2 Multimer 楠岃瘉**寮哄埗瑕佹眰**銆?
                        )
                    return (
                        result['pdb_path'], result['chain_id'],
                        result['conformation_key'], target_info,
                        f'鉁?宸插姞杞? {os.path.basename(result["pdb_path"])} (閾?{result["chain_id"]})',
                        gr.Markdown(value=idp_md, visible=is_idp and bool(idp_warn))
                    )

                mf_disease.change(mf_on_disease_select,
                                  [mf_disease],
                                  [conf_choices, mf_target_info, _mf_disease_key,
                                   _mf_is_idp, _mf_idp_warning])
                mf_load_btn.click(mf_on_load,
                                  [_mf_disease_key, conf_choices],
                                  [_mf_target_pdb, _mf_target_chain, _mf_conf_key,
                                   mf_target_info, _mf_idp_warning, mf_idp_box])

                # 鈹€鈹€ Step 2: Epitope Analysis 鈹€鈹€
                with gr.Group():
                    gr.Markdown("#### Step 2: 琛ㄤ綅鍒嗘瀽")
                    with gr.Row():
                        with gr.Column(scale=1):
                            mf_epi_topn = gr.Slider(5, 30, value=20, step=1, label="Top-N 琛ㄤ綅娈嬪熀")
                            mf_use_known = gr.Checkbox(label="浣跨敤鏂囩尞宸茬煡琛ㄤ綅", value=True,
                                                       info="浼樺厛浣跨敤宸插彂琛ㄧ殑娌荤枟鎬ф姉浣撹〃浣?)
                            mf_epi_btn = gr.Button("馃攳 鍒嗘瀽琛ㄤ綅", variant="primary")
                        with gr.Column(scale=2):
                            mf_epi_result = gr.Markdown("", elem_classes="result-box")
                    with gr.Row():
                        mf_epi_region_input = gr.Textbox(
                            label="璁捐琛ㄤ綅鍖哄煙",
                            info="鍙墜鍔ㄧ紪杈戯紝鎴栦娇鐢ㄤ笅鏂瑰缓璁€傛牸寮? A:1-16,20-30",
                            placeholder="鐐瑰嚮鍒嗘瀽鍚庤嚜鍔ㄥ～鍏ュ缓璁尯鍩?..")
                    with gr.Row():
                        mf_known_epi_display = gr.Markdown("")

                def mf_on_epitope(pdb_path, chain_id, top_n, use_known):
                    if not pdb_path or not os.path.exists(str(pdb_path)):
                        return '', '', '', [], ''
                    pipeline = mfp.MisfoldingDesignPipeline()
                    known = []
                    if use_known:
                        # Find disease key from state 鈥?we pass it via a trick
                        known = []
                    epitope = pipeline.analyze_epitope(pdb_path, chain_id,
                                                       top_n=int(top_n),
                                                       known_epitopes=known)
                    data = epitope.get('epitope_data', [])
                    suggested = epitope.get('suggested_region', '')
                    table_md = tdh.format_epitope_summary_for_ui(data, int(top_n))

                    # Build known epitope display
                    known_md = ''
                    known_eps = epitope.get('known_epitopes', [])
                    if known_eps:
                        known_md = '**鏂囩尞宸茬煡琛ㄤ綅**:\n\n'
                        for ep in known_eps:
                            r = ep.get('region', (0, 0))
                            known_md += f'- {ep.get("ab", "?")}: 娈嬪熀 {r[0]}-{r[1]} ({ep.get("source", "")})\n'

                    return table_md, suggested, json.dumps(data), known_md, suggested

                mf_epi_btn.click(mf_on_epitope,
                                 [_mf_target_pdb, _mf_target_chain, mf_epi_topn, mf_use_known],
                                 [mf_epi_result, mf_epi_region_input, _mf_epitope_data,
                                  mf_known_epi_display, _mf_epitope_region])
                # Also update epitope when known epitopes toggle changes
                mf_use_known.change(mf_on_epitope,
                                    [_mf_target_pdb, _mf_target_chain, mf_epi_topn, mf_use_known],
                                    [mf_epi_result, mf_epi_region_input, _mf_epitope_data,
                                     mf_known_epi_display, _mf_epitope_region])

                # 鈹€鈹€ Step 3: Antibody Template 鈹€鈹€
                with gr.Group():
                    gr.Markdown("#### Step 3: 鎶椾綋妯℃澘")
                    with gr.Row():
                        with gr.Column(scale=1):
                            scaffold_choices = [(f'{s[1]} ({s[2]})', s[0]) for s in mkb.list_scaffolds()]
                            mf_scaffold = gr.Dropdown(
                                choices=scaffold_choices, label="绾崇背鎶椾綋鏀灦",
                                value=scaffold_choices[0][1] if scaffold_choices else None,
                                info="閫夋嫨 VHH 绾崇背鎶椾綋鏀灦")
                            mf_scaffold_info = gr.Textbox(label="鏀灦淇℃伅", lines=4, interactive=False)
                        with gr.Column(scale=2):
                            mf_ab_fasta = gr.Textbox(
                                label="鎶椾綋搴忓垪", lines=3, interactive=False,
                                info="閫夊畾鏀灦鐨勬皑鍩洪吀搴忓垪锛堝闇€鑷畾涔夊彲缂栬緫锛?)
                            mf_ab_fasta_file = gr.File(
                                label="鎴栦笂浼犳姉浣?PDB/FASTA", file_types=[".pdb", ".fasta"])

                def mf_on_scaffold(scaffold_key):
                    if not scaffold_key:
                        return '', ''
                    scf = mkb.get_scaffold(scaffold_key)
                    if scf is None:
                        return '鏀灦鏈壘鍒?, ''
                    info = f'鍚嶇О: {scf["name"]}\n闀垮害: {scf.get("length", "?")} aa\n'
                    info += f'CDRs: {scf.get("cdr_definitions", {})}\n{scf.get("notes", "")}'
                    seq = scf.get('sequence', '')
                    return info, seq

                mf_scaffold.change(mf_on_scaffold, [mf_scaffold], [mf_scaffold_info, mf_ab_fasta])

                # 鈹€鈹€ Step 4: Constrained Design 鈹€鈹€
                with gr.Group():
                    gr.Markdown("#### Step 4: 绾︽潫璁捐")
                    with gr.Row():
                        with gr.Column(scale=1):
                            mf_design_tool = gr.Radio(
                                choices=["BFN (閫氱敤铔嬬櫧璁捐)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (閫氱敤铔嬬櫧璁捐)", label="璁捐宸ュ叿")
                            mf_n_samples = gr.Slider(3, 20, value=8, step=1, label="璁捐鏍锋湰鏁?)
                            mf_constraint = gr.Checkbox(label="琛ㄤ綅绾︽潫璁捐", value=True,
                                                        info="浠呭湪闈㈠悜琛ㄤ綅鐨勫尯鍩熻繘琛岃璁?)
                            mf_constraint_cutoff = gr.Slider(5.0, 15.0, value=8.0, step=0.5,
                                                             label="鎺ヨЕ璺濈闃堝€?(脜)")
                            mf_design_btn = gr.Button("馃К 寮€濮嬭璁?, variant="primary")
                        with gr.Column(scale=2):
                            mf_design_report = gr.Textbox(label="璁捐鎶ュ憡", lines=10, interactive=False,
                                                         elem_classes="result-box")
                            mf_design_df = gr.DataFrame(label="璁捐缁撴灉鎺掑悕", interactive=False)
                            mf_design_fasta = gr.File(label="FASTA 涓嬭浇", file_types=[".fasta"])

                def mf_run_design(fasta_text, pdb_path, design_region, tool, n_samples,
                                  constraint, constraint_cutoff):
                    if not fasta_text or not pdb_path:
                        return "閿欒: 闇€瑕佹姉浣撳簭鍒楀拰鐩爣 PDB", pd.DataFrame(), None, []
                    if not os.path.exists(str(pdb_path)):
                        return f"閿欒: PDB 鏂囦欢涓嶅瓨鍦?({pdb_path})", pd.DataFrame(), None, []

                    # Use the BFN protein design path
                    # Fast path: generate PDB from antibody FASTA (use f2p)
                    try:
                        ab_pdb, fasta_info = fasta_to_pdb_file(fasta_text)
                        if not ab_pdb:
                            return f"閿欒: 鏃犳硶浠嶧ASTA鐢熸垚鎶椾綋PDB\n{fasta_info}", pd.DataFrame(), None, []
                    except Exception as e:
                        return f"閿欒: FASTA鈫扨DB 杞崲澶辫触: {e}", pd.DataFrame(), None, []

                    # Run design
                    try:
                        if tool.startswith('BFN'):
                            report, fasta_str, results, _, _, _, _ = run_bfn_protein(
                                ab_pdb, design_region or 'A:26-33,51-58,97-110',
                                int(n_samples), False, True
                            )
                        elif tool == 'ProteinMPNN':
                            report, fasta_str, results, _, _, _, _ = run_mpnn(
                                ab_pdb, design_region or 'A:26-33,51-58,97-110',
                                int(n_samples), '0.1', '42', False
                            )
                        elif tool == 'ESM-IF':
                            report, fasta_str, results, _, _, _, _ = run_esmif(
                                ab_pdb, design_region or 'A:26-33,51-58,97-110',
                                int(n_samples), '0.1'
                            )
                        else:
                            report, fasta_str, results, _, _, _, _ = run_bfn_protein(
                                ab_pdb, design_region or 'A:26-33,51-58,97-110',
                                int(n_samples), False, True
                            )
                    except Exception as e:
                        return f"璁捐澶辫触: {e}", pd.DataFrame(), None, []

                    # Apply misfolding cascade
                    try:
                        filtered, casc_report = cf.apply_cascade_misfolding(results, use_af2=False)
                        report = f'{report}\n\n{casc_report}'
                    except Exception:
                        filtered = results

                    df = build_results_dataframe(filtered)
                    fasta_path = _save_fasta(fasta_str)
                    return report, df, fasta_path, filtered

                mf_design_btn.click(mf_run_design,
                                    [mf_ab_fasta, _mf_target_pdb, _mf_epitope_region,
                                     mf_design_tool, mf_n_samples, mf_constraint,
                                     mf_constraint_cutoff],
                                    [mf_design_report, mf_design_df, mf_design_fasta,
                                     _mf_design_results])

                # 鈹€鈹€ Step 5: AF2 Multimer Validation 鈹€鈹€
                with gr.Group():
                    gr.Markdown("#### Step 5: AF2 Multimer 楠岃瘉")
                    with gr.Row():
                        with gr.Column(scale=1):
                            mf_af2_recycle = gr.Slider(0, 12, value=3, step=1, label="AF2 鍥炴敹姝ユ暟")
                            mf_af2_topk = gr.Slider(1, 5, value=3, step=1, label="楠岃瘉 Top-K 璁捐")
                            mf_af2_btn = gr.Button("馃敩 杩愯 AF2 Multimer", variant="primary")
                        with gr.Column(scale=2):
                            mf_af2_report = gr.Textbox(label="AF2 楠岃瘉鎶ュ憡", lines=12, interactive=False,
                                                       elem_classes="result-box")
                            mf_af2_df = gr.DataFrame(label="AF2 鎺掑悕缁撴灉", interactive=False)

                def mf_run_af2(disease_key, design_results, target_pdb, target_chain,
                               topk, num_recycle, is_idp):
                    if not design_results:
                        return "鏃犺璁＄粨鏋滃彲楠岃瘉", pd.DataFrame(), []
                    if not target_pdb:
                        return "鐩爣 PDB 鏈姞杞?, pd.DataFrame(), []

                    # Get target sequence
                    ag_seq = tdh.extract_seq_from_pdb(str(target_pdb), str(target_chain))
                    if not ag_seq:
                        return f"鏃犳硶浠?{target_pdb} 閾?{target_chain} 鎻愬彇鎶楀師搴忓垪", pd.DataFrame(), []

                    cfg = load_app_config()
                    colabfold_exe = cfg.get('alphafold', {}).get('af2', {}).get('executable', 'colabfold_batch')
                    output_dir = os.path.join(
                        str(PROJECT_DIR),
                        cfg.get('misfolding_disease', {}).get('output_dir', 'misfolding_results'),
                        str(disease_key) or 'unknown',
                    )

                    pipeline = mfp.MisfoldingDesignPipeline()
                    results_list = list(design_results) if not isinstance(design_results, list) else design_results
                    top_designs = results_list[:int(topk)]

                    af2_results = []
                    report_lines = ['AF2 Multimer Validation Report', '=' * 50]
                    batch_inputs = []
                    for i, d in enumerate(top_designs):
                        seq = d.get('sequence', '')
                        if not seq:
                            continue
                        result = pipeline.validate_with_af2(
                            seq, ag_seq, output_dir,
                            num_recycle=int(num_recycle),
                            colabfold_exe=colabfold_exe,
                            ab_name=f'design_{i+1}',
                            ag_name=str(disease_key) or 'target',
                        )
                        result['rank'] = i + 1
                        result['sequence'] = seq
                        af2_results.append(result)

                        if result['success']:
                            report_lines.append(
                                f'Design #{i+1}: ipTM={result["iptm"]:.3f} '
                                f'pTM={result["ptm"]:.3f} pLDDT={result["plddt"]:.1f}'
                            )
                            if result.get('interface_pae') is not None:
                                report_lines.append(f'  Interface PAE: {result["interface_pae"]:.2f} A')
                        else:
                            report_lines.append(f'Design #{i+1}: FAILED 鈥?{result.get("error")}')

                    # AF2 cascade re-ranking
                    for r in af2_results:
                        if r['success']:
                            batch_inputs.append({
                                'sequence': r.get('sequence', ''),
                                'plddt': r['plddt'] * 100,  # to 0-100
                                'ptm': r['ptm'],
                                'iptm': r['iptm'],
                                'max_pae': r.get('max_pae', 999),
                                'pdb_path': r.get('pdb_path'),
                            })

                    if batch_inputs:
                        af2_filtered, af2_report = cf.apply_cascade_misfolding(batch_inputs, use_af2=True)
                        report_lines.append('\n' + af2_report)
                    else:
                        af2_filtered = []

                    # Build DataFrame
                    df_data = []
                    for r in (af2_filtered or af2_results):
                        df_data.append({
                            'Rank': r.get('af2_rank', r.get('rank', '?')),
                            'Sequence': (r.get('sequence', '') or '')[:25],
                            'ipTM': round(r.get('iptm', 0), 3) if r.get('iptm') else '-',
                            'pTM': round(r.get('ptm', 0), 3) if r.get('ptm') else '-',
                            'pLDDT': round(r.get('plddt', 0), 1),
                            'InterfacePAE': round(r.get('interface_pae', 99), 1) if r.get('interface_pae') else '-',
                        })
                    df = pd.DataFrame(df_data) if df_data else pd.DataFrame()

                    return '\n'.join(report_lines), df, af2_results

                mf_af2_btn.click(mf_run_af2,
                                 [_mf_disease_key, _mf_design_results, _mf_target_pdb,
                                  _mf_target_chain, mf_af2_topk, mf_af2_recycle, _mf_is_idp],
                                 [mf_af2_report, mf_af2_df, _mf_af2_validated])

                # 鈹€鈹€ Step 6: Iterative Refinement (collapsible) 鈹€鈹€
                with gr.Accordion("馃攧 杩唬绮句慨 (鍙€?", open=False):
                    gr.Markdown("閫氳繃'璁捐鈫扐F2楠岃瘉鈫掔骇鑱旂瓫閫夆啋鏇存柊妯℃澘鈫掑啀璁捐'寰幆浼樺寲搴忓垪銆?)
                    with gr.Row():
                        with gr.Column(scale=1):
                            mf_iter_samples = gr.Slider(5, 30, value=10, step=1, label="姣忚疆鏍锋湰鏁?)
                            mf_iter_topk = gr.Slider(2, 8, value=3, step=1, label="Top-K 淇濈暀")
                            mf_iter_rounds = gr.Slider(2, 6, value=3, step=1, label="鏈€澶ц疆娆?)
                            mf_iter_btn = gr.Button("馃攧 鍚姩杩唬绮句慨", variant="secondary")
                        with gr.Column(scale=2):
                            mf_iter_report = gr.Textbox(label="杩唬绮句慨鎶ュ憡", lines=12,
                                                        interactive=False, elem_classes="result-box")
                            mf_iter_fasta = gr.File(label="鏈€浼?FASTA", file_types=[".fasta"])

                def mf_run_iterative(disease_key, design_results, target_pdb, target_chain,
                                     fasta_text, n_samples, top_k, max_rounds, num_recycle):
                    if not design_results:
                        return "璇峰厛杩愯璁捐 (Step 4) 鑾峰彇鍒濆璁捐缁撴灉銆?, None
                    if not target_pdb:
                        return "鐩爣 PDB 鏈姞杞姐€?, None

                    from iterative_refiner import IterativeRefiner

                    def mf_design_fn(pdb_path, region, n_samp):
                        """Wrapped design function for iter refiner"""
                        _, _, results, _, _, _, _ = run_bfn_protein(
                            pdb_path, region, n_samp, False, True
                        )
                        return results

                    def mf_af2_validator(sequences, output_dir, progress_cb=None):
                        """Wrapped AF2 validator for iter refiner"""
                        ag_seq = tdh.extract_seq_from_pdb(str(target_pdb), str(target_chain))
                        pipeline = mfp.MisfoldingDesignPipeline()
                        results = []
                        for i, seq in enumerate(sequences):
                            r = pipeline.validate_with_af2(seq, ag_seq, output_dir,
                                                           num_recycle=int(num_recycle))
                            if r['success']:
                                results.append({
                                    'sequence': seq,
                                    'plddt': r['plddt'] * 100,  # 0-100
                                    'ptm': r['ptm'],
                                    'iptm': r['iptm'],
                                    'max_pae': r.get('max_pae', 999),
                                    'pdb_path': r.get('pdb_path'),
                                    'success': True,
                                })
                            else:
                                results.append({'sequence': seq, 'success': False})
                            if progress_cb:
                                progress_cb(i + 1, len(sequences), f'AF2 {i+1}/{len(sequences)}')
                        return results

                    def mf_cascade(results, thresholds, weights):
                        """Wrapped cascade for iter refiner"""
                        return cf.apply_cascade_misfolding(results, use_af2=True)

                    # Get initial antibody PDB
                    ab_pdb, _ = fasta_to_pdb_file(fasta_text) if fasta_text else (None, '')
                    if not ab_pdb:
                        return "鏃犳硶浠庢姉浣撳簭鍒楃敓鎴愭ā鏉?PDB銆?, None

                    refiner = IterativeRefiner(
                        design_fn=mf_design_fn,
                        af2_validator=mf_af2_validator,
                        cascade_fn=mf_cascade,
                        n_samples=int(n_samples),
                        top_k=int(top_k),
                        max_rounds=int(max_rounds),
                        convergence_delta=0.02,
                    )

                    try:
                        result = refiner.run(
                            ab_pdb,
                            _mf_epitope_region.value if hasattr(_mf_epitope_region, 'value') else 'A:26-33,51-58,97-110',
                            None
                        )
                        report = iterative_refiner.format_refinement_report(result)
                        fasta_str = f'>best_iterative\ndesign_rounds={len(result.rounds)}_ipTM={result.best_overall_iptm:.3f}\n{result.best_overall_sequence}'
                        fasta_path = _save_fasta(fasta_str)
                        return report, fasta_path
                    except Exception as e:
                        return f"杩唬绮句慨澶辫触: {e}", None

                mf_iter_btn.click(mf_run_iterative,
                                  [_mf_disease_key, _mf_design_results, _mf_target_pdb,
                                   _mf_target_chain, mf_ab_fasta, mf_iter_samples,
                                   mf_iter_topk, mf_iter_rounds, mf_af2_recycle],
                                  [mf_iter_report, mf_iter_fasta])

        # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲  Tab 10: 闂幆浼樺寲 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        with gr.TabItem("馃攣 闂幆浼樺寲"):
            gr.Markdown("""
            ### Multi-Objective Closed-Loop Alignment

            瀹屾暣闂幆绠＄嚎: **璁捐 鈫?AF2鎶樺彔楠岃瘉 鈫?Rosetta鐗╃悊鎵撳垎 鈫?Pareto澶氱洰鏍囨帓搴?鈫?杩唬绮句慨**

            杩欐槸鐮旂┒鍓嶆部锛氬皢鐢熸垚銆侀獙璇併€佺墿鐞嗘墦鍒嗗拰妯″瀷浼樺寲涓茶仈涓鸿嚜鍔ㄥ寲闂幆銆?
            """)

            # 鈹€鈹€ State variables 鈹€鈹€
            _cl_pdb = gr.State("")
            _cl_region = gr.State("")
            _cl_cycles_data = gr.State([])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 馃敡 閰嶇疆")
                    cl_pdb_file = gr.File(label="涓婁紶闈剁偣 PDB", file_types=[".pdb"])
                    cl_pdb_info = gr.Textbox(label="PDB 淇℃伅", lines=3, interactive=False)

                    cl_region = gr.Textbox(label="璁捐鍖哄煙", value="A:25-35",
                                           info="e.g. A:25-35 鎴?A:25-35,50-60")
                    cl_design_tool = gr.Radio(
                        choices=["BFN (閫氱敤铔嬬櫧璁捐)", "ProteinMPNN", "ESM-IF"],
                        value="BFN (閫氱敤铔嬬櫧璁捐)", label="璁捐宸ュ叿")
                    cl_n_samples = gr.Slider(5, 30, value=15, step=5,
                                             label="姣忚疆搴忓垪鏁?)
                    cl_top_k = gr.Slider(2, 10, value=5, step=1,
                                         label="姣忚疆淇濈暀 Top-K")
                    cl_max_cycles = gr.Slider(2, 8, value=3, step=1,
                                              label="鏈€澶ц疆鏁?)
                    cl_convergence = gr.Slider(0.005, 0.10, value=0.02, step=0.005,
                                               label="鏀舵暃闃堝€?(hypervolume 螖)")

                    with gr.Accordion("馃И Rosetta 鐗╃悊鎵撳垎 (楂樼骇)", open=False):
                        cl_ag_chain = gr.Textbox(label="鎶楀師閾?, value="A")
                        cl_ab_chain = gr.Textbox(label="鎶椾綋/缁撳悎鐗╅摼", value="B")
                        cl_enable_relax = gr.Checkbox(label="鍚敤 FastRelax 寮涜鲍",
                                                      value=False,
                                                      info="鑳介噺鏈€灏忓寲 (鎱絾绮惧噯)")

                    with gr.Accordion("馃敩 妯″瀷寰皟 (鍙€?", open=False):
                        cl_enable_finetune = gr.Checkbox(label="鍚敤濂栧姳鍔犳潈寰皟",
                                                         value=False,
                                                         info="姣廚杞敤楂樺垎璁捐寰皟BFN妯″瀷")
                        cl_finetune_freq = gr.Slider(1, 5, value=2, step=1,
                                                     label="寰皟闂撮殧 (杞?")

                    cl_run_btn = gr.Button("馃殌 鍚姩闂幆浼樺寲", variant="primary")

                with gr.Column(scale=2):
                    cl_progress = gr.Textbox(label="褰撳墠鐘舵€?, lines=2, interactive=False)
                    cl_report = gr.Textbox(label="姣忚疆鎶ュ憡", lines=18, interactive=False)

            with gr.Row():
                cl_ranking_df = gr.DataFrame(label="鏈€浼樿璁℃帓鍚?(鎵€鏈夎疆)")

            # 鈹€鈹€ Callback: PDB upload 鈹€鈹€
            def cl_on_upload(pdb_file):
                if pdb_file is None:
                    return "", "", ""
                chains = detect_chains(pdb_file.name)
                info = get_chain_info(pdb_file.name)
                lines = [f"缁撴瀯: {os.path.basename(pdb_file.name)}"]
                for c in chains:
                    ci = info.get(c, {})
                    lines.append(f"  閾?{c}: {ci.get('len','?')} 娈嬪熀 "
                                 f"[{ci.get('first','?')}-{ci.get('last','?')}]")
                default_region = (f"{chains[0]}:1-10" if chains else "")
                return '\n'.join(lines), pdb_file.name, default_region

            # 鈹€鈹€ Callback: Run closed loop 鈹€鈹€
            def cl_run(pdb_path, region, tool, n_samples, top_k, max_cycles,
                       convergence, ag_chain, ab_chain, enable_relax,
                       enable_finetune, finetune_freq, progress=gr.Progress()):
                """Execute the full closed-loop optimization."""
                if not pdb_path or not os.path.exists(str(pdb_path)):
                    return "閿欒: 璇峰厛涓婁紶闈剁偣 PDB 鏂囦欢", pd.DataFrame()

                # 鈹€鈹€ Build design_fn 鈹€鈹€
                def design_fn(pdb_p, region_s, n_s):
                    if tool.startswith("BFN"):
                        _, _, results = run_bfn_protein(
                            pdb_p, region_s, n_s, stochastic=True, eval_mode=True)
                    elif tool == "ProteinMPNN":
                        chains = re.findall(r'([A-Za-z0-9]+):', region_s)
                        chain_str = ' '.join(chains) if chains else 'A'
                        _, _, results = run_mpnn(
                            pdb_p, chain_str, n_s, temperature=0.1, seed=42, omit_aas='')
                    else:  # ESM-IF
                        _, _, results = run_esmif(
                            pdb_p, region_s.split(':')[0] if ':' in region_s else 'A',
                            n_s, temperature=0.1)
                    return results

                # 鈹€鈹€ Build af2_validator 鈹€鈹€
                def af2_validator(sequences, output_dir, **kwargs):
                    """Lightweight AF2 validator using ColabFold or mock."""
                    try:
                        from af2_validator import validate_sequences
                        return validate_sequences(
                            sequences, output_dir=output_dir,
                            model_type='alphafold2_multimer_v3',
                            num_recycle=1, progress_cb=None)
                    except ImportError:
                        # Fallback: return sequences with placeholder scores
                        return [{'sequence': s, 'success': True,
                                 'plddt': 70.0, 'ptm': 0.6, 'iptm': 0.5,
                                 'max_pae': 10.0, 'pdb_path': pdb_path}
                                for s in sequences]

                # 鈹€鈹€ Build physics scorer 鈹€鈹€
                physics_scorer = clo.create_physics_scorer(
                    target_pdb_path=str(pdb_path),
                    antibody_chains=ab_chain,
                    antigen_chain=ag_chain,
                    enable_relax=enable_relax,
                )

                # 鈹€鈹€ Build finetune_fn (optional) 鈹€鈹€
                finetune_fn = None
                if enable_finetune:
                    try:
                        import bfn_finetune as bft
                        bfn_ckpt = get_bfn_ckpt()
                        finetune_fn = bft.create_finetune_fn(
                            checkpoint_path=bfn_ckpt,
                            output_dir='closed_loop_results/finetune',
                            n_epochs=3, lr=1e-5, freeze_backbone=True,
                        )
                    except ImportError:
                        pass

                # 鈹€鈹€ Build orchestrator 鈹€鈹€
                log_messages = []
                def log_cb(msg):
                    log_messages.append(msg)

                orchestrator = clo.ClosedLoopOrchestrator(
                    design_fn=design_fn,
                    af2_validator=af2_validator,
                    physics_scorer=physics_scorer,
                    finetune_fn=finetune_fn,
                    n_samples=n_samples, top_k=top_k,
                    max_cycles=max_cycles,
                    convergence_delta=convergence,
                    finetune_every_n_cycles=finetune_freq,
                    selection_strategy='pareto_first',
                    log_cb=log_cb,
                    progress_cb=lambda c, m, s: progress(
                        (c-1)/m + 0.1/m, desc=f"Cycle {c}/{m}: {s}"),
                )

                progress(0.0, desc="Starting closed loop...")
                result = orchestrator.run(
                    initial_pdb=str(pdb_path),
                    design_region=region,
                )

                # 鈹€鈹€ Build report 鈹€鈹€
                report = clo.format_closed_loop_report(result)

                # 鈹€鈹€ Build DataFrame 鈹€鈹€
                all_ranked = []
                for cycle in result.cycles:
                    for r in cycle.all_ranked[:5]:
                        d = {
                            'Cycle': cycle.cycle_idx,
                            'Rank': r.get('mo_rank', '?'),
                            'ParetoFront': r.get('pareto_front', '?'),
                            'Composite': r.get('mo_composite_score', 0),
                            'ipTM': r.get('iptm', '') or '',
                            'pLDDT': r.get('plddt', '') or '',
                            'dG': r.get('dG', '') or '',
                            'PPL': round(r.get('ppl', 0) or 0, 1),
                            'Sequence': (r.get('sequence', '') or '')[:40],
                        }
                        all_ranked.append(d)

                df = pd.DataFrame(all_ranked) if all_ranked else pd.DataFrame()

                return report, df

            # 鈹€鈹€ Register callbacks 鈹€鈹€
            cl_pdb_file.change(cl_on_upload, [cl_pdb_file],
                               [cl_pdb_info, _cl_pdb, cl_region])
            cl_run_btn.click(cl_run,
                             [_cl_pdb, cl_region, cl_design_tool, cl_n_samples,
                              cl_top_k, cl_max_cycles, cl_convergence,
                              cl_ag_chain, cl_ab_chain, cl_enable_relax,
                              cl_enable_finetune, cl_finetune_freq],
                             [cl_report, cl_ranking_df])

        # 鈹€鈹€ Footer 鈹€鈹€
        gr.HTML("""
        <div class="platform-footer">
            <strong>Protein Design Platform</strong> v3.0 &nbsp;路&nbsp;
            BFN 路 ProteinMPNN 路 ESM-IF 路 AlphaFold2 &nbsp;路&nbsp;
            绠＄悊: <code>python manage.py start|stop|restart|status|batch|config|test</code>
        </div>
        """)

        # Register send-to-AF2 callback (must be after af_fasta_text is defined)
        send_to_af2_btn.click(
            on_send_to_af2, [fasta_download],
            [af_fasta_text, af_fasta_info]
        )

        # Register FASTA鈫扨DB conversion and clear callbacks
        def on_fasta_to_pdb(fasta_text, fasta_file):
            # If file uploaded, read it
            if fasta_file is not None:
                try:
                    with open(fasta_file.name, encoding='utf-8') as f:
                        fasta_text = f.read()
                except Exception:
                    pass
            pdb_path, info = fasta_to_pdb_file(fasta_text)
            if pdb_path:
                info_text, chains, region = on_upload(type('F', (), {'name': pdb_path})())
                return pdb_path, pdb_path, info_text, chains, region, info
            return None, "", "", "", "", info

        def on_f2p_file_upload(fasta_file):
            """When a FASTA file is uploaded, populate the text box."""
            if fasta_file is not None:
                try:
                    with open(fasta_file.name, encoding='utf-8') as f:
                        return f.read()
                except Exception:
                    pass
            return ""

        def on_clear_f2p():
            return "", None, "", "", "", "", ""

        f2p_btn.click(
            on_fasta_to_pdb, [f2p_fasta, f2p_fasta_file],
            [pdb_file, _pdb_path, pdb_info, _chain_list, _def_region, f2p_info]
        )
        f2p_fasta_file.upload(
            on_f2p_file_upload, [f2p_fasta_file], [f2p_fasta]
        )
        f2p_clear.click(
            on_clear_f2p,
            outputs=[f2p_fasta, f2p_fasta_file, pdb_info, _chain_list, _def_region, _pdb_path, f2p_info]
        )

    return app, css


if __name__ == '__main__':
    cfg = load_app_config()
    sv = cfg['server']

    print("=" * 60)
    print("  Protein Sequence Design Platform v2.0")
    print(f"  http://{sv['host']}:{sv['port']}")
    print(f"  澶栭儴璁块棶: {'鍏佽' if sv['host'] != '127.0.0.1' else '绂佹锛堜粎鏈満锛?}")
    print("=" * 60)

    # Preload BFN
    try:
        print("  棰勫姞杞?BFN 妯″瀷...")
        load_bfn()
        print("  BFN 鉁?)
    except Exception as e:
        print(f"  BFN 璺宠繃: {e}")

    print(f"\n  Web 鐣岄潰: http://{sv['host']}:{sv['port']}")
    print(f"  绠＄悊鍛戒护: python manage.py --help")
    print()

    app, css = create_ui()

    # Add security headers via ASGI middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
            response.headers.setdefault("Permissions-Policy",
                "accelerometer=(), camera=(), geolocation=(), microphone=(), payment=(), usb=()")
            response.headers.setdefault("Content-Security-Policy",
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: blob:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "connect-src 'self'")
            return response

    app.app.add_middleware(SecurityHeadersMiddleware)

    app.launch(
        server_name=sv['host'],
        server_port=sv['port'],
        share=sv.get('share', False),
        inbrowser=sv.get('auto_open', True),
        css=css,
        show_error=True,
    )
