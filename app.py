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
import target_design_helpers as tdh
import cascade_filter as cf

PROJECT_DIR = Path(__file__).parent
CONFIG_FILE = PROJECT_DIR / 'app_config.yaml'
DEFAULT_MODEL_CONFIG = PROJECT_DIR / 'configs' / 'demo_design.yml'

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

# ── Global caches ──
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


# ── PDB Helpers ──

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
        return "请上传 PDB 文件", "", ""
    try:
        chains = detect_chains(pdb_file.name)
        info = get_chain_info(pdb_file.name)
        lines = [f"检测到 {len(chains)} 条链: {', '.join(chains)}"]
        for c in chains:
            ci = info.get(c, {})
            s = ci.get('seq', '')
            lines.append(f"  链 {c}: {ci.get('len','?')}残基 [{ci.get('first','?')}-{ci.get('last','?')}] {s[:60]}{'...' if len(s)>60 else ''}")
        return '\n'.join(lines), ', '.join(chains), f"{chains[0]}:1-10" if chains else ""
    except Exception as e:
        return f"解析错误: {e}", "", ""


# ── Model Loaders ──

def load_bfn():
    global _bfn_model, _bfn_config
    if _bfn_model is not None:
        return _bfn_model, _bfn_config
    from antibodydesignbfn.models import get_model
    from antibodydesignbfn.utils.misc import load_config as _lc
    ckpt_path = get_bfn_ckpt()
    config, _ = _lc(DEFAULT_MODEL_CONFIG)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)
    model = get_model(mc).to('cpu')
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
    global _esmif_model
    if _esmif_model is not None:
        return _esmif_model
    from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
    model, _ = esm_if1_gvp4_t16_142M_UR50()
    model = model.to('cpu').eval()
    _esmif_model = model
    return model


# ── Confidence Helpers ──

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


# ── Standalone Confidence Evaluation ──

def run_bfn_confidence_evaluation(pdb_path, region_spec):
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform

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
    batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
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


# ── Design Functions ──

def run_bfn_antibody(pdb_path, heavy, light, num_samples, stochastic, eval_mode):
    from antibodydesignbfn.datasets.custom import preprocess_antibody_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform

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

    return '\n'.join(lines), fasta_str, results_list


def run_bfn_protein(pdb_path, region_spec, num_samples, stochastic, eval_mode):
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.misc import seed_all
    from antibodydesignbfn.utils.data import PaddingCollate
    from antibodydesignbfn.utils.transforms import get_transform

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
        return f"区域格式错误: {region_spec}", "", []

    model, config = load_bfn()
    seed_all(getattr(config.sampling, 'seed', 42))
    structure = preprocess_protein_structure(pdb_path, chain_ids=list(regions.keys()))
    if structure is None:
        return "错误：无法解析结构", "", []

    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
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
    return '\n'.join(lines), fasta_str, results_list


def run_esmif(pdb_path, chain, temperature, num_samples):
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


# ── FASTA → PDB Conversion ──

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
    # N-CA-C ≈ 111° (standard tetrahedral geometry).
    # N and C straddle the forward (tangent) direction in the plane formed
    # by forward × helix-axis.  Since the peptide plane parallels the helix
    # axis in an α-helix, the perpendicular is (0, 0, 1).
    half_n_ca_c = math.radians(55.5)  # half of 111°
    cos_half = math.cos(half_n_ca_c)   # ≈ 0.566
    sin_half = math.sin(half_n_ca_c)   # ≈ 0.824

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
        # is approximately parallel to the helix axis in an α-helix.
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

        # O: placed trans to N relative to CA-C, ~120° from CA-C bond.
        # CA-C-O angle ≈ 120° (standard peptide geometry).
        ca_to_c_x = x_c - cx
        ca_to_c_y = y_c - cy
        ca_to_c_z = z_c - cz
        cc_norm = math.sqrt(ca_to_c_x**2 + ca_to_c_y**2 + ca_to_c_z**2)
        ca_to_c_x, ca_to_c_y, ca_to_c_z = ca_to_c_x / cc_norm, ca_to_c_y / cc_norm, ca_to_c_z / cc_norm

        # O direction: rotate CA→C by ~120° in the peptide plane (forward × helix-axis)
        ox = fy * pz - fz * py  # forward × z_axis
        oy = fz * px - fx * pz
        oz = fx * py - fy * px
        o_norm = math.sqrt(ox * ox + oy * oy + oz * oz)
        if o_norm > 1e-6:
            ox, oy, oz = ox / o_norm, oy / o_norm, oz / o_norm
        else:
            ox, oy, oz = 1.0, 0.0, 0.0

        # C→O = bond_c_o * (cos_60 * u + sin_60 * w) where u = CA→C, w ⟂ peptide
        # This gives CA-C-O ≈ 120° (standard peptide geometry).
        cos60 = 0.5
        sin60 = math.sqrt(3.0) / 2.0  # ≈ 0.866
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
        return None, "请提供 FASTA 序列"

    seqs = parse_fasta(fasta_text)
    if not seqs:
        return None, "无效的 FASTA 格式（需要 > 开头）"

    # Use the first sequence
    header, seq = seqs[0]
    pdb_content = generate_pdb_from_sequence(seq)

    if pdb_content is None:
        return None, "序列中没有有效的氨基酸"

    # Use header as filename base
    name = re.sub(r'[^A-Za-z0-9_-]', '_', header.split()[0] if header else 'protein')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False,
                                      encoding='utf-8', prefix=f'{name}_')
    tmp.write(pdb_content)
    tmp.close()

    info = (f"✅ PDB 已生成: {name}\n"
            f"   链 A: {len(seq)} 个残基\n"
            f"   文件: {tmp.name}\n"
            f"   注意: 此 PDB 为 α-螺旋模板骨架，非真实折叠结构\n"
            f"   可用于 BFN / ProteinMPNN / ESM-IF 进行序列设计")
    return tmp.name, info


# ── AlphaFold Structure Prediction ──

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
        return "", "请提供 FASTA 序列或上传 .fasta 文件"
    seqs = parse_fasta(fasta_text)
    if not seqs:
        return "", "未检测到有效 FASTA 格式序列（需要 > 开头）"
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
        return "请提供 FASTA 序列", None, None

    seqs = parse_fasta(fasta_text)
    if not seqs:
        return "无效的 FASTA 格式", None, None

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

    progress(0.05, desc="启动 AlphaFold2 (ColabFold)...")
    log_lines = [f"🔮 AlphaFold2 (ColabFold) 结构预测",
                 f"{'='*50}",
                 f"序列: {seqs[0][0]} ({len(seqs[0][1])} aa)",
                 f"模型: {model_type}  |  回收: {int(num_recycle)}  |  达标分: {int(stop_at)}",
                 f"输出: {out_dir / pid}/", f"",
                 f"⏱ CPU 预测较慢，请耐心等待（小蛋白约 5-15 分钟）"]

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


# ── Batch Processing ──

def run_batch(pdb_files, tool, region_spec, chains_spec, temperature, num_samples, progress=gr.Progress()):
    if not pdb_files:
        return "请上传 PDB 文件", None, None

    results = []
    all_results_lists = []
    total = len(pdb_files)
    output_dir = Path('batch_results')
    output_dir.mkdir(exist_ok=True)

    for i, pf in enumerate(pdb_files):
        pdb_name = Path(pf.name).stem
        progress((i+1)/total, desc=f"处理 {pdb_name}...")

        try:
            if tool == "BFN (通用蛋白设计)":
                text, fasta_str, results_list = run_bfn_protein(pf.name, region_spec, num_samples, False, True)
            elif tool == "ProteinMPNN":
                text, fasta_str, results_list = run_mpnn(pf.name, chains_spec, num_samples, temperature, 42, "")
            elif tool == "ESM-IF":
                text, fasta_str, results_list = run_esmif(pf.name, chains_spec, float(temperature), num_samples)
            else:
                text = f"未知工具: {tool}"
                fasta_str = ""
                results_list = []
            results.append({'pdb': pdb_name, 'status': 'OK', 'text': text, 'fasta': fasta_str})
            all_results_lists.extend(results_list)
        except Exception as e:
            results.append({'pdb': pdb_name, 'status': 'ERROR', 'error': str(e)})

    # Build summary
    lines = [f"批量处理完成: {total} 个文件\n"]
    ok = sum(1 for r in results if r['status'] == 'OK')
    err = sum(1 for r in results if r['status'] == 'ERROR')
    lines.append(f"成功: {ok}  失败: {err}\n")
    lines.append("=" * 60)

    for r in results:
        lines.append(f"\n{'─'*60}")
        lines.append(f"📁 {r['pdb']}  [{r['status']}]")
        if r['status'] == 'OK':
            lines.append(r['text'])
        else:
            lines.append(f"错误: {r['error']}")

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

    lines.append(f"\n结果已保存到: {output_dir}/")

    df = build_results_dataframe(all_results_lists)
    return '\n'.join(lines), str(json_path), df


# ── Target Design Pipeline ──

def run_target_design(target_pdb, target_chain, epitope_region,
                       antibody_pdb, ab_heavy, ab_light,
                       design_tool, constraint_mode, constraint_cutoff,
                       bfn_samples, bfn_stochastic, bfn_eval,
                       mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
                       esmif_temp, esmif_samples):
    """Target-aware constrained design pipeline."""
    if not target_pdb or not os.path.exists(str(target_pdb)):
        return "请先完成步骤1：上传或预测靶点结构", "", "", None

    if not epitope_region or not epitope_region.strip():
        return "请先完成步骤2：分析并选择表位区域", "", "", None

    if not antibody_pdb or not os.path.exists(str(antibody_pdb)):
        return "请先完成步骤3：提供抗体结构", "", "", None

    # Parse epitope region
    epitope_residues = tdh.parse_region_spec(epitope_region)
    if not epitope_residues:
        return f"表位区域格式错误: {epitope_region}", "", "", None

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
            design_lines.append(f"距离约束模式 (cutoff={constraint_cutoff}Å):")
            design_lines.append(f"  面向表位的重链残基: {len(facing)} 个")
            if ab_light and ab_light.strip():
                design_lines.append(f"  面向表位的轻链残基: {len(facing_l)} 个")

            if not facing and not facing_l:
                return ("未找到面向表位的抗体残基 — 请检查距离阈值或表位/抗体结构是否正确",
                        "", "", None)

            # Build region spec from facing residues
            region_parts = []
            if facing:
                region_parts.append(tdh.residues_to_region_spec(ab_heavy, facing))
            if facing_l:
                region_parts.append(tdh.residues_to_region_spec(ab_light, facing_l))
            design_region = ' '.join(region_parts)
            design_lines.append(f"  设计区域: {design_region}")
            design_lines.append("")
        else:
            # Without constraint, use CDR defaults
            design_region = f"{ab_heavy}:95-102"  # CDR H3 range
            if ab_light and ab_light.strip():
                design_region += f" {ab_light}:89-97"  # CDR L3 range
            design_lines.append(f"非约束模式 — 默认CDR区域: {design_region}")
            design_lines.append("")

        # Run selected design tool
        results_list = []
        if design_tool == "BFN (抗体CDR设计)":
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
            return f"未知工具: {design_tool}", "", "", None

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
                validation_text = f"未检测到 {ab_heavy}-{target_chain} 之间的接触 (距离阈值 8Å)"
        except Exception as val_err:
            validation_text = f"验证跳过: {val_err}"

    except Exception as e:
        import traceback
        return f"靶点设计错误:\n{traceback.format_exc()}", "", "", None

    results_df = build_results_dataframe(results_list)
    return '\n'.join(design_lines), fasta_str, validation_text, results_df


# ── Main Dispatch ──

def run_design(pdb_path, tool,
               bfn_ab_heavy, bfn_ab_light, bfn_ab_samples, bfn_ab_stochastic, bfn_ab_eval,
               bfn_pro_region, bfn_pro_samples, bfn_pro_stochastic, bfn_pro_eval,
               mpnn_chains, mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
               esmif_chain, esmif_temp, esmif_samples):
    if not pdb_path or not os.path.exists(str(pdb_path)):
        return "请先上传 PDB 文件", None, None, gr.update(visible=False)
    try:
        results_list = []
        if tool == "BFN (抗体CDR设计)":
            text, fasta_str, results_list = run_bfn_antibody(pdb_path, bfn_ab_heavy, bfn_ab_light,
                                    int(bfn_ab_samples), bool(bfn_ab_stochastic), bool(bfn_ab_eval))
        elif tool == "BFN (通用蛋白设计)":
            text, fasta_str, results_list = run_bfn_protein(pdb_path, bfn_pro_region,
                                   int(bfn_pro_samples), bool(bfn_pro_stochastic), bool(bfn_pro_eval))
        elif tool == "ProteinMPNN":
            text, fasta_str, results_list = run_mpnn(pdb_path, mpnn_chains, int(mpnn_samples), mpnn_temp, int(mpnn_seed), mpnn_omit)
        elif tool == "ESM-IF":
            text, fasta_str, results_list = run_esmif(pdb_path, esmif_chain, float(esmif_temp), int(esmif_samples))
        else:
            return f"未知工具: {tool}", None, None, gr.update(visible=False)
    except Exception as e:
        import traceback
        return f"运行错误:\n{traceback.format_exc()}", None, None, gr.update(visible=False)

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


# ── Unified Pipeline ──

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


# ── Config Viewer ──

def get_config_display():
    try:
        cfg = load_app_config()
        return yaml.dump(cfg, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        return f"配置读取错误: {e}"


def save_config_from_text(config_text):
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

    status = f"""服务状态面板
{'='*50}
监听地址:     {host}:{port}
端口状态:     {'✓ 运行中' if port_open else '✗ 未启动'}
外部访问:     {'禁止 (仅本机)' if host == '127.0.0.1' else '允许'}
{'='*50}
BFN 模型:      {'✓ 加载就绪' if bfn_loaded else ('✓ 文件存在' if bfn_ok else '✗ 未找到')}
ProteinMPNN:   {'✓ 就绪' if mpnn_ok else '✗ 未找到'}
ESM-IF:        {'✓ 已加载' if esmif_loaded else '○ 按需加载'}
AlphaFold2:    {'✓ 就绪' if af2_ok else '✗ 未找到'}
{'='*50}
配置文件:      {CONFIG_FILE}
Python:        {sys.version.split()[0]}
PyTorch:       {torch.__version__}
CUDA:          {torch.cuda.is_available()}
"""
    return status


# ── UI ──

def create_ui():
    css = """
    /* ══════════════════════════════════════════════════════════════════
       Protein Design Platform — Layered Functional Depth Theme
       ══════════════════════════════════════════════════════════════════ */

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

    /* ── All tabs same height ── */
    .tabs > .tabitem { min-height:680px !important; }

    /* ── Header ── */
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

    /* ── Tabs nav ── */
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

    /* ── Functional section blocks ── */
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

    /* ── Accordion ── */
    .accordion > .label-wrap {
        background:var(--input) !important; padding:10px 18px !important;
        font-weight:600 !important; color:var(--text) !important;
        border-bottom:1px solid var(--border); border-radius:16px 16px 0 0 !important;
    }

    /* ── Buttons ── */
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

    /* ── Inputs (inset depth) ── */
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

    /* ── Radio / Checkbox ── */
    input[type="radio"]:checked + label { color:var(--accent) !important; font-weight:600 !important; }
    .gr-checkbox label, .gr-checkbox span { color:var(--text) !important; }
    input[type="range"] { accent-color:var(--accent) !important; }

    /* ── File upload ── */
    .file-preview, .file-upload {
        border-radius:14px !important; border:1px dashed var(--border) !important;
        background:var(--input) !important; box-shadow:inset 0 1px 4px rgba(0,0,0,0.25) !important;
    }

    /* ── Result boxes ── */
    .result-box textarea, .result-box, .result-box input, .result-box .gr-textbox textarea {
        font-family:'JetBrains Mono','Cascadia Code','Fira Code','Consolas',monospace !important;
        font-size:0.82em !important; background:#0a0a10 !important; color:#b2baf6 !important;
        border-radius:12px !important; border:1px solid var(--border) !important; line-height:1.5 !important;
        box-shadow:inset 0 2px 8px rgba(0,0,0,0.45) !important;
    }
    .result-box label { color:var(--muted) !important; }

    /* ── Footer ── */
    .platform-footer {
        text-align:center; color:var(--muted); font-size:0.75em; margin-top:16px;
        padding:10px 16px; background:var(--surface); border-radius:16px;
        border:1px solid var(--border); box-shadow:0 2px 8px rgba(0,0,0,0.15);
    }
    .platform-footer code { background:var(--input); color:var(--text); padding:2px 7px; border-radius:8px; font-size:0.9em; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width:7px; height:7px; }
    ::-webkit-scrollbar-track { background:var(--bg); border-radius:4px; }
    ::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
    ::-webkit-scrollbar-thumb:hover { background:var(--muted); }

    /* ── Transitions & overrides ── */
    button, input, textarea, select { transition:all 0.2s ease; }
    .gr-box, .gr-form, .gr-panel, .gr-group, .gr-accordion, .gr-tab-item, .gr-tabs {
        background:transparent !important; border:none !important; box-shadow:none !important;
    }
    .app { background:var(--bg) !important; }
    .contain { background:transparent !important; }
    footer { display:none !important; }

    /* ── Target Design Tab: larger layout ── */
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
            <h1>🧬 蛋白质序列设计平台</h1>
            <p class="subtitle">Protein Sequence Design Platform — BFN · ProteinMPNN · ESM-IF · AlphaFold2</p>
            <div class="meta-row">
                <span class="badge"><span class="badge-dot"></span> 服务运行中</span>
                <span class="badge">🔒 仅本机访问</span>
                <span class="badge">http://127.0.0.1:7860</span>
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ═══════════════════════  Tab 1: Single Design ═══════════════════════
            with gr.TabItem("🎯 单步设计"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # FASTA → PDB converter
                        with gr.Accordion("🧬 FASTA → PDB 转换", open=False):
                            gr.Markdown(
                                "输入序列或上传 FASTA 文件，自动生成 **α-螺旋模板 PDB** 供设计使用。\n\n"
                                "几何参数：3.6 残基/圈 · 螺距 1.5 Å · 半径 2.3 Å · CA-CA ≈ 3.8 Å\n"
                                "骨架原子 N/CA/C/O 均备，键长键角符合标准蛋白质几何。\n\n"
                                "✅ 兼容 **BFN** / **ProteinMPNN** / **ESM-IF** 三款设计工具"
                            )
                            f2p_fasta = gr.Textbox(
                                label="FASTA 序列", lines=4,
                                placeholder=">my_protein\nMSDRPTARRWGK..."
                            )
                            f2p_fasta_file = gr.File(label="或上传 FASTA 文件",
                                                     file_types=[".fasta", ".fa", ".txt"])
                            with gr.Row():
                                f2p_btn = gr.Button("🔄 生成 α-螺旋模板 PDB", variant="secondary", size="sm")
                                f2p_clear = gr.Button("✕ 清空", size="sm")
                            f2p_info = gr.Textbox(label="转换信息", lines=3, interactive=False)

                        with gr.Group():
                            gr.Markdown("### 📂 1. 上传 PDB")
                            pdb_file = gr.File(label="选择 PDB 文件", file_types=[".pdb"])
                            pdb_info = gr.Textbox(label="结构信息", lines=6, interactive=False)
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
                            gr.Markdown("### ⚙️ 2. 选择工具")
                            tool_sel = gr.Radio(
                                ["BFN (抗体CDR设计)", "BFN (通用蛋白设计)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (通用蛋白设计)", label="设计工具"
                            )

                    with gr.Column(scale=2):
                        # BFN Antibody
                        with gr.Group(visible=False) as pan_bfn_ab:
                            gr.Markdown("**BFN 抗体 CDR 设计**")
                            with gr.Row():
                                bfn_ab_heavy = gr.Textbox(label="重链ID", value="H")
                                bfn_ab_light = gr.Textbox(label="轻链ID", value="L")
                            with gr.Row():
                                bfn_ab_samples = gr.Slider(1, 20, value=3, step=1, label="序列数")
                                bfn_ab_stochastic = gr.Checkbox(label="随机采样", value=False)
                                bfn_ab_eval = gr.Checkbox(label="评估模式", value=True)

                        # BFN Protein
                        with gr.Group(visible=True) as pan_bfn_pro:
                            gr.Markdown("**BFN 通用蛋白质设计**")
                            bfn_pro_region = gr.Textbox(label="设计区域", value="A:10-25",
                                                        placeholder="A:10-25 或 A:10-25 B:5-15")
                            with gr.Row():
                                bfn_pro_samples = gr.Slider(1, 20, value=3, step=1, label="序列数")
                                bfn_pro_stochastic = gr.Checkbox(label="随机采样", value=False)
                                bfn_pro_eval = gr.Checkbox(label="评估模式", value=True)

                        # ProteinMPNN
                        with gr.Group(visible=False) as pan_mpnn:
                            gr.Markdown("**ProteinMPNN**")
                            with gr.Row():
                                mpnn_chains = gr.Textbox(label="设计链", value="A")
                                mpnn_temp = gr.Textbox(label="温度", value="0.1")
                                mpnn_samples = gr.Slider(1, 20, value=3, step=1, label="序列数")
                            with gr.Row():
                                mpnn_seed = gr.Number(label="随机种子", value=42, precision=0)
                                mpnn_omit = gr.Textbox(label="排除AA", value="", placeholder="如: CX")

                        # ESM-IF
                        with gr.Group(visible=False) as pan_esmif:
                            gr.Markdown("**ESM-IF**")
                            with gr.Row():
                                esmif_chain = gr.Textbox(label="目标链", value="A")
                                esmif_temp = gr.Slider(0.05, 1.0, value=0.1, step=0.05, label="温度")
                                esmif_samples = gr.Slider(1, 10, value=3, step=1, label="序列数")

                        with gr.Group():
                            gr.Markdown("### 🚀 3. 开始")
                            run_btn = gr.Button("▶ 开始设计", variant="primary", size="lg")

                # Result
                with gr.Group():
                    gr.Markdown("### 📊 结果")
                    result_out = gr.Textbox(label="", lines=18, interactive=False,
                                            elem_classes="result-box", placeholder="结果将在此显示...")
                    result_df = gr.DataFrame(label="设计结果排名", interactive=False,
                                             wrap=True)

                # FASTA action bar (hidden until design completes)
                with gr.Group(visible=False) as fasta_action_bar:
                    gr.Markdown("### 📁 设计序列导出")
                    with gr.Row():
                        fasta_download = gr.File(label="💾 FASTA 文件下载", file_types=[".fasta"],
                                                 elem_classes="result-box")
                        send_to_af2_btn = gr.Button("🔮 发送到 AlphaFold2 预测",
                                                     variant="secondary", size="lg")
                    _fasta_for_af2 = gr.State("")

                # Tool switch
                def switch_tool(t):
                    return (
                        gr.update(visible=(t == "BFN (抗体CDR设计)")),
                        gr.update(visible=(t == "BFN (通用蛋白设计)")),
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

            # ═══════════════════════  Tab 2: Batch ═══════════════════════
            with gr.TabItem("📦 批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 📂 1. 上传 PDB")
                            batch_files = gr.File(label="上传 PDB 文件（可多选）", file_types=[".pdb"],
                                                 file_count='multiple')

                        with gr.Group():
                            gr.Markdown("### ⚙️ 2. 选择工具与参数")
                            batch_tool = gr.Radio(
                                ["BFN (通用蛋白设计)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (通用蛋白设计)", label="设计工具"
                            )
                            batch_region = gr.Textbox(label="设计区域 (BFN)", value="A:10-25")
                            batch_chains = gr.Textbox(label="目标链 (MPNN/ESM-IF)", value="A")
                            batch_temp = gr.Textbox(label="温度", value="0.1")
                            batch_samples = gr.Slider(1, 10, value=3, step=1, label="每文件序列数")

                        with gr.Group():
                            gr.Markdown("### 🚀 3. 开始")
                            batch_btn = gr.Button("▶ 开始批量处理", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📊 结果")
                            batch_result = gr.Textbox(label="", lines=22, interactive=False,
                                                      elem_classes="result-box", placeholder="结果将在此显示...")
                        batch_json = gr.File(label="汇总 JSON", visible=False)
                        batch_df = gr.DataFrame(label="📊 综合结果排名", interactive=False)

                batch_btn.click(
                    run_batch,
                    [batch_files, batch_tool, batch_region, batch_chains, batch_temp, batch_samples],
                    [batch_result, batch_json, batch_df]
                )

            # ═══════════════════════  Tab 3: Structure Prediction ═══════════════════════
            with gr.TabItem("🔮 结构预测"):
                gr.Markdown("### AlphaFold2 结构预测 —— 从序列到结构，一键接入设计流程")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### 📝 输入序列")
                            af_fasta_text = gr.Textbox(
                                label="FASTA 序列", lines=8,
                                placeholder=">protein_name\nMSDRPTARRWGKCGPLCTRENIMV..."
                            )
                            af_fasta_file = gr.File(label="或上传 FASTA 文件", file_types=[".fasta", ".fa", ".txt"])
                            af_fasta_info = gr.Textbox(label="序列预览", lines=4, interactive=False)
                            af_fasta_text.change(on_fasta_input, [af_fasta_text, af_fasta_file],
                                                [af_fasta_text, af_fasta_info])
                            af_fasta_file.upload(on_fasta_input, [af_fasta_text, af_fasta_file],
                                                [af_fasta_text, af_fasta_info])

                        with gr.Group():
                            gr.Markdown("#### ⚙️ AlphaFold2 (ColabFold) 预测参数")
                            with gr.Row():
                                af_num_models = gr.Slider(1, 5, value=1, step=1, label="模型数",
                                                         info="1=快速(推荐) 3=标准 5=最慢最准")
                                af_num_recycle = gr.Slider(0, 12, value=1, step=1, label="回收次数",
                                                          info="1=快速(推荐) 3=标准 更多=更精确但更慢")
                            af_use_dropout = gr.Checkbox(label="随机采样 (多样性)", value=False,
                                                        info="开启可产生多样结构但更慢")

                        af_run_btn = gr.Button("▶ 开始预测", variant="primary", size="lg")
                        af_send_btn = gr.Button("📤 发送到设计", variant="secondary", size="sm",
                                                visible=False)

                    with gr.Column(scale=2):
                        af_result = gr.Textbox(label="预测日志", lines=20, interactive=False,
                                              elem_classes="result-box")
                        af_pdb_file = gr.File(label="预测结构 PDB", visible=True,
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
                        return None, "", "无可用 PDB 结构", "", ""
                    if isinstance(pdb_file_val, str):
                        path = pdb_file_val
                    elif isinstance(pdb_file_val, dict):
                        path = pdb_file_val.get('name', '')
                    else:
                        path = getattr(pdb_file_val, 'name', '')
                    if not path or not os.path.exists(path):
                        return None, "", f"PDB 文件不存在: {path}", "", ""
                    info_text, chains, region = on_upload(type('F', (), {'name': path})())
                    return path, path, info_text, chains, region

                af_send_btn.click(
                    on_send_to_design, [af_pdb_file],
                    [pdb_file, _pdb_path, pdb_info, _chain_list, _def_region]
                )

            # ═══════════════════════  Tab 4: Settings ═══════════════════════
            with gr.TabItem("⚙️ 设置"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 📊 系统状态")
                            status_display = gr.Textbox(label="", value=get_system_status(),
                                                        lines=18, interactive=False,
                                                        elem_classes="result-box")
                            refresh_btn = gr.Button("🔄 刷新状态", variant="secondary")

                        with gr.Group():
                            gr.Markdown("### 🔧 模型预加载")
                            with gr.Row():
                                preload_esmif_btn = gr.Button("加载 ESM-IF", size="sm")
                                preload_bfn_btn = gr.Button("重载 BFN", size="sm")

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📝 配置文件编辑")
                            gr.Markdown(f"路径: `{CONFIG_FILE}`")
                            config_display = gr.Textbox(
                                label="", value=get_config_display(),
                                lines=22, interactive=True, elem_classes="result-box"
                            )
                            with gr.Row():
                                save_cfg_btn = gr.Button("💾 保存配置", variant="primary", size="sm")
                                reload_cfg_btn = gr.Button("🔄 重新加载", variant="secondary", size="sm")
                                cfg_msg = gr.Textbox(label="", visible=True, interactive=False, container=False)

                def do_save_config(text):
                    return save_config_from_text(text)

                def do_reload_config():
                    global _app_config
                    _app_config = None
                    return get_config_display(), "✅ 已重新加载"

                save_cfg_btn.click(do_save_config, [config_display], [cfg_msg])
                reload_cfg_btn.click(do_reload_config, outputs=[config_display, cfg_msg])

                def do_preload_esmif():
                    try:
                        load_esmif()
                        return get_system_status(), "✅ ESM-IF 已加载"
                    except Exception as e:
                        return get_system_status(), f"❌ 失败: {e}"

                def do_reload_bfn():
                    global _bfn_model, _bfn_config
                    _bfn_model = None; _bfn_config = None
                    try:
                        load_bfn()
                        return get_system_status(), "✅ BFN 已重载"
                    except Exception as e:
                        return get_system_status(), f"❌ 失败: {e}"

                preload_esmif_btn.click(do_preload_esmif, outputs=[status_display, cfg_msg])
                preload_bfn_btn.click(do_reload_bfn, outputs=[status_display, cfg_msg])

            # ═══════════════════════  Tab 5: Target Design ═══════════════════════
            with gr.TabItem("🎯 靶点设计"):
                gr.HTML('<div class="td-wrapper">')
                gr.Markdown("### 靶点导向的抗体/蛋白设计 —— 从靶点分析到结合界面优化")

                # Tab 5 state variables
                _t5_target_pdb = gr.State("")
                _t5_target_chain = gr.State("A")
                _t5_epitope_data = gr.State(None)
                _t5_epitope_region = gr.State("")
                _t5_antibody_pdb = gr.State("")
                _t5_design_fasta = gr.State("")

                # ── Step 1: Target Input ──
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🎯 1. 靶点输入")
                            with gr.Accordion("FASTA → AlphaFold2 预测靶点结构", open=False):
                                t5_fasta_text = gr.Textbox(
                                    label="靶点序列 (FASTA)", lines=4,
                                    placeholder=">target_protein\nMSDRPTARRWGKCG..."
                                )
                                t5_fasta_file = gr.File(label="或上传 FASTA 文件",
                                                        file_types=[".fasta", ".fa", ".txt"])
                                with gr.Row():
                                    t5_predict_btn = gr.Button("🔮 AlphaFold2 预测", variant="primary", size="sm")
                                    t5_fasta_info_btn = gr.Button("📋 解析序列", size="sm")
                                t5_fasta_info = gr.Textbox(label="序列预览", lines=3, interactive=False)

                            t5_pdb_file = gr.File(label="或直接上传靶点 PDB", file_types=[".pdb"])
                            t5_chain_selector = gr.Dropdown(label="选择分析链", choices=[], interactive=True)
                            t5_pdb_info = gr.Textbox(label="结构信息", lines=4, interactive=False)

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 📋 步骤日志")
                            t5_step1_log = gr.Textbox(label="", lines=14, interactive=False,
                                                      elem_classes="result-box",
                                                      placeholder="靶点结构信息将在此显示...")

                # ── Step 2: Epitope Analysis ──
                with gr.Group():
                    gr.Markdown("### 🔬 2. 表位分析")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                t5_epitope_run_btn = gr.Button("▶ 分析表位", variant="primary")
                                t5_top_n = gr.Slider(5, 50, value=20, step=5, label="显示Top-N")
                            t5_epitope_methods = gr.CheckboxGroup(
                                choices=["SASA (溶剂可及性)", "亲水性", "凸出指数"],
                                value=["SASA (溶剂可及性)", "亲水性", "凸出指数"],
                                label="评分方法"
                            )
                            t5_epitope_region_input = gr.Textbox(
                                label="确认表位区域",
                                placeholder="如: A:25-35,40-50 或 A:25,30,35,40",
                                value=""
                            )
                            t5_epitope_select_btn = gr.Button("✓ 确认表位区域", variant="secondary")

                        with gr.Column(scale=2):
                            t5_epitope_result = gr.Textbox(
                                label="表位评分结果", lines=18, interactive=False,
                                elem_classes="result-box",
                                placeholder="点击'分析表位'查看候选表位残基..."
                            )

                # ── Step 3 + 4: Antibody Setup + Constrained Design ──
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🧬 3. 抗体设置")
                            with gr.Accordion("FASTA → 模板 PDB (α-螺旋)", open=False):
                                t5_ab_fasta = gr.Textbox(
                                    label="抗体序列 (FASTA)", lines=3,
                                    placeholder=">antibody_scfv\nEVQLVESGG..."
                                )
                                t5_ab_gen_pdb_btn = gr.Button("🔄 生成 α-螺旋模板 PDB", size="sm")

                            t5_ab_pdb_file = gr.File(label="或直接上传抗体 PDB", file_types=[".pdb"])
                            t5_ab_pdb_info = gr.Textbox(label="抗体结构信息", lines=3, interactive=False)
                            with gr.Row():
                                t5_ab_heavy = gr.Textbox(label="重链ID", value="H", scale=1)
                                t5_ab_light = gr.Textbox(label="轻链ID", value="L", scale=1)

                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 🚀 4. 约束设计")
                            t5_design_tool = gr.Radio(
                                ["BFN (抗体CDR设计)", "ProteinMPNN", "ESM-IF"],
                                value="BFN (抗体CDR设计)", label="设计工具"
                            )

                            # BFN params
                            with gr.Group(visible=True) as t5_pan_bfn:
                                gr.Markdown("**BFN 参数**")
                                with gr.Row():
                                    t5_bfn_samples = gr.Slider(1, 10, value=3, step=1, label="序列数")
                                    t5_bfn_stochastic = gr.Checkbox(label="随机采样", value=False)
                                    t5_bfn_eval = gr.Checkbox(label="评估模式", value=True)

                            # MPNN params
                            with gr.Group(visible=False) as t5_pan_mpnn:
                                gr.Markdown("**ProteinMPNN 参数**")
                                with gr.Row():
                                    t5_mpnn_temp = gr.Textbox(label="温度", value="0.1")
                                    t5_mpnn_samples = gr.Slider(1, 10, value=3, step=1, label="序列数")
                                with gr.Row():
                                    t5_mpnn_seed = gr.Number(label="随机种子", value=42, precision=0)
                                    t5_mpnn_omit = gr.Textbox(label="排除AA", value="", placeholder="如: CX")

                            # ESM-IF params
                            with gr.Group(visible=False) as t5_pan_esmif:
                                gr.Markdown("**ESM-IF 参数**")
                                with gr.Row():
                                    t5_esmif_temp = gr.Slider(0.05, 1.0, value=0.1, step=0.05, label="温度")
                                    t5_esmif_samples = gr.Slider(1, 10, value=3, step=1, label="序列数")

                            with gr.Row():
                                t5_constraint_mode = gr.Checkbox(
                                    label="启用表位距离约束", value=True,
                                    info="仅设计与表位接触的抗体残基"
                                )
                                t5_constraint_cutoff = gr.Slider(
                                    5, 20, value=10, step=1,
                                    label="距离阈值 (Å)",
                                    info="CA-CA距离阈值，确定面向表位的残基"
                                )
                            t5_run_design_btn = gr.Button("▶ 开始靶点设计", variant="primary", size="lg")

                # ── Step 5: Validation ──
                with gr.Group():
                    gr.Markdown("### ✅ 5. 结果与验证")
                    with gr.Row():
                        t5_design_result = gr.Textbox(
                            label="设计结果", lines=14, interactive=False,
                            elem_classes="result-box", placeholder="设计结果将在此显示..."
                        )
                        t5_validation_result = gr.Textbox(
                            label="验证报告", lines=14, interactive=False,
                            elem_classes="result-box", placeholder="接触分析和界面特性..."
                        )
                    with gr.Group(visible=False) as t5_action_bar:
                        with gr.Row():
                            t5_fasta_download = gr.File(label="💾 FASTA 下载", file_types=[".fasta"])
                            t5_send_af2_btn = gr.Button("🔮 发送到 AF2 Multimer 预测", variant="secondary")
                            t5_send_design_btn = gr.Button("📤 发送序列到单步设计", variant="secondary")
                    t5_result_df = gr.DataFrame(label="📊 设计结果排名", interactive=False)

                # ── Tab 5 internal callbacks ──

                def t5_on_parse_fasta(fasta_text, fasta_file):
                    if fasta_file is not None:
                        try:
                            with open(fasta_file.name, encoding='utf-8') as f:
                                fasta_text = f.read()
                        except Exception:
                            pass
                    if not fasta_text or not fasta_text.strip():
                        return "", "请提供 FASTA 序列"
                    seqs = parse_fasta(fasta_text)
                    if not seqs:
                        return "", "未检测到有效 FASTA"
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
                    return result, "", gr.update(choices=[], value="A"), "A", "预测失败，未生成 PDB"

                def t5_on_epitope_analyze(pdb_path, chain, top_n, methods):
                    if not pdb_path or not os.path.exists(str(pdb_path)):
                        return None, "请先提供靶点结构 (步骤1)", ""
                    try:
                        weights = {
                            'sasa': 0.4 if "SASA" in ' '.join(methods) else 0.0,
                            'hydrophilicity': 0.35 if "亲水性" in ' '.join(methods) else 0.0,
                            'protrusion': 0.25 if "凸出指数" in ' '.join(methods) else 0.0,
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
                        return None, f"分析错误: {traceback.format_exc()}", ""

                def t5_on_select_epitope(region_str):
                    regions = tdh.parse_region_spec(region_str)
                    if not regions:
                        return region_str, gr.update(value="⚠ 区域格式无效")
                    total = sum(len(r) for r in regions.values())
                    return region_str, gr.update(value=f"✓ 已选 {total} 个表位残基")

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
                    return f"转换失败", None

                def t5_tool_switch(tool):
                    return (
                        gr.update(visible=(tool == "BFN (抗体CDR设计)")),
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
                        return "", "无设计序列"
                    seqs = parse_fasta(fasta_str)
                    if not seqs:
                        return "", "无法解析设计序列"
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
                        return fasta_str, "无法提取靶点序列，已返回原始 FASTA"
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
                        return "", None, "", "", "", "", "无设计序列"
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

            # ══════════════════════  Tab 6: Iterative Refinement ══════════════════════
            with gr.TabItem("🔄 迭代精修"):
                gr.Markdown("""
                ### AF2 驱动的迭代精修循环
                设计 → AF2 验证 → 级联筛选 Top-N → 用 AF2 结构作为模板再设计 → 重复至收敛

                ⚠️ AF2 预测较慢（每条序列约 5-15 分钟），建议先用小样本数验证流程。
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        ir_pdb = gr.File(label="上传 PDB 模板", file_types=[".pdb"])
                        ir_pdb_info = gr.Textbox(label="结构信息", lines=4, interactive=False)
                        ir_region = gr.Textbox(label="设计区域", value="A:10-30",
                                               placeholder="A:10-30")
                        with gr.Row():
                            ir_n_samples = gr.Slider(3, 20, value=5, step=1, label="每轮样本数")
                            ir_top_k = gr.Slider(1, 5, value=2, step=1, label="保留 Top-K")
                        ir_max_rounds = gr.Slider(1, 5, value=2, step=1, label="最大轮次")
                        ir_af2_recycle = gr.Slider(0, 3, value=1, step=1, label="AF2 回收步数")
                        ir_btn = gr.Button("🚀 开始迭代精修", variant="primary")
                        ir_btn_stop = gr.Button("✕ 停止", variant="stop", visible=False)

                    with gr.Column(scale=2):
                        ir_log = gr.Textbox(label="运行日志", lines=22, interactive=False,
                                           placeholder="点击'开始迭代精修'运行...")
                        ir_report = gr.Textbox(label="精修报告", lines=8, interactive=False)

                def on_ir_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", ""
                    info, chains, region = on_upload(pdb_file)
                    return info, pdb_file.name

                ir_pdb.upload(on_ir_pdb_upload, [ir_pdb], [ir_pdb_info, _pdb_path])

                def on_iterative_refine(pdb_path, region, n_samples, top_k, max_rounds, af2_recycle,
                                        progress=gr.Progress()):
                    if not pdb_path or not os.path.exists(str(pdb_path)):
                        return "请先上传 PDB 文件", ""

                    log_msgs = []
                    def log(msg):
                        log_msgs.append(msg)
                        return '\n'.join(log_msgs)

                    # Wrapper design function
                    def design_fn(pdb, region_spec, n):
                        from antibodydesignbfn.datasets.protein import preprocess_protein_structure
                        from antibodydesignbfn.utils.train import recursive_to
                        from antibodydesignbfn.utils.misc import seed_all
                        from antibodydesignbfn.utils.data import PaddingCollate
                        from antibodydesignbfn.utils.transforms import get_transform

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
                        batch = recursive_to(PaddingCollate()([transform(structure)]), 'cpu')
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
                        progress_cb=lambda r, mr, s: progress((r-1)/mr, desc=f'第{r}轮: {s}'),
                        log_cb=lambda msg: log_msgs.append(msg),
                    )

                    try:
                        result = refiner.run(pdb_path, region)
                        report = format_refinement_report(result)
                        return '\n'.join(log_msgs), report
                    except Exception as e:
                        import traceback
                        log(f"❌ 错误: {e}")
                        log(traceback.format_exc())
                        return '\n'.join(log_msgs), ""

                ir_btn.click(
                    on_iterative_refine,
                    [_pdb_path, ir_region, ir_n_samples, ir_top_k, ir_max_rounds, ir_af2_recycle],
                    [ir_log, ir_report]
                )

            # ══════════════════════  Tab 7: Confidence Evaluation ══════════════════════
            with gr.TabItem("📊 置信度评估"):
                gr.Markdown("""
                ### BFN 内置置信度评估
                直接使用训练好的 pLDDT/ipTM/PAE 头评估任意蛋白结构的置信度，无需进行序列设计。
                使用 V6 Phase 2 模型（3 层 pLDDT + context-only ipTM）。
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        conf_pdb = gr.File(label="上传 PDB 文件", file_types=[".pdb"])
                        _conf_pdb_path = gr.State("")
                        conf_pdb_info = gr.Textbox(label="结构信息", lines=3, interactive=False)
                        conf_region = gr.Textbox(label="评估区域", value="A:10-25",
                                                 placeholder="A:10-25 或 A:1-50")
                        conf_btn = gr.Button("🔍 评估置信度", variant="primary")
                    with gr.Column(scale=2):
                        conf_report = gr.Textbox(label="评估报告", lines=14, interactive=False,
                                                 elem_classes="result-box")
                        conf_residue_df = gr.DataFrame(label="逐残基 pLDDT", interactive=False)

                def on_conf_pdb_upload(pdb_file):
                    if pdb_file is None:
                        return "", ""
                    info, chains, region = on_upload(pdb_file)
                    return info, pdb_file.name

                conf_pdb.upload(on_conf_pdb_upload, [conf_pdb], [conf_pdb_info, _conf_pdb_path])

                conf_btn.click(run_bfn_confidence_evaluation,
                              [_conf_pdb_path, conf_region],
                              [conf_report, conf_residue_df])

            # ══════════════════════  Tab 8: Unified Pipeline ══════════════════════
            with gr.TabItem("🚀 统一工作流"):
                gr.Markdown("""
                ### 一键设计 → 过滤 → 验证流水线
                将 BFN 设计、级联过滤和可选 AF2 验证整合为一个工作流，自动完成排名和筛选。
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        pipe_pdb = gr.File(label="上传 PDB 模板", file_types=[".pdb"])
                        _pipe_pdb_path = gr.State("")
                        pipe_pdb_info = gr.Textbox(label="结构信息", lines=3, interactive=False)
                        pipe_region = gr.Textbox(label="设计区域", value="A:10-25")
                        with gr.Row():
                            pipe_samples = gr.Slider(3, 20, value=8, step=1, label="设计样本数")
                            pipe_stochastic = gr.Checkbox(label="随机采样", value=True)
                        with gr.Row():
                            pipe_enable_af2 = gr.Checkbox(label="启用 AF2 验证 (慢)", value=False)
                            pipe_af2_recycle = gr.Slider(0, 3, value=1, step=1, label="AF2 回收步数")
                        pipe_btn = gr.Button("🚀 运行统一工作流", variant="primary")
                    with gr.Column(scale=2):
                        pipe_report = gr.Textbox(label="流水线报告", lines=18, interactive=False,
                                                elem_classes="result-box")
                        pipe_df = gr.DataFrame(label="最终排名结果", interactive=False)
                        pipe_fasta = gr.File(label="FASTA 下载", file_types=[".fasta"])

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

        # ── Footer ──
        gr.HTML("""
        <div class="platform-footer">
            <strong>Protein Design Platform</strong> v3.0 &nbsp;·&nbsp;
            BFN · ProteinMPNN · ESM-IF · AlphaFold2 &nbsp;·&nbsp;
            管理: <code>python manage.py start|stop|restart|status|batch|config|test</code>
        </div>
        """)

        # Register send-to-AF2 callback (must be after af_fasta_text is defined)
        send_to_af2_btn.click(
            on_send_to_af2, [fasta_download],
            [af_fasta_text, af_fasta_info]
        )

        # Register FASTA→PDB conversion and clear callbacks
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
    print(f"  外部访问: {'允许' if sv['host'] != '127.0.0.1' else '禁止（仅本机）'}")
    print("=" * 60)

    # Preload BFN
    try:
        print("  预加载 BFN 模型...")
        load_bfn()
        print("  BFN ✓")
    except Exception as e:
        print(f"  BFN 跳过: {e}")

    print(f"\n  Web 界面: http://{sv['host']}:{sv['port']}")
    print(f"  管理命令: python manage.py --help")
    print()

    app, css = create_ui()
    app.launch(
        server_name=sv['host'],
        server_port=sv['port'],
        share=sv.get('share', False),
        inbrowser=sv.get('auto_open', True),
        css=css,
        show_error=True,
    )
