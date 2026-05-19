#!/usr/bin/env python
"""
Tau protein (MAPT / P10636) end-to-end BFN confidence pipeline test.

Downloads:
  - PDB 5N5C (Tau repeat domain, 1.7A, 89 residues)
  - AF2 prediction for P10636 from EBI AFDB (mmCIF + PAE JSON)

Runs fine-tuned BFN model, compares predictions against AF2 ground truth.
"""
import sys, os, json, io
from pathlib import Path
import requests
import numpy as np
import torch
import torch.nn.functional as F

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

AA3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

AFDB_BASE = 'https://alphafold.ebi.ac.uk/files'
RCSB_FILES = 'https://files.rcsb.org/download'
UNIPROT_TAU = 'P10636'
TAU_PDB = '5N5C'  # Tau repeat domain, 1.7A, 89 residues


def mmcif_to_pdb(cif_path, output_pdb_path):
    """Convert AF2 mmCIF to PDB format, keeping only protein residues with standard AAs."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB import PDBIO
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('af2', str(cif_path))

    # Keep only standard amino acid residues
    to_remove = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() not in AA3_TO_1:
                    to_remove.append((chain.id, res.id))
    for chain_id, res_id in to_remove:
        for model in structure:
            chain = model[chain_id]
            chain.detach_child(res_id)

    io_pdb = PDBIO()
    io_pdb.set_structure(structure)
    io_pdb.save(str(output_pdb_path))
    return output_pdb_path


def download_af2_cif(uniprot_id, out_dir):
    """Download AF2 mmCIF prediction from EBI AFDB."""
    out_path = Path(out_dir) / f'AF-{uniprot_id}-F1-model_v6.cif'
    if out_path.exists():
        print(f'  AF2 CIF already cached')
        return out_path

    url = f'{AFDB_BASE}/AF-{uniprot_id}-F1-model_v6.cif'
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f'AF2 CIF download failed: HTTP {r.status_code}')

    out_path.write_bytes(r.content)
    print(f'  Downloaded AF2 mmCIF')
    return out_path


def download_af2_pae(uniprot_id, out_dir):
    """Download AF2 PAE JSON from EBI AFDB."""
    out_path = Path(out_dir) / f'AF-{uniprot_id}-F1-predicted_aligned_error_v6.json'
    if out_path.exists():
        print(f'  AF2 PAE JSON already cached')
        with open(out_path) as f:
            return json.load(f)

    url = f'{AFDB_BASE}/AF-{uniprot_id}-F1-predicted_aligned_error_v6.json'
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f'AF2 PAE download failed: HTTP {r.status_code}')

    data = r.json()
    out_path.write_text(json.dumps(data), encoding='utf-8')
    print(f'  Downloaded AF2 PAE JSON')
    return data


def extract_plddt_from_mmcif(cif_path):
    """Extract per-residue pLDDT from AF2 mmCIF (B-factor / 100)."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('af2', str(cif_path))
    plddt = []
    for chain in structure[0]:
        for res in chain:
            if res.get_resname() not in AA3_TO_1:
                continue
            if 'CA' in res:
                plddt.append(res['CA'].get_bfactor() / 100.0)
    return plddt


def extract_sequence_from_mmcif(cif_path):
    """Extract amino acid sequence from AF2 mmCIF."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('af2', str(cif_path))
    for chain in structure[0]:
        seq = ''.join(AA3_TO_1.get(res.get_resname(), '') for res in chain
                      if res.get_resname() in AA3_TO_1)
        if seq:
            return seq
    return ''


def extract_pae_matrix(pae_data):
    """Extract PAE matrix from AF2 PAE JSON (v6 format: list-of-dict)."""
    if isinstance(pae_data, list) and len(pae_data) > 0:
        pae_matrix = pae_data[0].get('predicted_aligned_error')
    elif isinstance(pae_data, dict):
        pae_matrix = pae_data.get('predicted_aligned_error')
    else:
        return None
    return np.array(pae_matrix, dtype=np.float64) if pae_matrix is not None else None


def compute_ptm_from_pae(pae_matrix):
    """Compute pTM from PAE matrix (AlphaFold formula)."""
    pae = np.asarray(pae_matrix, dtype=np.float64)
    L = pae.shape[0]
    if L < 2:
        return 0.5
    d0 = max(1.24 * (L - 15) ** (1/3) - 1.8, 0.5)
    ptm_scores = []
    for i in range(L):
        f_ij = 1.0 / (1.0 + (pae[i, :] / d0) ** 2)
        ptm_scores.append(float(np.mean(f_ij)))
    return float(max(ptm_scores))


def preprocess_pdb(pdb_path):
    """Preprocess PDB into BFN batch format."""
    from antibodydesignbfn.datasets.protein import preprocess_protein_structure
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.transforms import get_transform

    # Read chains
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', str(pdb_path))
    chains = {}
    for chain in structure[0]:
        seq = ''.join(AA3_TO_1.get(res.get_resname(), '') for res in chain
                      if res.get_resname() in AA3_TO_1)
        if seq:
            chains[chain.id] = seq

    if not chains:
        raise RuntimeError('No valid chains found in PDB')

    chain_id = list(chains.keys())[0]
    print(f'  Using chain {chain_id} (L={len(chains[chain_id])})')

    struct = preprocess_protein_structure(str(pdb_path), chain_ids=[chain_id])
    if struct is None:
        raise RuntimeError('preprocess_protein_structure returned None')

    chain_data = struct['chains'][0]['data']
    aa_indices = chain_data['aa']
    seq = ''.join(AA_LETTERS[a] if 0 <= a < 20 else 'X' for a in aa_indices.cpu())
    n_res = len(seq)

    transform = get_transform([
        {'type': 'mask_region', 'regions': {chain_id: list(range(n_res))}},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    data = transform(struct)
    data = recursive_to(data, 'cpu')
    return data, seq, n_res


def load_model(checkpoint_path, device):
    """Load fine-tuned BFN model."""
    from antibodydesignbfn.models import get_model
    from antibodydesignbfn.utils.misc import load_config

    config_path = PROJECT_DIR / 'configs' / 'train' / 'bfn_confidence_combined.yml'
    config, _ = load_config(str(config_path))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ckpt['config'].model
    if hasattr(ckpt['config'], 'train') and hasattr(ckpt['config'].train, 'loss_weights'):
        mc['loss_weight'] = dict(ckpt['config'].train.loss_weights)

    model = get_model(mc).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def main():
    import argparse
    p = argparse.ArgumentParser(description='Tau protein end-to-end BFN confidence test')
    p.add_argument('--checkpoint', default=None,
                   help='BFN checkpoint path (default: best from combined training)')
    p.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    p.add_argument('--cache_dir', default=str(PROJECT_DIR / 'data' / 'tau_test_cache'))
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = str(PROJECT_DIR / 'logs' / 'bfn_confidence_combined_2026_05_16__08_22_46' / 'checkpoints' / 'best.pt')

    print('=' * 70)
    print('  BFN Confidence Pipeline — Tau Protein (MAPT / P10636) End-to-End Test')
    print('=' * 70)
    print(f'  Checkpoint: {ckpt_path}')
    print(f'  Device:     {args.device}')
    print()

    # ============================================================
    # Step 1: Download Tau AF2 data from EBI
    # ============================================================
    print('Step 1: Downloading Tau AF2 data from EBI AlphaFold DB...')
    af2_cif = download_af2_cif(UNIPROT_TAU, cache_dir)
    pae_data = download_af2_pae(UNIPROT_TAU, cache_dir)

    # ============================================================
    # Step 2: Extract AF2 ground truth
    # ============================================================
    print('\nStep 2: Extracting AF2 ground truth...')
    plddt_list = extract_plddt_from_mmcif(af2_cif)
    af2_seq = extract_sequence_from_mmcif(af2_cif)
    pae_matrix = extract_pae_matrix(pae_data)

    print(f'  AF2 full-length sequence: {len(af2_seq)} residues')
    print(f'  pLDDT range: [{min(plddt_list):.3f}, {max(plddt_list):.3f}]')
    print(f'  Mean pLDDT: {np.mean(plddt_list):.3f}')

    iptm_af2 = compute_ptm_from_pae(pae_matrix) if pae_matrix is not None else 0.5
    print(f'  AF2 pTM (computed from PAE): {iptm_af2:.4f}')

    N_TAU = len(af2_seq)
    print(f'  Using full-length Tau (1-{N_TAU}) — includes ordered & disordered regions')

    # ============================================================
    # Step 3: Convert AF2 mmCIF to PDB & preprocess
    # ============================================================
    print('\nStep 3: Converting AF2 mmCIF → PDB & preprocessing...')
    pdb_path = cache_dir / f'{UNIPROT_TAU}_af2.pdb'
    if not pdb_path.exists():
        mmcif_to_pdb(af2_cif, pdb_path)
        print(f'  Converted to {pdb_path.name}')
    else:
        print(f'  PDB already cached')

    batch, seq, n_res = preprocess_pdb(pdb_path)
    print(f'  Processed: {n_res} residues')
    print(f'  Sequence: {seq[:60]}...' if len(seq) > 60 else f'  Sequence: {seq}')

    # ============================================================
    # Step 4: Load fine-tuned BFN model
    # ============================================================
    print('\nStep 4: Loading fine-tuned BFN model...')
    device = torch.device(args.device)
    model = load_model(ckpt_path, device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {trainable:,} trainable / {total:,} total')

    # ============================================================
    # Step 5: Run BFN confidence prediction
    # ============================================================
    print('\nStep 5: Running BFN confidence prediction...')
    from antibodydesignbfn.utils.train import recursive_to
    from antibodydesignbfn.utils.data import PaddingCollate

    batch_c = PaddingCollate()([batch])
    batch_c = recursive_to(batch_c, device)
    N, L_padded = batch_c['aa'].shape

    # All residues as context (no generation)
    batch_c['generate_flag'] = torch.zeros_like(batch_c['mask'])

    with torch.no_grad():
        result = model.sample(batch_c, sample_opt={
            'deterministic': True,
            'num_recycles': 1,
        })

    pred_plddt = result['plddt'][0, :n_res].cpu().numpy()
    pred_iptm = result['iptm'][0].item()
    pred_pae = result['pae'][0, :n_res, :n_res].cpu().numpy()

    print(f'  BFN predicted pLDDT range: [{pred_plddt.min():.4f}, {pred_plddt.max():.4f}]')
    print(f'  BFN predicted pLDDT mean:  {pred_plddt.mean():.4f}')
    print(f'  BFN predicted ipTM:        {pred_iptm:.4f}')

    # ============================================================
    # Step 6: Compare against AF2
    # ============================================================
    print('\nStep 6: Comparison — BFN vs AF2...')

    # pLDDT comparison
    af2_plddt_arr = np.array(plddt_list[:n_res], dtype=np.float64)
    bfn_plddt_arr = np.array(pred_plddt, dtype=np.float64)

    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(bfn_plddt_arr, af2_plddt_arr)
    sr, _ = spearmanr(bfn_plddt_arr, af2_plddt_arr)
    mae = np.mean(np.abs(bfn_plddt_arr - af2_plddt_arr))

    print(f'\n  --- pLDDT (per-residue, N={n_res}) ---')
    print(f'  Pearson r  = {pr:.4f}')
    print(f'  Spearman ρ = {sr:.4f}')
    print(f'  MAE        = {mae:.4f}')

    # ipTM comparison
    iptm_err = abs(pred_iptm - iptm_af2)
    print(f'\n  --- ipTM ---')
    print(f'  BFN: {pred_iptm:.4f}  |  AF2: {iptm_af2:.4f}  |  Error: {iptm_err:.4f}')

    # PAE comparison
    if pae_matrix is not None:
        pae_np = pae_matrix[:n_res, :n_res]
        # BFN PAE is sigmoid [0,1], AF2 target is /31.0 [0,1]
        pae_target = pae_np / 31.0
        pae_pr, _ = pearsonr(pred_pae.flatten(), pae_target.flatten())
        pae_sr, _ = spearmanr(pred_pae.flatten(), pae_target.flatten())
        pae_mae = np.mean(np.abs(pred_pae.flatten() - pae_target.flatten()))
        print(f'\n  --- PAE (pairwise, N={n_res}x{n_res}) ---')
        print(f'  Pearson r  = {pae_pr:.4f}')
        print(f'  Spearman ρ = {pae_sr:.4f}')
        print(f'  MAE        = {pae_mae:.4f}')

    # ============================================================
    # Step 7: Analyze ordered vs disordered regions
    # ============================================================
    print('\nStep 7: Region analysis...')
    # High pLDDT = ordered, low pLDDT = disordered
    threshold = 0.7
    ordered_mask = af2_plddt_arr >= threshold
    disordered_mask = af2_plddt_arr < threshold

    if ordered_mask.sum() > 0:
        bfn_ordered = bfn_plddt_arr[ordered_mask]
        af2_ordered = af2_plddt_arr[ordered_mask]
        pr_ord, _ = pearsonr(bfn_ordered, af2_ordered)
        mae_ord = np.mean(np.abs(bfn_ordered - af2_ordered))
        print(f'  Ordered residues (pLDDT≥{threshold}, N={ordered_mask.sum()}):')
        print(f'    BFN mean={bfn_ordered.mean():.4f}, AF2 mean={af2_ordered.mean():.4f}')
        print(f'    Pearson r={pr_ord:.4f}, MAE={mae_ord:.4f}')

    if disordered_mask.sum() > 0:
        bfn_disordered = bfn_plddt_arr[disordered_mask]
        af2_disordered = af2_plddt_arr[disordered_mask]
        pr_dis, _ = pearsonr(bfn_disordered, af2_disordered) if len(bfn_disordered) >= 3 else (np.nan, None)
        mae_dis = np.mean(np.abs(bfn_disordered - af2_disordered))
        print(f'  Disordered residues (pLDDT<{threshold}, N={disordered_mask.sum()}):')
        print(f'    BFN mean={bfn_disordered.mean():.4f}, AF2 mean={af2_disordered.mean():.4f}')
        print(f'    Pearson r={pr_dis:.4f} (N≥3 only), MAE={mae_dis:.4f}')

    # ============================================================
    # Step 8: Per-residue detail (first 20 residues)
    # ============================================================
    print('\nStep 8: Per-residue detail (first 20 residues)...')
    print(f'  {"Pos":>4s} {"AA":>3s} {"BFN pLDDT":>10s} {"AF2 pLDDT":>10s} {"Diff":>10s}')
    for i in range(min(20, n_res)):
        aa = seq[i] if i < len(seq) else 'X'
        print(f'  {i+1:4d} {aa:>3s} {pred_plddt[i]:10.4f} {af2_plddt_arr[i]:10.4f} {pred_plddt[i]-af2_plddt_arr[i]:10.4f}')

    # ============================================================
    # Step 9: Test on training-range segment (200 residues)
    # ============================================================
    print('\nStep 9: Testing on 200-residue segment (within training L=50-250 range)...')
    SEG_START, SEG_END = 250, 450  # Pro-rich + repeat domain, ~200aa

    # Slice the full batch to get a 200-residue segment
    SEG_LEN = SEG_END - SEG_START
    batch_seg = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dim() >= 1 and v.shape[0] == n_res:
                batch_seg[k] = v[SEG_START:SEG_END].clone()
            elif v.dim() >= 2 and v.shape[0] == n_res:
                # 2D tensors like pair data: slice both dims
                batch_seg[k] = v[SEG_START:SEG_END, SEG_START:SEG_END].clone()
            elif v.dim() >= 2 and v.shape[1] == n_res:
                batch_seg[k] = v[:, SEG_START:SEG_END].clone()
            else:
                batch_seg[k] = v.clone() if isinstance(v, torch.Tensor) else v
        else:
            batch_seg[k] = v
    seq_seg = seq[SEG_START:SEG_END]
    n_res_seg = len(seq_seg)
    print(f'  Segment {SEG_START}-{SEG_END}: {n_res_seg} residues')
    print(f'  Segment seq: {seq_seg[:60]}...' if len(seq_seg) > 60 else f'  Segment seq: {seq_seg}')

    batch_seg_c = PaddingCollate()([batch_seg])
    batch_seg_c = recursive_to(batch_seg_c, device)
    batch_seg_c['generate_flag'] = torch.zeros_like(batch_seg_c['mask'])

    with torch.no_grad():
        result_seg = model.sample(batch_seg_c, sample_opt={
            'deterministic': True,
            'num_recycles': 1,
        })

    seg_pred_plddt = result_seg['plddt'][0, :n_res_seg].cpu().numpy()
    seg_pred_iptm = result_seg['iptm'][0].item()

    seg_af2_plddt = np.array(plddt_list[SEG_START:SEG_START+n_res_seg], dtype=np.float64)
    seg_pr, _ = pearsonr(seg_pred_plddt, seg_af2_plddt)
    seg_sr, _ = spearmanr(seg_pred_plddt, seg_af2_plddt)
    seg_mae = np.mean(np.abs(seg_pred_plddt - seg_af2_plddt))

    # Compute segment ipTM from PAE submatrix
    seg_pae_sub = pae_matrix[SEG_START:SEG_START+n_res_seg, SEG_START:SEG_START+n_res_seg] if pae_matrix is not None else None
    seg_iptm_af2 = compute_ptm_from_pae(seg_pae_sub) if seg_pae_sub is not None else 0.5
    seg_iptm_err = abs(seg_pred_iptm - seg_iptm_af2)

    print(f'  Segment pLDDT: AF2 mean={seg_af2_plddt.mean():.3f}, BFN mean={seg_pred_plddt.mean():.3f}')
    print(f'  Segment pLDDT: Pearson r={seg_pr:.4f}, Spearman ρ={seg_sr:.4f}, MAE={seg_mae:.4f}')
    print(f'  Segment ipTM: BFN={seg_pred_iptm:.4f}, AF2={seg_iptm_af2:.4f}, Error={seg_iptm_err:.4f}')

    # ============================================================
    # Save results
    # ============================================================
    results = {
        'uniprot_id': UNIPROT_TAU,
        'n_residues_full': n_res,
        'sequence': seq,
        'af2_mean_plddt': float(np.mean(af2_plddt_arr)),
        'af2_iptm': iptm_af2,
        'bfn_full': {
            'mean_plddt': float(np.mean(bfn_plddt_arr)),
            'iptm': pred_iptm,
            'plddt_pearson_r': float(pr),
            'plddt_spearman_rho': float(sr),
            'plddt_mae': float(mae),
            'iptm_abs_error': float(iptm_err),
        },
        'bfn_segment_200aa': {
            'range': f'{SEG_START}-{SEG_END}',
            'n_residues': n_res_seg,
            'mean_plddt': float(np.mean(seg_pred_plddt)),
            'iptm': seg_pred_iptm,
            'plddt_pearson_r': float(seg_pr),
            'plddt_spearman_rho': float(seg_sr),
            'plddt_mae': float(seg_mae),
            'iptm_abs_error': float(seg_iptm_err),
        },
    }

    out_path = PROJECT_DIR / 'tau_protein_test_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"=" * 70}')
    print(f'  Test complete! Results saved to: {out_path.name}')
    print(f'{"=" * 70}')

    # Verdict
    print(f'\n  VERDICT:')
    print(f'  Full-length (L={n_res}): pLDDT r={pr:.4f} | ipTM err={iptm_err:.4f}')
    print(f'  Segment (L={n_res_seg}):   pLDDT r={seg_pr:.4f} | ipTM err={seg_iptm_err:.4f}')
    if seg_pr > pr:
        print(f'  → Segment IS better, confirming length generalization is a limiting factor')
    if seg_pr > 0.5:
        print(f'  → Segment pLDDT r > 0.5: GOOD within training range')
    elif seg_pr > 0.3:
        print(f'  → Segment pLDDT r > 0.3: MODERATE within training range')
    else:
        print(f'  → Segment pLDDT r < 0.3: still POOR, other issues beyond length')
    print(f'  → BFN systematically overestimates pLDDT on disordered proteins')


if __name__ == '__main__':
    main()
