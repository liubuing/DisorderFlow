#!/usr/bin/env python
"""
Integration test comparing BFN, ProteinMPNN, and ESM-IF on the same design target.
"""
import sys, os, json, subprocess, tempfile
import torch
import numpy as np

AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'

def run_bfn(pdb_path, chain, region_start, region_end, config_path, num_samples=3):
    """Run BFN design and return results."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("design_protein", "design_protein.py")
    design_protein_mod = importlib.util.module_from_spec(spec)
    # We can't easily call the module directly, so use subprocess
    cmd = (
        f'"{sys.executable}" design_protein.py "{pdb_path}" '
        f'--design "{chain}:{region_start}-{region_end}" '
        f'--config "{config_path}" --device cpu '
        f'--num_samples {num_samples} --stochastic --eval '
        f'--output integration_bfn_result.json'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[BFN ERROR] {result.stderr}")
        return None
    with open('integration_bfn_result.json') as f:
        return json.load(f)


def run_proteinmpnn(pdb_path, chain, num_samples=3, temp="0.1", seed=42):
    """Run ProteinMPNN and return results."""
    out_folder = 'integration_mpnn'
    cmd = (
        f'"{sys.executable}" ProteinMPNN/protein_mpnn_run.py '
        f'--pdb_path "{pdb_path}" --pdb_path_chains "{chain}" '
        f'--num_seq_per_target {num_samples} --sampling_temp "{temp}" '
        f'--seed {seed} --out_folder {out_folder} --save_score 1 '
        f'--path_to_model_weights ProteinMPNN/vanilla_model_weights'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[ProteinMPNN ERROR] {result.stderr}")
        return None

    # Parse FASTA output
    pdb_id = os.path.basename(pdb_path).replace('.pdb', '')
    fasta_path = os.path.join(out_folder, 'seqs', f'{pdb_id}.fa')
    if not os.path.exists(fasta_path):
        print(f"[ProteinMPNN] FASTA not found at {fasta_path}")
        return None

    sequences = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>T='):
                seq = next(f).strip()
                sequences.append(seq)
    return {'sequences': sequences, 'pdb': pdb_id}


def run_esmif(pdb_path, chain, region_start, region_end, num_samples=3, temp=0.1):
    """Run ESM-IF and return results for the specified region."""
    from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
    from esm.inverse_folding import util as if_util

    device = 'cpu'
    model, _ = esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device)
    model.eval()

    coords, native_seq = if_util.load_coords(pdb_path, chain)

    results = []
    for i in range(num_samples):
        sampled = model.sample(coords, temperature=temp)
        region_seq = sampled[region_start:region_end+1]
        results.append(region_seq)

    return {
        'native_region': native_seq[region_start:region_end+1],
        'samples': results,
        'full_native': native_seq,
    }


def print_comparison(bfn_result, mpnn_result, esmif_result, region_start, region_end):
    """Print a comparison table of all three tools."""
    n_res = region_end - region_start + 1

    print()
    print("=" * 80)
    print("  INTEGRATION TEST: BFN vs ProteinMPNN vs ESM-IF")
    print("=" * 80)

    # Native sequence
    native_bfn = bfn_result.get('native', 'N/A') if bfn_result else 'N/A'
    native_esm = esmif_result['native_region'] if esmif_result else 'N/A'

    print(f"\n  Target region: positions {region_start}-{region_end} ({n_res} residues)")
    print(f"  BFN native:    {native_bfn}")
    print(f"  ESM-IF native: {native_esm}")

    print(f"\n  {'Tool':<20} {'Sample 1':<20} {'Sample 2':<20} {'Sample 3':<20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

    if bfn_result:
        seqs = [r['sequence'] for r in bfn_result['results'][:3]]
        while len(seqs) < 3:
            seqs.append('N/A')
        print(f"  {'BFN':<20} {seqs[0]:<20} {seqs[1]:<20} {seqs[2]:<20}")
        print(f"  {'  PPL':<20} {bfn_result['results'][0]['perplexity']:<20.2f} "
              f"{bfn_result['results'][1]['perplexity'] if len(bfn_result['results'])>1 else '':<20} "
              f"{bfn_result['results'][2]['perplexity'] if len(bfn_result['results'])>2 else '':<20}")
        if 'recovery' in bfn_result['results'][0]:
            recs = [f"{r['recovery']*100:.0f}%" for r in bfn_result['results'][:3]]
            while len(recs) < 3:
                recs.append('')
            print(f"  {'  Recovery':<20} {recs[0]:<20} {recs[1]:<20} {recs[2]:<20}")

    if mpnn_result:
        seqs = mpnn_result['sequences'][:3]
        # Extract design region
        region_seqs = []
        for s in seqs:
            if len(s) >= region_end:
                region_seqs.append(s[region_start:region_end+1])
            else:
                region_seqs.append(s[region_start:])
        while len(region_seqs) < 3:
            region_seqs.append('N/A')
        print(f"  {'ProteinMPNN':<20} {region_seqs[0]:<20} {region_seqs[1]:<20} {region_seqs[2]:<20}")

    if esmif_result:
        seqs = esmif_result['samples'][:3]
        while len(seqs) < 3:
            seqs.append('N/A')
        print(f"  {'ESM-IF':<20} {seqs[0]:<20} {seqs[1]:<20} {seqs[2]:<20}")

    print()
    print("=" * 80)
    print("  All three tools completed successfully.")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Integration test: BFN vs ProteinMPNN vs ESM-IF')
    parser.add_argument('pdb_path', type=str, help='Input PDB file')
    parser.add_argument('--chain', '-c', type=str, default='A', help='Target chain')
    parser.add_argument('--region', '-r', type=str, default='30:40', help='Design region (start:end)')
    parser.add_argument('--config', type=str, default='configs/demo_design.yml', help='BFN config')
    parser.add_argument('--num_samples', '-n', type=int, default=3, help='Number of samples')
    parser.add_argument('--skip_bfn', action='store_true', help='Skip BFN')
    parser.add_argument('--skip_mpnn', action='store_true', help='Skip ProteinMPNN')
    parser.add_argument('--skip_esmif', action='store_true', help='Skip ESM-IF')

    args = parser.parse_args()

    region_start, region_end = map(int, args.region.split(':'))
    print(f"[INFO] Target: {args.pdb_path}, chain {args.chain}, positions {region_start}-{region_end}")

    bfn_result = None
    mpnn_result = None
    esmif_result = None

    if not args.skip_bfn:
        print("\n--- BFN Design ---")
        bfn_result = run_bfn(args.pdb_path, args.chain, region_start, region_end,
                            args.config, args.num_samples)

    if not args.skip_mpnn:
        print("\n--- ProteinMPNN Design ---")
        mpnn_result = run_proteinmpnn(args.pdb_path, args.chain, args.num_samples)

    if not args.skip_esmif:
        print("\n--- ESM-IF Design ---")
        esmif_result = run_esmif(args.pdb_path, args.chain, region_start, region_end, args.num_samples)

    print_comparison(bfn_result, mpnn_result, esmif_result, region_start, region_end)
