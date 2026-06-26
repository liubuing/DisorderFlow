#!/usr/bin/env python3
"""§3.3C Fallback: ProteinMPNN (generator) + BFN (scorer) hybrid pipeline.

When BFN cannot learn antigen-specific CDR design, ProteinMPNN provides
the CDR generation (antigen-aware by design) while BFN supplies unique
scoring capabilities: disorder head, confidence heads, PPL/entropy.

Usage:
    python mpnn_bfn_hybrid.py \
        --scaffold data/.../5IMK.pdb --scaffold-chain B \
        --antigen data/.../abeta42.pdb --antigen-chain A \
        --cdr B:26-33,B:51-58,B:97-113 \
        --samples 100 --top 10 --af2

Output: Top-10 designs ranked by BFN composite score, optionally
        validated by AF2.
"""
import sys, os, json, tempfile, subprocess, argparse, time, shutil
import numpy as np, torch, yaml
AA = 'ACDEFGHIKLMNPQRSTVWY'


def build_complex_pdb(scaffold_pdb, scaffold_chain, antigen_pdb, antigen_chain):
    """Merge scaffold + antigen into a single PDB for ProteinMPNN."""
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure('scaffold', scaffold_pdb)
    s2 = parser.get_structure('antigen', antigen_pdb)
    merged = Structure.Structure('complex')
    model = Model.Model(0)
    for struct, chain_id in [(s1, scaffold_chain), (s2, antigen_chain)]:
        for c in struct.get_chains():
            if c.id == chain_id:
                c.id = chain_id
                c.detach_parent()
                model.add(c)
    merged.add(model)
    out = tempfile.mktemp(suffix='.pdb')
    io = PDBIO(); io.set_structure(merged); io.save(out)
    return out


def run_proteinmpnn(complex_pdb, chains, num_samples, temperature, seed, omit_aas=''):
    """Run ProteinMPNN, return list of {sequence, score, recovery}."""
    out_dir = tempfile.mkdtemp(prefix='mpnn_')
    cmd = [
        sys.executable, 'ProteinMPNN/protein_mpnn_run.py',
        '--pdb_path', complex_pdb, '--pdb_path_chains', chains,
        '--num_seq_per_target', str(num_samples),
        '--sampling_temp', str(temperature),
        '--seed', str(seed),
        '--out_folder', out_dir, '--save_score', '1',
        '--path_to_model_weights', 'ProteinMPNN/vanilla_model_weights',
    ]
    if omit_aas:
        cmd.extend(['--omit_AAs', omit_aas])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'ProteinMPNN failed:\n{r.stderr[:2000]}')
    fa = os.path.join(out_dir, 'seqs',
                      os.path.basename(complex_pdb).replace('.pdb', '.fa'))
    if not os.path.exists(fa):
        raise FileNotFoundError(f'MPNN output not found: {fa}')
    results = []
    with open(fa) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>T='):
                parts = {p.split('=')[0].strip(): p.split('=')[1].strip()
                         for p in line.split(',') if '=' in p}
                seq = next(f).strip()
                results.append({
                    'sequence': seq,
                    'mpnn_score': float(parts.get('score', 0)),
                    'mpnn_recovery': float(parts.get('seq_recovery', 0)),
                    'sample': int(parts.get('sample', 0)),
                })
    shutil.rmtree(out_dir, ignore_errors=True)
    return results


def score_with_bfn(designs, complex_pdb, region_spec):
    """Score a list of CDR designs using BFN confidence heads.

    Returns designs list with added keys:
        ppl, entropy, plddt_mean, iptm, disorder_mean, composite
    """
    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, cfg = bfn_loader.load_bfn('cpu')

    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.misc import seed_all

    # Parse region spec into CDR mask
    def _parse_regions(region_spec):
        pairs = []
        for part in region_spec.split(','):
            chain, rng = part.split(':')
            s, e = rng.split('-')
            pairs.append((chain, int(s) - 1, int(e)))
        return pairs

    regions = _parse_regions(region_spec)

    # Preprocess structure once
    batch = preprocess_protein_structure(complex_pdb, chain_ids=None)
    L = batch['aa'].shape[0]

    # Build fixed generate_flag mask for this scaffold
    gen_flag = torch.zeros(L, dtype=torch.bool)
    for chain, start, end in regions:
        # chain_nb mapping depends on the preprocessor
        gen_flag[start:end] = True
    batch['generate_flag'] = gen_flag.unsqueeze(0)

    # For each design, graft CDR into scaffold sequence, then run BFN inference
    transform = get_transform([{'type': 'merge_chains'}, {'type': 'patch_around_anchor'}])

    scored = []
    for d in designs:
        # Graft designed CDR into scaffold
        aa = batch['aa'].clone()
        cdr_seq = d['sequence']
        pos = 0
        for _, start, end in regions:
            length = end - start
            for j in range(length):
                if pos < len(cdr_seq):
                    aa[start + j] = AA.index(cdr_seq[pos])
                    pos += 1

        # Create batch entry
        entry = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        entry['aa'] = aa
        entry = transform(entry)
        entry = {k: v.unsqueeze(0) for k, v in entry.items()
                 if isinstance(v, torch.Tensor)}

        # Run BFN confidence evaluation
        with torch.no_grad():
            try:
                pred = model(entry)
            except Exception:
                scored.append({**d, 'ppl': 999, 'entropy': 3.0,
                               'plddt_mean': 0, 'iptm': 0,
                               'disorder_mean': 0, 'composite': 0})
                continue

        # Extract metrics
        pred_seq = pred.get('pred_seq_logits', pred.get('pred_seq',
                              torch.zeros(1, L, 20)))
        probs = torch.softmax(pred_seq, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)
        max_prob = probs.max(dim=-1).values

        cdr_mask = gen_flag.unsqueeze(0)
        cdr_entropy = entropy[cdr_mask].mean().item() if cdr_mask.any() else 3.0
        cdr_max_prob = max_prob[cdr_mask].mean().item() if cdr_mask.any() else 0.05

        # PPL approximation
        ppl = torch.exp(-log_probs.max(dim=-1).values[cdr_mask].mean()).item()

        plddt = pred.get('pred_plddt', torch.zeros(1, L))
        iptm = pred.get('pred_iptm', torch.zeros(1))
        disorder = pred.get('pred_disorder', torch.zeros(1, L))

        plddt_mean = plddt[cdr_mask].mean().item() if cdr_mask.any() else 0
        iptm_val = iptm.mean().item()
        disorder_mean = disorder[cdr_mask].mean().item() if cdr_mask.any() else 0

        # Composite score (same weights as cascade_filter)
        composite = (0.35 * iptm_val + 0.25 * plddt_mean +
                     0.15 * (1.0 / max(ppl, 1e-6)) +
                     0.10 * (1.0 - cdr_entropy / 3.0) +
                     0.15 * (1.0 - disorder_mean))

        scored.append({**d, 'ppl': ppl, 'entropy': cdr_entropy,
                       'plddt_mean': plddt_mean, 'iptm': iptm_val,
                       'disorder_mean': disorder_mean, 'composite': composite})

    # Sort by composite score descending
    scored.sort(key=lambda x: x['composite'], reverse=True)
    return scored


def main():
    parser = argparse.ArgumentParser(description='ProteinMPNN + BFN hybrid CDR designer')
    parser.add_argument('--scaffold', required=True, help='Scaffold PDB path')
    parser.add_argument('--scaffold-chain', default='B', help='Scaffold chain ID')
    parser.add_argument('--antigen', required=True, help='Antigen PDB path')
    parser.add_argument('--antigen-chain', default='A', help='Antigen chain ID')
    parser.add_argument('--cdr', required=True,
                        help='CDR regions, e.g. "B:26-33,B:51-58,B:97-113"')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of ProteinMPNN designs (default 100)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='ProteinMPNN sampling temperature')
    parser.add_argument('--top', type=int, default=10,
                        help='Output top-K designs (default 10)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None, help='Output JSON path')
    parser.add_argument('--af2', action='store_true',
                        help='Also run AF2 validation on top designs')
    args = parser.parse_args()

    print(f'Building complex PDB...')
    complex_pdb = build_complex_pdb(
        args.scaffold, args.scaffold_chain,
        args.antigen, args.antigen_chain)
    print(f'  → {complex_pdb}')

    print(f'Running ProteinMPNN ({args.samples} designs, T={args.temperature})...')
    t0 = time.time()
    designs = run_proteinmpnn(complex_pdb, f'{args.scaffold_chain} {args.antigen_chain}',
                              args.samples, args.temperature, args.seed)
    print(f'  → {len(designs)} designs in {time.time()-t0:.1f}s')

    print(f'Scoring with BFN...')
    t0 = time.time()
    scored = score_with_bfn(designs, complex_pdb, args.cdr)
    print(f'  → scored {len(scored)} in {time.time()-t0:.1f}s')

    os.unlink(complex_pdb)

    top = scored[:args.top]
    print(f'\n=== Top-{args.top} ===')
    for i, d in enumerate(top):
        print(f'#{i+1}: composite={d["composite"]:.4f}  '
              f'iptm={d["iptm"]:.4f}  plddt={d["plddt_mean"]:.4f}  '
              f'ppl={d["ppl"]:.1f}  entropy={d["entropy"]:.3f}  '
              f'disorder={d["disorder_mean"]:.3f}')
        print(f'    CDR: {d["sequence"][:80]}{"..." if len(d["sequence"])>80 else ""}')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'top': top, 'all': scored}, f, indent=2)
        print(f'\nSaved to {args.output}')

    if args.af2:
        print('\nAF2 validation not yet implemented in this script.')
        print('Use: python run_oc_v14_p1.py with the output JSON')
        print('  or python af2_wsl_batch.py on the grafted PDBs')


if __name__ == '__main__':
    main()
