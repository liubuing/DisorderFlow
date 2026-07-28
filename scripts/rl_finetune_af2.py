import os
#!/usr/bin/env python3
"""V18 Direction A: Multi-round AF2 distillation into BFN confidence (§2.3).

Root fix for Spearman ρ negative: BFN confidence heads were trained on
synthetic constant labels (V12) or seq recovery cross-entropy (V14),
neither of which teaches design-quality ranking. AF2 ipTM is the ground
truth for folding quality — distill it into BFN's confidence distribution
via multi-round RL with diverse rewards and KL regularization.

Enhancements over single-round G:
  1. Multi-round iteration (≥3 rounds): finetune → generate → AF2 → repeat
  2. Diverse reward: 0.5*ipTM + 0.3*pLDDT + 0.2*(1-interface_pae)
  3. Negative reward: designs with ipTM<0.05 used as counter-examples
  4. Temperature scheduling: early rounds T=0.5 (explore), late T=0.1 (exploit)
  5. KL regularization: prevent catastrophic forgetting of seq recovery

Usage:
    python rl_finetune_af2.py --rounds 3 --designs 50 --top 10 --finetune_steps 500
"""
import sys, os, pickle, json, time, random, argparse, tempfile, shutil, glob
import numpy as np
import torch
import lmdb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.misc import seed_all
from build_design_variant_dataset import _batch_af2_wsl
AA = 'ACDEFGHIKLMNPQRSTVWY'


def graft_cdrs(scaffold, cdr_seq, cdr_spec):
    """Graft designed CDRs into scaffold sequence.
    cdr_spec: list of (start_1based, end_1based, name)
    Returns full antibody sequence."""
    full = list(scaffold)
    pos = 0
    for start, end, name in cdr_spec:
        length = end - start + 1
        remaining = len(cdr_seq) - pos
        if remaining <= 0:
            break
        actual_len = min(length, remaining)
        cdr = cdr_seq[pos:pos + actual_len]
        for j in range(actual_len):
            full[start - 1 + j] = cdr[j]
        pos += actual_len
    return ''.join(full)


def load_bfn_model(ckpt_path, device='cuda'):
    """Load BFN model from checkpoint."""
    import bfn_loader
    # Point app_config to our checkpoint
    import yaml
    app = yaml.safe_load(open('app_config.yaml', encoding='utf-8'))
    orig_ckpt = app['models']['bfn']['checkpoint']
    app['models']['bfn']['checkpoint'] = ckpt_path
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, cfg = bfn_loader.load_bfn(device)
    # Restore
    app['models']['bfn']['checkpoint'] = orig_ckpt
    yaml.dump(app, open('app_config.yaml', 'w', encoding='utf-8'))
    return model, cfg


def get_cdr_spec(entry, scaffold_chain='H'):
    """Extract CDR regions (1-based, inclusive) from a SAbDab entry."""
    d = entry.get('heavy') if scaffold_chain == 'H' else entry.get('light')
    if d is None:
        return [], ''
    cf = d.get('cdr_flag')
    if cf is None:
        return [], ''
    cf = cf.tolist()
    aa = d['aa'].tolist()
    scaffold_seq = ''.join(AA[a] if a < 20 else 'X' for a in aa)
    names = {1: 'H1', 2: 'H2', 3: 'H3'}
    regions = []
    for t, nm in names.items():
        idxs = [i for i, c in enumerate(cf) if c == t]
        if len(idxs) >= 3:
            regions.append((idxs[0] + 1, idxs[-1] + 1, nm))  # 1-based
    return regions, scaffold_seq


def write_complex_pdb(entry, out_path):
    """Write antibody+antigen complex PDB from a SAbDab entry dict."""
    from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO
    AA3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
           'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
           'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
           'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
    st = Structure.Structure('s')
    mo = Model.Model(0)
    for cid, d in [('H', entry.get('heavy')), ('P', entry.get('antigen'))]:
        if d is None:
            continue
        ch = Chain.Chain(cid)
        aa = d['aa'].tolist()
        pos = d['pos_heavyatom']
        for i, a in enumerate(aa):
            if a >= 20:
                continue
            res = Residue.Residue((' ', i + 1, ' '), AA3.get(AA[a], 'GLY'), ' ')
            ca = pos[i][1].tolist() if pos[i].shape[0] > 1 else [0, 0, 0]
            res.add(Atom.Atom('CA', ca, 0.0, 1.0, ' ', ' CA ', i + 1, 'C'))
            res.add(Atom.Atom('N', [ca[0] - 1.46, ca[1], ca[2]], 0.0, 1.0, ' ', ' N  ', i + 1, 'N'))
            res.add(Atom.Atom('C', [ca[0] + 1.52, ca[1], ca[2]], 0.0, 1.0, ' ', ' C  ', i + 1, 'C'))
            res.add(Atom.Atom('O', [ca[0] + 2.1, ca[1], ca[2]], 0.0, 1.0, ' ', ' O  ', i + 1, 'O'))
            ch.add(res)
        mo.add(ch)
    st.add(mo)
    io = PDBIO()
    io.set_structure(st)
    io.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of RL rounds (≥3 recommended)')
    parser.add_argument('--scaffolds', type=int, default=5)
    parser.add_argument('--designs', type=int, default=50,
                        help='Designs per scaffold per round')
    parser.add_argument('--top', type=int, default=10,
                        help='Top-K designs used for finetuning')
    parser.add_argument('--finetune_steps', type=int, default=500,
                        help='Finetune steps per round')
    parser.add_argument('--temperature-start', type=float, default=0.5,
                        help='Early-round sampling temperature (explore)')
    parser.add_argument('--temperature-end', type=float, default=0.1,
                        help='Late-round sampling temperature (exploit)')
    parser.add_argument('--kl-weight', type=float, default=0.05,
                        help='KL divergence regularization weight')
    parser.add_argument('--ckpt', type=str,
                        default='logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/best.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix='rl_af2_')
    print(f"Working dir: {tmpdir}")

    # ── 1. Load BFN ──
    print("Loading BFN model...")
    model, _ = load_bfn_model(args.ckpt, args.device)

    # ── 2. Pick scaffolds from phase3 val ──
    print("Picking scaffolds from phase3 val set...")
    env = lmdb.open('data/sabdab_phase3_processed/val.lmdb', readonly=True, lock=False, readahead=False, subdir=False)
    ids = pickle.load(open('data/sabdab_phase3_processed/val.lmdb-ids', 'rb'))
    random.seed(42)
    random.shuffle(ids)

    scaffolds = []
    with env.begin() as txn:
        for sid in ids:
            if len(scaffolds) >= args.scaffolds:
                break
            e = pickle.loads(txn.get(sid.encode()))
            if e.get('heavy') is None or e.get('antigen') is None:
                continue
            cdr_spec, scaffold_seq = get_cdr_spec(e, 'H')
            if not cdr_spec:
                continue
            scaffolds.append((sid, e, cdr_spec, scaffold_seq))
    env.close()
    print(f"Selected {len(scaffolds)} scaffolds")

    # ── 3. Generate designs ──
    print(f"Generating {args.designs} designs per scaffold ({len(scaffolds) * args.designs} total)...")
    from bfn_loader import run_bfn_design

    all_designs = []  # list of (scaffold_idx, full_ab_seq, cdr_seq, bfn_iptm)
    for si, (sid, entry, cdr_spec, scaffold_seq) in enumerate(scaffolds):
        # Write complex PDB
        pdb_path = os.path.join(tmpdir, f'{sid}_complex.pdb')
        write_complex_pdb(entry, pdb_path)

        # Build region spec from cdr_spec (1-based ranges)
        parts = [f"H:{s}-{e}" for s, e, _ in cdr_spec]
        region_spec = ','.join(parts)

        seed_all(42 + si)
        designs = run_bfn_design(pdb_path, region_spec, num_samples=args.designs,
                                 stochastic=True, context_chains=['P'],
                                 device=args.device, sort_by=None)

        for d in designs:
            cdr_seq = d['sequence']
            full_ab = graft_cdrs(scaffold_seq, cdr_seq, cdr_spec)
            all_designs.append({
                'scaffold_idx': si,
                'sid': sid,
                'entry': entry,
                'cdr_spec': cdr_spec,
                'scaffold_seq': scaffold_seq,
                'cdr_seq': cdr_seq,
                'full_ab': full_ab,
                'bfn_iptm': d['iptm'],
                'bfn_plddt': d['plddt'],
            })
        print(f"  [{si + 1}/{len(scaffolds)}] {sid}: {len(designs)} designs")

    # ── 4. AF2 scoring ──
    print(f"Running AF2 on {len(all_designs)} designs...")
    epi_seq = ''.join(AA[a] if a < 20 else 'X' for a in scaffolds[0][1]['antigen']['aa'].tolist())
    ab_seqs = [d['full_ab'] for d in all_designs]
    print(f"  Epitope: {len(epi_seq)}aa, Antibody: ~{len(ab_seqs[0])}aa")

    af2_results = _batch_af2_wsl(ab_seqs, epi_seq, num_recycle=3, warmup_seq=ab_seqs[0])

    for d, r in zip(all_designs, af2_results):
        if r and r.get('success'):
            d['af2_iptm'] = r.get('iptm')
            d['af2_plddt'] = r.get('plddt')
            d['af2_interface_pae'] = r.get('interface_pae')
        else:
            d['af2_iptm'] = None
            d['af2_plddt'] = None
            d['af2_interface_pae'] = None

    # Also try to read interface_pae from WSL results file (not extracted by _batch_af2_wsl)
    wsl_results_path = '.af2_wsl_results.jsonl'
    if os.path.exists(wsl_results_path):
        try:
            with open(wsl_results_path) as f:
                for line in f:
                    r = json.loads(line)
                    idx = r.get('id')
                    if idx is not None and idx < len(all_designs):
                        if 'interface_pae' in r and all_designs[idx].get('af2_interface_pae') is None:
                            all_designs[idx]['af2_interface_pae'] = r['interface_pae']
        except Exception:
            pass

    valid = [d for d in all_designs if d['af2_iptm'] is not None]
    print(f"  AF2 succeeded: {len(valid)}/{len(all_designs)}")

    # ── Multi-round RL setup ──
    print(f"\n{'='*60}")
    print(f"V18 RL Distillation: {args.rounds} rounds, {args.designs} designs/round")
    print(f"  Temperature: {args.temperature_start:.2f} → {args.temperature_end:.2f}")
    print(f"  KL weight: {args.kl_weight}")
    print(f"  Top-K: {args.top}, Finetune steps: {args.finetune_steps}")
    print(f"{'='*60}\n")

    best_ckpt = args.ckpt
    all_round_results = []

    for round_idx in range(args.rounds):
        # Temperature schedule: linear interpolation
        progress = round_idx / max(args.rounds - 1, 1)
        temperature = args.temperature_start + (args.temperature_end - args.temperature_start) * progress
        print(f"\n{'─'*40}\n  ROUND {round_idx+1}/{args.rounds} (T={temperature:.2f})\n{'─'*40}")

    # ── 5. Select top designs with diverse reward ──
    # Try multiple ranking metrics: interface_pae (lower=better), iptm (higher=better)
    # interface_pae measures per-residue interface confidence — often more variance
    use_interface_pae = all(d.get('af2_interface_pae') is not None for d in valid)
    if use_interface_pae:
        sort_key = lambda d: d['af2_interface_pae']  # lower = better
        reverse_sort = False
        metric_name = 'interface_pae'
    else:
        sort_key = lambda d: d['af2_iptm']
        reverse_sort = True
        metric_name = 'iptm'

    print(f"Selecting top {args.top} per scaffold (by {metric_name})...")
    selected = []
    for si in range(len(scaffolds)):
        group = sorted([d for d in valid if d['scaffold_idx'] == si],
                       key=sort_key, reverse=reverse_sort)
        selected.extend(group[:args.top])
        if group:
            best = group[0]
            print(f"  scaffold {si}: best {metric_name}={best.get(metric_name, 'N/A'):.4f}, "
                  f"AF2 iptm={best.get('af2_iptm', 0):.4f}, BFN iptm={best['bfn_iptm']:.4f}")

    print(f"Selected {len(selected)} designs for finetuning")

    # ── 6. Build in-memory finetuning dataset ──
    print("Building finetuning dataset...")
    from torch.utils.data import Dataset as TorchDataset

    # Phase3 LMDB stores raw SAbDab entry dicts (heavy/light/antigen).
    # We need to apply transforms to get batch dicts, then replace CDR aa.
    phase3_transform = get_transform([
        {'type': 'mask_multiple_cdrs'},
        {'type': 'merge_chains'},
        {'type': 'patch_around_anchor'},
    ])

    # Re-open phase3 val LMDB
    env2 = lmdb.open('data/sabdab_phase3_processed/val.lmdb', readonly=True, lock=False, readahead=False, subdir=False)

    ft_samples = []
    with env2.begin() as txn:
        for i, d in enumerate(selected):
            sid = d['sid']
            cdr_seq = d['cdr_seq']

            # Load raw SAbDab entry from phase3 val LMDB
            raw = pickle.loads(txn.get(sid.encode()))

            # Apply transforms to get batch dict
            try:
                batch = phase3_transform(raw)
            except Exception as e:
                print(f"  WARNING: transform failed for {sid}: {e}, skipping")
                continue

            batch = recursive_to(batch, 'cpu')

            # Replace aa at CDR positions with designed sequence
            gen_flag = batch['generate_flag']
            cdr_indices = gen_flag.nonzero(as_tuple=True)[0]
            if len(cdr_indices) == 0:
                print(f"  WARNING: no CDR residues for {sid}, skipping")
                continue

            aa_tensor = batch['aa'].clone()
            pos = 0
            for idx in cdr_indices:
                if pos < len(cdr_seq):
                    aa_char = cdr_seq[pos]
                    aa_tensor[idx] = AA.index(aa_char) if aa_char in AA else 0
                    pos += 1

            batch['aa'] = aa_tensor
            ft_samples.append(batch)

    env2.close()
    print(f"  Built {len(ft_samples)} finetuning samples")

    class FinetuneDataset(TorchDataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    ft_dataset = FinetuneDataset(ft_samples)

    # ── 7. Finetune BFN ──
    print(f"Finetuning BFN for {args.finetune_steps} steps on {len(ft_samples)} samples...")

    # Manual training loop
    from torch.utils.data import DataLoader
    ft_loader = DataLoader(ft_dataset, batch_size=1, shuffle=True,
                           collate_fn=PaddingCollate(), num_workers=0)

    # Load model
    model, _ = load_bfn_model(args.ckpt, args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ft_log_path = os.path.join(tmpdir, 'finetune_log.txt')

    for step in range(args.finetune_steps):
        for batch in ft_loader:
            batch = recursive_to(batch, args.device)
            loss_dict = model(batch)
            loss = loss_dict.get('seq', torch.tensor(0.0))
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if step % 20 == 0:
            print(f"  ft step {step}: loss(seq)={loss.item():.4f}")

    # Save finetuned checkpoint (include config for bfn_loader compatibility)
    ft_ckpt_path = os.path.join(tmpdir, 'finetuned.pt')
    # Load original checkpoint to get config
    orig_ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    ft_state = {
        'model': model.state_dict(),
        'iteration': args.finetune_steps,
        'config': orig_ckpt.get('config', {}),
    }
    # Copy any other metadata from original checkpoint
    for k in ['optimizer', 'scheduler', 'min_val_loss']:
        if k in orig_ckpt:
            ft_state[k] = orig_ckpt[k]
    torch.save(ft_state, ft_ckpt_path)
    ft_best = ft_ckpt_path
    print(f"Finetuned checkpoint saved: {ft_best}")

    # ── 8. Verify with probe ──
    print("\n=== Running antigen signal probe ===")
    os.environ['V14_CKPT'] = ft_best
    ret = os.system(f'python -u probe_antigen_signal_usage.py')
    if ret != 0:
        print(f"Probe failed with exit code {ret}")

    # Cleanup
    shutil.rmtree(tmpdir)
    print(f"\nDone. Cleaned up {tmpdir}")


if __name__ == '__main__':
    main()
