#!/usr/bin/env python3
"""Direction G: AF2 ipTM-guided Best-of-N finetuning (single round).

BFN generates CDR designs → AF2 scores them → top designs used as
supervised targets to finetune BFN → probe verifies antigen signal Δ > 0.

Usage:
    python rl_finetune_af2.py [--scaffolds N] [--designs M] [--top K] [--finetune_steps S]
"""
import sys, os, pickle, json, time, random, argparse, tempfile, shutil, glob
import numpy as np
import torch
import lmdb
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')

from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from disorderflow.utils.transforms import get_transform
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.misc import seed_all
from build_design_variant_dataset import _batch_af2_wsl
from ad_af2_validate import graft_cdrs
AA = 'ACDEFGHIKLMNPQRSTVWY'


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
    parser.add_argument('--scaffolds', type=int, default=5)
    parser.add_argument('--designs', type=int, default=20)
    parser.add_argument('--top', type=int, default=2)
    parser.add_argument('--finetune_steps', type=int, default=200)
    parser.add_argument('--ckpt', type=str,
                        default='logs/v17c_zeroinit/bfn_v17c_zeroinit_xpu_2026_06_26__23_22_18/checkpoints/best.pt')
    parser.add_argument('--device', type=str, default='cuda')
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
        d['af2_iptm'] = r.get('iptm') if r.get('success') else None
        d['af2_plddt'] = r.get('plddt') if r.get('success') else None

    valid = [d for d in all_designs if d['af2_iptm'] is not None]
    print(f"  AF2 succeeded: {len(valid)}/{len(all_designs)}")

    # ── 5. Select top designs ──
    print(f"Selecting top {args.top} per scaffold...")
    selected = []
    for si in range(len(scaffolds)):
        group = sorted([d for d in valid if d['scaffold_idx'] == si],
                       key=lambda d: d['af2_iptm'], reverse=True)
        selected.extend(group[:args.top])
        if group:
            print(f"  scaffold {si}: best AF2 ipTM = {group[0]['af2_iptm']:.4f} (BFN ipTM = {group[0]['bfn_iptm']:.4f})")

    print(f"Selected {len(selected)} designs for finetuning")

    # ── 6. Create finetuning dataset (in-memory) ──
    print("Building finetuning dataset...")
    ft_data_dir = os.path.join(tmpdir, 'finetune_pdbs')
    os.makedirs(ft_data_dir, exist_ok=True)

    for i, d in enumerate(selected):
        # Create a modified PDB with the designed CDR sequence
        entry = d['entry']
        cdr_spec = d['cdr_spec']
        cdr_seq = d['cdr_seq']

        # Build modified aa array
        heavy = entry['heavy']
        aa_orig = heavy['aa'].clone()
        pos = 0
        for start, end, name in cdr_spec:
            length = end - start + 1
            for j in range(length):
                aa_val = AA.index(cdr_seq[pos + j]) if cdr_seq[pos + j] in AA else 0
                aa_orig[start - 1 + j] = aa_val
            pos += length

        # Write PDB with modified sequence
        pdb_path = os.path.join(ft_data_dir, f'design_{i:03d}.pdb')
        modified_entry = {
            'heavy': {**{k: v for k, v in heavy.items()}, 'aa': aa_orig},
            'antigen': entry.get('antigen'),
        }
        write_complex_pdb(modified_entry, pdb_path)

    print(f"  Created {len(selected)} PDB files in {ft_data_dir}")

    # ── 7. Finetune BFN ──
    print(f"Finetuning BFN for {args.finetune_steps} steps...")
    # Create finetune config
    ft_config_path = os.path.join(tmpdir, 'finetune_config.yml')
    with open(ft_config_path, 'w') as f:
        f.write(f"""
model:
  type: antibody_bfn
  res_feat_dim: 256
  pair_feat_dim: 128
  diffusion:
    num_steps: 20
    beta: 1.0
    schedule: cosine
    eps_net_opt:
      num_layers: 6
  train_structure: true
  train_sequence: true
  ot_opt:
    num_iters: 5
  loss_weight:
    seq: 1.0
    dist: 0.0
    ang: 0.0
    plddt: 0.0
    iptm: 0.0
    pae: 0.0
    disorder: 0.0
    conf_variance: 0.0
    conf_ranking: 0.0
    grouped_margin: 0.0
    conf_anticollapse: 0.0
    huber_beta: 0.1
    train_recycles: 1
    mask_aug_prob: 0.0
    label_smoothing: 0.0
    encoder_dropout: 0.0

train:
  loss_weights:
    seq: 1.0
  freeze_backbone: false
  max_iters: {args.finetune_steps}
  val_freq: 50
  log_freq: 10
  batch_size: 1
  warmup_steps: 20
  seed: 2042
  max_grad_norm: 5.0
  ema_decay: 0.999
  optimizer:
    type: adamw
    lr: 1.e-4
    weight_decay: 1.0e-4
    beta1: 0.9
    beta2: 0.999
  scheduler:
    type: plateau
    factor: 0.5
    patience: 5
    min_lr: 1.e-7

dataset:
  train:
    type: custom
    structure_dir: {ft_data_dir.replace(chr(92), '/')}
    transform:
      - type: mask_multiple_cdrs
      - type: merge_chains
      - type: patch_around_anchor
  val:
    type: custom
    structure_dir: {ft_data_dir.replace(chr(92), '/')}
    transform:
      - type: mask_multiple_cdrs
      - type: merge_chains
      - type: patch_around_anchor
""")

    ft_logdir = os.path.join(tmpdir, 'finetune_logs')
    os.makedirs(ft_logdir, exist_ok=True)

    ret = os.system(
        f'python train.py {ft_config_path} '
        f'--finetune {args.ckpt} '
        f'--accum_steps 2 '
        f'--logdir {ft_logdir} '
        f'--device {args.device}'
    )
    if ret != 0:
        print(f"ERROR: Finetuning failed with exit code {ret}")
        sys.exit(1)

    # Find the best checkpoint
    ft_ckpt_dirs = glob.glob(os.path.join(ft_logdir, '*', 'checkpoints'))
    if not ft_ckpt_dirs:
        print("ERROR: No finetune checkpoint found")
        sys.exit(1)
    ft_best = os.path.join(ft_ckpt_dirs[0], 'best.pt')
    print(f"Finetuned checkpoint: {ft_best}")

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
