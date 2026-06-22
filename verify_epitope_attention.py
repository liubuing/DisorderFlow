#!/usr/bin/env python
"""Verify: does BFN actually READ epitope residue identities during design?

Creates controlled epitope variants (same backbone geometry, different residues)
and checks whether BFN produces different CDR designs.

Three tests:
  1. Native vs All-Ala epitope (same positions, all Alanine)
  2. Native vs All-Gly epitope (same positions, all Glycine)
  3. Native vs Charge-flipped epitope (swap D/E with K/R)

If BFN truly attends to epitope → designs should differ between variants.
If BFN only sees backbone → designs will be identical regardless of residue changes.
"""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.'); sys.path.insert(0, 'modules')

import yaml, time, tempfile, shutil, random
import numpy as np

V11_CKPT = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'
CDR_SPEC = 'B:26-33,51-58,97-113'

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
AA1TO3 = {v: k for k, v in AA3TO1.items()}


def auto_detect_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_variant_pdb(template_pdb, chain_id, mutation_map, output_path):
    """Create a variant PDB with specified residue mutations on one chain.

    mutation_map: dict {resnum: new_aa_3letter} or a function aa -> aa
    """
    with open(template_pdb) as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21]
                if chain == chain_id:
                    resnum = int(line[22:26].strip())
                    old_resname = line[17:20].strip()

                    if callable(mutation_map):
                        new_aa3 = mutation_map(old_resname)
                    else:
                        new_aa3 = mutation_map.get(resnum, old_resname)

                    if new_aa3 != old_resname:
                        line = line[:17] + f'{new_aa3:>3s}' + line[20:]

            f.write(line)


def build_complex_and_design(epitope_pdb, label):
    """Build 5IMK+epitope complex and run BFN design."""
    from antibody_epitope_complex import position_epitope
    from bfn_loader import run_bfn_design, load_bfn

    complex_info = position_epitope(
        'data/misfolding_targets/5IMK.pdb', epitope_pdb,
        scaffold_chain='B', epitope_chain='A', distance=6.0,
    )

    designs = run_bfn_design(
        complex_info['pdb_path'], CDR_SPEC,
        num_samples=10, stochastic=True,
        context_chains=['A'],  # Complex mode
        device=auto_detect_device(),
    )

    ppls = [d.get('ppl', 0) for d in designs]
    seqs = [d['sequence'] for d in designs]

    # Cleanup temp dir
    tmp_dir = os.path.dirname(complex_info['pdb_path'])
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        'label': label,
        'n_designs': len(designs),
        'ppl_mean': float(np.mean(ppls)),
        'ppl_range': [float(min(ppls)), float(max(ppls))],
        'sequences': seqs,
        'unique_sequences': len(set(seqs)),
    }


def run_verify():
    # Setup checkpoint
    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    orig_ckpt = app_cfg['models']['bfn']['checkpoint']
    app_cfg['models']['bfn']['checkpoint'] = V11_CKPT
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    try:
        import bfn_loader
        bfn_loader._bfn_model = None
        bfn_loader._bfn_config = None
        from bfn_loader import load_bfn
        load_bfn(auto_detect_device())  # pre-load

        # Use Abeta42 as template (small, easy to modify)
        native_pdb = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'

        # Create variant PDBs
        tmpdir = tempfile.mkdtemp(prefix='epi_variants_')

        # Variant 1: All Alanine
        ala_pdb = os.path.join(tmpdir, 'abeta_all_ala.pdb')
        create_variant_pdb(native_pdb, 'A', lambda r: 'ALA', ala_pdb)

        # Variant 2: All Glycine
        gly_pdb = os.path.join(tmpdir, 'abeta_all_gly.pdb')
        create_variant_pdb(native_pdb, 'A', lambda r: 'GLY', gly_pdb)

        # Variant 3: Charge-flipped (D/E ↔ K/R)
        def flip_charge(resname):
            flip = {'ASP': 'LYS', 'GLU': 'ARG', 'LYS': 'ASP', 'ARG': 'GLU',
                    'HIS': 'ASP'}
            return flip.get(resname, resname)
        charge_pdb = os.path.join(tmpdir, 'abeta_charge_flip.pdb')
        create_variant_pdb(native_pdb, 'A', flip_charge, charge_pdb)

        # Variant 4: Shuffled residues (same composition, different order)
        # Read native residues, shuffle, write back
        native_residues = []
        with open(native_pdb) as f:
            for line in f:
                if line.startswith('ATOM') and line[21] == 'A':
                    rn = int(line[22:26].strip())
                    rname = line[17:20].strip()
                    if (rn, rname) not in native_residues:
                        native_residues.append((rn, rname))

        shuffled_resnames = [r[1] for r in native_residues]
        random.seed(42)
        random.shuffle(shuffled_resnames)
        shuffle_map = {native_residues[i][0]: shuffled_resnames[i]
                       for i in range(len(native_residues))}
        shuffle_pdb = os.path.join(tmpdir, 'abeta_shuffled.pdb')
        create_variant_pdb(native_pdb, 'A', shuffle_map, shuffle_pdb)

        # ── Run tests ──
        print('=' * 70)
        print('  VERIFICATION: Does BFN read epitope residue identities?')
        print('=' * 70)
        print(f'  Template epitope: Abeta42 (42aa)')
        print()

        results = {}
        for label, pdb in [
            ('Native (Abeta42)', native_pdb),
            ('All-Ala (same backbone)', ala_pdb),
            ('All-Gly (same backbone)', gly_pdb),
            ('Charge-flipped', charge_pdb),
            ('Shuffled residues', shuffle_pdb),
        ]:
            print(f'  Testing: {label}...', end=' ', flush=True)
            t0 = time.time()
            r = build_complex_and_design(pdb, label)
            dt = time.time() - t0
            print(f'{dt:.0f}s  PPL={r["ppl_mean"]:.0f}  unique_seqs={r["unique_sequences"]}')
            results[label] = r

        # ── Comparison ──
        print(f'\n{"=" * 70}')
        print(f'  COMPARISON: Sequence overlap between epitope variants')
        print(f'{"=" * 70}')

        native_seqs = set(results['Native (Abeta42)']['sequences'])

        for label, r in results.items():
            seqs = set(r['sequences'])
            overlap = len(native_seqs & seqs)
            unique_to_this = len(seqs - native_seqs)
            print(f'  {label:<30s}: overlap with native={overlap}/{len(seqs)}  '
                  f'unique={unique_to_this}')

        # The KEY question
        all_seq_sets = [set(r['sequences']) for r in results.values()]
        total_unique = len(set().union(*all_seq_sets))

        if total_unique <= len(native_seqs) * 1.2:
            print(f'\n  ❌ VERDICT: BFN IGNORES epitope residue identities!')
            print(f'     All variants produce essentially the same CDR designs.')
            print(f'     Only {total_unique} unique sequences across ALL 5 epitope variants.')
            print(f'     → Epitope is "a blob at distance 6Å", not a specific protein.')
        else:
            print(f'\n  ✅ VERDICT: BFN RESPONDS to epitope residue identities!')
            print(f'     {total_unique} unique sequences across variants (vs {len(native_seqs)} per variant).')

        # Show some sequences side by side
        print(f'\n  Top sequences from each variant:')
        for label, r in results.items():
            top_seq = r['sequences'][0] if r['sequences'] else 'N/A'
            print(f'    {label:<30s}: {top_seq[:50]}...')

        return results

    finally:
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    run_verify()
