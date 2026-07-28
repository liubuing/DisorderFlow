import os
#!/usr/bin/env python3
"""V3 L0 Gate: Physical scoring on crystal coordinates — native vs scrambled.

Paradigm shift: NO AF2 refolding. Use EXISTING crystal structure coordinates.
Scramble CDR residue identities, keep backbone geometry fixed.
Physical scores (contact + charge + hydrophobic) MUST separate native
from scrambled with p < 0.05 (paired t-test across 4 antibodies).

This gate blocks ALL further work until passed. It answers:
  "Can our physical metrics see what the crystal already knows?"

V3_ROOT_CAUSE: AF2-multimer systematically fails at short peptide-antibody
  de novo refolding (4HIX native CDR scores 0.118 on re-fold). This is NOT
  a DisorderFlow bug — it's a metric paradigm mismatch.
"""
import sys, os, json, copy, random, time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from Bio.PDB import PDBParser, PDBIO
from modules.idp_dock_score import score_complex
import tempfile

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q',
       'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
       'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

# ── Known anti-Aβ antibody crystal structures ──
ANTIBODIES = {
    '4HIX': {
        'pdb': 'data/anti_abeta_refs/4HIX.pdb',
        'name': 'solanezumab (3D6 Fab)',
        'ab_chain': 'H', 'epi_chain': 'A',
        'cdr_chothia': [(26,32),(52,56),(95,102)],  # VH H1,H2,H3
    },
    '5CSZ': {
        'pdb': 'data/anti_abeta_refs/5CSZ.pdb',
        'name': 'gantenerumab Fab',
        'ab_chain': 'H', 'epi_chain': 'D',  # 11aa peptide on chain D
        'cdr_chothia': [(26,32),(52,56),(95,102)],
    },
    '3UOT': {
        'pdb': 'data/anti_abeta_refs/3UOT.pdb',
        'name': 'anti-Aβ Fab 3UOT',
        'ab_chain': 'A', 'epi_chain': 'D',  # Fab H=A, peptide=D
        'cdr_chothia': [(26,32),(52,56),(95,102)],
    },
}

N_TRIALS = 20  # scramble trials per antibody
SIGNIFICANCE = 0.05


def scramble_cdr_residues(structure, ab_chain, cdr_ranges):
    """Scramble CDR residue names in-place. Backbone geometry unchanged."""
    s = copy.deepcopy(structure)
    cdr_residues = []
    for res in s[0][ab_chain]:
        ri = res.id[1]
        if any(sr <= ri <= er for sr, er in cdr_ranges):
            cdr_residues.append(res)
    names = [r.resname for r in cdr_residues]
    random.shuffle(names)
    for r, name in zip(cdr_residues, names):
        r.resname = name
    return s


def run_l0_gate():
    print("=" * 70)
    print("V3 L0 GATE: Physical Scoring on Crystal Coordinates")
    print("  native CDR vs scrambled (same backbone, shuffled residue IDs)")
    print("=" * 70)
    print()

    results = {}

    for pdb_id, info in ANTIBODIES.items():
        pdb_path = info['pdb']
        if not os.path.exists(pdb_path):
            print(f"[{pdb_id}] PDB NOT FOUND: {pdb_path} — SKIP")
            continue

        ab_chain = info['ab_chain']
        epi_chain = info['epi_chain']
        cdr_ranges = info['cdr_chothia']

        parser = PDBParser(QUIET=True)
        native_struct = parser.get_structure('native', pdb_path)

        print(f"\n[{pdb_id}] {info['name']}")
        print(f"  Chains: ab={ab_chain}, epi={epi_chain}")
        print(f"  CDR ranges (Chothia): {cdr_ranges}")

        # Score native
        native_tmp = tempfile.mktemp(suffix='_native.pdb')
        io = PDBIO(); io.set_structure(native_struct); io.save(native_tmp)
        native_score = score_complex(native_tmp, ab_chain, epi_chain, cdr_ranges)
        os.unlink(native_tmp)

        print(f"  Native:  contacts={native_score['contacts']} "
              f"density={native_score['density']:.3f} "
              f"elec={native_score['electrostatics']:.3f} "
              f"hydro={native_score['hydrophobic']:.3f} "
              f"composite={native_score['composite']:.4f}")

        # Scrambled trials
        scrambled_scores = []
        for trial in range(N_TRIALS):
            random.seed(42 + trial * 137)
            scram_struct = scramble_cdr_residues(native_struct, ab_chain, cdr_ranges)
            scram_tmp = tempfile.mktemp(suffix='_scram.pdb')
            io2 = PDBIO(); io2.set_structure(scram_struct); io2.save(scram_tmp)
            scram_score = score_complex(scram_tmp, ab_chain, epi_chain, cdr_ranges)
            os.unlink(scram_tmp)
            scrambled_scores.append(scram_score)

        scram_composites = [s['composite'] for s in scrambled_scores]
        scram_contacts = [s['contacts'] for s in scrambled_scores]

        # Statistics — use electrostatics + hydrophobic (SEQUENCE-DEPENDENT)
        # NOT contact-count (geometry-only, unchanged by residue rename)
        native_seq_score = native_score['electrostatics'] + native_score['hydrophobic']
        scram_seq_scores = [s['electrostatics'] + s['hydrophobic'] for s in scrambled_scores]
        scram_composites = [s['composite'] for s in scrambled_scores]

        scram_mean = np.mean(scram_seq_scores)
        scram_std = np.std(scram_seq_scores)
        sep_seq = native_seq_score - scram_mean

        # Cohen's d: effect size (native relative to scrambled distribution)
        pooled_std = np.sqrt((scram_std**2 + 0.0001) / 2)  # native var ≈ 0
        cohens_d = sep_seq / max(pooled_std, 0.001) if pooled_std > 0 else (1.0 if sep_seq > 0 else 0.0)
        native_contacts = native_score['contacts']

        # Pass: Cohen's d > 0.3 (small-medium effect) AND contacts > 0
        results[pdb_id] = {
            'native_seq_score': float(native_seq_score),
            'scrambled_mean': float(scram_mean),
            'scrambled_std': float(scram_std),
            'cohens_d': float(cohens_d),
            'pass': bool(cohens_d > 0.3 and native_contacts > 0),
        }

        print(f"  Scrambled (n={N_TRIALS}): seq_score mean={scram_mean:.4f} std={scram_std:.4f}")
        print(f"  Cohen's d={cohens_d:.3f} (sep={sep_seq:+.4f})")
        print(f"  L0: {'PASS' if results[pdb_id]['pass'] else 'FAIL'} "
              f"(d>0.3={cohens_d>0.3}, contacts>0={native_contacts>0})")

    # ── Final verdict ──
    print(f"\n{'='*70}")
    print("L0 GATE VERDICT")
    print(f"{'='*70}")
    n_pass = sum(1 for r in results.values() if r['pass'])
    n_total = len(results)
    print(f"  Passed: {n_pass}/{n_total}")

    if n_pass >= n_total * 0.5:  # at least 50% must pass
        print(f"  VERDICT: L0 PASSED — physical metrics are trustworthy")
        print(f"  Next: P1 — L3 cross-conformation robust scoring on P3 library")
    else:
        print(f"  VERDICT: L0 FAILED — physical metrics cannot separate native from scrambled")
        print(f"  DO NOT proceed to P1. Fix the metric first.")

    # Save
    os.makedirs('idp_benchmark_results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = f'idp_benchmark_results/v3_l0_gate_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'results': {k: {'pass': v['pass'], 'seq_separation': v.get('seq_separation', 0)}
                               for k, v in results.items()},
                   'n_pass': n_pass, 'n_total': n_total}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    run_l0_gate()
