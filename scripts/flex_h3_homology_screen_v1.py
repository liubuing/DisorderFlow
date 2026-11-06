"""FLEX H3 sequence-level homology screen (code-fix report 7.3).

Question: is the corrected J0a/J0b native advantage confounded by the judge
having seen homologous CDR-H3 (or antigen peptide) sequences during training?

The exact base-checkpoint training split (train 2950 / val 327) is not
recoverable locally, so this screen uses the most conservative available
universe: the FULL local SAbDab2 snapshot (data/sabdab2_current/splits.tar.gz,
15,647 antibody-antigen complexes, 2026 snapshot).  Any SAbDab-derived
training subset of the checkpoint lineage is a subset of (or heavily overlaps)
this universe, so a NEGATIVE result here bounds the memorization confound; a
POSITIVE result identifies the exact structures to audit further.

Screens, per J0 component:
1. exact substring: does the H3 / peptide sequence appear verbatim in any
   heavy chain / any chain of the snapshot;
2. 8-mer seed screen: max fraction of the query's distinct 8-mers shared with
   a single snapshot chain (contiguity-free homology proxy; a 12-mer H3 has
   5 distinct 8-mers, so 4/5 = 0.8 flags a near-exact local match).

Outputs (write-once): results/flex_h3_homology_screen_v1/.

Usage:
  python scripts/flex_h3_homology_screen_v1.py            # full screen
  python scripts/flex_h3_homology_screen_v1.py --limit 50 # smoke
"""
import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest  # noqa: E402

J0 = ROOT / 'results/flex_j0_v1'
SPLITS_TAR = ROOT / 'data/sabdab2_current/splits.tar.gz'
OUT = ROOT / 'results/flex_h3_homology_screen_v1'

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}

K = 8


def chain_sequences_from_cif_text(text):
    """Observed-residue sequences per auth chain from an mmCIF atom_site loop.

    Light parse: residues in first-occurrence order per auth_asym_id.  This is
    the observed (deposited) sequence, which is what a training loader sees.
    """
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith('_atom_site.'):
            i += 1
            continue
        headers = []
        while i < n and lines[i].startswith('_atom_site.'):
            headers.append(lines[i].strip())
            i += 1
        col = {name: k for k, name in enumerate(headers)}
        while i < n:
            row = lines[i].strip()
            if not row or row.startswith('#'):
                break
            parts = row.split()
            if len(parts) >= len(headers):
                aa = THREE_TO_ONE.get(parts[col['_atom_site.label_comp_id']])
                if aa is not None:
                    chain = parts[col['_atom_site.auth_asym_id']]
                    key = (chain, parts[col['_atom_site.auth_seq_id']])
                    if key not in seen:
                        seen.add(key)
                        chains.setdefault(chain, []).append(aa)
            i += 1
    return chains


def eightmers(seq):
    return {seq[i:i + K] for i in range(len(seq) - K + 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='debug only: screen only the first N members')
    parser.add_argument('--out', default=str(OUT),
                        help='debug only: redirect output (smoke tests)')
    args = parser.parse_args()
    out_dir = Path(args.out)

    cohort = read(J0 / 'cohort.json')
    prereg = read(J0 / 'preregistration.json')
    family_of = {}
    for family_key, members in prereg['cohort']['antigen_families'].items():
        for peptide in members:
            family_of[peptide] = family_key

    queries = []
    for entry in cohort['components']:
        queries.append({
            'kind': 'h3',
            'instance': entry['instance'],
            'sequence': entry['region_sequence'],
            'family': family_of.get(entry['peptide_sequence']),
        })
    seen_peptides = set()
    for entry in cohort['components']:
        pep = entry['peptide_sequence']
        if pep not in seen_peptides:
            seen_peptides.add(pep)
            queries.append({
                'kind': 'peptide',
                'instance': entry['instance'],
                'sequence': pep,
                'family': family_of.get(pep),
            })

    # Index: k-mer -> set of member indices; plus per-member chain sequences.
    # Members whose embedded PDB id belongs to the J0 cohort itself are
    # excluded: hitting your own deposited structure says nothing about
    # training-set memorization (the snapshot postdates the checkpoint).
    j0_pdb_ids = {e['pdb_id'].lower() for e in cohort['components']}
    member_count = 0
    n_excluded_self = 0
    excluded_ids: list[str] = []
    member_names: list[str] = []
    exact_hits = {q['sequence']: [] for q in queries}
    kmer_index: dict[str, set[int]] = {}
    chain_cache: dict[int, list[tuple[str, str]]] = {}

    import re
    with tarfile.open(SPLITS_TAR, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.cif'):
                continue
            match = re.match(r'splits_final/pdb_0*(.+?)_', member.name)
            if match and match.group(1).lower() in j0_pdb_ids:
                n_excluded_self += 1
                if match.group(1).lower() not in excluded_ids:
                    excluded_ids.append(match.group(1).lower())
                continue
            if args.limit and member_count >= args.limit:
                break
            member_index = member_count
            member_names.append(member.name)
            member_count += 1
            text = tar.extractfile(member).read().decode('utf-8', errors='replace')
            chains = chain_sequences_from_cif_text(text)
            chain_cache[member_index] = [
                (chain_id, ''.join(aas)) for chain_id, aas in chains.items()]
            for chain_id, seq in chain_cache[member_index]:
                for query in queries:
                    if query['sequence'] in seq:
                        exact_hits[query['sequence']].append(
                            {'member': member.name, 'chain': chain_id})
                for i in range(len(seq) - K + 1):
                    kmer_index.setdefault(seq[i:i + K], set()).add(member_index)
            if member_count % 1000 == 0:
                print(f'  indexed {member_count} members', flush=True)

    rows = []
    for query in queries:
        seq = query['sequence']
        query_kmers = eightmers(seq)
        member_votes: dict[int, int] = {}
        for kmer in query_kmers:
            for member_index in kmer_index.get(kmer, ()):
                member_votes[member_index] = member_votes.get(member_index, 0) + 1
        best_member, best_shared = None, 0
        if member_votes:
            best_member = max(member_votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            best_shared = member_votes[best_member]
        best_chain, best_chain_seq = None, ''
        if best_member is not None:
            chain_options = [(cid, s) for cid, s in chain_cache[best_member]
                             if query_kmers & eightmers(s)]
            if chain_options:
                best_chain, best_chain_seq = max(
                    chain_options, key=lambda cs: len(query_kmers & eightmers(cs)))
        rows.append({
            'kind': query['kind'],
            'instance': query['instance'],
            'family': query['family'],
            'sequence': seq,
            'length': len(seq),
            'n_distinct_kmers': len(query_kmers),
            'exact_substring_hits': exact_hits[seq],
            'best_kmer_member': best_member,
            'best_kmer_member_file': member_names[best_member]
                                      if best_member is not None else None,
            'best_shared_kmers': best_shared,
            'best_kmer_fraction': float(best_shared / len(query_kmers)) if query_kmers else 0.0,
            'best_member_chains': [c for c, _ in chain_cache.get(best_member, [])]
                                  if best_member is not None else None,
            'best_chain_id': best_chain,
        })

    h3_rows = [r for r in rows if r['kind'] == 'h3']
    pep_rows = [r for r in rows if r['kind'] == 'peptide']
    summary = {
        'screen': 'h3_and_peptide_homology_vs_sabdab2_snapshot',
        'universe': str(SPLITS_TAR.relative_to(ROOT)),
        'n_members_screened': member_count,
        'n_members_excluded_self': n_excluded_self,
        'excluded_j0_pdb_ids': excluded_ids,
        'limit_applied': args.limit or None,
        'k': K,
        'n_h3_queries': len(h3_rows),
        'n_peptide_queries': len(pep_rows),
        'h3_exact_hits': sum(1 for r in h3_rows if r['exact_substring_hits']),
        'peptide_exact_hits': sum(1 for r in pep_rows if r['exact_substring_hits']),
        'h3_max_kmer_fraction': max(r['best_kmer_fraction'] for r in h3_rows),
        'peptide_max_kmer_fraction': max(r['best_kmer_fraction'] for r in pep_rows),
        'h3_mean_kmer_fraction': float(np.mean([r['best_kmer_fraction'] for r in h3_rows])),
        'interpretation_bound': (
            'negative result here bounds the memorization confound only against '
            'the SAbDab2 2026 snapshot; the exact base-checkpoint split '
            '(train 2950 / val 327) is not locally recoverable and may contain '
            'entries absent from this snapshot'),
        'classification': 'development_screen_not_confirmatory_claim',
        'cohort_source': str(J0 / 'cohort.json'),
    }
    freeze(out_dir / 'screen_rows.json', {'queries': rows})
    freeze(out_dir / 'screen_summary.json', summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
