"""FLEX-BINDER Phase A1b: exposure cleanliness, isolation, and family split.

Takes the 36 A1-passing single-domain x peptide PDBs and answers the question
"how many independent data units is that really", with four gates:

1. biological identity review (v21 lesson: ANARCII success is not antibody
   identity) - RCSB titles and polymer-entity descriptions keyword scan;
2. exposure cleanliness - the peptide must contact no protein chain other than
   the confirmed heavy chain (4.5 A heavy-atom cutoff, v26 convention);
3. homology isolation - antigen and heavy sequences vs the full ECLS data
   universe (sealed + development + adaptation panels), 30% identity / 80%
   coverage (v5 homology convention), global BLOSUM62 alignment;
4. family split - single-linkage clustering of antigens and heavy chains at
   the same threshold, reporting the independent-unit count that Phase B/C
   power statements must be based on.

No model training, no candidate generation, no binding claims.
"""
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.materialize_pae_analog_structures_v26 import heavy_coords, strict_residues
from scripts.flex_a1_scaffold_audit_v1 import count_nonstandard

OUT = ROOT / 'results/flex_a1b_isolation_families_v1'
A1 = ROOT / 'results/flex_a1_scaffold_audit_v1'
SCAN = ROOT / 'results/flex_campaign_v1_scaffold_scan.json'
ECLS_CIF = ROOT / 'data/peptide_h3_temporal_split_v1/cif'
ECLS_SPLITS = {
    'sealed_final': ROOT / 'data/peptide_h3_publication_split_v4/sealed_final_manifest.csv',
    'development': ROOT / 'data/peptide_h3_publication_split_v4/development_manifest.csv',
    'adaptation': ROOT / 'data/peptide_h3_publication_split_v4/adaptation_manifest.csv',
}
USER_AGENT = 'DisorderFlow-flex-a1b-isolation/1'
CONTACT_CUTOFF = 4.5
AA = set('ACDEFGHIKLMNPQRSTVWY')
IDENTITY_GATE = 0.30
COVERAGE_GATE = 0.80

IDENTITY_FLAGS = {
    'tcr': re.compile(r'\b(t[- ]?cell receptor|tcr)\b', re.I),
    'mhc': re.compile(r'\b(mhc|histocompatibility|hla[- ])\b', re.I),
    'bence_jones': re.compile(r'bence[- ]?jones|light[- ]chain dimer', re.I),
    'non_immunoglobulin': re.compile(r'\b(albumin|transferrin|lipase|protease inhibitor)\b', re.I),
}
ANTIBODY_KEYWORDS = re.compile(r'nanobod\w*|vhh|single[- ]domain|camelid|\bfab\b|antibod\w*|immunoglobulin|fv\b', re.I)


def http_json(url, retries=4):
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except Exception as error:
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f'{url} failed: {last}')


def chain_sequences(cif_path):
    """Sequence per protein chain with >=5 strict standard-CA residues."""
    structure = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True).get_structure(
        cif_path.stem, str(cif_path))
    model = next(structure.get_models())
    result = {}
    for chain in model:
        rows = strict_residues(chain, allow_mse=True)
        if len(rows) < 5:
            continue
        result[chain.id] = {
            'seq': ''.join(r['aa'] for r in rows),
            'rows': rows,
            'nonstandard': count_nonstandard(chain.get_residues()),
            'length': len(rows),
        }
    return result


def contact_chain_ids(peptide_rows, chains, heavy_chain_id, peptide_chain_id):
    """Protein chains whose heavy atoms approach the peptide within cutoff."""
    peptide = heavy_coords(peptide_rows)
    contacting = []
    for chain_id, info in chains.items():
        if chain_id in (heavy_chain_id, peptide_chain_id) or info['length'] < 5:
            continue
        other = heavy_coords(info['rows'])
        minimum = float('inf')
        for coords in peptide:
            for block in other:
                distance = float(np.linalg.norm(
                    coords[:, None, :] - block[None, :, :], axis=-1).min())
                minimum = min(minimum, distance)
        if minimum <= CONTACT_CUTOFF:
            contacting.append(chain_id)
    return contacting


def alignment_identity(seq_a, seq_b):
    """(identity, coverage_a, coverage_b) over the global BLOSUM62 alignment."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    alignment = aligner.align(seq_a, seq_b)[0]
    aligned = 0
    matches = 0
    for (a_start, a_end), (b_start, b_end) in zip(alignment.aligned[0],
                                                  alignment.aligned[1]):
        aligned += int(a_end) - int(a_start)
        matches += sum(1 for offset in range(int(a_end) - int(a_start))
                       if seq_a[int(a_start) + offset] == seq_b[int(b_start) + offset])
    if aligned == 0:
        return 0.0, 0.0, 0.0
    return matches / aligned, aligned / len(seq_a), aligned / len(seq_b)


def homologous(seq_a, seq_b):
    identity, cov_a, cov_b = alignment_identity(seq_a, seq_b)
    return (identity >= IDENTITY_GATE and (cov_a >= COVERAGE_GATE or cov_b >= COVERAGE_GATE),
            round(identity, 3), round(cov_a, 3), round(cov_b, 3))


def single_linkage(sequences, keys):
    """Cluster sequence keys at the identity/coverage gate (single linkage)."""
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1:]:
            hit, *_ = homologous(sequences[key_a], sequences[key_b])
            if hit:
                ra, rb = find(key_a), find(key_b)
                if ra != rb:
                    parent[rb] = ra
    clusters = {}
    for key in keys:
        clusters.setdefault(find(key), []).append(key)
    return sorted(clusters.values(), key=len, reverse=True)


def parse_fasta_all_chains(text):
    """All (chain_id, sequence) pairs from an RCSB FASTA download.

    Headers look like >{pdb}_{entity}|Chain A|... or >{pdb}_{entity}|Chains A/B|...
    """
    blocks = []
    header = None
    lines = []
    for line in text.splitlines():
        if line.startswith('>'):
            if header is not None:
                blocks.append((header, ''.join(lines)))
            header, lines = line, []
        else:
            lines.append(line.strip())
    if header is not None:
        blocks.append((header, ''.join(lines)))
    result = []
    for block_header, seq in blocks:
        chain_match = re.search(r'\|Chains? ([A-Za-z0-9/]+)(?:\||$)', block_header)
        if not chain_match or not seq:
            continue
        for chain_id in chain_match.group(1).split('/'):
            result.append((chain_id.strip(), seq))
    return result


def load_ecls_universe():
    """Antibody-side and antigen-side sequences from the ECLS data universe:
    every PDB of the v4 publication manifests (sealed/development/adaptation)
    plus the local post-2021 structure pool. Sequences come from cached RCSB
    FASTA downloads; chain roles are unknown here, so all chains are kept and
    bucketed by length at comparison time (conservative, role-free)."""
    universe = {}

    def add(pdb_id, panel):
        if pdb_id in universe:
            universe[pdb_id]['panels'].add(panel)
        else:
            universe[pdb_id] = {'panels': {panel}}

    import csv
    for panel, manifest_path in ECLS_SPLITS.items():
        if not manifest_path.exists():
            continue
        with manifest_path.open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                pdb = row['official_instance'].split('_')[1].lstrip('0') or '0'
                add(pdb.lower(), panel)
    for cif_path in ECLS_CIF.glob('*.cif'):
        add(cif_path.stem.split('_')[1].lstrip('0') or '0', 'local_post2021_pool')

    fasta_dir = OUT / 'ecls_fasta'
    fasta_dir.mkdir(parents=True, exist_ok=True)
    all_chains = {}
    for pdb, info in sorted(universe.items()):
        path = fasta_dir / f'{pdb}.fasta'
        if not path.exists():
            try:
                request = urllib.request.Request(
                    f'https://www.rcsb.org/fasta/entry/{pdb}/download',
                    headers={'User-Agent': USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
                time.sleep(0.3)
            except Exception as error:
                all_chains[pdb] = {'error': str(error), 'panels': sorted(info['panels'])}
                continue
        chains = {}
        for chain_id, seq in parse_fasta_all_chains(
                path.read_text(encoding='utf-8', errors='replace')):
            if seq not in chains:
                chains[seq] = chain_id
        all_chains[pdb] = {'panels': sorted(info['panels']),
                           'chains': [{'chain': cid, 'seq': seq, 'length': len(seq)}
                                      for seq, cid in chains.items()]}
    return all_chains


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a1_manifest = read(A1 / 'manifest.json')
    passing = [r for r in a1_manifest['records'] if r['passes']]
    passing_pdbs = sorted({r['pdb_id'] for r in passing})
    scan_rows = {}
    for row in read(SCAN)['candidate_pdbs_no_light_peptide']:
        scan_rows.setdefault(row['pdb'].lower(), row)

    protocol = {
        'classification': 'isolation_and_family_split_not_binding_evidence',
        'a1_manifest_sha256': digest(A1 / 'manifest.json'),
        'passing_pdbs': len(passing_pdbs),
        'identity_gate': IDENTITY_GATE, 'coverage_gate': COVERAGE_GATE,
        'contact_cutoff_angstrom': CONTACT_CUTOFF,
        'ecls_universe_panels': sorted(ECLS_SPLITS),
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)
    print('passing PDBs:', len(passing_pdbs), flush=True)

    # 1. biological identity review via RCSB metadata.
    identity_reviews = {}
    for pdb in passing_pdbs:
        try:
            entry = http_json(f'https://data.rcsb.org/rest/v1/core/entry/{pdb}')
            texts = [entry.get('struct', {}).get('title', '') or '']
            for entity_index in range(1, entry.get('rcsb_entry_info', {})
                                      .get('polymer_entity_count', 0) + 1):
                entity = http_json(
                    f'https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity_index}')
                texts.append(json.dumps(entity)[:4000])
            blob = ' '.join(texts)
            identity_reviews[pdb] = {
                'title': texts[0],
                'antibody_keyword': bool(ANTIBODY_KEYWORDS.search(blob)),
                'flags': {name: bool(rx.search(blob)) for name, rx in IDENTITY_FLAGS.items()},
            }
        except Exception as error:
            identity_reviews[pdb] = {'error': str(error)}
        time.sleep(0.4)
    freeze(OUT / 'identity_review.json', identity_reviews)
    flagged = {p: r for p, r in identity_reviews.items()
               if r.get('flags') or not r.get('antibody_keyword', True)}
    print('identity flagged:', sorted(flagged), flush=True)

    # 2. exposure cleanliness per passing component.
    cif_dir = A1 / 'cif'
    exposure = {}
    for record in passing:
        pdb = record['pdb_id']
        heavy_id = record['chains']['heavy']
        peptide_id = record['chains']['peptide']
        cache_key = f"{record['instance']}"
        if cache_key in exposure:
            continue
        try:
            chains = chain_sequences(cif_dir / f'{pdb}.cif')
            peptide_rows = chains[peptide_id]['rows'] if peptide_id in chains else None
            if peptide_rows is None:
                exposure[cache_key] = {'status': 'peptide_chain_not_found'}
                continue
            others = contact_chain_ids(peptide_rows, chains, heavy_id, peptide_id)
            exposure[cache_key] = {'status': 'ok', 'contacting_other_chains': others,
                                   'exposure_clean': not others}
        except Exception as error:
            exposure[cache_key] = {'status': f'error: {error}'}
    freeze(OUT / 'exposure.json', exposure)
    clean = [k for k, v in exposure.items() if v.get('exposure_clean')]
    print('exposure clean components:', len(clean), '/', len(exposure), flush=True)

    # 3. isolation vs the ECLS universe (v4 manifests + local post-2021 pool).
    universe = load_ecls_universe()
    heavy_like = {}
    peptide_like = {}
    fetch_errors = 0
    for pdb, info in universe.items():
        if 'chains' not in info:
            fetch_errors += 1
            continue
        for chain in info['chains']:
            if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', chain['seq']):
                continue
            bucket = heavy_like if chain['length'] >= 80 else peptide_like
            bucket.setdefault(chain['seq'], []).append(
                {'pdb': pdb, 'chain': chain['chain'], 'panels': info['panels']})
    print('ECLS universe PDBs:', len(universe),
          '| heavy-like unique seqs:', len(heavy_like),
          '| peptide-like unique seqs:', len(peptide_like),
          '| fetch errors:', fetch_errors, flush=True)
    my_antigens = {}
    my_heavies = {}
    for record in passing:
        my_antigens.setdefault(record['sequences']['peptide'], []).append(record['instance'])
        my_heavies.setdefault(record['sequences']['heavy'], []).append(record['instance'])

    def best_match(my_seq, reference_bucket):
        best = None
        for ref_seq, references in reference_bucket.items():
            hit, identity, cov_a, cov_b = homologous(my_seq, ref_seq)
            if hit and (best is None or identity > best['identity']):
                best = {'identity': identity, 'coverage_self': cov_a,
                        'coverage_ref': cov_b,
                        'references': [{'pdb': r['pdb'], 'chain': r['chain'],
                                        'panels': r['panels']} for r in references]}
        return best

    isolation = {'antigen_violations': [], 'heavy_violations': []}
    for my_seq, instances in sorted(my_antigens.items()):
        best = best_match(my_seq, peptide_like)
        if best:
            best['instances'] = instances
            isolation['antigen_violations'].append(best)
    for my_seq, instances in sorted(my_heavies.items()):
        best = best_match(my_seq, heavy_like)
        if best:
            best['instances'] = instances
            isolation['heavy_violations'].append(best)
    def bucket_index(bucket):
        index = {}
        for seq, refs in bucket.items():
            key = 'seq_' + hashlib.sha256(seq.encode()).hexdigest()[:16]
            index[key] = {'sequence': seq, 'length': len(seq), 'references': refs}
        return index

    freeze(OUT / 'isolation.json', {
        'universe_pdbs': len(universe),
        'universe_fetch_errors': fetch_errors,
        'heavy_like_unique_seqs': bucket_index(heavy_like),
        'peptide_like_unique_seqs': bucket_index(peptide_like),
        'isolation': isolation})
    print('isolation violations: antigens', len(isolation['antigen_violations']),
          'heavies', len(isolation['heavy_violations']), flush=True)

    # 4. family split among the passing set.
    antigen_clusters = single_linkage({k: k for k in my_antigens}, list(my_antigens))
    heavy_clusters = single_linkage({k: k for k in my_heavies}, list(my_heavies))
    summary = {
        'passing_pdbs': len(passing_pdbs),
        'unique_antigen_sequences': len(my_antigens),
        'unique_heavy_sequences': len(my_heavies),
        'antigen_families': len(antigen_clusters),
        'heavy_lineage_families': len(heavy_clusters),
        'antigen_cluster_sizes': [len(c) for c in antigen_clusters],
        'heavy_cluster_sizes': [len(c) for c in heavy_clusters],
        'exposure_clean_components': len(clean),
        'exposure_total_components': len(exposure),
        'identity_flagged_pdbs': sorted(flagged),
        'isolation_antigen_violations': len(isolation['antigen_violations']),
        'isolation_heavy_violations': len(isolation['heavy_violations']),
        'classification': 'independent_unit_counts_for_power_statements',
    }
    freeze(OUT / 'families.json', {
        'antigen_clusters': antigen_clusters, 'heavy_clusters': heavy_clusters})
    freeze(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
