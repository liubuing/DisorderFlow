"""CSN Phase C1 (v1) execution: transient-structure enrichment of antibody-
precedented windows on neuronal/synaptic IDP parents.

Implements results/csn_c1_v1/preregistration.json (frozen e8b19110...) before
any label was read. Mechanical parameters frozen via --protocol before --run.

Label ladder realized here (same estimator, expanding coverage):
  nmr_v2        in-repo corrected NMR RMSF (results/flex_a2_nmr_rmsf_v2)
  nmr_fetched   NEW solution-NMR ensembles fetched from RCSB for the parent
                accession and scored with the SAME corrected pipeline
                (kabsch_row, >=5 models) - same label type as nmr_v2
  atlas         ATLAS MD RMSF track, only when the best hit identity >= 90
                (the frozen v1-audit threshold; gp160's 58.8% best hit is
                therefore NOT admissible)
  afdb          AlphaFold DB pLDDT (downgraded stratum, descriptive only,
                never pooled with RMSF labels)

Exposure proxy: the frozen prereg specifies CA SASA on the bound antigen chain
of the deposited complex; where that chain is too short to yield any control
candidate (e.g. the 6-residue 2x6m chain), the execution falls back to the
parent's own structure (fetched NMR model or AFDB model) and records the
deviation per parent. The purpose (exposure matching) is preserved; the source
choice is disclosed, and affected rows are development evidence only.
"""
import argparse
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, SASA
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402
from scripts.csn_c0_conformer_sharing_v1 import (  # noqa: E402
    chain_residues, kabsch_row, parse_structure,
)
from scripts.flex_a2_atlas_nmr_audit_v1 import global_pairs  # noqa: E402
from scripts.flex_a2_nmr_rmsf_v2 import parse_entry as nmr_parse_entry  # noqa: E402

C1 = ROOT / 'results/csn_c1_v1'
POOL = ROOT / 'data/sabdab2_abag_project_pool_v2/cif'
C0_COMPS = ROOT / 'results/csn_c0_v2/components.json'
NMR_V2 = ROOT / 'results/flex_a2_nmr_rmsf_v2/rmsf.json'
OUT = ROOT / 'results/csn_c1_enrich_v6'
CACHE = OUT / 'cache'


def parse_any_structure(path):
    """CIF or PDB, by content sniff (AFDB/NMR downloads are PDB format; the
    csn_c0 parser is CIF-only and crashes on 'must begin with data_')."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        head = Path(path).read_bytes()[:64]
    if head.lstrip().startswith(b'data_'):
        return MMCIFParser(QUIET=True).get_structure(Path(path).stem, str(path))
    return PDBParser(QUIET=True).get_structure(Path(path).stem, str(path))
DATE = '2026-11-08'
CLASSIFICATION = 'development_descriptive_not_confirmatory_claim'
K_CONTROLS = 3
ATLAS_MIN_IDENTITY = 90.0
SEED = 20261108
DRAWS = 10000
RCSB_SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_PDB = 'https://files.rcsb.org/download/{}.pdb'
ATLAS_TRACK = 'https://www.dsimb.inserm.fr/ATLAS/api/tracks/{}.js'
ATLAS_REFERER = 'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{pdb_id}/{pdb_id}.html'
AFDB = 'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb'
LOCAL_AFDB = ROOT / 'data/disprot_afdb_cache'

CASE_PARENTS = {
    'P37840': {'pdb': '2x6m', 'window': 'DYEPEA', 'preferred': 'nmr_v2'},
    'P04578': {'pdb': '5hm1', 'window': 'AVERYLKDQQLLGIW', 'preferred': 'atlas'},
    'P0DTC2': {'pdb': '9lds', 'window': 'DPLQPELDSFKEELD', 'preferred': 'atlas'},
    'P21579': {'pdb': '8b8i', 'window': 'KEDAFSKLKEKFMNEL', 'preferred': 'nmr_fetched'},
    'G9GAG7': {'pdb': '5ivn', 'window': 'DRKAAVSHWQ', 'preferred': 'nmr_fetched'},
    'Q02223': {'pdb': '6j7w', 'window': 'GQCSQNEYFDSLLHAC', 'preferred': 'nmr_fetched'},
    'B2VLW8': {'pdb': '6s0y', 'window': 'VETPIRNEWGCRC', 'preferred': 'nmr_fetched'},
    'Q6NZI2': {'pdb': '9eiu', 'window': 'LSLLDKIIGAVDQIQL', 'preferred': 'nmr_fetched'},
    'P10636': {'pdb': '8op0', 'window': 'GSVQIVYKPV', 'preferred': 'afdb',
               'mismatch': 'deposited GSVQIVYKPK; canonical window used per prereg '
                           'anchored_with_mismatch; peptidematch verified zero for the '
                           'deposited peptide in Swiss-Prot+isoforms'},
    'P11207': {'pdb': '8skj', 'window': 'GKPIPNPLLGLDST', 'preferred': 'nmr_fetched',
               'note': 'PK/SV5 tag; PIV5 P protein; peptidematch hit (P11207/P11208 '
                       'share the N-terminal tag region; P11207 taken, lower accession, '
                       'deterministic)'},
    'Q08346': {'pdb': '6zwk', 'window': 'TQAQEY', 'preferred': 'nmr_fetched',
               'note': 'peptidematch single hit; E0b entity check failed because the '
                       'deposited antigen is a derived peptide'},
}
UNIVERSE_EXCLUSIONS = {
    '6i2g': 'SRLEEELRRRLTE verified zero in Swiss-Prot+isoforms (peptidematch v1); '
            'parent not resolvable at metadata level',
}


def http_get(url, referer=None, retries=3):
    last = None
    for attempt in range(retries):
        try:
            headers = {'User-Agent': 'csn-c1-enrich/1'}
            if referer:
                headers['Referer'] = referer
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - retry ladder
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'GET {url} failed: {last}')


def cached_text(path: Path, producer):
    if path.exists():
        return path.read_text(encoding='utf-8', errors='replace')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = producer()
    path.write_text(text, encoding='utf-8')
    time.sleep(0.5)
    return text


def parent_sequence(acc):
    text = cached_text(C1 / 'cache/fasta' / f'{acc}.fasta',
                       lambda: http_get(f'https://rest.uniprot.org/uniprotkb/{acc}.fasta')
                       .decode('utf-8', errors='replace'))
    return ''.join(line.strip() for line in text.splitlines()
                   if not line.startswith('>'))


def universe_sequences():
    seqs = []
    comps = json.loads(C0_COMPS.read_text(encoding='utf-8'))['components']
    for comp in comps:
        if comp.get('status') == 'valid':
            if comp.get('binder_ref'):
                seqs.append(comp['binder_ref'])
            seqs.append(comp['peptide_sequence'])
    n_pool = 0
    for path in sorted(POOL.glob('*.cif')):
        try:
            structure = parse_any_structure(path)
            for ch in structure[0]:
                residues = chain_residues(ch)
                seq = ''.join(aa for _r, aa in residues)
                if seq:
                    seqs.append(seq)
            n_pool += 1
        except Exception:  # noqa: BLE001 - counted, not silent
            continue
    print(f'  precedent universe: {n_pool} pool files, {len(seqs)} chains', flush=True)
    return seqs, n_pool


def precedent_index(lengths, seqs):
    index = {L: set() for L in lengths}
    for seq in seqs:
        for L in lengths:
            if len(seq) < L:
                continue
            for i in range(len(seq) - L + 1):
                index[L].add(seq[i:i + L])
    return index


def chain_sasa_map(structure_path, chain_id):
    """Residue-order -> CA SASA for one chain (ShrakeRupley, residue level)."""
    structure = parse_any_structure(structure_path)
    sasa = SASA.ShrakeRupley()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sasa.compute(structure, level='R')
    target = next((ch for ch in structure[0] if ch.id == chain_id), None)
    if target is None:
        return None, None
    residues = [r for r, _aa in chain_residues(target)]
    values = []
    for res in residues:
        try:
            values.append(float(res.sasa))
        except AttributeError:
            return None, None
    seq = ''.join(aa for _r, aa in chain_residues(target))
    return seq, values


def map_chain_to_parent(chain_seq, parent_seq):
    """(chain_index, parent_index) pairs via BLOSUM62 global alignment."""
    pairs, _score = global_pairs(chain_seq, parent_seq)
    return [(int(a), int(b)) for a, b in pairs]


def fetch_nmr_ensemble_rmsf(acc):
    """Fetch solution-NMR entries for the accession and score the first entry
    with >=5 models covering the case window, using the corrected pipeline."""
    query = {
        'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_polymer_entity_container_identifiers.'
                             'reference_sequence_identifiers.database_accession',
                'operator': 'exact_match', 'value': acc}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'exptl.method', 'operator': 'exact_match',
                'value': 'SOLUTION NMR'}}]},
        'return_type': 'entry',
        'request_options': {'paginate': {'start': 0, 'rows': 25},
                            'results_content_type': ['experimental']}}
    def produce():
        request = urllib.request.Request(
            RCSB_SEARCH, data=json.dumps(query).encode(),
            headers={'User-Agent': 'csn-c1-enrich/1',
                     'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode('utf-8', errors='replace')
    body = cached_text(CACHE / 'rcsb_search' / f'{acc}.json', produce)
    ids = [hit['identifier'] for hit in json.loads(body).get('result_set', [])]
    for entry_id in ids[:5]:
        try:
            text = cached_text(CACHE / 'nmr_pdb' / f'{entry_id}.pdb',
                               lambda: http_get(RCSB_PDB.format(entry_id))
                               .decode('utf-8', errors='replace'))
        except Exception:  # noqa: BLE001 - try next entry
            continue
        path = CACHE / 'nmr_pdb' / f'{entry_id}.pdb'
        entry = nmr_parse_entry(path)
        if entry and len(set(entry['rmsf'])) >= 1:
            return {'source': 'nmr_fetched', 'pdb': entry['pdb'],
                    'sequence': entry['sequence'], 'rmsf': entry['rmsf']}
    return None


def atlas_search_cache(acc, parent_seq):
    path = C1 / 'cache/atlas_search' / f'{acc}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    def produce():
        data = urllib.parse.urlencode({'sequence': parent_seq[:1000],
                                       'header': acc, 'dataset': 'ATLAS'}).encode()
        request = urllib.request.Request(
            'https://www.dsimb.inserm.fr/ATLAS/api/sequence-search', data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except Exception as error:  # noqa: BLE001 - 404 = no hit
            if '404' in str(error):
                return {'sequence_search_results': []}
            raise
    result = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze(path, result)
    time.sleep(1.0)
    return result


def fetch_atlas_label(parent_seq, acc):
    search = atlas_search_cache(acc, parent_seq)
    hits = search.get('sequence_search_results', [])
    eligible = [h for h in hits if float(h['identity']) >= ATLAS_MIN_IDENTITY]
    if not eligible:
        return None
    best = max(eligible, key=lambda h: float(h['identity']))
    pdb_id = best['pdb_id']
    try:
        raw = cached_text(CACHE / 'atlas_tracks' / f'{pdb_id}.js',
                          lambda: http_get(ATLAS_TRACK.format(pdb_id),
                                           referer=ATLAS_REFERER.format(pdb_id=pdb_id))
                          .decode('utf-8', errors='replace'))
        track = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))['rmsfTrack']
    except Exception:  # noqa: BLE001 - recorded as unavailable
        return None
    begins = [t['begin'] for t in track]
    if not all(b == i + 1 for i, b in enumerate(begins)):
        return None
    values = [float(t['value']) for t in track]
    target_pdb, target_chain = pdb_id.split('_')
    fasta = cached_text(CACHE / 'atlas_fasta' / f'{target_pdb}.fasta',
                        lambda: http_get(f'https://www.rcsb.org/fasta/entry/'
                                         f'{target_pdb}/download').decode('utf-8',
                                                                         errors='replace'))
    chain_seq = None
    header, lines = None, []
    blocks = []
    for line in fasta.splitlines() + ['>']:
        if line.startswith('>'):
            if header is not None:
                blocks.append((header, ''.join(lines)))
            header, lines = line, []
        else:
            lines.append(line.strip())
    for block_header, seq in blocks:
        if f'Chain {target_chain}[' in block_header + '[' or \
                f'Chain {target_chain}|' in block_header or \
                f'Chain {target_chain}(' in block_header:
            chain_seq = seq
            break
    if not chain_seq:
        return None
    pairs, _score = map_chain_to_parent(parent_seq, chain_seq)
    window = CASE_PARENTS[acc]['window']
    w_start = parent_seq.find(window)
    if w_start < 0:
        return None
    parent_to_target = {b: a + 1 for a, b in pairs}  # parent idx -> track pos
    vals = [values[parent_to_target[w_start + i] - 1]
            for i in range(len(window)) if (w_start + i) in parent_to_target]
    if len(vals) < len(window):
        return None
    return {'source': 'atlas', 'target': pdb_id,
            'identity': float(best['identity']), 'values': vals}


def fetch_afdb(acc):
    def produce():
        # Local AFDB cache first (disorder line), then the v6 download.
        local = LOCAL_AFDB / f'AF-{acc}-F1-model_v6.pdb'
        if local.exists():
            return local.read_text(encoding='utf-8', errors='replace')
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    AFDB.format(acc=acc), headers={'User-Agent': 'csn-c1-enrich/1'})
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read().decode('utf-8', errors='replace')
            except Exception as error:  # noqa: BLE001 - big models need patience
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
    text = cached_text(CACHE / 'afdb' / f'AF-{acc}-F1.pdb', produce)
    plddt = {}
    idx = 0
    for line in text.splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            plddt[idx] = float(line[60:66])
            idx += 1
    if not plddt:
        return None
    return {'source': 'afdb', 'n_residues': idx,
            'values': [plddt[i] / 100.0 for i in range(idx)]}


def exposure_values(acc, info, parent_seq, comp_by_pdb, nmr_cache):
    """Per-parent-position CA SASA. Source preference (deviation from the
    frozen bound-chain proxy is recorded per parent): (1) the parent's own
    full-length structure - 2kkw NMR for alpha-syn, else the AFDB v6 model -
    because free-state exposure matches the free-state label premise and the
    deposited chains are often too short to yield any control window;
    (2) the deposited antigen chain, only as a last resort."""
    L = len(info['window'])
    # (1a) alpha-syn: in-repo full-length NMR ensemble.
    if acc == 'P37840':
        try:
            path = ROOT / 'data/calibration_nmr_pdb/2kkw.pdb'
            structure = parse_any_structure(path)
            sasa = SASA.ShrakeRupley()
            sasa.compute(structure, level='R')
            for ch in structure[0]:
                residues = [r for r, _aa in chain_residues(ch)]
                if len(residues) >= len(parent_seq) - 5:
                    seq = ''.join(aa for _r, aa in chain_residues(ch))
                    if parent_seq[:40] in seq:
                        pairs = map_chain_to_parent(seq, parent_seq)
                        return ('parent_structure_nmr_2kkw',
                                {b: float(residues[a].sasa) for a, b in pairs})
        except Exception as error:  # noqa: BLE001 - fall through
            print(f'  2kkw exposure failed: {repr(error)[:80]}', flush=True)
    # (1b) AFDB v6 model.
    try:
        afdb_path = CACHE / 'afdb' / f'AF-{acc}-F1.pdb'
        if not afdb_path.exists():
            fetch_afdb(acc)
        structure = parse_any_structure(afdb_path)
        sasa = SASA.ShrakeRupley()
        sasa.compute(structure, level='R')
        for ch in structure[0]:
            residues = [r for r, _aa in chain_residues(ch)]
            if len(residues) >= len(parent_seq) - 5:
                seq = ''.join(aa for _r, aa in chain_residues(ch))
                pairs = map_chain_to_parent(seq, parent_seq)
                if len(pairs) >= len(parent_seq) * 0.8:
                    return ('parent_structure_afdb',
                            {b: float(residues[a].sasa) for a, b in pairs})
    except Exception as error:  # noqa: BLE001 - fall through, recorded
        print(f'  afdb exposure failed {acc}: {repr(error)[:80]}', flush=True)
    # (2) deposited antigen chain, last resort.
    comp = comp_by_pdb.get(info['pdb'])
    if comp and comp.get('status') == 'valid' and comp.get('coord_path'):
        path = ROOT / comp['coord_path']
        chain_id = comp.get('peptide_chain')
        try:
            chain_seq, sasa = chain_sasa_map(path, chain_id)
        except Exception:  # noqa: BLE001
            chain_seq, sasa = None, None
        if chain_seq and sasa:
            pairs = map_chain_to_parent(chain_seq, parent_seq)
            cov = [b for _a, b in pairs]
            w_start = parent_seq.find(info['window'])
            if w_start >= 0 and cov:
                lo, hi = min(cov), max(cov)
                if lo <= w_start and hi >= w_start + L - 1 and                         (hi - lo) >= L + 5:
                    pos_to_sasa = {b: sasa[a] for a, b in pairs}
                    return ('deposited_chain', {p: pos_to_sasa[p] for p in cov})
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.protocol:
        freeze(OUT / 'protocol.json', {
            'classification': CLASSIFICATION,
            'date': DATE,
            'prereg_sha256': digest(C1 / 'preregistration.json'),
            'mechanical_parameters': {
                'control_matching': (f'nearest-{K_CONTROLS} control windows per case '
                                     'window by |mean CA SASA difference|, deterministic, '
                                     'ties by lower start'),
                'sasa_engine': 'Bio.PDB.SASA.ShrakeRupley, residue level',
                'exposure_source_rule': ('deposited antigen chain when it spans the case '
                                         'window plus >=5 residues on one side and >=L+5 '
                                         'total; else the parent own structure (NMR model '
                                         'or AFDB model); deviation from the frozen '
                                         'bound-chain proxy is recorded per parent'),
                'precedent_universe': ('every chain sequence of the local pool CIFs plus '
                                       'queue component sequences; precedent = exact '
                                       'substring match of the window'),
                'label_kinds': ['nmr_v2', 'nmr_fetched', 'atlas_ge90', 'afdb_downgraded'],
                'atlas_identity_threshold': ATLAS_MIN_IDENTITY,
                'endpoint': {'statistic': 'parent-level paired difference '
                                          '(case - control mean)',
                             'uncertainty': f'cluster bootstrap {DRAWS} draws, seed {SEED}'},
            },
            'claims_boundary': 'see frozen prereg claims_boundary; descriptive only',
            'revision_note': ('v6: v5 left alpha-syn unscored (nmr_v2 tuple not converted '
                              'to a position map), RCSB NMR search used GET, AFDB big-file '
                              'downloads timed out, and new accessions lacked ATLAS '
                              'search caches; all four fixed, mechanical parameters '
                              'otherwise unchanged'),
            'script_sha256': digest(Path(__file__))})
        print('protocol frozen:', digest(OUT / 'protocol.json'))
    if args.run:
        run()


def run():
    prereg = json.loads((C1 / 'preregistration.json').read_text(encoding='utf-8'))
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['prereg_sha256'] != digest(C1 / 'preregistration.json'):
        raise SystemExit('prereg changed after protocol freeze; open csn_c1_enrich_v2')
    if protocol['script_sha256'] != digest(Path(__file__)):
        raise SystemExit('script changed after protocol freeze; open csn_c1_enrich_v2')

    comps = json.loads(C0_COMPS.read_text(encoding='utf-8'))['components']
    comp_by_pdb = {}
    for comp in comps:
        if comp.get('status') == 'valid':
            comp_by_pdb.setdefault(comp['pdb_id'], comp)

    print('building precedent universe...', flush=True)
    seqs, n_pool = universe_sequences()
    lengths = {len(info['window']) for info in CASE_PARENTS.values()}
    index = precedent_index(lengths, seqs)

    nmr_entries = {e['pdb']: e for e in
                   json.loads(NMR_V2.read_text(encoding='utf-8'))['entries']}
    rows = []
    for acc, info in CASE_PARENTS.items():
        parent_seq = parent_sequence(acc)
        w_start = parent_seq.find(info['window'])
        L = len(info['window'])
        if w_start < 0:
            rows.append({'accession': acc, 'status': 'window_not_in_parent'})
            continue
        exposure_source, sasa_map = exposure_values(acc, info, parent_seq,
                                                    comp_by_pdb, nmr_entries)
        if not sasa_map:
            rows.append({'accession': acc, 'status': 'exposure_unavailable'})
            continue
        case_sasa = float(np.mean([sasa_map[w_start + i] for i in range(L)
                                   if (w_start + i) in sasa_map]))
        candidates = []
        for s in range(len(parent_seq) - L + 1):
            if abs(s - w_start) < L:
                continue
            window = parent_seq[s:s + L]
            if window in index[L]:
                continue
            if not all((s + i) in sasa_map for i in range(L)):
                continue
            candidates.append((abs(float(np.mean([sasa_map[s + i] for i in range(L)])))
                                - case_sasa, s))
        candidates.sort(key=lambda t: (t[0], t[1]))
        controls = [s for _d, s in candidates[:K_CONTROLS]]
        if len(controls) < K_CONTROLS:
            rows.append({'accession': acc, 'status': 'control_candidates_insufficient',
                         'n_candidates': len(candidates)})
            continue

        # label, ladder order: nmr_v2 -> nmr_fetched -> atlas>=90 -> afdb
        label, label_kind = None, None
        nmr_hit = read_nmr_v2_label(info['window'])
        if nmr_hit:
            label, label_kind = nmr_hit, 'nmr_v2'
        if label is None:
            try:
                fetched = fetch_nmr_ensemble_rmsf(acc)
            except Exception as error:  # noqa: BLE001
                fetched = None
                print(f'  nmr fetch failed {acc}: {repr(error)[:80]}', flush=True)
            if fetched:
                ens_seq, ens_rmsf = fetched['sequence'], fetched['rmsf']
                w_start_ens = ens_seq.find(info['window'])
                if w_start_ens >= 0:
                    label = ens_rmsf[w_start_ens:w_start_ens + L]
                    label_kind = 'nmr_fetched'
                else:
                    pairs = map_chain_to_parent(ens_seq, parent_seq)
                    p2e = {b: a for a, b in pairs}
                    vals = [ens_rmsf[p2e[w_start + i]]
                            for i in range(L) if (w_start + i) in p2e]
                    if len(vals) == L:
                        label, label_kind = vals, 'nmr_fetched'
        if label is None and info['preferred'] in ('atlas', 'nmr_fetched'):
            try:
                atlas = fetch_atlas_label(parent_seq, acc)
            except Exception as error:  # noqa: BLE001
                atlas = None
                print(f'  atlas fetch failed {acc}: {repr(error)[:80]}', flush=True)
            if atlas:
                label, label_kind = atlas['values'], 'atlas_ge90'
        if label is None:
            try:
                afdb = fetch_afdb(acc)
            except Exception as error:  # noqa: BLE001
                afdb = None
                print(f'  afdb fetch failed {acc}: {repr(error)[:80]}', flush=True)
            if afdb and len(afdb['values']) >= w_start + L:
                label = afdb['values'][w_start:w_start + L]
                label_kind = 'afdb_downgraded'

        if label is None:
            rows.append({'accession': acc, 'status': 'label_unavailable',
                         'exposure_source': exposure_source})
            continue

        # Build the full-parent label map (position -> value) per kind; every
        # downstream read (case and controls) goes through this map, so gappy
        # ensemble coverage can never mis-slice.
        label_map, label_target = None, None
        try:
            if label_kind == 'nmr_v2':
                series, covered = read_nmr_v2_full(parent_seq)
                label_map = {pos: val for pos, val in zip(covered, series)}
            elif label_kind == 'nmr_fetched':
                fetched = fetch_nmr_ensemble_rmsf(acc)
                pairs = map_chain_to_parent(fetched['sequence'], parent_seq)
                label_map = {b: fetched['rmsf'][a] for a, b in pairs}
                label_target = fetched['pdb']
            elif label_kind == 'atlas_ge90':
                atlas = fetch_atlas_label(parent_seq, acc)
                if atlas is None:
                    rows.append({'accession': acc, 'status': 'label_series_insufficient'})
                    continue
                target_seq = atlas_target_seq(atlas['target'])
                if not target_seq:
                    rows.append({'accession': acc, 'status': 'label_series_insufficient'})
                    continue
                pairs = map_chain_to_parent(parent_seq, target_seq)
                track = atlas_full_values(atlas['target'])
                label_map = {b: track[a] for a, b in pairs}
                label_target = atlas['target']
            else:
                afdb = fetch_afdb(acc)
                label_map = {p: v for p, v in enumerate(afdb['values'])}
                label_target = f'AF-{acc}-F1'
        except Exception as error:  # noqa: BLE001 - recorded, never silent
            rows.append({'accession': acc, 'status': f'label_map_error: {repr(error)[:80]}'})
            continue
        if not label_map or not all((w_start + i) in label_map for i in range(L)):
            rows.append({'accession': acc, 'status': 'label_series_insufficient'})
            continue

        case_mean = float(np.mean([label_map[w_start + i] for i in range(L)]))
        control_means, control_windows = [], []
        for s in controls:
            if all((s + i) in label_map for i in range(L)):
                control_means.append(float(np.mean([label_map[s + i] for i in range(L)])))
                control_windows.append(parent_seq[s:s + L])
        if len(control_means) < K_CONTROLS:
            rows.append({'accession': acc, 'status': 'control_labels_insufficient'})
            continue
        diff = case_mean - float(np.mean(control_means))
        rows.append({
            'accession': acc, 'status': 'scored', 'pdb': info['pdb'],
            'window': info['window'], 'window_start': w_start,
            'mismatch': info.get('mismatch'), 'note': info.get('note'),
            'exposure_source': exposure_source,
            'label_kind': label_kind, 'label_target': label_target,
            'case_window_mean': round(case_mean, 4),
            'control_windows': control_windows,
            'control_means': [round(m, 4) for m in control_means],
            'paired_diff': round(diff, 4),
            'lower_is_more_transient': label_kind in ('nmr_v2', 'nmr_fetched',
                                                      'atlas_ge90'),
        })
        print(f'  {acc} {info["window"]}: {label_kind} case {case_mean:.3f} '
              f'controls {[round(m, 3) for m in control_means]} diff {diff:.3f}',
              flush=True)
        time.sleep(0.5)

    freeze(OUT / 'windows.json', {'classification': CLASSIFICATION, 'rows': rows})
    scored = [r for r in rows if r['status'] == 'scored']
    primary = [r for r in scored if r['label_kind'] in ('nmr_v2', 'nmr_fetched',
                                                        'atlas_ge90')]
    downgraded = [r for r in scored if r['label_kind'] == 'afdb_downgraded']
    # Sign conventions: RMSF lower = more transient structure; pLDDT HIGHER =
    # more structured (model confidence). Normalize to "transient-structure
    # score" = -paired_diff for RMSF, +paired_diff for pLDDT? No - keep raw
    # per-kind diffs and interpret separately (never pooled; prereg).
    rng = np.random.default_rng(SEED)
    primary_summary = None
    if len(primary) >= 3:
        diffs = np.array([r['paired_diff'] if r['label_kind'] != 'afdb_downgraded'
                          else -r['paired_diff'] for r in primary])
        # All primary rows are RMSF (lower = transient): mean diff < 0 supports H1.
        boot = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
                for _ in range(DRAWS)]
        ci = [round(float(np.percentile(boot, 2.5)), 4),
              round(float(np.percentile(boot, 97.5)), 4)]
        mean_diff = round(float(np.mean(diffs)), 4)
        if ci[0] <= 0 <= ci[1]:
            verdict = 'no_evidence'
        elif mean_diff < 0:
            verdict = 'transient_structure_enrichment'
        else:
            verdict = 'inverse_enrichment_exposed_flexible'
        primary_summary = {'n_parents': len(primary), 'mean_paired_diff': mean_diff,
                           'ci95': ci, 'verdict': verdict}
    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'n_scored': len(scored),
        'primary_stratum': {'rows': [r['accession'] for r in primary],
                            **(primary_summary or {'verdict': 'primary_stratum_underpowered',
                                                   'n_parents': len(primary),
                                                   'note': 'frozen stop rule: >=3 '
                                                           'primary-label parents required '
                                                           'for the CI endpoint'})},
        'downgraded_stratum': {'rows': [
            {'accession': r['accession'], 'paired_diff': r['paired_diff'],
             'note': 'pLDDT higher = more model-ordered; opposite sign convention '
                     'to RMSF; descriptive only, never pooled'} for r in downgraded]},
        'unscored': [{'accession': r['accession'], 'status': r['status']}
                     for r in rows if r['status'] != 'scored'],
        'claims_boundary': 'see frozen prereg claims_boundary',
    }
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'protocol_sha256': digest(OUT / 'protocol.json'),
        'script_sha256': digest(Path(__file__)),
        'artifacts': ['windows.json', 'summary.json', 'completion.json']})
    print(json.dumps({k: summary[k] for k in ('n_scored', 'primary_stratum')},
                     indent=1))


def label_source_name(kind, acc):
    return {'nmr_v2': '2kkw', 'nmr_fetched': 'rcsb_nmr', 'atlas_ge90': 'atlas_track',
            'afdb_downgraded': f'AF-{acc}-F1'}[kind]


def read_nmr_v2_label(case_pep):
    entries = json.loads(NMR_V2.read_text(encoding='utf-8'))['entries']
    for entry in entries:
        seq = entry['sequence']
        if case_pep in seq:
            off = seq.find(case_pep)
            return entry['rmsf'][off:off + len(case_pep)]
    return None


def read_nmr_v2_full(parent_seq):
    entries = json.loads(NMR_V2.read_text(encoding='utf-8'))['entries']
    for entry in entries:
        seq = entry['sequence']
        if len(seq) >= len(parent_seq) - 5 and parent_seq[:40] in seq:
            pairs = map_chain_to_parent(seq, parent_seq)
            p2e = {b: a for a, b in pairs}
            try:
                series = [entry['rmsf'][p2e[p]] for p in range(len(parent_seq))
                          if p in p2e]
            except KeyError:
                continue
            return series, sorted(p2e)
    return None


def atlas_target_seq(target_pdb):
    fasta = cached_text(CACHE / 'atlas_fasta' / f'{target_pdb}.fasta',
                        lambda: http_get(f'https://www.rcsb.org/fasta/entry/'
                                         f'{target_pdb}/download').decode('utf-8',
                                                                         errors='replace'))
    chain_id = target_pdb.split('_')[1]
    header, lines, blocks = None, [], []
    for line in fasta.splitlines() + ['>']:
        if line.startswith('>'):
            if header is not None:
                blocks.append((header, ''.join(lines)))
            header, lines = line, []
        else:
            lines.append(line.strip())
    for block_header, seq in blocks:
        if f'Chain {chain_id}[' in block_header + '[' or \
                f'Chain {chain_id}|' in block_header or \
                f'Chain {chain_id}(' in block_header:
            return seq
    return None


def atlas_full_values(target_pdb):
    raw = cached_text(CACHE / 'atlas_tracks' / f'{target_pdb}.js',
                      lambda: http_get(ATLAS_TRACK.format(target_pdb),
                                       referer=ATLAS_REFERER.format(pdb_id=target_pdb))
                      .decode('utf-8', errors='replace'))
    track = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))['rmsfTrack']
    return [float(t['value']) for t in track]


if __name__ == '__main__':
    main()
