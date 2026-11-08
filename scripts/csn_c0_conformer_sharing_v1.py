"""CSN Phase C0 (v1): cross-binder conformer sharing on disordered-epitope peptides.

Question: do independently deposited binders (different PDB entries) of the same
disordered-epitope peptide report the same backbone conformation (a
conformer-selection signal), or distinct ones (an induced-fit signal)?

Coordinate bookkeeping only: no model inference, no AF2, no network, no scores
of any kind. Development-descriptive; never confirmatory (M7). Verdicts are a
closed set; narrative upgrades are forbidden (N2 discipline, CSN campaign doc).

Universe: the frozen flex_j0 28-component queue plus a mechanical substring
sweep of the local SAbDab2 projected pool, restricted to peptide-like antigen
chains (<=40 residues). Protein-embedded matches are recorded as context
observations only, never as components.

Frozen-first discipline: `--prereg` freezes preregistration.json (recording this
script's sha256) BEFORE any outcome metric exists; `--run` refuses to start
unless the current script sha256 equals the prereg-recorded value (M3).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import calc_dihedral
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402


def kabsch_row(mobile, reference):
    """Row-convention Kabsch superposition, unit-tested against rigid roundtrips.

    NOT the legacy ``scripts.flex_a2_nmr_rmsf_v1.kabsch``: that function returns
    the transpose of the correct rotation and applies it as ``mc @ rotation``
    (a perfect rigid motion yields 5-9 A RMSD instead of 0). Finding disclosed
    in docs/CSN_CAMPAIGN_V1.md section 6; frozen downstream artifacts that used
    the legacy function (flex_a2_nmr_rmsf_v1/rmsf.json, IDP ensemble pose
    diversity audits) require *_v2 recomputation before further use.
    """
    mc = mobile - mobile.mean(axis=0)
    rc = reference - reference.mean(axis=0)
    u, _s, vt = np.linalg.svd(mc.T @ rc)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return mc @ rotation + reference.mean(axis=0)

QUEUE = ROOT / 'results/flex_j0_v1/preregistration.json'
OUT = ROOT / 'results/csn_c0_v2'
POOL = ROOT / 'data/sabdab2_abag_project_pool_v2/cif'
MULTI = ROOT / 'data/multiscaffold_confirmatory_v1/cif'
CAL = ROOT / 'data/candidate_interface_external_calibration_v1/structures_v3b/cif'

DATE = '2026-11-08'
CLASSIFICATION = 'development_descriptive_not_confirmatory_claim'
RMSD_MIN_POINTS = 4
OVERLAP_MIN = 5
PEPTIDE_LIKE_MAX = 40
FLOOR_EXPECTED = 0.5
FLOOR_INVALIDATION = 1.0

# Closed verdict set (N2): nothing outside these strings may appear in summary.json.
VERDICTS = (
    'conformer_sharing', 'related_conformers', 'distinct_induced_fit', 'mixed',
    'no_cross_entry_evidence', 'no_valid_coordinates', 'coordinate_starvation',
)


def tier(rmsd):
    """Preregistered tiers, grounded on the frozen T2.1 v2 moderate tier (1.5-3.0 A)."""
    if rmsd <= 1.0:
        return 'same_bound_conformer'
    if rmsd <= 2.5:
        return 'related_conformers'
    return 'distinct_induced_fit'


def dihedral_class(phi, psi):
    """Descriptive Ramachandran class; priority extended > helical > ppii > other."""
    if phi is None or psi is None:
        return 'terminal_na'
    if -180.0 <= phi <= -80.0 and 80.0 <= psi <= 180.0:
        return 'extended'
    if -140.0 <= phi <= -55.0 and -70.0 <= psi <= -10.0:
        return 'helical'
    if -85.0 <= phi <= -35.0 and 115.0 <= psi <= 165.0:
        return 'ppii'
    return 'other'


def family_groups(sequences):
    """Union-find over the substring relation; family id = longest member."""
    parent = {s: s for s in sequences}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    seqs = sorted(set(sequences), key=len, reverse=True)
    for i, a in enumerate(seqs):
        for b in seqs[i + 1:]:
            if b in a or a in b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    keep, drop = (ra, rb) if len(ra) >= len(rb) else (rb, ra)
                    parent[drop] = keep
    groups = {}
    for s in set(sequences):
        groups.setdefault(find(s), set()).add(s)
    return {max(members, key=len): sorted(members) for members in groups.values()}


AA = set('ACDEFGHIKLMNPQRSTVWY')


def chain_residues(chain):
    """Standard residues (order-preserving) with seq1 names, MSE->M."""
    residues = []
    for res in chain:
        if res.id[0] != ' ':
            continue
        aa = seq1(res.resname, custom_map={'MSE': 'M'})
        if aa in AA:
            residues.append((res, aa))
    return residues


def parse_structure(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))


def backbone_atoms(res):
    try:
        return {name: res[name].coord.astype(float).copy()
                for name in ('N', 'CA', 'C', 'O')}
    except KeyError:
        return None


def ca_rmsd(coords_a, coords_b):
    """Kabsch-superposed CA RMSD over index-aligned coordinate pairs."""
    mob = np.asarray(coords_a, dtype=float)
    ref = np.asarray(coords_b, dtype=float)
    if len(mob) != len(ref) or len(mob) < RMSD_MIN_POINTS:
        raise ValueError(f'invalid overlap: {len(mob)} vs {len(ref)} points')
    aligned = kabsch_row(mob, ref)
    return float(np.sqrt(((aligned - ref) ** 2).sum(axis=1).mean()))


def copy_classes(selected):
    """Per-residue dihedral class for one bound copy (list of residue objects)."""
    classes = []
    n = len(selected)

    def vec(res, name):
        try:
            return res[name].get_vector()
        except KeyError:
            return None

    for i, res in enumerate(selected):
        nv, cav, cv_ = vec(res, 'N'), vec(res, 'CA'), vec(res, 'C')
        if nv is None or cav is None or cv_ is None:
            classes.append('atom_missing')
            continue
        phi = psi = None
        if i > 0:
            prev_c = vec(selected[i - 1], 'C')
            if prev_c is not None:
                phi = float(calc_dihedral(prev_c, nv, cav, cv_) * 180.0 / np.pi)
        if i < n - 1:
            nxt_n = vec(selected[i + 1], 'N')
            if nxt_n is not None:
                psi = float(calc_dihedral(nv, cav, cv_, nxt_n) * 180.0 / np.pi)
        classes.append(dihedral_class(phi, psi))
    return classes


def resolve_cif(pdb_id, heavy_chain):
    cands = []
    p1 = POOL / f'pdb_0000{pdb_id}_{heavy_chain}_+.cif'
    if p1.exists():
        cands.append(('sabdab2_pool', p1))
    p2 = MULTI / f'pdb_0000{pdb_id}-{heavy_chain}.cif'
    if p2.exists():
        cands.append(('multiscaffold', p2))
    p3 = CAL / f'{pdb_id.upper()}.cif'
    if p3.exists():
        cands.append(('calibration_full_entry', p3))
    return cands


def locate_peptide(structure, target, peptide_chain_id, binder_head):
    """Choose the antigen chain containing `target` as an exact substring.

    Disambiguation: prefer the preregistered peptide chain id (non-binder),
    then any non-binder chain (shortest), never a binder chain.
    """
    found = []
    for ch in structure[0]:
        residues = chain_residues(ch)
        seq = ''.join(aa for _r, aa in residues)
        if target in seq:
            found.append({'chain': ch.id, 'chain_len': len(seq),
                          'is_binder_chain': bool(binder_head) and binder_head in seq,
                          'residues': residues})
    if not found:
        return None, found
    pref = [f for f in found if f['chain'] == peptide_chain_id and not f['is_binder_chain']]
    nonbinder = [f for f in found if not f['is_binder_chain']]
    if pref:
        return pref[0], found
    if nonbinder:
        return sorted(nonbinder, key=lambda f: (f['chain_len'], f['chain']))[0], found
    return None, found


def extract_from_path(path, source, comp, structure=None):
    rec = {'instance': comp['instance'], 'pdb_id': comp['pdb_id'],
           'peptide_sequence': comp['peptide_sequence'], 'status': None,
           'coord_source': source,
           'coord_path': str(path.relative_to(ROOT))}
    if structure is None:
        structure = parse_structure(path)
    binder_head = (comp.get('heavy_sequence') or '')[:25]
    pick, found = locate_peptide(structure, comp['peptide_sequence'],
                                 comp.get('peptide_chain'), binder_head)
    rec['candidate_chains'] = [{'chain': f['chain'], 'chain_len': f['chain_len'],
                                'is_binder_chain': f['is_binder_chain']} for f in found]
    if pick is None:
        rec['status'] = ('peptide_substring_not_found' if not found
                         else 'peptide_only_in_binder_chain')
        return rec
    seq = ''.join(aa for _r, aa in pick['residues'])
    offset = seq.find(comp['peptide_sequence'])
    selected = [r for r, _aa in pick['residues'][offset:offset + len(comp['peptide_sequence'])]]
    cas, classes = [], []
    for res in selected:
        atoms = backbone_atoms(res)
        cas.append(atoms['CA'] if atoms else None)
    classes = copy_classes(selected)
    binder_ref = comp.get('heavy_sequence')
    if not binder_ref:
        # Expansion components: binder identity = longest chain that does not
        # contain the peptide substring (mechanical, recorded).
        other = []
        for ch in structure[0]:
            residues = chain_residues(ch)
            seq = ''.join(aa for _r, aa in residues)
            if seq and comp['peptide_sequence'] not in seq:
                other.append(seq)
        binder_ref = max(other, key=len) if other else None
    rec.update({
        'status': 'valid', 'peptide_chain': pick['chain'],
        'chain_len': pick['chain_len'], 'offset': offset,
        'visible_auth': [r.id[1] for r in selected],
        'n_ca': sum(c is not None for c in cas),
        'ca': [None if c is None else c.tolist() for c in cas],
        'dihedral_classes': classes,
        'binder_ref': binder_ref,
        'model_count': len(structure),
        'disambiguation': 'preregistered_peptide_chain' if pick['chain'] == comp.get('peptide_chain') else 'non_binder_shortest',
    })
    return rec


def extract_component(comp):
    rec = {'instance': comp['instance'], 'pdb_id': comp['pdb_id'],
           'peptide_sequence': comp['peptide_sequence'], 'status': None}
    cands = resolve_cif(comp['pdb_id'], comp['heavy_chain'])
    if not cands:
        rec['status'] = 'coordinates_unavailable'
        return rec
    return extract_from_path(cands[0][1], cands[0][0], comp)


def usable_positions(comp_rec):
    return [i for i, c in enumerate(comp_rec['ca']) if c is not None]


def class_of(comp_rec, i):
    c = comp_rec['dihedral_classes'][i]
    return c if c in ('extended', 'helical', 'ppii', 'other') else None


def pair_record(comp_a, comp_b, family_parent):
    inst_a, inst_b = comp_a['instance'], comp_b['instance']
    a_seq, b_seq = comp_a['peptide_sequence'], comp_b['peptide_sequence']
    off_a = family_parent.find(a_seq)
    off_b = family_parent.find(b_seq)
    pos_a = {off_a + i for i in usable_positions(comp_a)}
    pos_b = {off_b + i for i in usable_positions(comp_b)}
    identical = a_seq == b_seq
    common = sorted(pos_a & pos_b)
    need = len(a_seq) if identical else OVERLAP_MIN
    rec = {'a': inst_a, 'b': inst_b, 'peptide_a': a_seq, 'peptide_b': b_seq,
           'parent': family_parent, 'n_overlap_required': need,
           'n_overlap_usable': len(common), 'relation': None, 'status': None}
    rec['relation'] = ('cross_entry' if comp_a['pdb_id'] != comp_b['pdb_id']
                       else ('ncs_copy' if comp_a['binder_ref'] == comp_b['binder_ref']
                             else 'same_entry_diff_binder'))
    if len(common) < need or len(common) < RMSD_MIN_POINTS:
        rec['status'] = 'skipped_overlap_insufficient'
        return rec
    coords_a = [comp_a['ca'][p - off_a] for p in common]
    coords_b = [comp_b['ca'][p - off_b] for p in common]
    rmsd = ca_rmsd(coords_a, coords_b)
    shared = [p for p in common
              if class_of(comp_a, p - off_a) and class_of(comp_b, p - off_b)]
    agree = (sum(1 for p in shared if class_of(comp_a, p) == class_of(comp_b, p))
             / len(shared)) if shared else None
    rec.update({'status': 'scored', 'rmsd': rmsd, 'tier': tier(rmsd),
                'n_dihedral_shared': len(shared),
                'dihedral_agreement': None if agree is None else round(agree, 4)})
    return rec


def family_verdict(cross_pairs):
    if not cross_pairs:
        return 'no_cross_entry_evidence'
    tiers = {p['tier'] for p in cross_pairs}
    if tiers == {'same_bound_conformer'}:
        return 'conformer_sharing'
    if tiers == {'distinct_induced_fit'}:
        return 'distinct_induced_fit'
    if tiers <= {'same_bound_conformer', 'related_conformers'}:
        return 'related_conformers'
    return 'mixed'


def sweep_pool(parents, covered):
    """Mechanical substring sweep of the local projected pool (1,146 files)."""
    hits, errors = [], []
    files = sorted(POOL.glob('*.cif'))
    t0 = time.time()
    for k, path in enumerate(files, 1):
        pdb_id = path.name[len('pdb_0000'):].split('_')[0]
        try:
            structure = parse_structure(path)
            for ch in structure[0]:
                residues = chain_residues(ch)
                seq = ''.join(aa for _r, aa in residues)
                if not seq:
                    continue
                for pep in parents:
                    if pep in seq:
                        context = ('peptide_like' if len(seq) <= PEPTIDE_LIKE_MAX
                                   else 'protein_embedded')
                        hits.append({'file': path.name, 'pdb_id': pdb_id, 'chain': ch.id,
                                     'chain_len': len(seq), 'peptide': pep,
                                     'context': context,
                                     'covered_by_queue': (pdb_id, ch.id) in covered})
        except Exception as exc:  # noqa: BLE001 - one-off sweep, every failure is recorded
            errors.append({'file': path.name, 'error': repr(exc)})
        if k % 200 == 0:
            print(f'  sweep {k}/{len(files)} files, {len(hits)} hits, '
                  f'{time.time() - t0:.0f}s', flush=True)
    return hits, errors, {'n_files': len(files), 'runtime_s': round(time.time() - t0, 1)}


def build_prereg():
    comps = json.loads(QUEUE.read_text(encoding='utf-8'))['cohort']['components']
    uniq = sorted({c['peptide_sequence'] for c in comps})
    groups = family_groups(uniq)
    parents = sorted(groups)
    script_sha = digest(Path(__file__))
    return {
        'classification': CLASSIFICATION,
        'campaign': 'CSN v1',
        'phase': 'C0',
        'date': DATE,
        'question': ('Do independently deposited binders (different PDB entries) of the '
                     'same disordered-epitope peptide report the same backbone '
                     'conformation (conformer-selection signal) or distinct ones '
                     '(induced-fit signal)?'),
        'universe': {
            'queue_components': len(comps),
            'queue_pdbs': len({c['pdb_id'] for c in comps}),
            'unique_peptides': uniq,
            'sweep_parents': parents,
            'sweep_pool': 'data/sabdab2_abag_project_pool_v2/cif (local, frozen)',
            'sweep_rule': ('peptide-like antigen chains (<=40 residues) become expansion '
                           'components; protein-embedded matches recorded as context only'),
            'resolution_order': ['sabdab2_pool', 'multiscaffold', 'calibration_full_entry'],
            'known_missing_coordinates': ['5ivn_A_B', '6zwk_B_H', '6zwk_D_J',
                                          '6zwk_E_K', '6zwk_F_L', '8skj_G_H'],
        },
        'metric': ('C-alpha RMSD after Kabsch superposition restricted to overlapped '
                   'residues; identical peptides use full length; >=4 usable CA points'),
        'tiers': {
            'same_bound_conformer': 'rmsd <= 1.0',
            'related_conformers': '1.0 < rmsd <= 2.5',
            'distinct_induced_fit': 'rmsd > 2.5',
            'grounding': ('T2.1 v2 moderate-perturbation tier 1.5-3.0 A (frozen, '
                          'read-only reference); thresholds fixed before any pair RMSD '
                          'was computed'),
        },
        'floor_control': {
            'rule': 'same-entry same-binder (NCS-copy) pairs expected mean <= 0.5 A',
            'invalidation': ('mean > 1.0 A -> floor_check_failed; all same-entry '
                             'diversity interpretations downgraded'),
        },
        'dihedral_windows': {
            'extended': 'phi in [-180,-80] and psi in [80,180]',
            'helical': 'phi in [-140,-55] and psi in [-70,-10]',
            'ppii': 'phi in [-85,-35] and psi in [115,165]',
            'priority': ['extended', 'helical', 'ppii'],
            'purpose': 'descriptive class per bound copy; no mechanism claim',
        },
        'validity_gates': [
            'peptide chain located by exact substring, never a binder chain',
            'overlap >= 5 residues for distinct windows; full length for identical peptides',
            '>= 4 usable CA points in overlap',
            'multiple candidate chains are recorded, never silently dropped',
        ],
        'verdict_closed_set': list(VERDICTS),
        'family_verdict_rules': {
            'conformer_sharing': 'all valid cross-entry pairs tier same_bound_conformer',
            'related_conformers': 'all cross-entry pairs <= related_conformers',
            'distinct_induced_fit': 'all cross-entry pairs tier distinct_induced_fit',
            'mixed': 'any other tier mix',
        },
        'stop_rules': {
            'starvation': '0 families with a valid cross-entry pair -> coordinate_starvation',
            'generalization': ('<3 cross-entry families -> single_or_few_family_descriptive, '
                               'insufficient for generalization'),
        },
        'claims_boundary': ('Coordinate bookkeeping only; no model, no scores, no '
                            'network. Verdicts describe deposited bound conformations; '
                            'they do not establish conformer selection as a mechanism, '
                            'binding, affinity, or any design claim. Read-only '
                            'references: flex_j0 preregistration, T2.1 v2 tier, '
                            'INTEGRATIVE_ANALYSIS section 2.2.'),
        'revision_note': ('v2 of results/csn_c0_v1: the v1 run froze components.json and '
                          'sweep.json (extraction unaffected) and crashed in pair_record '
                          'before pairs/summary/completion; dihedral-agreement positions '
                          'were indexed in the parent frame instead of per-component local '
                          'indices (off_a/off_b mapping now applied). Prereg scientific '
                          'content is unchanged; only the script sha256 and OUT directory '
                          'moved. v1 partial artifacts remain write-once preserved.'),
        'provenance': {'script_sha256': script_sha, 'queue_sha256': digest(QUEUE)},
    }


def run():
    prereg_path = OUT / 'preregistration.json'
    if not prereg_path.exists():
        raise SystemExit('preregistration.json missing; run --prereg first (M4)')
    prereg = json.loads(prereg_path.read_text(encoding='utf-8'))
    if prereg['provenance']['script_sha256'] != digest(Path(__file__)):
        raise SystemExit('script changed after prereg freeze; open results/csn_c0_v2 (M4)')
    comps = json.loads(QUEUE.read_text(encoding='utf-8'))['cohort']['components']

    print('extracting 28 queue components...', flush=True)
    records = [extract_component(c) for c in comps]
    freeze(OUT / 'components.json', {'classification': CLASSIFICATION,
                                     'components': records})
    valid = [r for r in records if r['status'] == 'valid']
    print(f'  valid {len(valid)}/{len(records)}: '
          f"{sorted({r['status'] for r in records if r['status'] != 'valid'})}")

    covered = {(r['pdb_id'], r['peptide_chain']) for r in valid}
    print('sweeping pool...', flush=True)
    parents = sorted(family_groups({c['peptide_sequence'] for c in comps}))
    hits, errors, sweep_stats = sweep_pool(parents, covered)
    new_hits = [h for h in hits if not h['covered_by_queue']
                and h['context'] == 'peptide_like']
    expansion = []
    seen = set()
    for h in new_hits:
        key = (h['pdb_id'], h['chain'], h['peptide'])
        if key in seen:
            continue
        seen.add(key)
        expansion.append({'instance': f"EXP_{h['pdb_id']}_{h['chain']}",
                          'pdb_id': h['pdb_id'], 'peptide_sequence': h['peptide'],
                          'heavy_chain': None, 'heavy_sequence': None,
                          'expansion': True, 'source_file': h['file']})
    print(f'  hits {len(hits)} (peptide-like new {len(new_hits)}), '
          f'expansion components {len(expansion)}, parse errors {len(errors)}')

    exp_records = []
    for comp in expansion:
        path = POOL / comp['source_file']
        rec = extract_from_path(path, 'sabdab2_pool_expansion', comp)
        rec['expansion'] = True
        exp_records.append(rec)
    all_records = records + exp_records
    freeze(OUT / 'sweep.json', {'classification': CLASSIFICATION, 'stats': sweep_stats,
                                'hits': hits, 'parse_errors': errors,
                                'expansion_records': exp_records})

    usable = [r for r in all_records if r['status'] == 'valid']
    groups = family_groups({r['peptide_sequence'] for r in usable})
    families, pairs = [], []
    for parent, members in sorted(groups.items()):
        fam_comps = [r for r in usable if r['peptide_sequence'] in members]
        fam_pairs = []
        for i in range(len(fam_comps)):
            for j in range(i + 1, len(fam_comps)):
                p = pair_record(fam_comps[i], fam_comps[j], parent)
                if p['status'] == 'scored':
                    fam_pairs.append(p)
                    pairs.append(p)
        cross = [p for p in fam_pairs if p['relation'] == 'cross_entry']
        ncs = [p for p in fam_pairs if p['relation'] == 'ncs_copy']
        families.append({
            'parent': parent, 'members': sorted(members),
            'components': [r['instance'] for r in fam_comps],
            'n_pairs_scored': len(fam_pairs),
            'n_cross_entry': len(cross), 'n_ncs_copy': len(ncs),
            'n_same_entry_diff_binder': len([p for p in fam_pairs
                                             if p['relation'] == 'same_entry_diff_binder']),
            'verdict': family_verdict(cross),
        })
    freeze(OUT / 'pairs.json', {'classification': CLASSIFICATION, 'pairs': pairs})

    ncs_all = [p for p in pairs if p['relation'] == 'ncs_copy']
    floor_mean = (float(np.mean([p['rmsd'] for p in ncs_all])) if ncs_all else None)
    floor_flag = None if floor_mean is None and not ncs_all else (
        'ok' if floor_mean is not None and floor_mean <= FLOOR_INVALIDATION
        else 'floor_check_failed')
    cross_families = [f for f in families if f['n_cross_entry'] > 0]
    if not cross_families:
        global_verdict, global_class = 'coordinate_starvation', 'terminal'
    else:
        tier_sets = []
        for f in cross_families:
            tp = [p['tier'] for p in pairs if p['relation'] == 'cross_entry'
                  and p['parent'] == f['parent']]
            tier_sets.append(set(tp))
        if all(ts == {'same_bound_conformer'} for ts in tier_sets):
            global_verdict = 'conformer_sharing'
        elif all(ts == {'distinct_induced_fit'} for ts in tier_sets):
            global_verdict = 'distinct_induced_fit'
        else:
            global_verdict = 'mixed'
        global_class = ('population_eligible' if len(cross_families) >= 3
                        else 'single_or_few_family_descriptive')
    tau = next((f for f in families if 'GSVQIVYKPK' in f['members']), None)
    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'n_components_queue': len(records),
        'n_components_valid_queue': len(valid),
        'n_expansion_components': len(exp_records),
        'n_expansion_valid': len([r for r in exp_records if r['status'] == 'valid']),
        'n_families': len(families),
        'n_cross_entry_families': len(cross_families),
        'n_pairs_scored': len(pairs),
        'n_cross_entry_pairs': len([p for p in pairs if p['relation'] == 'cross_entry']),
        'floor_control': {'n_ncs_pairs': len(ncs_all),
                          'mean_rmsd': None if floor_mean is None else round(floor_mean, 4),
                          'flag': floor_flag,
                          'expected_mean_le': FLOOR_EXPECTED,
                          'invalidation_gt': FLOOR_INVALIDATION},
        'global_verdict': global_verdict,
        'global_classification': global_class,
        'families': families,
        'neuro_annotations': {
            'tau_phf6_star': {
                'family_parent': 'GSVQIVYKPK' if tau else None,
                'note': ('extended-class fraction per bound copy is reported in '
                         'components.json dihedral_classes; descriptive only'),
            },
            'alpha_syn_c_term': {
                'family_parent': 'DYEPEA',
                'note': ('single deposited complex (2x6m); no cross-binder evidence '
                         'available; classes recorded descriptively'),
            },
        },
        'claims_boundary': 'see preregistration.json claims_boundary (frozen)',
    }
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'prereg_sha256': digest(prereg_path),
        'script_sha256': digest(Path(__file__)),
        'queue_sha256': digest(QUEUE),
        'sweep_stats': sweep_stats,
        'artifacts': ['components.json', 'sweep.json', 'pairs.json',
                      'summary.json', 'completion.json'],
    })
    print('C0 frozen:', global_verdict, '/', global_class,
          f'({len(cross_families)} cross-entry families)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prereg', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prereg:
        freeze(OUT / 'preregistration.json', build_prereg())
        print('preregistration frozen:', digest(OUT / 'preregistration.json'))
    if args.run:
        run()


if __name__ == '__main__':
    main()
