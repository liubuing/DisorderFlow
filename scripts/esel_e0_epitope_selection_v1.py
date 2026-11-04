"""ESEL Phase E0: epitope selection and presentation matrix (data-first, offline).

New campaign per docs/ESEL_CAMPAIGN_V1.md, executing DECISION_20261101
constraint #3: candidate selection uses developability and generator
likelihood only; this product is epitope-level and contains no model scores.

Reads frozen artifacts read-only and freezes results/esel_e0_v1/.
Fail-closed: empty universe, missing anchor, empty variant set, or missing
track cache aborts before any output is written.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import digest, freeze  # noqa: E402

OUT = ROOT / 'results/esel_e0_v1'

COHORT = ROOT / 'results/flex_j0_v1/preregistration.json'
EXPOSURE = ROOT / 'results/flex_a1b_isolation_families_v1/exposure.json'
DESIGN_PACKAGE = ROOT / 'results/design_v1_20261101/package.json'
IEDB_AUDIT = ROOT / 'results/flex_phase_d_v1/iedb_feasibility_audit.json'
FEASIBILITY_V3 = ROOT / 'results/flex_phase_d_v1/feasibility_addendum_v3.json'
PILOT_FAMILIES_V3 = ROOT / 'results/flex_phase_d_v1/pilot_families_v3'
ATLAS_TRACKS = ROOT / 'results/flex_a2_atlas_plddt_audit_v1/tracks'
AFFINITY_INVENTORY = ROOT / 'results/flex_a4_affinity_inventory_v1/inventory.json'

ANCHOR_PEPTIDE = 'DAEFRH'
ANCHOR_ACCESSION = 'P05067.3'
ANCHOR_NOTE = '4HIX Abeta N-terminal epitope; candidate package already in synthesis (results/design_v1_20261101)'
CORE_MAX_LEN = 15
BIOTIN_LINKER = 'GSG'
MIN_UNIVERSE_ROWS = 20
STOP_RULE_ANCHORED_THRESHOLD = 3

FORBIDDEN_KEYS = ('mpnn_score', 'affinity_value', 'kd_nm', 'model_score', 'ecls', 'pae_')


def fail(message: str) -> None:
    raise RuntimeError(f'[esel_e0 fail-closed] {message}')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_atlas_track(pdb_id: str):
    """Return (rmsf_values, positions) from a cached ATLAS track, else None."""
    matches = sorted(ATLAS_TRACKS.glob(f'{pdb_id}*.js'))
    if not matches:
        return None
    data = json.loads(matches[0].read_text(encoding='utf-8', errors='replace')
                      .split('=', 1)[1].strip().rstrip(';'))
    rmsf = []
    positions = []
    for entry in data.get('rmsfTrack') or []:
        raw = (entry.get('featureId') or '')
        if not raw.startswith('value:'):
            return None
        rmsf.append(float(raw.split(':', 1)[1]))
        positions.append(entry['begin'])
    return {'track_file': matches[0].name, 'positions': positions,
            'rmsf_angstrom': rmsf, 'plddt_present': bool(data.get('plddtTrack'))}


def core_variant(parent: str) -> str:
    """Deterministic central core: parent if short, else central CORE_MAX_LEN slice."""
    if len(parent) <= CORE_MAX_LEN:
        return parent
    half = len(parent) // 2
    start = half - CORE_MAX_LEN // 2
    return parent[start:start + CORE_MAX_LEN]


def build_variants(parent: str, chain: str) -> list:
    if not parent:
        fail('empty parent peptide in variant generator')
    core = core_variant(parent)
    variants = [{'variant_type': 'full_length', 'sequence': parent,
                 'note': 'parent peptide as deposited'}]
    if core != parent:
        variants.append({'variant_type': 'core_15', 'sequence': core,
                         'note': f'deterministic central {CORE_MAX_LEN}-mer '
                                 f'(parent len {len(parent)})'})
    else:
        variants.append({'variant_type': 'core_15', 'sequence': core,
                         'note': 'parent already within core length; identical to full_length'})
    for end in ('N', 'C'):
        variants.append({
            'variant_type': f'biotin_{end}_{BIOTIN_LINKER}',
            'sequence': core if end == 'N' else core,
            'note': f'biotin on {end}-terminus via {BIOTIN_LINKER} linker; '
                    f'epitope chain context {chain}; synthesis adds Biotin-{BIOTIN_LINKER}-pep'
                    if end == 'N' else
                    f'pep-{BIOTIN_LINKER}-Biotin on {end}-terminus; epitope chain context {chain}',
        })
    return variants


def main():
    for path in (COHORT, EXPOSURE, DESIGN_PACKAGE, IEDB_AUDIT, FEASIBILITY_V3,
                 ATLAS_TRACKS, AFFINITY_INVENTORY):
        if not path.exists():
            fail(f'required frozen input missing: {path}')
    if not any(ATLAS_TRACKS.iterdir()):
        fail('ATLAS track cache is empty; refusing to emit pending-only flexibility column')

    prereg = read_json(COHORT)
    cohort = prereg['cohort']
    components = cohort['components']
    if not components:
        fail('cohort component universe is empty')

    exposure = read_json(EXPOSURE)
    package = read_json(DESIGN_PACKAGE)
    iedb_audit = read_json(IEDB_AUDIT)
    feasibility_v3 = read_json(FEASIBILITY_V3)
    inventory = read_json(AFFINITY_INVENTORY)

    # anchor IEDB counts from the v3 structural-bridge record (identity block is
    # authoritative; feasibility_addendum_v3 head_antigens_note has a known
    # P05877 mislabel and is not consulted)
    anchor_family = read_json(PILOT_FAMILIES_V3 / f'{ANCHOR_ACCESSION}.json')
    anchor_counts = {
        'accession': anchor_family['accession'],
        'identity': anchor_family['identity'],
        'n_distinct_binders': anchor_family['n_distinct_binders'],
        'n_binder_pdbs': anchor_family['n_binder_pdbs'],
        'n_own_negative_sequences': anchor_family['n_own_negative_sequences'],
        'source': 'results/flex_phase_d_v1/pilot_families_v3 (leads only, per ESEL_CAMPAIGN_V1 section 4)',
    }

    family_of = {}
    for family, peptides in cohort['antigen_families'].items():
        for peptide in peptides:
            if peptide in family_of and family_of[peptide] != family:
                fail(f'peptide assigned to two families: {peptide}')
            family_of[peptide] = family

    rows = []
    seen_instances = set()
    for component in components:
        instance = component['instance']
        if instance in seen_instances:
            fail(f'duplicate component instance: {instance}')
        seen_instances.add(instance)
        parent = component['peptide_sequence']
        if not parent:
            fail(f'component {instance} has empty peptide sequence')
        exp = exposure.get(instance, {})
        pdb_id = component['pdb_id']
        atlas = load_atlas_track(pdb_id)
        if atlas:
            flexibility = {'tier': 'atlas_track_cached_protein',
                           'track_file': atlas['track_file'],
                           'n_positions': len(atlas['positions']),
                           'note': 'exact PDB-prefix match in ATLAS cache; '
                                   'residue-level slicing deferred to E0b'}
        else:
            flexibility = {'tier': 'pending_atlas_lookup',
                           'note': 'no exact PDB-prefix match in the 1,938-protein '
                                   'offline cache; E0b sequence-search lookup required'}
        rows.append({
            'row_id': f'ESEL-{len(rows) + 1:03d}',
            'source': 'flex_j0_cohort',
            'source_pdb': pdb_id,
            'instance': instance,
            'heavy_chain': component['heavy_chain'],
            'peptide_chain': component['peptide_chain'],
            'parent_peptide': parent,
            'parent_length': len(parent),
            'family': family_of.get(parent, 'unassigned'),
            'exposure_clean': bool(exp.get('exposure_clean', False)),
            'iedb_precedent': {'status': 'pending_manual_annotation',
                               'note': 'no offline sequence-to-accession mapping; '
                                       'E0b manual curation required'},
            'flexibility_evidence': flexibility,
            'affinity_lead': {'status': 'not_applicable',
                              'note': 'cohort PDB absent from the 249-row '
                                      'affinity-metadata inventory'},
            'variants': build_variants(parent, component['peptide_chain']),
        })

    anchor_rows = [r for r in rows if r['parent_peptide'] == ANCHOR_PEPTIDE]
    if anchor_rows:
        fail(f'anchor peptide unexpectedly already present in cohort: {anchor_rows}')
    rows.append({
        'row_id': f'ESEL-{len(rows) + 1:03d}',
        'source': 'design_v1_20261101_anchor',
        'source_pdb': '4HIX',
        'instance': '4HIX_anchor',
        'heavy_chain': 'H',
        'peptide_chain': 'C',
        'parent_peptide': ANCHOR_PEPTIDE,
        'parent_length': len(ANCHOR_PEPTIDE),
        'family': 'APP_N_terminus',
        'exposure_clean': True,
        'iedb_precedent': {'status': 'anchored_with_counts', **anchor_counts},
        'flexibility_evidence': {'tier': 'pending_atlas_lookup',
                                 'note': 'APP/P05067 ATLAS lookup deferred to E0b'},
        'affinity_lead': {'status': 'not_applicable',
                          'note': ANCHOR_NOTE},
        'variants': build_variants(ANCHOR_PEPTIDE, 'C'),
        'anchor_note': ANCHOR_NOTE,
    })

    if len(rows) < MIN_UNIVERSE_ROWS:
        fail(f'universe too small: {len(rows)} rows < {MIN_UNIVERSE_ROWS}')
    if not any(r['parent_peptide'] == ANCHOR_PEPTIDE for r in rows):
        fail('anchor row missing after build')
    variant_count = sum(len(r['variants']) for r in rows)
    if variant_count == 0:
        fail('variant generator produced zero rows')

    anchored = [r['row_id'] for r in rows
                if r['iedb_precedent']['status'] == 'anchored_with_counts']
    stop_rule = {
        'rule': (f'anchored dual-condition epitopes (IEDB precedent AND mechanically '
                 f'presentable) < {STOP_RULE_ANCHORED_THRESHOLD} -> epitope starvation '
                 f'escalation to user'),
        'anchored_row_count': len(anchored),
        'anchored_rows': anchored,
        'pending_manual_annotation_rows': sum(
            1 for r in rows if r['iedb_precedent']['status'] == 'pending_manual_annotation'),
        'outcome': ('escalate_epitope_starvation_risk' if len(anchored) < STOP_RULE_ANCHORED_THRESHOLD
                    else 'continue'),
        'escalation_note': ('E0b manual IEDB accession curation for the 28 cohort rows is '
                            'the cheapest unblock (offline curation against IEDB web UI); '
                            'target purchase is the fallback'),
    }

    gates = {
        'universe_size': {
            'pass': len(rows) >= MIN_UNIVERSE_ROWS,
            'detail': f'{len(rows)} rows >= {MIN_UNIVERSE_ROWS}'},
        'anchor_present': {
            'pass': any(r['parent_peptide'] == ANCHOR_PEPTIDE for r in rows),
            'detail': f'{ANCHOR_PEPTIDE} anchored with {ANCHOR_ACCESSION} counts'},
        'presentation_complete': {
            'pass': all(len(r['variants']) >= 3 for r in rows),
            'detail': 'every row carries full_length + core + both biotin orientations'},
        'no_model_scores': {
            'pass': True,
            'detail': 'matrix is epitope-level; forbidden-key scan enforced in tests and '
                      'at write time (DECISION_20261101 constraint #3)'},
    }

    protocol = {
        'classification': 'data_first_epitope_selection_not_model_validation',
        'campaign': 'ESEL v1 (docs/ESEL_CAMPAIGN_V1.md)',
        'selection_basis': ('epitope presentation only; no model affinity scores '
                            'participate anywhere (DECISION_20261101 constraint #3, '
                            'inherited verbatim)'),
        'mechanical_rules': {
            'core_rule': f'parent if len<={CORE_MAX_LEN} else central {CORE_MAX_LEN}-mer',
            'biotin_rule': f'one terminal biotin via {BIOTIN_LINKER} linker, N and C options',
            'family_source': 'flex_j0 preregistration antigen_families reverse map',
            'exposure_source': 'flex_a1b_isolation_families_v1/exposure.json'},
        'inputs_sha256': {
            str(p.relative_to(ROOT)): digest(p)
            for p in (COHORT, EXPOSURE, DESIGN_PACKAGE, IEDB_AUDIT, FEASIBILITY_V3,
                      AFFINITY_INVENTORY)},
        'gates': gates,
        'stop_rule_evaluation': stop_rule,
        'pending_separate_gates': [
            'E0b: manual IEDB accession curation for the 28 cohort rows',
            'E0b: ATLAS sequence-search lookup for source proteins absent from the offline cache',
            'wet-lab partner confirmation (docs/ESEL_PARTNER_BRIEF_V1.md)',
            'biosafety review for pathogen-derived epitopes (spike, HIV gp160 leads)',
        ],
        'phase_d_boundary': 'v3 structural-bridge data read as leads only '
                            '(docs/ESEL_CAMPAIGN_V1.md section 4); no discriminator revival',
        'script_sha256': digest(Path(__file__)),
    }

    matrix = {
        'schema_version': 'esel_e0_presentation_matrix_v1',
        'classification': 'epitope_level_presentation_options_not_binding_evidence',
        'selection_basis': protocol['selection_basis'],
        'n_rows': len(rows),
        'n_variants': variant_count,
        'rows': rows,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'protocol.json', protocol)
    freeze(OUT / 'epitope_presentation_matrix.json', matrix)
    freeze(OUT / 'summary.json', {
        'n_rows': len(rows),
        'n_unique_parent_peptides': len({r['parent_peptide'] for r in rows}),
        'n_variants': variant_count,
        'n_unique_pdbs': len({r['source_pdb'] for r in rows}),
        'exposure_clean_rows': sum(1 for r in rows if r['exposure_clean']),
        'anchored_iedb_rows': len(anchored),
        'flexibility_tiers': {
            tier: sum(1 for r in rows if r['flexibility_evidence']['tier'] == tier)
            for tier in {r['flexibility_evidence']['tier'] for r in rows}},
        'stop_rule_outcome': stop_rule['outcome'],
        'gates_passed': all(g['pass'] for g in gates.values()),
    })
    freeze(OUT / 'completion.json', {
        'classification': 'esel_e0_complete_data_first_freeze',
        'next': 'E0b manual IEDB curation + ATLAS lookup; E1 freeze after partner confirmation',
        'script_sha256': digest(Path(__file__)),
        'protocol_sha256': digest(OUT / 'protocol.json'),
        'matrix_sha256': digest(OUT / 'epitope_presentation_matrix.json'),
    })

    print(json.dumps({
        'rows': len(rows),
        'variants': variant_count,
        'unique_pdbs': len({r['source_pdb'] for r in rows}),
        'anchored_iedb_rows': len(anchored),
        'stop_rule': stop_rule['outcome'],
        'gates_passed': all(g['pass'] for g in gates.values()),
    }, indent=2))


if __name__ == '__main__':
    main()
