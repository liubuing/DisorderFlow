"""ESEL E0b contract tests.

Bound to two incident patterns found during the first E0b run (M3):
(1) IEDB's curated_source_antigen accessions carry sequence versions
    (P37840.1 -> 300 records) while bare UniProt IDs return 0 silently;
(2) anchor identity drift: the Phase D head_antigens_note mislabeled P05877
    as alpha-synuclein (it is HIV gp160; alpha-syn is P37840). Tests pin the
    correct identities and the uniform anchored-row schema.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'results/esel_e0b_v1'
E0_PROTOCOL = ROOT / 'results/esel_e0_v1/protocol.json'


def _read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def test_anchored_counts_use_versioned_accession_forms_and_are_nonzero():
    counts = _read('iedb_counts.json')['counts_by_accession']
    nonzero = 0
    for accession, entry in counts.items():
        form = entry['accession_form_used']
        if entry['total_5_50'] > 0:
            nonzero += 1
            assert '.' in form, (
                f'nonzero counts must come from a versioned accession form: {accession}')
    assert nonzero >= 3, (
        'at least three antigens must carry real IEDB records; a full-zero run '
        'means the bare-vs-versioned accession bug regressed')


def test_anchor_identities_are_the_frozen_ones_not_the_mislabel():
    counts = _read('iedb_counts.json')['counts_by_accession']
    forms = {entry['accession_form_used'] for entry in counts.values()}
    assert 'P37840.1' in forms, 'alpha-synuclein must anchor as P37840.1'
    assert all(not form.startswith('P05877') for form in forms), (
        'P05877 is HIV gp160 (per its frozen identity block); the Phase D '
        'head_antigens_note mislabel must never reappear as alpha-syn')


def test_stop_rule_flip_is_recorded_with_threshold():
    stop = _read('stop_rule.json')
    assert stop['threshold'] == 3
    assert stop['outcome'] == 'continue', (
        'E0b ran precisely because E0 escalated; the rerun must record the flip')
    assert stop['anchored_row_count'] >= 3


def test_all_resolution_statuses_are_explicit():
    mapping = _read('accession_map.json')['mapping']
    allowed = {'anchored', 'unresolved_candidates', 'chain_not_found',
               'peptide_not_in_entity_sequence'}
    assert len(mapping) == 14
    assert all(m.get('status') in allowed for m in mapping.values())
    for instance, entry in mapping.items():
        if entry.get('status') == 'anchored':
            assert entry['anchored_accessions'], f'{instance} anchored but empty'


def test_iedb_access_stays_metadata_level():
    text = json.dumps(_read('iedb_counts.json'))
    assert 'red_line_note' in text
    assert 'linear_sequence' not in json.dumps(_read('atlas_lookup.json')), (
        'ATLAS lookup stores tiers/summaries only, not epitope record sequences')


def test_e0_protocol_stop_rule_was_the_input():
    e0 = json.loads(E0_PROTOCOL.read_text(encoding='utf-8'))
    assert e0['stop_rule_evaluation']['outcome'] == 'escalate_epitope_starvation_risk', (
        'E0b is only licensed to run after an E0 escalation; E0 must still '
        'record the original escalation for provenance')


def test_atlas_lookup_module_defines_track_path():
    # Incident 2026-11-09: atlas_lookup() referenced ATLAS_TRACKS without
    # defining it; the surrounding except Exception silently demoted every
    # verified ATLAS hit to tier 'atlas_error' instead of failing closed.
    # The module must import cleanly and keep the same track cache as the
    # E0 epitope-selection script (results/flex_a2_atlas_plddt_audit_v1/tracks).
    from scripts.esel_e0b_curation_v1 import ATLAS_TRACKS
    expected = ROOT / 'results/flex_a2_atlas_plddt_audit_v1/tracks'
    assert ATLAS_TRACKS == expected
    assert ATLAS_TRACKS.is_dir() and any(ATLAS_TRACKS.glob('*.js')), (
        'ATLAS track cache missing; atlas_lookup would demote all hits '
        'to hit_track_not_cached')
