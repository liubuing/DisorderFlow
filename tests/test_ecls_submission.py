"""Integrity regressions for the saved-score manuscript package."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_ecls_submission import portable_copy
from scripts.verify_ecls_saved_results import read, verify_scores, verify_ranking, verify_manifest
from scripts.validate_publication_alignment import publication_blockers

ROOT = Path(__file__).resolve().parents[1]


def test_portable_copy_changes_only_paths_and_records_provenance():
    source = {'score': .172281, 'n': 31, 'claim': 'native versus shuffle',
              'nested': [r'C:\biological\DisorderFlow\results\evidence.json',
                         r'C:\Users\private\python.exe']}
    original = copy.deepcopy(source)
    changes = []
    public = portable_copy(source, changes)
    assert source == original
    assert public == {**original,'nested':['results/evidence.json','python']}
    assert len(changes) == 2
    assert changes[0]['original_string_sha256'] == hashlib.sha256(source['nested'][0].encode()).hexdigest()
    assert 'private' not in json.dumps(changes)


def test_saved_adaptation_accepts_finite_permutation_controls():
    data = read(ROOT/'release/ecls_v1/prepared/adaptation_results_portable.json')
    counts = [len(r['composition_shuffles']) for r in data['results']]
    assert counts.count(119) == 2 and counts.count(200) == 44
    result = verify_scores(data,2281,read(ROOT/'release/ecls_v1/prepared/adaptation_cluster_assignments.json'))
    assert result['n_inference_units'] == 46


def test_score_sign_corruption_is_rejected():
    data = read(ROOT/'results/publication/h3_ecls_temporal_final_v1/results.json')
    data['results'][0]['native']['epitope_delta_nll'] += .1
    with pytest.raises(ValueError,match='arithmetic'):
        verify_scores(data,2309)


def test_cluster_double_counting_is_rejected():
    data = read(ROOT/'results/publication/h3_ecls_temporal_final_v1/results.json')
    data['inference_units'][0]['record_ids'].append(data['inference_units'][0]['record_ids'][0])
    with pytest.raises(ValueError,match='partition'):
        verify_scores(data,2309)


def test_candidate_rank_change_is_rejected():
    data = read(ROOT/'results/publication/h3_candidate_reranking_dev_v1/results.json')
    calibration = read(ROOT/'results/publication/h3_generator_calibration_v1/analysis.json')
    assert verify_ranking(data,calibration)['n_pools'] == 63
    data['results'][0]['metrics']['ecls_nnr'] = .123
    with pytest.raises(ValueError,match='native rank'):
        verify_ranking(data,calibration)


def test_manifest_detects_stale_bytes_and_unexpected_files(tmp_path):
    body = b'current manuscript'
    path = tmp_path/'manuscript.md';path.write_bytes(body)
    manifest = {'files':[{'path':'manuscript.md','bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}]}
    (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
    assert verify_manifest(tmp_path) == 1
    path.write_bytes(b'old manuscript')
    with pytest.raises(ValueError,match='Checksum'):
        verify_manifest(tmp_path)
    path.write_bytes(body)
    (tmp_path/'old.zip').write_bytes(b'old archive')
    with pytest.raises(ValueError,match='unexpected'):
        verify_manifest(tmp_path)


def test_declarations_are_not_inferred_from_valid_content():
    blockers = publication_blockers(read(ROOT/'release/ecls_v1/submission_metadata.json'))
    assert 'funding' in blockers
    assert 'conflicts_of_interest' in blockers
    assert 'all_authors_approve_public_archive' in blockers
