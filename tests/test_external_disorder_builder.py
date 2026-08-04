import csv
import pickle

import numpy as np

from scripts.build.build_external_disorder_lookup import parse_scores


def test_parse_metapredict_scores(tmp_path):
    path = tmp_path / 'scores.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample-a', '0.1', '0.8'])
    profiles = parse_scores(path)
    assert np.allclose(profiles['sample-a'], [0.1, 0.8])


def test_parse_metapredict_v3_scores_checks_sequence(tmp_path):
    path = tmp_path / 'scores.csv'
    path.write_text('sample-a, AC, 0.1, 0.8\n', encoding='utf-8')
    profiles = parse_scores(path, {'sample-a': 'AC'})
    assert np.allclose(profiles['sample-a'], [0.1, 0.8])


def test_external_artifact_split_is_not_implicit_val(tmp_path):
    artifact = {'schema_version': 2, 'split': 'external_blind', 'profiles': {}}
    path = tmp_path / 'artifact.pkl'
    path.write_bytes(pickle.dumps(artifact))
    assert pickle.loads(path.read_bytes())['split'] == 'external_blind'
