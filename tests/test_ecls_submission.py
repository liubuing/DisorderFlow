"""Integrity regressions for the saved-score manuscript package."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_ecls_submission import portable_copy
from scripts.verify_ecls_saved_results import read, verify_scores, verify_ranking, verify_manifest
from scripts.validate_publication_alignment import publication_blockers, publication_state_errors

ROOT = Path(__file__).resolve().parents[1]


# 可移植拷贝：只允许改写路径类字符串并记录原串 sha256 溯源，其余数值/文本逐位不变且不泄漏私有路径。
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


# 冻结产物核对：已保存结果的置换对照数量（119/200 口径）与推断单元数必须与冻结时一致，
# 防止重算结果悄悄偏离已发布数字。
def test_saved_adaptation_accepts_finite_permutation_controls():
    data = read(ROOT/'release/ecls_v1/prepared/adaptation_results_portable.json')
    counts = [len(r['composition_shuffles']) for r in data['results']]
    assert counts.count(119) == 2 and counts.count(200) == 44
    result = verify_scores(data,2281,read(ROOT/'release/ecls_v1/prepared/adaptation_cluster_assignments.json'))
    assert result['n_inference_units'] == 46


# 符号篡改检测：对已保存分数做微小数值改动必须触发算术一致性错误，防止结果文件被事后修改。
def test_score_sign_corruption_is_rejected():
    data = read(ROOT/'results/publication/h3_ecls_temporal_final_v1/results.json')
    data['results'][0]['native']['epitope_delta_nll'] += .1
    with pytest.raises(ValueError,match='arithmetic'):
        verify_scores(data,2309)


# 双重计数检测：推断单元中重复的 record ID 必须触发 partition 错误，防止同一记录被重复计入证据。
def test_cluster_double_counting_is_rejected():
    data = read(ROOT/'results/publication/h3_ecls_temporal_final_v1/results.json')
    data['inference_units'][0]['record_ids'].append(data['inference_units'][0]['record_ids'][0])
    with pytest.raises(ValueError,match='partition'):
        verify_scores(data,2309)


# 排名篡改检测：改写候选指标导致 native 排名改变必须报错，池数量仍须与冻结一致（63 池）。
def test_candidate_rank_change_is_rejected():
    data = read(ROOT/'results/publication/h3_candidate_reranking_dev_v1/results.json')
    calibration = read(ROOT/'results/publication/h3_generator_calibration_v1/analysis.json')
    assert verify_ranking(data,calibration)['n_pools'] == 63
    data['results'][0]['metrics']['ecls_nnr'] = .123
    with pytest.raises(ValueError,match='native rank'):
        verify_ranking(data,calibration)


# 清单校验：字节被回滚（sha256 失配）或出现清单外文件都必须报错——守护稿件包与 MANIFEST 严格一致。
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


# 申报独立：资助/利益冲突/存档授权等声明必须由显式申报提供，不得从合格内容推断补齐——
# 防止发表合规项被静默"自动通过"。
def test_declarations_are_not_inferred_from_valid_content():
    blockers = publication_blockers(read(ROOT/'release/ecls_v1/submission_metadata.json'))
    assert 'funding' in blockers
    assert 'conflicts_of_interest' in blockers
    assert 'all_authors_approve_public_archive' in blockers


# DOI 状态机：预留 DOI 不因稿件正文提及"已存档"而升级为已发表——文字不得替代事实确认。
def test_reserved_doi_cannot_be_promoted_to_publication_by_prose():
    submission = {'remote_publication_confirmed':False, 'reserved_doi':'10.5281/zenodo.23037055'}
    documents = {'manuscript':'The public archive is deposited on Zenodo (DOI: 10.5281/zenodo.23037055).',
                 'cover':'The corresponding immutable code revision is tagged in the source repository at publication.'}
    assert len(publication_state_errors(submission,documents)) == 2


# 一致放行：计划中的 DOI 与"存档待完成"的表述互相一致时不得报错，防止状态机误报。
def test_planned_doi_and_pending_release_are_consistent():
    text = 'Planned Zenodo DOI: 10.5281/zenodo.23037055; public release and remote file verification remain pending.'
    assert publication_state_errors({'remote_publication_confirmed':False},{'manuscript':text}) == []
