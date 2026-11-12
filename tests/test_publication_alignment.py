"""发布对齐校验的契约测试：守护 Zenodo 上传构建在身份不一致时 fail-closed 且不动暂存区、
旧产物归档不丢失，以及当前仓库发布对齐无错误——防止发布包内容与稿件身份错位。"""
import json
from pathlib import Path

import pytest

from scripts import build_zenodo_upload as builder
from scripts.validate_publication_alignment import validate


# 辅助函数：搭建最小发布目录结构（发布线清单、稿件草稿、Zenodo 元数据）。
def prepare(tmp_path):
    line = {'primary_line':'ecls','release_id':'disorderflow-ecls-v1',
            'manuscript':'publication/MANUSCRIPT_DRAFT.md','manuscript_title':'ECLS title'}
    for folder in ('release/ecls_v1','release/zenodo_v1/upload','publication'):
        (tmp_path/folder).mkdir(parents=True,exist_ok=True)
    (tmp_path/builder.LINE).write_text(json.dumps(line))
    (tmp_path/'publication/MANUSCRIPT_DRAFT.md').write_text('# ECLS title\n')
    (tmp_path/'release/zenodo_v1/zenodo_metadata.json').write_text(json.dumps({
        'title':'ECLS title (ECLS v1 reproducibility package)'}))


# 身份守卫：元数据标题与发布线身份不符时构建必须报错，且暂存区既有文件字节原样不动——
# 防止错误身份的产物混入上传或覆盖历史文件。
def test_mixed_identity_rejected_before_staging_changes(tmp_path,monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(builder,'INCLUDE',['publication/MANUSCRIPT_DRAFT.md'])
    metadata = tmp_path/'release/zenodo_v1/zenodo_metadata.json'
    metadata.write_text(json.dumps({'title':'PAE surrogate'}))
    marker = tmp_path/'release/zenodo_v1/upload/old.pdf'
    marker.write_bytes(b'history')
    with pytest.raises(ValueError,match='identity'):
        builder.build(tmp_path)
    assert marker.read_bytes()==b'history'


# 归档无损失：旧 PAE 文件移入 archive（逐字节相同）后上传区只剩稿件——防止构建静默丢弃历史产物。
def test_old_pae_file_archived_without_loss(tmp_path,monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(builder,'INCLUDE',['publication/MANUSCRIPT_DRAFT.md'])
    marker = tmp_path/'release/zenodo_v1/upload/af2_pae.pdf'
    marker.write_bytes(b'historical PAE manuscript')
    builder.build(tmp_path)
    assert not marker.exists()
    archived = list((tmp_path/'release/zenodo_v1/archive').glob('*/af2_pae.pdf'))
    assert len(archived)==1
    assert archived[0].read_bytes()==b'historical PAE manuscript'
    assert {p.name for p in marker.parent.iterdir()}=={'MANUSCRIPT_DRAFT.md'}


# 仓库现状：对当前仓库执行完整对齐校验必须零错误，防止发布线声明与实际产物脱节。
# 干净的 CI checkout 不含 dist/ 与 upload/ 本地暂存区（均被 .gitignore 排除），
# 此时跳过；tracked 源文件的完整性由 test_release_lineage.py 覆盖。
def test_current_publication_alignment():
    root = Path(__file__).resolve().parents[1]
    if not (root/'release/zenodo_v1/upload').is_dir() or not (root/'dist').exists():
        pytest.skip('local publication staging absent (clean checkout)')
    assert validate(root)['errors']==[]
