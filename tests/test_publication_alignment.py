import json
from pathlib import Path

import pytest

from scripts import build_zenodo_upload as builder
from scripts.validate_publication_alignment import validate


def prepare(tmp_path):
    line = {'primary_line':'ecls','release_id':'disorderflow-ecls-v1',
            'manuscript':'publication/MANUSCRIPT_DRAFT.md','manuscript_title':'ECLS title'}
    for folder in ('release/ecls_v1','release/zenodo_v1/upload','publication'):
        (tmp_path/folder).mkdir(parents=True,exist_ok=True)
    (tmp_path/builder.LINE).write_text(json.dumps(line))
    (tmp_path/'publication/MANUSCRIPT_DRAFT.md').write_text('# ECLS title\n')
    (tmp_path/'release/zenodo_v1/zenodo_metadata.json').write_text(json.dumps({
        'title':'ECLS title (ECLS v1 reproducibility package)'}))


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


def test_current_publication_alignment():
    assert validate(Path(__file__).resolve().parents[1])['errors']==[]
