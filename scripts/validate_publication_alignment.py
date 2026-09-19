"""Check primary paper identity, active upload contents, and source integrity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def validate(root=ROOT):
    root = Path(root)
    errors = []
    base = root/'release/zenodo_v1'
    line_path = root/'release/ecls_v1/publication_line.json'
    line = read(line_path)
    metadata = read(base/'zenodo_metadata.json')
    manifest = read(base/'UPLOAD_MANIFEST.json')
    title = (root/line['manuscript']).read_text(encoding='utf-8-sig').splitlines()[0].removeprefix('# ')
    if line['primary_line']!='ecls' or manifest.get('primary_line')!='ecls':
        errors.append('Primary line must be ECLS')
    if title!=line['manuscript_title'] or metadata['title']!=title+' (ECLS v1 reproducibility package)':
        errors.append('Manuscript/deposit title mismatch')
    if manifest.get('release_id')!=line['release_id'] or manifest.get('manuscript')!=line['manuscript']:
        errors.append('Manifest release/manuscript mismatch')
    if manifest.get('publication_line_sha256')!=sha256(line_path):
        errors.append('Publication identity hash mismatch')
    if manifest.get('metadata_sha256')!=sha256(base/'zenodo_metadata.json'):
        errors.append('Deposit metadata hash mismatch')
    out = root/manifest['upload_directory']
    expected = {e['path'] for e in manifest['files']}
    actual = {p.name for p in out.iterdir()}
    if actual!=expected:
        errors.append(f'Unexpected/missing upload entries: {sorted(actual^expected)}')
    sources = {e['source'] for e in manifest['files']}
    for key in ('manuscript','frozen_scope','primary_decision','primary_results','reviewer_package'):
        if line[key] not in sources:
            errors.append(f'Missing primary ECLS source: {key}')
    for entry in manifest['files']:
        if 'pae' in entry['path'].lower() or 'af2_' in entry['path'].lower():
            errors.append(f'PAE artifact mixed into ECLS upload: {entry["path"]}')
        for path in (out/entry['path'],root/entry['source']):
            if not path.is_file():
                errors.append(f'Missing file: {path}')
            elif path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
                errors.append(f'Stale upload/source hash: {path}')
    decision = read(root/line['primary_decision'])
    if sha256(root/decision['source_result'])!=decision['source_result_sha256']:
        errors.append('Frozen ECLS result checksum mismatch')
    if decision.get('rerun_permitted') is not False:
        errors.append('Frozen final rerun policy changed')
    return {'status':'valid' if not errors else 'invalid','primary_line':line['primary_line'],
            'files_checked':len(manifest['files']),'remote_publication_confirmed':False,'errors':errors}


if __name__=='__main__':
    result = validate()
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(result['errors']))
