"""Check primary paper identity, active upload contents, and source integrity."""
import hashlib
import json
from pathlib import Path
import zipfile

try:
    from .prepare_ecls_submission import portable_copy
except ImportError:
    from prepare_ecls_submission import portable_copy

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def archive_errors(root, line):
    errors = []
    path = root/line['reviewer_package']
    summary = read(root/'release/ecls_v1/submission_package_manifest.json')
    if sha256(path) != summary['sha256'] or path.stat().st_size != summary['bytes']:
        errors.append('Reviewer archive summary is stale')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            errors.append('Unsafe or duplicate archive entry')
        manifest = json.loads(archive.read('MANIFEST.json'))
        expected = {e['path'] for e in manifest['files']} | {'MANIFEST.json'}
        if set(names) != expected:
            errors.append('Reviewer archive file set differs from inner manifest')
        for entry in manifest['files']:
            data = archive.read(entry['path'])
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                errors.append('Reviewer archive checksum mismatch: '+entry['path'])
        current = {
            line['manuscript']:'manuscript/ECLS_MANUSCRIPT.md',
            line['manuscript_pdf']:'manuscript/ECLS_MANUSCRIPT.pdf',
            line['supplement']:'manuscript/ECLS_SUPPLEMENT.md',
            'publication/ECLS_SUPPLEMENT.pdf':'manuscript/ECLS_SUPPLEMENT.pdf',
            'scripts/verify_ecls_saved_results.py':'verify_ecls_saved_results.py',
            'scripts/plot_ecls_narrative.py':'plot_ecls_narrative.py',
            'release/ecls_v1/figure_source_data.json':'figure_source_data.json',
        }
        for figure in ('ecls_study_design','ecls_frozen_evidence'):
            for suffix in ('png','pdf','svg'):
                current[f'publication/figures/{figure}.{suffix}'] = f'manuscript/figures/{figure}.{suffix}'
        for local, nested in current.items():
            if archive.read(nested) != (root/local).read_bytes():
                errors.append('Stale manuscript or figure inside archive: '+nested)
        ledger = json.loads(archive.read('SOURCE_PROVENANCE.json'))
        for source, entry in ledger['sources'].items():
            if sha256(root/source) != entry['original_sha256']:
                errors.append('Original source digest differs from provenance: '+source)
            public = archive.read(entry['public_path'])
            if hashlib.sha256(public).hexdigest() != entry['public_sha256']:
                errors.append('Portable derivative digest differs from provenance: '+source)
            if 'path_changes' in entry:
                changes = []
                if portable_copy(read(root/source), changes) != json.loads(public) or changes != entry['path_changes']:
                    errors.append('Undocumented change in portable evidence: '+source)
    return errors


def publication_blockers(submission):
    blockers = []
    if any(not a['english_spelling_confirmed'] or not a['affiliation_confirmed'] for a in submission['authors']):
        blockers.append('Author English spellings and affiliation mapping')
    for field in ('author_contributions','funding','conflicts_of_interest'):
        if submission.get(field) is None or submission.get(field) == '':
            blockers.append(field)
    for field in ('all_authors_approve_submission','all_authors_approve_public_archive',
                  'related_manuscripts_status_confirmed','manuscript_data_license_confirmed'):
        if submission.get(field) is not True:
            blockers.append(field)
    if not submission.get('remote_publication_confirmed') or not submission.get('remote_doi') or not submission.get('remote_record_url'):
        blockers.append('Published archive, DOI and downloaded checksum verification')
    return blockers


def validate(root=ROOT):
    root = Path(root)
    errors = []
    base = root/'release/zenodo_v1'
    line_path = root/'release/ecls_v1/publication_line.json'
    line = read(line_path)
    metadata = read(base/'zenodo_metadata.json')
    submission = read(root/line['submission_metadata'])
    manifest = read(base/'UPLOAD_MANIFEST.json')
    manuscript = (root/line['manuscript']).read_text(encoding='utf-8-sig')
    title = manuscript.splitlines()[0].removeprefix('# ')
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
    authors = sorted(submission['authors'],key=lambda a:a['order'])
    if [a['name'] for a in metadata['creators']] != [a['deposit_name'] for a in authors]:
        errors.append('Deposit author order differs from author metadata')
    author_line = ', '.join(a['name_en'] + (' (corresponding author)' if a['corresponding'] else '') for a in authors)
    if author_line not in manuscript:
        errors.append('Manuscript author order differs from author metadata')
    if any(a['email'] not in manuscript for a in authors if a['corresponding']):
        errors.append('Corresponding email differs from author metadata')
    out = root/manifest['upload_directory']
    expected = {e['path'] for e in manifest['files']}
    actual = {p.name for p in out.iterdir()}
    if actual!=expected:
        errors.append(f'Unexpected/missing upload entries: {sorted(actual^expected)}')
    sources = {e['source'] for e in manifest['files']}
    for key in ('manuscript','manuscript_pdf','supplement','frozen_scope','primary_decision','public_primary_results','reviewer_package'):
        if line[key] not in sources:
            errors.append(f'Missing primary ECLS source: {key}')
    if line['primary_results'] in sources:
        errors.append('Raw machine-specific results must not replace the portable public derivative')
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
    errors.extend(archive_errors(root,line))
    blockers = publication_blockers(submission)
    if blockers and manifest.get('publication_ready'):
        errors.append('Publication marked ready while author declarations or remote verification are missing')
    return {'status':'valid' if not errors else 'invalid','primary_line':line['primary_line'],
            'files_checked':len(manifest['files']),'remote_publication_confirmed':submission['remote_publication_confirmed'],
            'publication_ready':not errors and not blockers,'publication_blockers':blockers,'errors':errors}


if __name__=='__main__':
    result = validate()
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(result['errors']))
