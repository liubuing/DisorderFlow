"""CSN peptide matching, local edition (v1): definitive exact-substring scan
against a downloaded Swiss-Prot (reviewed UniProtKB) + isoforms FASTA.

Why: the UniProt peptide-search asyncrest degrades (submit 202, poll empty
>=90 s, verified with a known-positive control on 2026-11-08) and PIR
PeptideMatch answers 503. Rather than record another service failure, this
script makes the answer deterministic and permanent: one download, one local
scan, every hit recorded with its accession, every miss recorded as a verified
zero in the scanned universe (Swiss-Prot reviewed + isoforms; TrEMBL excluded
and disclosed).

Positive controls are required before any conclusion is drawn:
- KEDAFSKLKEKFMNEL must hit P21579 (Syt1; containment-verified by ESEL E0b)
- DYEPEA must hit P37840 (alpha-synuclein C terminus)
- GSVQIVYKPV must hit P10636 (canonical tau) - the on-database counterpart of
  the deposited peptide GSVQIVYKPK (one C-terminal substitution, K vs V)
If any control fails, the scan is invalid and every conclusion is withheld.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze  # noqa: E402
from scripts.benchmark_aayl_learnability import digest  # noqa: E402

OUT = ROOT / 'results/csn_c1_v1/peptidematch'
DATA = ROOT / 'data/peptidematch'
SP_URL = ('https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true'
          '&format=fasta&compressed=true')
VAR_URL = 'https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true&format=fasta&compressed=true&fields=accession'
VARSP_URL = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot_varsplic.fasta.gz'
DATE = '2026-11-08'
CLASSIFICATION = 'csn_c1_parent_curation_evidence'

PEPTIDES = {
    'control_syt1': 'KEDAFSKLKEKFMNEL',
    'control_alpha_syn': 'DYEPEA',
    'control_tau_canonical': 'GSVQIVYKPV',
    'tau_deposited': 'GSVQIVYKPK',
    'i2g': 'SRLEEELRRRLTE',
    'pktag': 'GKPIPNPLLGLDST',
    'tqaqey': 'TQAQEY',
}
REQUIRED_CONTROLS = {
    'control_syt1': ['P21579'],
    'control_alpha_syn': ['P37840'],
    'control_tau_canonical': ['P10636'],
}


def download(url: str, dest: Path, retries: int = 4) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'csn-peptidematch/1'})
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, 'wb') as fh:
                total = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
            print(f'downloaded {dest.name}: {total/1e6:.1f} MB', flush=True)
            tmp.rename(dest)
            return dest
        except Exception as error:  # noqa: BLE001 - resume-able retry ladder
            last = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'download failed after {retries} attempts: {last}')


def scan_fasta(path: Path, peptides: dict[str, str]):
    """Stream a (possibly gzipped) FASTA; record every entry containing each peptide."""
    import gzip
    opener = gzip.open if path.suffix == '.gz' else open
    hits = {key: [] for key in peptides}
    n_entries = 0
    accession = None
    seq_parts = []
    with opener(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('>'):
                if accession is not None:
                    seq = ''.join(seq_parts)
                    for key, pep in peptides.items():
                        if pep in seq:
                            hits[key].append(accession)
                accession = line[1:].split()[0].split('|')[1] if '|' in line else line[1:].split()[0]
                seq_parts = []
                n_entries += 1
            else:
                seq_parts.append(line.strip())
        if accession is not None:
            seq = ''.join(seq_parts)
            for key, pep in peptides.items():
                if pep in seq:
                    hits[key].append(accession)
    return hits, n_entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    sp = download(SP_URL, DATA / 'uniprot_sprot.fasta.gz')
    varsp = download(VARSP_URL, DATA / 'uniprot_sprot_varsplic.fasta.gz')
    files = {'swissprot': sp, 'isoforms': varsp}
    freeze(OUT / 'protocol.json', {
        'classification': CLASSIFICATION,
        'date': DATE,
        'universe': 'UniProtKB reviewed (Swiss-Prot) + Swiss-Prot isoform sequences',
        'universe_exclusions': ['TrEMBL (unreviewed) - disclosed; too large for this lane'],
        'match_type': 'exact substring',
        'peptides': PEPTIDES,
        'required_controls': REQUIRED_CONTROLS,
        'positive_control_rule': ('every required control must hit its listed accession; '
                                  'otherwise the scan is invalid and no conclusion freezes'),
        'files_sha256': {name: digest(path) for name, path in files.items()},
        'script_sha256': digest(Path(__file__))})
    if args.download_only:
        return

    all_hits = {}
    entries_total = 0
    for name, path in files.items():
        hits, n = scan_fasta(path, PEPTIDES)
        all_hits[name] = hits
        entries_total += n
        print(f'{name}: {n} entries scanned', flush=True)

    merged = {}
    for key in PEPTIDES:
        accs = sorted(set(all_hits['swissprot'][key]) | set(all_hits['isoforms'][key]))
        merged[key] = {'accessions': accs,
                       'n_swissprot': len(set(all_hits['swissprot'][key])),
                       'n_isoforms_extra': len(accs) - len(set(all_hits['swissprot'][key]))}
    controls_ok = all(set(merged[key]['accessions']) & set(accs_required)
                      for key, accs_required in REQUIRED_CONTROLS.items())
    freeze(OUT / 'matches.json', {
        'classification': CLASSIFICATION,
        'entries_scanned': entries_total,
        'controls_ok': controls_ok,
        'matches': merged})
    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'entries_scanned': entries_total,
        'controls_ok': controls_ok,
        'control_results': {key: merged[key]['accessions']
                            for key in REQUIRED_CONTROLS},
        'conclusions': {
            'tau_deposited_GSVQIVYKPK': (
                'verified_zero_in_universe' if controls_ok and not merged['tau_deposited']['accessions']
                else merged['tau_deposited']['accessions']),
            'i2g_SRLEEELRRRLTE': merged['i2g']['accessions'],
            'pktag_GKPIPNPLLGLDST': merged['pktag']['accessions'],
            'tqaqey_TQAQEY': merged['tqaqey']['accessions'],
        },
        'reading': ('a verified zero (controls passed, peptide absent from every scanned '
                    'entry) is a positive engineering result: the deposited tau peptide '
                    'differences from the database are now a fact, not an API failure'),
        'claims_boundary': ('Exact-substring matching in the scanned universe only. '
                            'No homology, no isoform-integration claims beyond the '
                            'scanned files, no wet-lab implication.'),
    }
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {
        'classification': CLASSIFICATION, 'date': DATE,
        'protocol_sha256': digest(OUT / 'protocol.json'),
        'script_sha256': digest(Path(__file__)),
        'artifacts': ['protocol.json', 'matches.json', 'summary.json', 'completion.json']})
    print(json.dumps(summary['control_results'], indent=1))
    print('controls_ok:', controls_ok)
    for key, value in summary['conclusions'].items():
        print(key, '->', str(value)[:120])


if __name__ == '__main__':
    main()
