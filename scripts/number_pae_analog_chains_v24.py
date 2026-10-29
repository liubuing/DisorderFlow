"""Chain-level identity verification of v22 analog cohorts via ANARCII numbering.

TCR pool is numbered with the TCR model (IMGT; beta chains give CDR3-beta 105-117).
VHH pool is numbered with the antibody model; entries with any numbered light chain
are excluded from the single-domain scope; suspected shark VNAR chains are labeled
with the shark model rather than left as generic errors.

Numbering-based roles are not proof of direct peptide contact, family independence
or training-overlap absence; those gates are separate and follow.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/pae_analog_chain_identity_v24'
SOURCE = ROOT / 'data/pae_analog_scope_v22'
V23 = ROOT / 'results/pae_analog_audit_v23'
MIN_LEN, MAX_LEN = 80, 600


def key(seq):
    return 'seq_' + hashlib.sha256(seq.encode()).hexdigest()[:16]


def cdr3(numbering, lo, hi):
    if not numbering:
        return None
    residues = [res for (pos, ins), res in numbering if lo <= pos <= hi]
    seq = ''.join(r for r in residues if r in 'ACDEFGHIKLMNPQRSTVWY')
    return seq or None


def surviving(source_audit, cohort):
    data = read(source_audit)
    return [r['pdb_id'] for r in data[cohort]['rows'] if not r['errors']]


def collect_chains(meta, pdbs):
    """Receptor-side candidate chains per entry; peptide side is 5-50aa and handled separately."""
    meta_all = read(meta)
    chains = {}
    for pdb in pdbs:
        entry = meta_all.get(pdb)
        if entry is None:
            chains[pdb] = []
            continue
        rows = []
        for pe in entry.get('polymer_entities', []):
            seq = (pe.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can') or '').replace('\n', '').replace(' ', '')
            length = len(seq)
            if not MIN_LEN <= length <= MAX_LEN:
                continue
            rows.append({'entity': pe['rcsb_id'], 'seq': seq, 'len': length,
                         'desc': pe['rcsb_polymer_entity']['pdbx_description']})
        chains[pdb] = rows
    return chains


def number_pool(pool, seq_type, mode='accuracy'):
    """Resume-able numbering; batch files frozen so reruns only fill gaps.

    The antibody pool stores both the raw IMGT numbering and a Chothia
    conversion so the VHH CDR3 can be read under either scheme.
    """
    outdir = OUT / ('numbering_' + seq_type)
    outdir.mkdir(parents=True, exist_ok=True)
    model = None
    labels = {}
    items = sorted(pool.items())
    for start in range(0, len(items), 32):
        path = outdir / f'batch_{start:04d}.json'
        if not path.exists():
            if model is None:
                from anarcii import Anarcii
                model = Anarcii(seq_type=seq_type, mode=mode, batch_size=32, cpu=True, ncpu=2, verbose=False)
            batch = dict(items[start:start + 32])
            numbered = model.number(batch)
            records = {}
            for name, r in numbered.items():
                records[batch[name]] = {k: r.get(k) for k in
                                        ('chain_type', 'score', 'query_start', 'query_end', 'error', 'numbering')}
            if seq_type == 'antibody':
                chothia = model.to_scheme('chothia')
                for name, r in chothia.items():
                    if batch[name] in records:
                        records[batch[name]]['chothia_numbering'] = r.get('numbering')
            freeze(path, {'records': records, 'seq_type': seq_type})
        labels.update(read(path)['records'])
        print(f'[{seq_type}] numbered {len(labels)}/{len(items)}', flush=True)
    return labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v23_out = V23 / 'audit.json'
    protocol = {
        'classification': 'chain_identity_numbering_gate_not_model_validation',
        'source_sha256': {name: digest(SOURCE / name) for name in
                          ('tcr_candidates.json', 'tcr_meta.json', 'nano_passed.json', 'nano_meta.json')},
        'v23_audit_sha256': digest(v23_out),
        'input_scope': 'entries passing v23 offline input audit; receptor-side chains 80-600 aa from rcsb metadata',
        'tcr_model': 'ANARCII seq_type=tcr accuracy CPU IMGT; CDR3-beta = IMGT positions 105-117',
        'vhh_model': 'ANARCII seq_type=antibody accuracy CPU; Chothia H3 = 95-102 recorded alongside IMGT CDR3 105-117',
        'vhh_single_domain_gate': 'no numbered K/L chain in entry; shark model labels antibody-error chains',
        'tcr_gate': 'at least one numbered beta chain with extractable CDR3-beta; alpha/gamma/delta recorded',
        'boundary': 'numbering roles are not direct-contact, homology-isolation, provenance or family-independence evidence',
        'script_sha256': digest(Path(__file__)),
    }
    freeze(OUT / 'protocol.json', protocol)

    pdbs = {
        'tcr': surviving(v23_out, 'tcr'),
        'vhh': surviving(v23_out, 'vhh'),
    }
    chains = {
        'tcr': collect_chains(SOURCE / 'tcr_meta.json', pdbs['tcr']),
        'vhh': collect_chains(SOURCE / 'nano_meta.json', pdbs['vhh']),
    }

    pools = {}
    for cohort, rows in chains.items():
        pool = {}
        for pdb, entries in rows.items():
            for row in entries:
                pool[key(row['seq'])] = row['seq']
        pools[cohort] = pool
        print(f'{cohort}: {len(pdbs[cohort])} entries, {len(pool)} unique chains', flush=True)

    labels = {
        'tcr': number_pool(pools['tcr'], 'tcr'),
        'vhh': number_pool(pools['vhh'], 'antibody'),
    }

    # Shark labeling for VHH-pool sequences the antibody model could not number.
    unresolved = [s for s, r in labels['vhh'].items() if r.get('error') or not r.get('chain_type')]
    shark_labels = {}
    if unresolved:
        shark = {key(seq): seq for seq in unresolved}
        shark_labels = number_pool(shark, 'shark')
        print(f'shark relabel: {len(shark_labels)} chains', flush=True)

    shark_by_seq = dict(shark_labels)

    output = {}
    for cohort in ('tcr', 'vhh'):
        rows = []
        for pdb in pdbs[cohort]:
            entry_rows = []
            for row in chains[cohort][pdb]:
                label = labels[cohort].get(row['seq'])
                record = {**row, 'numbering': label}
                if cohort == 'vhh' and label and (label.get('error') or not label.get('chain_type')):
                    shark = shark_by_seq.get(row['seq'])
                    if shark is not None:
                        record['shark_numbering'] = {k: shark.get(k) for k in ('chain_type', 'score', 'error')}
                if cohort == 'tcr':
                    record['cdr3_beta_imgt'] = (cdr3(label.get('numbering'), 105, 117)
                                                if label and label.get('chain_type') == 'B' and not label.get('error') else None)
                if cohort == 'vhh':
                    record['cdr3_imgt'] = (cdr3(label.get('numbering'), 105, 117)
                                           if label and label.get('chain_type') == 'H' and not label.get('error') else None)
                    record['chothia_h3'] = (cdr3(label.get('chothia_numbering'), 95, 102)
                                            if label and label.get('chothia_numbering') else None)
                entry_rows.append(record)
            numbered = [r for r in entry_rows if r['numbering'] and not r['numbering'].get('error') and r['numbering'].get('chain_type')]
            if cohort == 'tcr':
                betas = [r for r in numbered if r['numbering']['chain_type'] == 'B' and r.get('cdr3_beta_imgt')]
                alphas = [r['entity'] for r in numbered if r['numbering']['chain_type'] == 'A']
                gate = {'passes': bool(betas), 'beta_entities': [r['entity'] for r in betas],
                        'alpha_entities': alphas,
                        'cdr3_beta': [r['cdr3_beta_imgt'] for r in betas],
                        'fail_reason': None if betas else 'no_numbered_beta_with_cdr3'}
            else:
                heavies = [r for r in numbered if r['numbering']['chain_type'] == 'H']
                lights = [r['entity'] for r in numbered if r['numbering']['chain_type'] in ('K', 'L')]
                shark_hits = [r['entity'] for r in entry_rows
                              if r.get('shark_numbering') and not r['shark_numbering'].get('error')
                              and r['shark_numbering'].get('chain_type') == 'S']
                if heavies and not lights and heavies[0].get('cdr3_imgt'):
                    gate = {'passes': True, 'heavy_entities': [r['entity'] for r in heavies],
                            'light_entities': lights, 'shark_entities': shark_hits,
                            'cdr3_imgt': [r['cdr3_imgt'] for r in heavies],
                            'chothia_h3': [r['chothia_h3'] for r in heavies],
                            'fail_reason': None}
                else:
                    reason = ('light_chain_numbered' if lights else
                              'shark_vnar_not_camelid_vhh' if shark_hits and not heavies else
                              'no_numbered_heavy_with_cdr3')
                    gate = {'passes': False, 'heavy_entities': [r['entity'] for r in heavies],
                            'light_entities': lights, 'shark_entities': shark_hits,
                            'fail_reason': reason}
            rows.append({'pdb': pdb, 'chains': entry_rows, 'gate': gate})
        passed = [r['pdb'] for r in rows if r['gate']['passes']]
        fail_reasons = {}
        for r in rows:
            if not r['gate']['passes']:
                fail_reasons[r['gate']['fail_reason']] = fail_reasons.get(r['gate']['fail_reason'], 0) + 1
        output[cohort] = {'rows': rows, 'passed_pdbs': passed, 'fail_reasons': fail_reasons,
                          'counts': {'input_entries': len(rows), 'passed': len(passed)}}

    freeze(OUT / 'audit.json', output)
    summary = {c: output[c]['counts'] | {'fail_reasons': output[c]['fail_reasons']} for c in output}
    freeze(OUT / 'summary.json', summary)
    freeze(OUT / 'completion.json', {'status': 'completed',
                                     'model_training_started': False, 'af2_started': False,
                                     'audit_sha256': digest(OUT / 'audit.json'),
                                     'protocol_sha256': digest(OUT / 'protocol.json')})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
