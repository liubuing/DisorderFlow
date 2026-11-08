"""One-off patcher: csn_c1_enrich v6 -> v7 (label v2 amendment + AFDB fallback)."""
import ast
import re
from pathlib import Path

P = Path(__file__).resolve().parents[2] / 'scripts/csn_c1_enrich_v1.py'
t = P.read_text(encoding='utf-8')

# 0. imports: calc_dihedral
t = t.replace('from Bio.PDB import MMCIFParser, PDBParser, SASA',
              'from Bio.PDB import MMCIFParser, PDBParser, SASA\n'
              'from Bio.PDB.vectors import calc_dihedral')

# 1. AFDB template + version fallback
t = t.replace("AFDB = 'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb'",
              "AFDB_TPL = 'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{ver}.pdb'\n"
              "AFDB_VERSIONS = ('v6', 'v5', 'v4', 'v3', 'v2')")

old = """def fetch_afdb(acc):
    def produce():
        # Local AFDB cache first (disorder line), then the v6 download.
        local = LOCAL_AFDB / f'AF-{acc}-F1-model_v6.pdb'
        if local.exists():
            return local.read_text(encoding='utf-8', errors='replace')
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    AFDB.format(acc=acc), headers={'User-Agent': 'csn-c1-enrich/1'})
                with urllib.request.urlopen(request, timeout=300) as response:
                    return response.read().decode('utf-8', errors='replace')
            except Exception as error:  # noqa: BLE001 - big models need patience
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))"""
new = """def fetch_afdb(acc):
    def produce():
        # Local AFDB caches first (disorder line), then version fallback v6..v2.
        for ver in AFDB_VERSIONS:
            local = LOCAL_AFDB / f'AF-{acc}-F1-model_{ver}.pdb'
            if local.exists():
                return local.read_text(encoding='utf-8', errors='replace')
        last = None
        for ver in AFDB_VERSIONS:
            for attempt in range(2):
                try:
                    request = urllib.request.Request(
                        AFDB_TPL.format(acc=acc, ver=ver),
                        headers={'User-Agent': 'csn-c1-enrich/1'})
                    with urllib.request.urlopen(request, timeout=300) as response:
                        return response.read().decode('utf-8', errors='replace')
                except Exception as error:  # noqa: BLE001 - version/transport fallback
                    last = error
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(f'all AFDB versions failed for {acc}: {last}')"""
assert old in t, 'fetch_afdb block not found'
t = t.replace(old, new)

# 2. fetch_nmr_ensemble_rmsf returns the ensemble path
old2 = """            return {'source': 'nmr_fetched', 'pdb': entry['pdb'],
                    'sequence': entry['sequence'], 'rmsf': entry['rmsf']}"""
new2 = """            return {'source': 'nmr_fetched', 'pdb': entry['pdb'],
                    'sequence': entry['sequence'], 'rmsf': entry['rmsf'],
                    'path': str(path)}"""
assert old2 in t, 'nmr return block not found'
t = t.replace(old2, new2)

# 3. dihedral stability function (insert before def main)
v2fn = '''def circular_r(angles):
    a = np.asarray([x for x in angles if x is not None], dtype=float)
    if len(a) < 3:
        return None
    return float(np.abs(np.exp(1j * np.deg2rad(a)).mean()))


def dihedral_stability_map(path):
    """Label v2 (prereg amendment 8f904b81): per-residue circular mean resultant
    length of phi/psi across ensemble models; higher = more locally stable."""
    structure = parse_any_structure(path)
    models = list(structure)
    if len(models) < 5:
        return None, None
    best_ch, best_n = None, 0
    for ch in models[0]:
        n = len([r for r in ch if r.id[0] == ' '])
        if n > best_n:
            best_ch, best_n = ch.id, n
    per_model = []
    for model in models:
        ch = next((c for c in model if c.id == best_ch), None)
        if ch is None:
            return None, None
        residues = [r for r, _aa in chain_residues(ch)]
        n = len(residues)

        def vec(r, name):
            try:
                return r[name].get_vector()
            except KeyError:
                return None

        angles = []
        for i, res in enumerate(residues):
            nv, cav, cv_ = vec(res, 'N'), vec(res, 'CA'), vec(res, 'C')
            phi = psi = None
            if nv is not None and cav is not None and cv_ is not None:
                if i > 0:
                    pc = vec(residues[i - 1], 'C')
                    if pc is not None:
                        phi = float(calc_dihedral(pc, nv, cav, cv_) * 180.0 / np.pi)
                if i < n - 1:
                    nx = vec(residues[i + 1], 'N')
                    if nx is not None:
                        psi = float(calc_dihedral(nv, cav, cv_, nx) * 180.0 / np.pi)
            angles.append((phi, psi))
        per_model.append(angles)
    ch0 = next(c for c in models[0] if c.id == best_ch)
    seq = ''.join(aa for _r, aa in chain_residues(ch0))
    stab = {}
    for i in range(len(per_model[0])):
        vals = [circular_r([m[i][k] for m in per_model]) for k in (0, 1)]
        vals = [v for v in vals if v is not None]
        stab[i] = float(np.mean(vals)) if vals else None
    return seq, stab


def main():'''
assert 'def main():' in t
t = t.replace('def main():', v2fn, 1)

# 4. read_nmr_v2_full returns the ensemble pdb id too
old4 = """            return series, sorted(p2e)
    return None"""
new4 = """            return series, sorted(p2e), entry['pdb']
    return None"""
assert old4 in t
t = t.replace(old4, new4)
old4b = """                series, covered = read_nmr_v2_full(parent_seq)
                label_map = {pos: val for pos, val in zip(covered, series)}"""
new4b = """                series, covered, _nmr_pdb = read_nmr_v2_full(parent_seq)
                label_map = {pos: val for pos, val in zip(covered, series)}"""
assert old4b in t
t = t.replace(old4b, new4b)

# 5. v2 label per row for nmr kinds
old_row = """        rows.append({
            'accession': acc, 'status': 'scored', 'pdb': info['pdb'],
            'window': info['window'], 'window_start': w_start,
            'mismatch': info.get('mismatch'), 'note': info.get('note'),
            'exposure_source': exposure_source,
            'label_kind': label_kind, 'label_target': label_target,"""
new_row = """        paired_v2 = None
        if label_kind in ('nmr_v2', 'nmr_fetched'):
            try:
                if label_kind == 'nmr_v2':
                    _s, _c, nmr_pdb_id = read_nmr_v2_full(parent_seq)
                    ens_path = ROOT / 'data/calibration_nmr_pdb' / f'{nmr_pdb_id}.pdb'
                else:
                    fetched = fetch_nmr_ensemble_rmsf(acc)
                    ens_path = Path(fetched['path'])
                seq_ens, stab = dihedral_stability_map(ens_path)
                if seq_ens and stab:
                    pairs2 = map_chain_to_parent(seq_ens, parent_seq)
                    p2e = {b: a for a, b in pairs2}
                    if all((w_start + i) in p2e for i in range(L)):
                        v2_case = float(np.mean([stab[p2e[w_start + i]] for i in range(L)]))
                        v2_ctrl = [float(np.mean([stab[p2e[s + i]] for i in range(L)]))
                                   for s in controls
                                   if all((s + i) in p2e for i in range(L))]
                        if len(v2_ctrl) >= K_CONTROLS:
                            paired_v2 = {'case': round(v2_case, 4),
                                         'controls': [round(x, 4) for x in v2_ctrl],
                                         'paired_diff': round(v2_case - float(np.mean(v2_ctrl)), 4)}
            except Exception as error:  # noqa: BLE001 - v2 is additive, never fatal
                paired_v2 = {'error': repr(error)[:100]}
        rows.append({
            'accession': acc, 'status': 'scored', 'pdb': info['pdb'],
            'window': info['window'], 'window_start': w_start,
            'mismatch': info.get('mismatch'), 'note': info.get('note'),
            'exposure_source': exposure_source,
            'label_kind': label_kind, 'label_target': label_target,
            'paired_diff_v2_dihedral': paired_v2,"""
assert old_row in t, 'row block not found'
t = t.replace(old_row, new_row)

# 6. summary: v2 block for primary stratum
old_sum = """    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'n_scored': len(scored),
        'primary_stratum': {'rows': [r['accession'] for r in primary],
                            **(primary_summary or {'verdict': 'primary_stratum_underpowered',
                                                   'n_parents': len(primary),
                                                   'note': 'frozen stop rule: >=3 '
                                                           'primary-label parents required '
                                                           'for the CI endpoint'})},"""
new_sum = """    v2_rows = [r for r in primary
               if isinstance(r.get('paired_diff_v2_dihedral'), dict)
               and 'paired_diff' in r['paired_diff_v2_dihedral']]
    v2_summary = None
    if v2_rows:
        diffs2 = [r['paired_diff_v2_dihedral']['paired_diff'] for r in v2_rows]
        v2_summary = {'n_parents': len(v2_rows),
                      'mean_paired_diff': round(float(np.mean(diffs2)), 4),
                      'reading': ('v2 dihedral stability: positive diff = case window '
                                  'more locally stable than controls (H1-v2 supported); '
                                  'negative = case more disordered'),
                      'per_parent': {r['accession']: r['paired_diff_v2_dihedral']
                                     for r in v2_rows}}
    summary = {
        'classification': CLASSIFICATION,
        'date': DATE,
        'n_scored': len(scored),
        'label_v2_amendment': 'results/csn_c1_v1/prereg_label_v2_amendment.json '
                              '(frozen before any v7 label read)',
        'primary_stratum': {'rows': [r['accession'] for r in primary],
                            **(primary_summary or {'verdict': 'primary_stratum_underpowered',
                                                   'n_parents': len(primary),
                                                   'note': 'frozen stop rule: >=3 '
                                                           'primary-label parents required '
                                                           'for the CI endpoint'})},
        'label_v2_dihedral': v2_summary,"""
assert old_sum in t, 'summary block not found'
t = t.replace(old_sum, new_sum)

# 7. OUT -> v7 + revision note
t = t.replace("OUT = ROOT / 'results/csn_c1_enrich_v6'",
              "OUT = ROOT / 'results/csn_c1_enrich_v7'")
t = re.sub(r"'revision_note': \('v6:.*?'\),",
           "'revision_note': ('v7: label v2 amendment (dihedral circular stability) "
           "frozen before any read; AFDB version fallback v6..v2 fixes 404s; three "
           "stale RCSB search caches purged (documented in the amendment); v1 RMSF "
           "and v2 dihedral labels both reported, no gate shopping'),",
           t, flags=re.S)

P.write_text(t, encoding='utf-8')
ast.parse(t)
print('patched to v7, syntax OK')
