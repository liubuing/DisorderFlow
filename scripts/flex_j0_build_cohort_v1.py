"""FLEX-BINDER Gate J0 cohort builder.

Materializes the 28 preregistered components as scoring-ready artifacts:
component PDBs (H = heavy, P = peptide, renumbered 1..N per chain, v26 writer),
CDR-H3 region specs from the frozen A1 ANARCII numbering caches, the native
region sequence, and 50 composition-matched shuffles per component under the
preregistered RNG. No model inference, no network.

Fail-closed guards: region indices must be contiguous and reconstruct the
frozen region sequence exactly; component counts must match the frozen
preregistration exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.materialize_pae_analog_structures_v26 import (
    region_indices, write_component_pdb)
from scripts.flex_a1_scaffold_audit_v1 import strict_residues  # noqa: F401 (re-exported contract)

OUT = ROOT / 'results/flex_j0_v1'
PREREG = OUT / 'preregistration.json'
A1 = ROOT / 'results/flex_a1_scaffold_audit_v1'
N_SHUFFLES = 50
RNG_SEED = 20261029


def load_numbering():
    labels = {}
    for path in sorted((A1 / 'numbering').glob('batch_*.json')):
        labels.update(read(path)['records'])
    return labels


def main():
    prereg = read(PREREG)
    components = prereg['cohort']['components']
    if len(components) != prereg['cohort']['n_components']:
        raise RuntimeError('preregistration component count mismatch')

    labels = load_numbering()
    pdb_dir = OUT / 'pdb'
    pdb_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = A1 / 'cif'

    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)

    rng = np.random.default_rng(RNG_SEED)
    entries = []
    for component in sorted(components, key=lambda c: c['instance']):
        instance = component['instance']
        pdb_id = component['pdb_id']
        heavy_seq = component['heavy_sequence']
        label = labels.get(heavy_seq)
        if not label or label.get('error') or not label.get('chothia_numbering'):
            raise RuntimeError(f'{instance}: heavy chain missing chothia numbering')
        indices = region_indices(label['chothia_numbering'],
                                 label.get('chothia_query_start',
                                           label['query_start']), 93, 102)
        if not indices:
            raise RuntimeError(f'{instance}: empty H3 region')
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise RuntimeError(f'{instance}: non-contiguous H3 indices {indices} - '
                               'region_spec as a range would be wrong; failing closed')
        region_seq = ''.join(heavy_seq[i] for i in indices)
        if region_seq != component['region_sequence']:
            raise RuntimeError(f'{instance}: region sequence mismatch '
                               f'{region_seq} != {component["region_sequence"]}')

        cif_path = cif_dir / f'{pdb_id}.cif'
        structure = parser.get_structure(pdb_id, str(cif_path))
        model = next(structure.get_models())
        heavy_rows = strict_residues(model[component['heavy_chain']], allow_mse=True)
        peptide_rows = strict_residues(model[component['peptide_chain']], allow_mse=True)
        if len(heavy_rows) != len(heavy_seq):
            raise RuntimeError(f'{instance}: heavy row count {len(heavy_rows)} != '
                               f'sequence length {len(heavy_seq)}')
        out_path = pdb_dir / f'{instance}.pdb'
        write_component_pdb(out_path, [('H', heavy_rows), ('P', peptide_rows)])

        shuffles = []
        for _ in range(N_SHUFFLES):
            letters = list(region_seq)
            rng.shuffle(letters)
            shuffles.append(''.join(letters))

        entries.append({
            'instance': instance,
            'pdb_id': pdb_id,
            'component_pdb': f'results/flex_j0_v1/pdb/{instance}.pdb',
            'component_pdb_sha256': digest(out_path),
            'region_spec': f'H:{indices[0] + 1}-{indices[-1] + 1}',
            'region_sequence': region_seq,
            'peptide_sequence': component['peptide_sequence'],
            'shuffles': shuffles,
        })
        print(instance, 'region', entries[-1]['region_spec'], region_seq, flush=True)

    frozen_shuffles = sorted({s for e in entries for s in e['shuffles']
                              if s != e['region_sequence']})
    for entry in entries:
        entry['n_shuffles_distinct_from_native'] = sum(
            1 for s in entry['shuffles'] if s != entry['region_sequence'])
    cohort = {
        'preregistration_sha256': digest(PREREG),
        'rng_seed': RNG_SEED,
        'n_shuffles': N_SHUFFLES,
        'components': entries,
        'component_order': 'sorted by instance, shuffle RNG consumed in this order',
    }
    freeze(OUT / 'cohort.json', cohort)
    print('components built:', len(entries), flush=True)
    print('distinct shuffles overall:', len(frozen_shuffles), flush=True)


if __name__ == '__main__':
    main()
