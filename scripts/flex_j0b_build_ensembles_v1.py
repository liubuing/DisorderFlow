"""FLEX-BINDER Gate J0b, ensemble construction.

Runs the T1-style restrained OpenMM ensemble (original T1 settings, unchanged:
seeds 2701/2711/2721/2731, antibody heavy-atom restraint 5000 kJ/mol/nm^2,
peptide backbone restraint 25 kJ/mol/nm^2, CPU, implicit solvent) around each
of the 28 preregistered J0 component poses, via the audited
``generate_ensemble`` machinery with chains remapped to the J0 component PDBs
(H = heavy, P = peptide).

Declared BEFORE any ensemble-conditioned scoring (M4):
- a component's ensemble is VALID iff >= 2 conformers accepted AND
  dispersion.mean_pairwise_rmsd >= 0.25 (T1 valid-ensemble criterion);
- invalid ensembles are EXCLUDED from J0b scoring and disclosed; they do not
  silently fall back to single-pose values.
No model inference here: OpenMM sampling and QC only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest

OUT = ROOT / 'results/flex_j0b_v1'
COHORT = ROOT / 'results/flex_j0_v1/cohort.json'
MIN_ACCEPTED = 2
MIN_DISPERSION = 0.25

CONFIG_TEMPLATE = {
    'chains': {'antibody': ['H'], 'peptide': 'P'},
    'sampling': {
        'platform': 'CPU',
        'ph': 7.4,
        'temperature_kelvin': 300.0,
        'friction_per_ps': 1.0,
        'timestep_fs': 2.0,
        'minimization_tolerance_kj_mol_nm': 10.0,
        'minimization_iterations': 300,
        'snapshot_minimization_iterations': 150,
        'equilibration_steps': 50,
        'production_steps': 150,
        'antibody_heavy_atom_restraint': 5000.0,
        'peptide_backbone_restraint': 25.0,
        'contact_cutoff_angstrom': 4.5,
        'seeds': [2701, 2711, 2721, 2731],
        'quality_control': {
            'max_antibody_rmsd': 0.5,
            'min_peptide_rmsd': 0.05,
            'max_peptide_rmsd': 3.0,
            'min_contact_retention': 0.5,
        },
    },
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = read(COHORT)
    freeze(OUT / 'protocol.json', {
        'classification': 'ensemble_construction_not_scoring',
        'cohort_sha256': digest(COHORT),
        'config': CONFIG_TEMPLATE,
        'config_provenance': 'identical values to configs/benchmarks/4hix_t1_ensemble.json '
                             'sampling block; chains remapped to J0 component PDBs',
        'validity_rule': {'min_accepted_conformers': MIN_ACCEPTED,
                          'min_mean_pairwise_rmsd': MIN_DISPERSION},
        'invalid_ensemble_handling': 'excluded from j0b scoring and disclosed; '
                                     'no fallback to single-pose values',
        'script_sha256': digest(Path(__file__))})
    print('components:', len(cohort['components']), flush=True)

    sys.path.insert(0, str(ROOT / 'scripts'))
    from generate_h3_peptide_ensemble import generate_ensemble

    rows = []
    for entry in cohort['components']:
        instance = entry['instance']
        out_dir = OUT / 'ensembles' / instance
        audit_path = out_dir / 'ensemble_audit.json'
        if not audit_path.exists():
            generate_ensemble(str(ROOT / entry['component_pdb']), str(out_dir),
                              CONFIG_TEMPLATE, record_id=instance)
        audit = read(audit_path)
        accepted = audit['accepted']
        dispersion = (audit.get('dispersion') or {}).get('mean_pairwise_rmsd')
        valid = accepted >= MIN_ACCEPTED and dispersion is not None \
            and dispersion >= MIN_DISPERSION
        rows.append({'instance': instance, 'accepted': accepted,
                     'attempted': audit['attempted'],
                     'mean_pairwise_rmsd': dispersion,
                     'valid': bool(valid)})
        print(instance, 'accepted', accepted, 'dispersion', dispersion,
              'VALID' if valid else 'INVALID', flush=True)

    summary = {
        'components': len(rows),
        'valid_ensembles': sum(1 for r in rows if r['valid']),
        'invalid_ensembles': sum(1 for r in rows if not r['valid']),
        'invalid_list': [r['instance'] for r in rows if not r['valid']],
        'rows': rows,
        'classification': 'declared_before_scoring',
    }
    freeze(OUT / 'build_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'},
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
