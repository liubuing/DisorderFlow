#!/usr/bin/env python3
"""Add run-scoped provenance and execution/QC templates to the 5CSZ wet-lab package."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / 'results/v5_1_candidates/abeta42_biological_constructs/expression_package'


def main():
    package = json.loads((PACKAGE / 'expression_package.json').read_text(encoding='utf-8'))
    run_id = '5CSZ_v5_1_local_panel_v1'
    records = []
    for construct in package['constructs']:
        legacy = construct['construct_id']
        canonical = f"{run_id}_{legacy.removeprefix('5CSZ_')}__bg-N52H"
        records.append({
            'design_run_id': run_id,
            'legacy_construct_id': legacy,
            'canonical_construct_id': canonical,
            'background_mutations': 'A:52:N>H',
            'nas_status': 'rescued_N52H',
            'candidate_mutations': construct['mutations'],
            'heavy_length': construct['heavy_length'],
            'light_length': construct['light_length'],
            'order_heavy_id': f'{legacy}_heavy',
            'order_light_id': '5CSZ_common_light',
        })
    handoff = {
        'status': 'order_package_ready_experiment_execution_requires_vendor_and_assay_lot_fields',
        'design_run_id': run_id,
        'constructs': records,
        'computational_scope': {
            'de_novo_not_required': True,
            'multimer_iptm_policy': 'absolute rejection gate only; do not rank candidates by small differences',
            'disorder_head_policy': 'unsupported predictor; excluded from candidate decisions',
        },
        'replication_contract': {
            'independent_expression_purification_batches': 2,
            'technical_replicates_per_binding_condition': 3,
            'randomized_plate_layout': True,
            'blinded_sample_ids_for_binding_analysis': True,
        },
        'state_qc_gate': {
            'required_before_binding_interpretation': True,
            'monomer': ['SEC retention/monomer peak documented', 'DLS size distribution documented'],
            'oligomer': ['SEC/DLS batch profile documented', 'lot accepted before candidate comparison'],
            'fibril': ['ThT-positive batch', 'TEM or AFM morphology documented'],
            'rule': 'No state-specificity claim when the corresponding A-beta preparation fails state QC',
        },
        'procurement_fields_required': [
            'A-beta peptide vendor/catalog/lot/purity',
            'scrambled peptide exact sequence/vendor/catalog/lot',
            'isotype Fab vendor/catalog/lot and human IgG1/kappa identity',
            'BLI sensor type/vendor/catalog/lot',
        ],
    }
    (PACKAGE / 'wetlab_execution_handoff.json').write_text(json.dumps(handoff, indent=2), encoding='utf-8')
    sample_fields = [
        'sample_id', 'canonical_construct_id', 'expression_batch', 'purification_batch',
        'date', 'operator', 'yield_mg_L', 'sec_monomer_fraction', 'dls_pdi',
        'storage_buffer', 'storage_temperature_C', 'freeze_thaw_count', 'raw_data_path',
    ]
    with (PACKAGE / 'sample_tracking_template.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        for record in records:
            for batch in (1, 2):
                writer.writerow({
                    'sample_id': f"{record['canonical_construct_id']}__batch-{batch}",
                    'canonical_construct_id': record['canonical_construct_id'],
                    'expression_batch': batch,
                    'purification_batch': batch,
                })
    with (PACKAGE / 'procurement_bom_template.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['item', 'role', 'vendor', 'catalog', 'lot', 'sequence_or_identity', 'qc_document', 'status']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, role in (
            ('A-beta1-42 monomer source', 'target'), ('A-beta1-42 oligomer source', 'target'),
            ('A-beta1-42 fibril source', 'target'), ('A-beta1-11 peptide', 'epitope control'),
            ('scrambled peptide', 'specificity control'), ('human IgG1/kappa Fab', 'isotype control'),
            ('BLI sensors', 'assay consumable')):
            writer.writerow({'item': item, 'role': role, 'status': 'required_before_experiment'})
    print(json.dumps({'status': handoff['status'], 'constructs': len(records)}, indent=2))


if __name__ == '__main__':
    main()
