import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import hashlib

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1]/'scripts/audit_pae_analog_inputs_v23.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def entity(seq, description='peptide', kind=None):
    poly = {'pdbx_seq_one_letter_code_can': seq}
    if kind is not None:
        poly['type'] = kind
    return {'rcsb_id': 'TEST_1', 'rcsb_polymer_entity': {'pdbx_description': description}, 'entity_poly': poly}


class InputAuditTests(unittest.TestCase):
    def test_dna_alphabet_is_not_protein_evidence(self):
        seq = 'ACGTACTG'
        result = audit.peptide_checks({'seq': seq, 'len': 8, 'desc': 'DNA aptamer'}, [entity(seq)])
        self.assertIn('description_identifies_nucleic_acid', result['errors'])

    def test_missing_type_is_unknown_and_explicit_rna_fails(self):
        p = {'seq': 'ACDEFG', 'len': 6, 'desc': 'peptide'}
        self.assertIn('polymer_type_not_retrieved', audit.peptide_checks(p, [entity(p['seq'])])['pending'])
        self.assertIn('polymer_type_outside_standard_L_peptide_scope', audit.peptide_checks(p, [entity(p['seq'], kind='polyribonucleotide')])['errors'])

    def test_accessory_chain_and_length_errors(self):
        p = {'seq': 'ACDEFG', 'len': 7, 'desc': 'CD3 zeta chain'}
        errors = audit.peptide_checks(p, [entity(p['seq'])])['errors']
        self.assertIn('description_identifies_receptor_or_accessory_chain', errors)
        self.assertIn('reported_length_differs_from_sequence', errors)

    def test_duplicates_do_not_become_independent_families(self):
        rows = [{'pdb': x, 'peptides': [{'seq': s, 'len': len(s), 'desc': 'peptide'}]} for x, s in [('A', 'ACDEFG'), ('B', 'ACDEFG'), ('C', 'ACDEFGH')]]
        meta = {r['pdb']: {'polymer_entities': [entity(r['peptides'][0]['seq'])], 'struct': {'title': 'Single-domain enzyme'}} for r in rows}
        result = audit.audit_cohort(rows, meta, 'vhh')
        self.assertEqual(result['summary']['repeated_exact_sequence_groups'], 1)
        self.assertEqual(result['summary']['strict_containment_pairs'], 1)
        self.assertTrue(all(not r['eligible_for_model_run'] for r in result['rows']))
        self.assertIn('no_specific_VHH_evidence_beyond_search_keyword', result['rows'][0]['pending'])

    def test_receipt_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'x.json').write_bytes(b'changed')
            (p/'HARVEST_RECEIPT.json').write_text(json.dumps({'files': {'x.json': {'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}}}))
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                audit.verify_receipt(p)


if __name__ == '__main__':
    unittest.main()
