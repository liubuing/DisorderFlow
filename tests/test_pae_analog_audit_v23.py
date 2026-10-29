"""PAE 类似物输入审计 v23 的契约测试：守护肽证据判定的 fail-closed（DNA 字母/核酸类型/辅助链/
长度不符均拦截）、重复序列不得成为独立家族，以及收据哈希损坏时完整性校验必须失败。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import hashlib

# 模块级加载：审计脚本不在包内，通过 importlib 按路径导入。
spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1]/'scripts/audit_pae_analog_inputs_v23.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


# 辅助函数：构造最小 RCSB 聚合实体记录（序列/描述/可选聚合类型）。
def entity(seq, description='peptide', kind=None):
    poly = {'pdbx_seq_one_letter_code_can': seq}
    if kind is not None:
        poly['type'] = kind
    return {'rcsb_id': 'TEST_1', 'rcsb_polymer_entity': {'pdbx_description': description}, 'entity_poly': poly}


class InputAuditTests(unittest.TestCase):
    # DNA 字母序列配 aptamer 描述必须报错，防止核酸适配体被当作肽证据纳入类似物队列。
    def test_dna_alphabet_is_not_protein_evidence(self):
        seq = 'ACGTACTG'
        result = audit.peptide_checks({'seq': seq, 'len': 8, 'desc': 'DNA aptamer'}, [entity(seq)])
        self.assertIn('description_identifies_nucleic_acid', result['errors'])

    # 聚合类型缺失进 pending（待补证据）、显式 RNA 类型进 errors——类型不确定时不得放行。
    def test_missing_type_is_unknown_and_explicit_rna_fails(self):
        p = {'seq': 'ACDEFG', 'len': 6, 'desc': 'peptide'}
        self.assertIn('polymer_type_not_retrieved', audit.peptide_checks(p, [entity(p['seq'])])['pending'])
        self.assertIn('polymer_type_outside_standard_L_peptide_scope', audit.peptide_checks(p, [entity(p['seq'], kind='polyribonucleotide')])['errors'])

    # 受体/辅助链描述与"报告长度≠序列长度"都要报错，防止 CD3 zeta 等辅助链或元数据错误混入。
    def test_accessory_chain_and_length_errors(self):
        p = {'seq': 'ACDEFG', 'len': 7, 'desc': 'CD3 zeta chain'}
        errors = audit.peptide_checks(p, [entity(p['seq'])])['errors']
        self.assertIn('description_identifies_receptor_or_accessory_chain', errors)
        self.assertIn('reported_length_differs_from_sequence', errors)

    # 家族去重：完全相同序列计为一组重复、严格包含序列对计为一对包含，全部标记不合格——
    # 防止同一肽的多份 PDB 冒充独立家族抬高样本量。
    def test_duplicates_do_not_become_independent_families(self):
        rows = [{'pdb': x, 'peptides': [{'seq': s, 'len': len(s), 'desc': 'peptide'}]} for x, s in [('A', 'ACDEFG'), ('B', 'ACDEFG'), ('C', 'ACDEFGH')]]
        meta = {r['pdb']: {'polymer_entities': [entity(r['peptides'][0]['seq'])], 'struct': {'title': 'Single-domain enzyme'}} for r in rows}
        result = audit.audit_cohort(rows, meta, 'vhh')
        self.assertEqual(result['summary']['repeated_exact_sequence_groups'], 1)
        self.assertEqual(result['summary']['strict_containment_pairs'], 1)
        self.assertTrue(all(not r['eligible_for_model_run'] for r in result['rows']))
        self.assertIn('no_specific_VHH_evidence_beyond_search_keyword', result['rows'][0]['pending'])

    # 哈希校验：文件字节被改动后与收据 sha256 失配必须抛 integrity mismatch——守护采集产物不可篡改。
    def test_receipt_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p/'x.json').write_bytes(b'changed')
            (p/'HARVEST_RECEIPT.json').write_text(json.dumps({'files': {'x.json': {'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}}}))
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                audit.verify_receipt(p)


if __name__ == '__main__':
    unittest.main()
