"""Package public retrospective candidates without assigning validation status."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.audit_successor_v3_isolation import select_representative

OUT = ROOT / 'data/pae_public_crosscheck_v8'


def main():
    source = read(OUT / 'audit.json')
    structure = read(OUT / 'structures/structural_manifest.json')
    components = read(OUT / 'structure_sequence_audit/components.json')
    records = {r['instance']:r for r in structure['records']}
    groups = []
    for i, members in enumerate(components['mmseqs_clean_components'], 1):
        representative = select_representative([records[k] for k in members])
        groups.append({'component_id':f'PUB{i:03d}', 'members':members, 'representative':representative,
            'classification':'retrospective_candidate_not_independent_confirmation',
            'short_sequence_sensitivity_pass':False, 'pretraining_overlap':'unresolved', 'ready_for_scoring':False})
    freeze(OUT / 'retrospective_candidates.json', {'classification':'model_free_candidate_package',
        'components':groups, 'selection':'existing geometric contact then resolution ranking; no model scores',
        'source_hashes':{p:digest(OUT / p) for p in ['protocol.json','audit.json','sequence_audit.json','structures/structural_manifest.json','structure_sequence_audit/components.json']},
        'decision':'candidate search complete; independent status unresolved; no AF2 generation/scoring launched'})
    lines = ['# 公共数据交叉筛查 v8', '',
        '日期：2026-09-18。结论：找到可继续审查的7组公共回顾性候选；没有建立严格独立验证集，也没有启动新AF2评分。', '',
        '## 来源与范围', '',
        '[SAbDab2](https://sabdab.opig.stats.ox.ac.uk/) 的官方all-summary快照于2026-09-18取得，最新条目更新日期2026-09-11；22,264行、11,604个PDB，存在重复实例行，未按行数当独立样本。',
        '[ABAG-docking官方仓库](https://github.com/Zhaonan99/Antibody-antigen-complex-structure-benchmark-dataset) 固定到提交6891e6e6790c7a898a08471193f494adf6f16b48；112案例对应105个PDB。本次下载编号清单并与SAbDab及RCSB链序列交叉核对，未下载整个ABAG结构包。', '',
        '新方案明确为回顾性公开数据可行性核查。元数据出现过仅作标记；既有严格v7协议保持不变。5033条序列/结构参考混合了训练及历史结构审计记录，命中该参考不自动等于训练泄漏，未命中也不证明没有预训练重叠。', '',
        '## 筛选结果', '',
        '| 步骤 | 数量 |', '|---|---:|',
        '| SAbDab标记含PEPTIDE的PDB | 953 |',
        '| 与ABAG合并去重后核对RCSB的PDB | 1,058 |',
        '| 配对H/L、5–50aa抗原链及实验分辨率条件初步通过 | 776 |',
        '| 未直接命中既有序列/结构参考的PDB | 118 |',
        '| 序列注释足够执行五轴检查 | 117 PDB / 219实例 |',
        '| 元数据序列MMseqs通过 | 24 PDB / 43实例 |',
        '| 进入坐标核查（含1个注释缺失项） | 25 PDB |',
        '| 结构、编号、H3接触检查通过 | 14 PDB |',
        '| 坐标重注释后MMseqs通过 | 13 PDB / 7组件 |',
        '| 同时通过保守短序列敏感性检查 | 0 |', '',
        'ABAG的105个PDB均可映射至本次SAbDab快照，其中93个有配对H/L记录；没有符合当前短肽条件的案例，不能把它作为本轮短肽扩充成功。它仍可作为另立蛋白抗原范围时的来源。', '',
        '## 候选组', '', '| 组件 | 代表结构 | 解析肽长 | 发布日期 |', '|---|---|---:|---|']
    for g in groups:
        r = g['representative']
        lines.append(f"| {g['component_id']} | {r['pdb_id']} | {len(r['antigen_sequence'])} | {r['release_date'][:10]} |")
    lines += ['', '## 必须保留的限制', '',
        '- 7组是按原MMseqs操作规则聚类的候选组，不等于7个经确认独立的抗原家族。',
        '- 保守短序列检查要求双向覆盖率0.8、H3一致性0.5、抗原一致性0.3；14个结构均存在抗原短序列命中。短序列偶然匹配与进化同源性不同，不能把这些命中直接写成真实泄漏；也不能为获得通过结果而静默删掉这一检查。',
        '- 元数据阶段唯一通过两套序列检查的3QO0，在实际坐标中未满足5–50aa已解析肽长度要求；它没有被保留。',
        '- 结构代表选择使用既有H3接触及分辨率规则，无PAE或模型分数。缺注释的9RUG同样进入坐标检查并按结果记录排除，没有静默丢弃。',
        '- 当前结构发布于1999–2020年。仅凭日期不能确认其是否被上游模型使用，尚未完成检查点训练清单核对，不能宣称时间隔离或预训练独立。',
        '- 已有SAbDab CDR注释与项目Chothia定义存在潜在差异，本次对进入结构阶段的条目已重新编号；元数据阶段排除者未全部做坐标重注释，所以本次不是穷尽性证明。', '',
        '## 下一步可执行判断', '',
        '优先对7组候选追溯真正模型训练/开发记录，核对抗原生物学关系与短肽匹配性质。若能支持回顾性外部适用性评估，在读取新标签前另行冻结效应与敏感性分析；若只能支持已知家族评估，则按该范围命名。严格独立、回顾性、预训练重叠三个状态分别报告。', '',
        '本轮3项单元测试通过，完整执行了两次五轴MMseqs审计和25个结构资格检查。没有改动旧模型、旧论文冻结包或v7结果。', '',
        '机器可读入口：`data/pae_public_crosscheck_v8/retrospective_candidates.json`。原始快照、下载清单、链映射、排除原因及哈希均在同目录。', '']
    (ROOT / 'docs/PAE_PUBLIC_CROSSCHECK_V8.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Packaged', len(groups), 'retrospective candidate components; no confirmation status granted.')


if __name__ == '__main__':
    main()
