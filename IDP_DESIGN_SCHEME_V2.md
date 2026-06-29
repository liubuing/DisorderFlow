# 天然无序蛋白 (IDP) 抗体设计提升方案 V2

**目标靶点**: Aβ42(`DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA`,典型天然无序蛋白/淀粉样蛋白)
**当前基线 (§8 RESULTS.md)**: AF2 ipTM ≈ 0.06、BFN 过自信 12x、complex 模式在 IDP 上反而比 fixbb 差
**目标**: 把 IDP 抗体设计从"会识别/会避开"推进到"产出 AF2 可验证的真实结合"(ipTM > 0.4)
**交接对象**: DeepSeek V4

---

## 0. 核心诊断:IDP 抗体设计为什么更难(三重失效)

| 失效 | 现象 | 根因 |
|------|------|------|
| **A. 表位选错目标** | 当前按 `disorder<0.3` 选 13–25 段 | IDP 表位选择的目标应是**生物功能位点**(聚集核心/治疗性表位),不是"最低无序"。13–25 段恰好包含聚集核心 `16KLVFFAED21` 是**碰巧命中**,不是设计 |
| **B. 验证指标本身失效** | AF2 ipTM 0.06 既反映"设计差"也反映"AF2 不会 fold IDP" | AF2-multimer 对无序链天然 pLDDT 低(≈0.3),拖垮 ipTM。**ipTM 0.06 是设计质量与指标缺陷的混合产物**,不能单独作为判据 |
| **C. 生成器在 IDP 上退化更严重** | top 出现 `TTTTTTTVVVV` 低复杂度 CDR | 聚集核心是疏水芳香区(KLVFF),paratope 应富集 Tyr/Trp/Phe 做 π-堆叠/疏水识别;退化成 T/V 后**既不能识别又过自信**,形成 §4 的正反馈 |

**结论**: IDP 抗体设计不能沿用"给通用管线一个 IDP 结构"的框架,需要**问题重构**。

---

## 1. 核心战略:问题重构——从"全链 multimer 设计"到"线性表位/肽段中心设计"

> 这是整套方案的支点。现有管线把 Aβ42 完整 42 残基作为 multimer 抗原链交给 AF2——对 IDP 这是**错配**。

### 1.1 重构依据(数据 + 生物学)

1. Aβ 的治疗性表位**全部是短线性段**(已知临床抗体):
   - N-端 `1DAEFRHDSGY10` — aducanumab / bapineuzumab / donanemab(斑块清除导向)
   - 中段 `16KLVFFAED24` — solanezumab / crenezumab(单体/寡聚体导向)
   - C-端 `33GAIIGLMVGGVVIA42` — gantenerumab(部分)
2. 短肽段(10–15aa)在**结合态下会被诱导出局部结构**,AF2/Chai 一1/Boltz-1 对短肽-抗体复合物的预测可靠性远高于全长无序链。
3. 项目已有这些**已知 Aβ 抗体模板**:`4HIX`(solanezumab Fab)、`5CSZ`(aducanumab Fab)、`3STB`,可作为 scaffold + benchmark 双重资产。

### 1.2 重构后的管线(5 阶段)

```
[1 表位重构] 不再按 disorder 选表位,按"功能表位库"选
    └─ 输入: Aβ42 分段表位清单(N端/中段/聚集核心/寡聚体特异性)
       ↓
[2 构象系综扩展] 5-seed → 功能态区分(monomer vs oligomer-like vs fibril-core)
    └─ 对每个表位生成 K 个结合态构象(从 MD/已知复合物取,而非随机 CA)
       ↓
[3 生成器换源 + IDP先验] MPNN/diffab 生成 CDR,注入 amyloid-aware 氨基酸先验
    └─ paratope 偏置 Tyr/Trp/Phe,禁用退化序列(§4 anti-degeneration)
       ↓
[4 IDP-appropriate 验证] 肽段-抗体对接 + 跨构象鲁棒性,AF2 ipTM 降权为辅助
    └─ 主判据: 对接 score + 界面接触数 + 跨构象最小值;AF2 仅作 sanity
       ↓
[5 IDP置信校准] BFN 分数按 disorder 门控,IDP 段不直接可信
    └─ 报告"BFN 分(BF 标记 IDP-不可信) + 对接分 + 鲁棒分"三元组
```

---

## 2. 分阶段任务(每个任务含落点文件 + 验收)

### 阶段 P0:验证指标重建(先解决"测不准"——否则后续无法判进步)

> **关键认知**:在没修好度量前训任何模型都是盲跑。先做这一步。

| 任务 | 内容 | 落点 | 验收 |
|------|------|------|------|
| **P0-1 肽段-抗体 benchmark** | 抓取 4HIX/5CSZ 等已知 Aβ 抗体-肽复合物,把它们的 native CDR 喂回 BFN/MPNN,记录 AF2 ipTM | `idp_benchmark_known_abs.py`(新) | 已知抗体的 native CDR 在"肽段+抗体"模式下 AF2 ipTM 须 >0.6;若 <0.6 说明指标/构象仍错,**禁止继续** |
| **P0-2 三联验证器** | 统一产出 (a) 肽段对接分 (b) 跨构象 ipTM {min, median, max} (c) BFN 分+disorder 标记 | `modules/idp_validation_triplet.py`(新) | 对已知抗体给出三联分,且 (b)median 能区分 native vs scrambled |

**判据**:P0 必须先证明"已知好抗体在重建的指标下能得高分",否则后续所有"进步"都不可信。这正面对治 §0 的失效 B。

### 阶段 P1:表位重构(治失效 A)

| 任务 | 内容 | 落点 | 验收 |
|------|------|------|------|
| **P1-1 功能表位库** | 建立 Aβ 表位清单(N端1-10 / 聚集核心16-21 / 中段16-24 / C端33-42),每个标注**治疗导向**(斑块清除 vs 寡聚体中和) | `configs/idp/abeta_epitopes.yml`(新) | 每个表位有生物学依据标注,可按治疗目标筛选 |
| **P1-2 表位评分器** | 替换 `find_ordered_segments` 的"disorder<0.3"准则为复合分:`aggregation_propensity(如 TANGO/AGGRESCAN) × disorder × druggability` | `modules/idp_epitope_selector.py`(新/改) | 选出的表位必须含聚集核心或已知治疗表位;不再纯靠 disorder |

### 阶段 P2:构象系综(治"对什么设计")

| 任务 | 内容 | 落点 | 验收 |
|------|------|------|------|
| **P2-1 功能态构象** | 扩展 5-seed:对聚集核心表位额外生成 β-发夹/fibril-段构象(从已知 fibril PDB 如 2NAO/5O3L 切片段),对 N-端生成游离构象 | `build_abeta_conformations.py`(改) | 每表位 ≥3 个构象,涵盖该表位的功能态 |
| **P2-2 跨构象鲁棒分** | 一个设计在 ≥K 个构象下都预测结合才算合格,score = min/percentile(不取平均) | `modules/idp_validation_triplet.py` | 引入鲁棒分后,top20 中"只在单构象高分"的设计被过滤 |

### 阶段 P3:生成器换源 + amyloid 先验(治失效 C,与 §7 任务 B 合流)

| 任务 | 内容 | 落点 | 验收 |
|------|------|------|------|
| **P3-1 混合生成** | MPNN/diffab 生成 CDR,V14 BFN 头做筛选(不用 V17i 头,见 RESULTS §5) | `mpnn_bfn_hybrid.py`(扩) | 输出统一 schema:scaffold/epitope/conf/CDR/gen_score |
| **P3-2 amyloid-aware 先验** | 对聚集核心表位,采样偏置 Tyr/Trp/Phe;全管线注入 anti-degeneration 过滤(Shannon + k-mer 重复度,§7 任务 C) | `modules/idp_aa_prior.py`(新) | top20 无 >6 连续相同残基;对疏水表位 output paratope 芳香族占比 > 25% |
| **P3-3 模板 leverage** | 对 4HIX/5CSZ 等已知抗体,做 CDR-seed(保留已知热点接触残基,只重设计其余),作为强基线 | `run_template_seeded.py`(新) | template-seeded 设计的对接分应 ≥ 从头设计 |

### 阶段 P4:IDP 置信校准(治"自评不可信")

| 任务 | 内容 | 落点 | 验收 |
|------|------|------|------|
| **P4-1 disorder 门控** | 当 epitope 或 paratope 段 mean_disorder > 0.5,BFN ipTM 标记 "IDP-不可信"并降权,报告里以对接分+鲁棒分为主 | `modules/closed_loop_scorer.py`(改) | IDP 靶点报告不再以 BFN ipTM 为首列 |
| **P4-2 IDP 专用头(可选)** | 若算力允许,在 IDP/短肽复合物数据上单独微调一个置信头 | `configs/train/bfn_idp_peptide_xpu.yml`(新) | 该头在 P0-1 benchmark 上 OC<2x |

### 阶段 P5:disorder head 上 CAID(把已成立的能力做成可发表资产)

| 任务 | 内容 | 落点 | 验收 |
|------|------|------|------|
| **P5-1 CAID 评估** | disorder head vs IUPred3 / AF2-pLDDT,报 AUC-ROC/PR | `benchmark_disorder_caid.py`(新) | 这一环不依赖抗体设计成功,是**独立可发表成果**(RESULTS §8 已定位为唯一已成立差异化能力) |

---

## 3. 指标与验收总表

| 里程碑 | 指标 | 判据 | 依赖 |
|--------|------|------|------|
| **M0 度量可信** | 已知抗体 native CDR | 肽段模式 AF2 ipTM >0.6 | P0 |
| **M1 度量可信(鲁棒)** | native vs scrambled 三联分 | median/min 显著区分 | P0/P2 |
| **M2 表位对齐** | 选出表位 | 100% 含聚集核心或治疗表位 | P1 |
| **M3 不退化** | top20 CDR | 无低复杂度退化 + 芳香族合规 | P3 |
| **M4 设计可用(小目标)** | 头批设计 | 肽段模式 AF2 ipTM >0.3 | P0–P3 |
| **M5 设计可用(全目标)** | 最终候选 | ipTM >0.4 + 跨构象鲁棒 | 全 |
| **M6 可发表** | CAID | AUC 竞争 IUPred3 | P5(独立) |

**关键**:M0 是硬闸门。**若已知 Aβ 抗体在重建指标下都得不了高分,说明问题仍在度量/构象,禁止往后训模型**——这是避免重蹈"V14→V17i 盲跑 4 版本"覆辙的护栏。

---

## 4. DeepSeek V4 协作约定(IDP 专用)

| 约定 | 原因 |
|------|------|
| **任何 IDP 改动前先建/确认 P0 度量,再动模型** | §0 失效 B:IDP 上 ipTM 是混合信号,不修度量则进步不可证 |
| **IDP 报告主指标用"对接分+鲁棒分",不用裸 AF2 ipTM** | AF2 对无序链系统性低估,裸 ipTM 会误杀好设计 |
| **置信头在 IDP 上不直接可信,必须 disorder 门控** | 12x 过自信,RESULTS §8 已确认 |
| **表位选择写明生物学依据,不靠 disorder 单准则** | 失效 A:13–25 命中是巧合 |
| **保留已知抗体模板(4HIX/5CSZ)作 sanity 基线** | 若 native CDR 都拿不到分,问题在管线不在设计 |
| **新建 V18-IDP 系列而非续 V17** | 战略转向(RESULTS §5),需标明 IDP 路线 |
| **IDP 版本同时报:肽段 ipTM / 全链 ipTM / 对接分 / 鲁棒分 / OC(IDP) 五指标** | 单一指标在 IDP 上必然误判 |

---

## 5. 优先级与排期建议(RTX 5060 算力约束)

| 优先级 | 阶段 | 周期 | 理由 |
|--------|------|------|------|
| **P0** | P0 度量重建 + P5 CAID | 1 周 | 度量是闸门;CAID 是独立可发表,风险最低回报最高 |
| **P0** | P1 表位重构 | 2–3 天 | 纯逻辑改动,无训练成本,直接治失效 A |
| **P1** | P3 生成换源(模板 seed + anti-degeneration) | 1 周 | 复用 §7 已有资产,最快出候选 |
| **P1** | P2 构象系综 + 鲁棒分 | 1 周 | 治"对什么设计" |
| **P2** | P4 置信门控(轻量版:仅 disorder 门控) | 2 天 | 不需训练即可让报告诚实化 |
| **P3** | P4-2 IDP 专用头 + P3-3 从头 mpnn 生成 | 2–3 周 | 算力密集,放最后 |

---

## 6. 为什么这套方案预期能破 IDP 天花板(因果链)

```
现状:  IDP → 全链AF2-multimer → ipTM 0.06 (失效B:指标错 + 失效C:退化CDR)
                                          ↓
P0      重建肽段度量 → 已知抗体能得高分 → 度量可信
                                          ↓
P1      表位按功能选 → 对准聚集核心KLVFF (失效A修复)
                                          ↓
P2      多构象+鲁棒分 → 只保留跨态稳定设计 → 去掉单构象侥幸
                                          ↓
P3      MPNN生成+芳香先验+anti-degen → CDR能识别疏水核心 (失效C修复)
                                          ↓
P3      模板seed → 以已知抗体为起点,非从零
                                          ↓
P4      disorder门控 → 不再被12x过自信误导
                                          ↓
期望:  肽段度量下 ipTM>0.4 + 鲁棒,且OC(IDP)降到<3x
```

每一步都对应一个已确认的失效点,不是"加更多训练"的盲赌——这是与 V11→V17i 路线的根本区别。

---

## 附:与 RESULTS.md §7/§8 的衔接

- 本方案 §P3 与 RESULTS §7 任务 B/C 合流(BFN→MPNN 换源 + anti-degeneration),IDP 是该换源的首选受益场景。
- 本方案 §P0/P4 是 RESULTS §8 任务 I-2(IDP 置信校准)的具体落地。
- 本方案 §P5 = RESULTS §8 任务 I-1(CAID)。
- 新增的 IDP 专属杠杆:**问题重构(肽段中心)、功能表位、构象系综鲁棒分、amyloid 先验、模板 seed**——这五项是 RESULTS.md 未覆盖、IDP 方向的核心增量。
