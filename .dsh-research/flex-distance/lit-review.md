# 文献对标：无序/柔性表位结合体设计的最先进水平

**产出**：`.dsh-research/flex-distance/lit-review.md`
**用途**：为「项目距离目标还有多远」的评估提供外部基准（外部 = 已发表/已预印的他人工作，不含本项目内部结果）
**检索日期**：2026-11-01
**口径**：以 de novo 蛋白 binder 为主线，纳米抗体/单域抗体作为「柔性表位识别」的实验现实基准，抗体替代物（aptamer / peptoid / 分子印迹）作为备选口径
**标注约定**：`(abstract only)` = 只读了摘要；`(全文)` = 读过开放全文（PubMed Central 或出版社 OA）；所有题名/作者/年份/DOI 均来自 `paper_search` / `paper_get` / `arxiv` 工具返回，未凭记忆补写

> **检索与核验的诚实边界**：本轮 Semantic Scholar 持续限流、arXiv 多次超时，可用索引主要是 OpenAlex / Crossref / Europe PMC / PubMed。`reference_verify` 两次尝试对 26 条与 3 条参考文献全部返回 `fetch failed`（0 verified / 26 could not be checked），即本会话内 Crossref/DOI 注册中心不可达；因此**没有**拿到第三方交叉核验签名。下文每一条的元数据均逐字来自上述检索工具的输出（DOI、题名、作者、年份、期刊、引用数），这是本环境下可达到的最高证据等级。

---

## 0. 领域现状总表

| # | 工作 | 年份 | 靶点类型 | 设计方法 | 验证深度 | 关键数字 |
|---|------|------|----------|----------|----------|----------|
| 1 | [Liu et al., *Nature*](https://doi.org/10.1038/s41586-025-09248-9) | 2025 | **无序**（IDP：amylin、C-peptide、VP48、BRCA1_ARATH；IDR：G3BP1、IL-2RG、prion） | 序列输入 RFdiffusion，靶点与 binder 构象同时自由采样 + 双侧 partial diffusion | **体外 + 结构 + 细胞 + 功能**（无体内） | Kd 3–100 nM；amylin 96 设计→4 命中（100–454 nM），优化后 174 设计→107 结合、最好 3.8 nM；prion 48 设计→14 nM；G3BP1 晶体 Cα RMSD 0.8 Å；11×8 all-by-all 特异性 |
| 2 | [Vázquez Torres et al., *Nature*](https://doi.org/10.1038/s41586-023-06953-1) | 2023 | **柔性**（有螺旋倾向的生物活性肽：PTH、GCG、NPY、SCT、Bim） | RFdiffusion 扩展到 flexible target + 参数化凹槽骨架 + partial diffusion | **体外 + 功能**（MS 富集、生物传感器） | 从头噪声扩散得到 pM 级 binder 且无实验优化；threading 4/66 命中（最好 2.7 nM）；partial diffusion 25/96（GCG）、20/96（NPY） |
| 3 | [Watson et al., *Nature*](https://doi.org/10.1038/s41586-023-06415-8) `(abstract only)` | 2023 | 有序（方法学底座） | RFdiffusion 扩散生成 | 计算 + 多处实验 | 被后续工作反复引用的 de novo binder 设计底座 |
| 4 | [Bennett et al., *Nat Commun*](https://doi.org/10.1038/s41467-023-38328-5) `(abstract only)` | 2023 | 有序 | Rosetta 能量项 + AF2/RoseTTAFold 双过滤 + ProteinMPNN | 体外 | AF2/RF 过滤把设计成功率提高**近 10 倍** |
| 5 | [Pacesa et al. (BindCraft), *Nature*](https://doi.org/10.1038/s41586-025-09429-6) | 2025 | 有序（12 个挑战性靶点：受体、过敏原、Cas9 等） | AF2 权重反向传播 hallucination + MPNN 表面/核心优化 | **体外(SPR) + 细胞 + 病人样本 + 病毒改造** | 实验成功率 **10–100%，均值 46.3%**；nM 级；EGFR binder 82 nM 在社区竞赛第一；≥100 个过筛设计需采样 300–3,000 条轨迹 |
| 6 | [Zambaldi et al. (AlphaProteo), *arXiv*](https://arxiv.org/abs/2409.08022) `(abstract only)` | 2024 | 有序 | 专用 ML 模型族 | 体外（一轮中通量筛选） | 对 7 个靶点，亲和力比当时最好方法**好 3–300 倍**、成功率更高；"ready-to-use" |
| 7 | [Sappington et al., *Nat Commun*](https://doi.org/10.1038/s41467-025-67866-3) `(abstract only)` | 2026 | 有序（**亲水 β 边缘链**：KIT、PDGFRα、ALK-2/3、FCRL5、NRP1、α-CTX） | β-pairing 条件化 RFdiffusion | 体外 + **共晶** | pM–中 nM，成功率高于无条件 RFdiffusion；尽管共享 β 链相互作用仍高特异性；KIT 共晶与设计模型几乎一致 |
| 8 | [Jiang et al., *bioRxiv*](https://doi.org/10.1101/2025.02.07.636769) `(abstract only)` | 2025 | 有序 + 1 个肽标签（Strep-Tag II、STAT3、FGF4、EGF、PDGF-BB、CD4） | RFdiffusion | 体外（WB/表达） | **6 靶点 × 5 设计**：仅 2 个 Strep-Tag II binder 可用且不如商品抗体；其余 5 靶点全失败（低表达/非特异/测不到亲和力） |
| 9 | [Islam et al., *Protein Sci*](https://doi.org/10.1002/pro.4875) `(abstract only)` | 2023 | **无序**（α-syn 14 肽段） | 免疫/噬菌体展示 + 结构解析 | **共晶结构 + 功能** | α-syn 肽以伸展主链方式被识别，特异性落在 α-syn 48–52 区；截短后结合更紧、抑制聚集更好 |
| 10 | [Hmila et al., *FEBS J*](https://doi.org/10.1111/febs.16376) `(abstract only)` | 2022 | **无序/聚集态**（α-syn N 端区） | 免疫文库 + 噬菌体展示 | 体外(ITC) + 细胞 + **人脑组织** | ITC 定量；相对单体优先结合纤维；抑制种子化聚集与细胞毒性；识别 PD/DLB 脑内 Lewy 体 |
| 11 | [Jiang et al., *Sci Adv*](https://doi.org/10.1126/sciadv.adf3775) `(abstract only)` | 2023 | **无序/聚集态**（α-syn、tau） | 羊驼免疫 + 噬菌体展示 | **体内成像（小鼠）** | 无创区分 α-syn 与 tau 病理，脑信号与病灶负荷强相关 |
| 12 | [Abskharon et al., *PNAS*](https://doi.org/10.1073/pnas.2300258120) `(abstract only)` | 2023 | **构象表位**（tau 纤维端） | 把纤维封端抑制序列嫁接到 VHH 骨架 + 双特异设计 | 细胞 + **病人脑提取物 + 小鼠 BBB** | 阻断 AD 与 PSP 病人脑提取物种子化；双特异（BBB 受体 + tau capping）提高静脉给药后脑内穿透 |
| 13 | [McArthur et al., *Biomater Sci*](https://doi.org/10.1039/d4bm00707g) `(abstract only)` | 2024 | **构象表位**（tau 寡聚体） | 免疫筛选 + 表位作图 | 体外 + 人脑组织 | OT2.4/OT2.6 对超**不连续表位**特异；区分单体/纤维；识别 AD 与 PART 脑内聚集 |
| 14 | [Mahajan et al., *Sci Rep*](https://doi.org/10.1038/s41598-018-35464-7) `(abstract only)` | 2018 | **无序**（α-syn NAC 区） | 结构计算亲和力成熟（ab initio 界面建模） | 体外 | **首轮计算设计未提升亲和力**；最终靠单点突变经 k_on 提升——无晶体结构条件下的现实上限 |
| 15 | [Chenna et al., *bioRxiv*](https://doi.org/10.1101/2025.06.21.660895) `(abstract only)` | 2025 | **无序**（α-syn C 端 / LAG3） | 结构生物信息造 + 蛋白语言模型（pLM）排序 | **仅计算** | 同源 CDR 内 Spearman ρ=0.86；换全新 CDR 掉到 ρ=0.18；Precision@100 = 38 / 12 |
| 16 | [Bozkurt, *bioRxiv*](https://doi.org/10.64898/2026.04.13.718094) `(abstract only)` | 2026 | 有序（公开基准复盘） | 用 disorder/聚集/拓扑/PTM 五类"生物学先验"模型重排 | 复分析已有实验基准 | CAR-T CD20 竞赛 11,984 设计、Adaptyv EGFR 603 设计；叠加过滤把命中率 13.8%→38.6%（2.8×） |
| 17 | [Bouvier-Müller et al., *NAR*](https://doi.org/10.1093/nar/gkae544) `(全文)` | 2024 | **构象表位**（α-syn 纤维多形体） | SELEX（2′F-Py RNA aptamer） | 体外 + 病人来源多形体 | F30 Kd=6.6±0.3 nM、F124 Kd=8.8±1.3 nM；可区分 PD/DLB/MSA 来源多形体；重复性 R²=0.85/0.94 |
| 18 | [Akgönüllü et al., *Polymers*](https://doi.org/10.3390/polym15030629) `(全文，部分)` | 2023 | 蛋白质（整体/表位印迹） | 分子印迹聚合物（MIP，"塑料抗体"） | 传感器 LOD 级 | 表位印迹 BSA LOD 0.02 ng/mL（线性 1.0–150 ng/mL）；HSA 16.6 ng/mL；CRP 0.41×10⁻⁵ ng/mL |
| 19 | [Lee et al., *Sci Adv*](https://doi.org/10.1126/sciadv.adr1018) `(abstract only)` | 2025 | 有序（cadherin-2、anti-HA 抗体） | 1 亿成员大环肽库 + 亲和选择-MS | 体外 | 初筛 CBP 表观 Kd=53 nM，定向优化后 NCBP-4 29 nM |
| 20 | [Habault et al., *Prostate*](https://doi.org/10.1002/pros.24868) `(abstract only)` | 2025 | 有序（β-catenin/TCF） | 大环 peptoid–peptide 杂合体 | **体内**（前列腺癌模型） | 选择性调控 β-catenin/TCF 并在体内显效 |

---

## A. de novo binder 设计打无序/柔性靶点：现在到哪一步

- **只有一篇工作是真正的"无序靶点 de novo binder"系统论文**：[Liu et al. 2025, *Nature*](https://doi.org/10.1038/s41586-025-09248-9)。它证明序列输入 + 双端自由采样的 RFdiffusion 可以对 IDP/IDR 出 binder，Kd 落在 **3–100 nM**；并且把"无序靶点能不能设计"这个问题从"没有通用方法"推进到"有方法、但每个靶点都要重付代价"。**验证程度**：体外 BLI、共晶（G3BP1-11，整体 Cα RMSD 0.8 Å、界面 Cα 0.86 Å / 侧链 2.29 Å）、细胞内共定位、以及功能读出（amylin 纤维抑制与解聚、G3BP1 应激颗粒解体、LC-MS/MS 富集）。**对 DisorderFlow 意味着**：外部已经把"无序表位可设计"这件事做成了，所以"柔性表位结合体设计"本身不再是新颖性来源；新颖性只能来自方法差异（如打分/生成），而且必须与这条基准正面比较。

- **命中率是"个位数百分比 + best-of-N"**：amylin 先测 96 个非螺旋构象设计，只得到 4 个初始命中（100/317/431/454 nM）；VP48 测 95 个设计，最好 750 nM；prion 测 48 个设计，最好 PRI28 = 14 nM。**能做到什么**：从百量级设计里挑出 nM 级 binder。**验证到什么程度**：初始命中必须靠双侧 partial diffusion 再优化一轮（amylin：174 个设计里 107 个结合，最好 3.8 nM，命中率跳到 61%）。**对 DisorderFlow 意味着**：任何"N 个设计里挑最好"的数字都必须给出**分母**（测了多少、多少表达、多少结合）；只报最优 Kd 不报分母在文献里是不可比的。

- **柔性 ≠ 无序，柔性有构象倾向时先例更早**：[Vázquez Torres et al. 2023, *Nature*](https://doi.org/10.1038/s41586-023-06953-1) 把 RFdiffusion 显式扩展到 flexible target，并用 partial diffusion 从随机噪声直接得到 **pM 级**、无实验优化的 binder（PTH、Bim）。它把机理讲得很清楚：肽在游离态部分或完全无结构，**把肽折叠成特定构象要付熵代价**，直接吃掉结合自由能。**对 DisorderFlow 意味着**：这句"熵代价"就是柔性表位设计的第一性困难，项目若要论证自己的方法有价值，最自然的落点正是"如何为这笔熵代价定价/采样"，而不是再报一个排序相关性。

- **有序靶点上的成功率已经很高，但这不能外推到无序靶点**：[Pacesa et al. 2025](https://doi.org/10.1038/s41586-025-09429-6) 在 12 个**结构明确**的靶点上做到 10–100%（均值 46.3%）实验成功率、nM 级、无需高通量筛选；[AlphaProteo](https://arxiv.org/abs/2409.08022) 报 3–300 倍亲和力提升与更高成功率。**对 DisorderFlow 意味着**：用 46.3% 这类数字为柔性靶点背书是错的；文献里柔性/无序靶点的成功率明显更低（见上条与 A-2）。

- **反向证据同样重要**：[Jiang et al. 2025, *bioRxiv*](https://doi.org/10.1101/2025.02.07.636769) 用 RFdiffusion 做 6 个靶点 × 5 个设计，只有 Strep-Tag II 的 2 个能用、且不如商品抗体，其余 5 个靶点全部失败（低表达、非特异、测不到亲和力）。**验证到什么程度**：这是少见的、把"失败"写进结果的独立复现。**对 DisorderFlow 意味着**：外部文献里"RFdiffusion 名义上能做"和"小规模真做能成"是两件事；小规模战役（每靶点 5 个设计）在设计上就不具备统计功效。

- **β 链/伸展表位是可设计的，但已有工作把它做成"有序靶点的边缘链"**：[Sappington et al. 2026, *Nat Commun*](https://doi.org/10.1038/s41467-025-67866-3) 用 β-pairing 条件化 RFdiffusion 打亲水边缘 β 链，得到 pM–中 nM、高特异性，并有与设计几乎一致的 KIT 共晶。**对 DisorderFlow 意味着**：这是"线性/伸展表位"最接近的可比工作，但它的靶点是折叠蛋白的边缘链，**不是**真正无固定构象的肽段；引用它时必须区分这两类。

- **领域自评的口径**：[Yang et al. 2026, *Nature*](https://doi.org/10.1038/s41586-026-10328-7) `(abstract only)` 说结构、组装、binder 设计"接近解决"，但**成功率与活性仍有改进空间**。**对 DisorderFlow 意味着**：如果项目要投方法类期刊，"成功率"和"活性"就是审稿人默认会问的两个数。

---

## B. 纳米抗体 / VHH 打 α-syn、tau、淀粉样蛋白：实验侧的天花板

- **α-syn 的线性表位有原子级结构**：[Islam et al. 2023, *Protein Sci*](https://doi.org/10.1002/pro.4875) 解出 Nbα-syn01 与 α-syn 14 残基肽段的共晶，识别方式很特殊——肽段以**伸展主链**结合，并"顶替"了纳米抗体自身的 N 端区；特异性集中在 α-syn 48–52。结构指导的截短让结合更紧、抑制聚集更好。**对 DisorderFlow 意味着**：柔性表位确实能被高亲和力识别，但机制可能不是"凹槽包住肽"，而是"β 补体式主链配对"——这直接支持把"β-strand pairing"作为柔性表位设计的首选几何假设（与 A-6 呼应）。

- **N 端表位 + 功能读出是人家的标准配置**：[Hmila et al. 2022, *FEBS J*](https://doi.org/10.1111/febs.16376) 用免疫文库 + 噬菌体展示拿到 Nbα-syn01，ITC 定量，发现它相对单体**优先结合纤维**；能抑制种子化聚集、降低细胞毒性、识别 PD/DLB 尸检脑的 Lewy 体。**验证到什么程度**：体外 + 细胞 + 人组织三级。**对 DisorderFlow 意味着**："实现"的最低可比门槛是三级齐全，只有一级（结合）在文献里不算完成。

- **体内成像是"结合体真的能用"的硬证据**：[Jiang et al. 2023, *Sci Adv*](https://doi.org/10.1126/sciadv.adf3775) 用免疫羊驼 + 噬菌体展示得到 sdAb，做到小鼠**无创**区分 α-syn 与 tau 病理，脑信号与病灶负荷强相关。**对 DisorderFlow 意味着**：这是备选天花板之一——如果项目连体外 Kd 都没有，距离这条线还差三个量级的工作。

- **构象表位（纤维端/寡聚体）是另一类问题**：[Abskharon et al. 2023, *PNAS*](https://doi.org/10.1073/pnas.2300258120) 把已知纤维封端序列嫁接到 VHH 上，阻断 AD 与 PSP 病人脑提取物在 biosensor 细胞里的种子化，并用双特异（BBB 受体 + capping）在**小鼠**里改善静脉给药后的脑穿透；[McArthur et al. 2024, *Biomater Sci*](https://doi.org/10.1039/d4bm00707g) 的 OT2.4/OT2.6 针对 tau 寡聚体的**不连续表位**，能区分单体/纤维并识别 AD 与 PART 脑内聚集。**对 DisorderFlow 意味着**：文献明确区分"序列表位"（线性、无序）与"构象表位"（聚集体形貌）；项目若只做序列/单体肽段，谈"构象特异性"是不成立的；反之若做聚集体，就需要像它们一样的病人样本验证。

- **计算设计打无序靶点的历史最好成绩很朴素**：[Mahajan et al. 2018, *Sci Rep*](https://doi.org/10.1038/s41598-018-35464-7) 对 α-syn NAC 区做结构计算的亲和力成熟：**首轮设计策略没有提升亲和力**，最终靠单点突变（主要提高 k_on）才有效，而且是在没有晶体结构的情况下靠 ab initio 建模摸索出来的。**对 DisorderFlow 意味着**：这是"计算→实验"在无序靶点上最诚实的对照——计算方案第一次失败是常态，只有后续实验闭环才算数。

- **纳米抗体的体量参考**：[McArthur et al. 2024, *Biotechnol Prog*](https://doi.org/10.1002/btpr.3463) `(abstract only)` 做 pan-tau 多价纳米抗体，结合 tau 聚集基序并识别病理聚集体。**对 DisorderFlow 意味着**：多价/多表位是提高表观亲和力的常规工程手段，但它是"优化"而非"设计"，不能替代 de novo 声称。

---

## C. 线性表位 vs 构象表位：设计难度差在哪

- **几何不同**：线性/无序表位以**伸展主链**被识别（Islam 2023 的 α-syn 14 肽就是主链配对），设计上对应"β 补体 / β-pairing"而不是"凹槽包裹"；[Sappington et al. 2026](https://doi.org/10.1038/s41467-025-67866-3) 正是把这条路做成了方法，并证明可以被条件化生成。**对 DisorderFlow 意味着**：若项目的方法没有显式的 β 配对/伸展几何先验，"柔性表位"这一说法在结构层面缺少可检验的落地。

- **熵不同**：柔性靶点在游离态没有单一构象，[Vázquez Torres et al. 2023](https://doi.org/10.1038/s41586-023-06953-1) 明确写出"把肽结构化"要付熵代价，这会直接削弱净结合自由能；他们的对策是参数化凹槽骨架 + threading + partial diffusion。**对 DisorderFlow 意味着**：柔性表位设计的主要成本项是构象采样与熵罚，任何只用单一静态构象打分的方法在原理上就会漏掉这一项。

- **数据不同**：真正无序的抗体–肽复合物结构在 PDB 里很稀少——这正是本项目自己在稿件里报告的事实（SAbDab/RCSB 系统审计后"零个新的配对 H/L 抗体–短肽复合物"）。我没能在检索中找到一篇专门量化"线性 vs 构象表位设计难度差"的论文；文献是以**案例系列**（Islam/Hmila/Abskharon/McArthur）而非受控比较的方式呈现这个差别的。**对 DisorderFlow 意味着**：这是一个真实的方法学空位，但空位本身不是结果；要填它需要非天然/合成复合物数据或实验，而不是再换一个打分器。

- **构象特异性可以被"做出来"但不能被"算出来"**：[McArthur et al. 2024](https://doi.org/10.1039/d4bm00707g) 的不连续表位是靠免疫筛选 + 表位作图拿到的，不是算出来的；[Abskharon et al. 2023](https://doi.org/10.1073/pnas.2300258120) 的构象选择性来自"把封端序列嫁接到骨架上"这一经验设计。**对 DisorderFlow 意味着**：构象选择性目前的主要来源仍是实验筛选或结构先验，计算侧尚无可靠预测器；若项目声称"构象选择性设计"，需要先证明其打分器能区分同一序列的不同构象态。

- **计算侧有一个具体的负结果值得记住**：Liu 2025 报告，用序列输入 RFdiffusion 得到的 **23 个成功案例，靶点全部处于 β-strand 构象**；螺旋或无结构态需要额外指定二级结构或双侧 partial diffusion 才拿到。**对 DisorderFlow 意味着**：不受约束的生成模型会偏向"伸展/链状"解，这既解释了为什么线性表位相对容易，也说明"柔性"不是单一难度档位——螺旋倾向肽和纯无规卷曲是两种不同的难。

---

## D. 什么算"实现"：门槛逐级拆解

> 文献里没有任何一篇把"计算设计一个结合体"直接等同于"做出了结合体"。分级如下，每级都给出文献中的代表数字。

**L0 计算生成 + 计算排序（没有湿实验）**
- 代表：AlphaProteo 的 in silico 报告；BindCraft 的过滤链；[Sinclair et al. 2026, *PASC*](https://doi.org/10.1145/3815572.3815764) `(abstract only)` 的 StructBioReasoner（Der f 21 的 787 个候选里 >50% 在**in silico 结合自由能**上优于文献里人工设计的参照 binder；NMNAT-2 从 97,066 个 binder 中归纳出 3 种结合模式）。
- 这一级的可信度上限：Bozkurt 2026 明确指出"**大多数设计在实验上失败，而当前以结构和亲和力为中心的评价不能可靠预测哪一个会成功**"。
- 对 DisorderFlow：项目当前处于 L0（ECLS/PAE 都是打分与排序），L0 的产出只能被称为"筛选信号"或"边界研究"。

**L1 可表达、可纯化（连"能拿到蛋白"都不保证）**
- 代表数字：Jiang 2025 的失败原因里**低表达**是首因之一；BindCraft 把"binder 表面疏水性 <35%""界面 Lys/Met 少于 3 个""结合态/游离态 RMSD <3.5 Å"直接写进过滤条件，就是为了过这一关。
- 对 DisorderFlow：任何设计不得不先报"多少条序列成功表达"的分母。

**L2 体外测到 Kd（BLI / SPR / ITC / FP）**
- 文献中的常见区间（同一批工具输出）：
  - de novo **无序/IDR** binder：**3–100 nM**（Liu 2025，amylin/C-peptide/VP48/BRCA1_ARATH；G3BP1/IL-2RG/prion 10–100 nM）
  - **螺旋倾向柔性肽** binder：pM–nM，例：2.7 nM（Vázquez Torres 2023）
  - **β 边缘链** binder：pM–中 nM（Sappington 2026）
  - **纳米抗体**：ITC 定量（Hmila 2022）；单域抗体常规在 nM 档（Islam 2023 结构指导截短后"结合更紧"）
  - **大环肽**：29–53 nM（Lee 2025）
  - **aptamer**：6.6–8.8 nM（Bouvier-Müller 2024，对 F 型 α-syn 纤维）
- 方法学要求：单浓度筛选 ≠ Kd；文献里凡称 Kd 的都给了曲线与拟合（BLI/SPR/ITC/FP），并注明是否多价（多价体只给表观 Kd）。
- 对 DisorderFlow：这是"结合体设计"的最低法定门槛；没有实测 Kd 就没有这一级的任何证据。

**L3 特异性 / 反筛**
- 代表做法与数字：Liu 2025 做了 **11 个 binder × 8 个靶点的 all-by-all BLI**，只有两例在极高浓度（>800 nM、2 µM）出现弱交叉；Sappington 2026 强调"尽管共享 β 链相互作用仍高特异性"；Chenna 2025 用 pLM 排序时，把"新 CDR 序列"上的 ρ 掉到 0.18 视为泛化失败。
- 对 DisorderFlow：**没有交叉结合矩阵的特异性声明不算证据**；最低可接受版本是"近缘抗原 + 组成打乱表位 + 非靶点蛋白"三类反筛。

**L4 结构验证（设计模型 = 真实结构）**
- 代表数字：G3BP1-11 共晶 vs 设计**整体 Cα RMSD 0.8 Å**（界面 Cα 0.86 Å、侧链 2.29 Å）(Liu 2025)；KIT 共晶"几乎与设计模型一致"(Sappington 2026)。Vázquez Torres 2023 用晶体结构比较界面形状互补（0.62 vs 0.67）与接触面积（431 Å² vs 522 Å²）。
- 对 DisorderFlow：结构验证是"设计"一词的支撑；退一步的最低版本是**定点突变破坏结合**（Liu 2025 做了 L12T/L16T、L26T/L30T、M10K/L24T 等界面突变）。

**L5 功能 / 细胞 / 病人样本**
- 代表：amylin binder 在 1:4、1:40、1:400 摩尔比下浓度依赖地抑制纤维形成并解聚已存在纤维（Liu 2025）；G3BP1 binder 在细胞内解体应激颗粒；tau capping 纳米抗体阻断 AD/PSP 病人脑提取物的种子化（Abskharon 2023）；纳米抗体降低种子化聚集与细胞毒性并识别 Lewy 体（Hmila 2022）；BindCraft 的 binder 减少病人来源样本中 IgE 与桦树过敏原结合（Pacesa 2025）。
- 对 DisorderFlow：功能读出是"结合体真的改变了生物学"的唯一证据，且大多不贵（细胞模型 + 已建立的聚集/毒性读出）。

**L6 体内 / 成药性**
- 代表：小鼠无创成像（Jiang 2023）；静脉给药后脑穿透改善（Abskharon 2023）；热稳定性 **95 °C**（Liu 2025 的全套 binder；Vázquez Torres 2023 的 CP-35/VP48-2）；peptoid–peptide 杂合体在体内显效（Habault 2025）。
- **检索范围内未见** de novo 无序靶点 binder 的体内 PK/免疫原性数据；这一步在文献里仍是空白（不是"没人做"，而是我检索到的这批工作没报）。
- 对 DisorderFlow：L6 距离最远；在 L2 之前谈成药性没有意义。

**通量与成功率（跨级对照表）**

| 指标 | 文献值 | 来源 |
|---|---|---|
| 单个靶点需要采样的生成轨迹 | ~300–3,000 条 → ≥100 个过筛设计 | Pacesa 2025 |
| 进入实验的设计数 | 取 top 20 中的 10 个；或每靶点 48–96 个 | Pacesa 2025；Liu 2025 |
| 小规模失败案例的设计数 | 每靶点仅 5 个 → 基本必然失败 | Jiang 2025 |
| 结构化靶点实验成功率 | 10–100%，均值 46.3% | Pacesa 2025 |
| 无序靶点初始命中 | 4/96（amylin），最好 750 nM（VP48，95 个） | Liu 2025 |
| 无序靶点优化后命中 | 107/174 结合（61%），最好 3.8 nM | Liu 2025 |
| 公开基准的设计池规模 | 11,984（CAR-T CD20）与 603（Adaptyv EGFR） | Bozkurt 2026 |
| 好的"生物学先验"过滤能带来的提升 | 命中率 13.8% → 38.6%（2.8×） | Bozkurt 2026 |

---

## E. 蛋白语言模型 / disorder-aware 打分器在 binder 排序上的已发表表现

- **有先例，但只在同源范围内成立**：[Chenna et al. 2025, *bioRxiv*](https://doi.org/10.1101/2025.06.21.660895) 针对 α-syn 无序 C 端 / LAG3 界面，用 pLM 序列嵌入训练排序器：**训练与评测 CDR 有重叠时 Spearman ρ=0.86、Precision@100=38**；换成差异很大的 CDR 序列，**ρ 掉到 0.18、Precision@100=12**。**验证到什么程度**：抽象为纯计算管线，摘要中**未报告任何湿实验**。**对 DisorderFlow 意味着**：这是"只靠打分排序"最强的已发表正例，但它自己就给出了泛化崩塌的数字；任何排序声称都必须报"同源 vs 非同源"两组。

- **"超越结构与亲和力"的先验特征确实有增益，但增益有上限**：[Bozkurt 2026, *bioRxiv*](https://doi.org/10.64898/2026.04.13.718094) 在两个真实基准上检验 disorder、amyloidogenicity、拓扑、PTM、蛋白分类五类模型：**聚集倾向是最稳健的跨基准信号**；disorder 特征"两个数据集都显著但方向相反"，其作用取决于架构（CAR 胞外域 vs 独立 binder）。叠加过滤后命中率 13.8%→38.6%。**对 DisorderFlow 意味着**：disorder 特征不是单调有益的方向；把它们当"更聪明的打分维度"直接加权，可能在不同体系上翻转符号——这正好对应项目内部"简单基线常常胜出"的处境。

- **纯粹的 in-silico 优越性不能当成功**：[Sinclair et al. 2026](https://doi.org/10.1145/3815572.3815764) 的多智能体系统在 Der f 21 上报告 787 个"design and validated"候选里 >50% 优于人工参照——但优势指标是**in silico 结合自由能**。**对 DisorderFlow 意味着**：这类数字（以及本项目内部的 ΔG/PAE/ECLS 提升）在文献坐标里都属于 L0；"validated"这个词在这里是内部含义，不是实验验证。

- **打分器在工程上真正起过的作用是"过滤"，不是"排序"**：[Bennett et al. 2023](https://doi.org/10.1038/s41467-023-38328-5) 用 AF2/RF 双过滤把成功率提高近 10 倍；[Pacesa et al. 2025](https://doi.org/10.1038/s41586-025-09429-6) 用 i_pTM 排序取 top 20 中的 10 个去做湿实验。**对 DisorderFlow 意味着**：文献里打分器的成功叙事是"提高命中率的过滤器"；"在候选池里把最优者排到第一"是更强的主张，而本项目自己的柔性线数据显示 native 从未排到第一（家族均值低至 0.00 相对近天然竞争者）。这与文献并不矛盾——文献也没证明过这种排序能力。

- **一个可借用的对照基线**：ProteinMPNN 条件似然在项目内部被当作 baseline，而 [Liu 2025](https://doi.org/10.1038/s41586-025-09248-9) 与 [Sappington 2026](https://doi.org/10.1038/s41467-025-67866-3) 的工作流里，ProteinMPNN 的角色是**序列设计器**而非排序器——这一点在写"我们优于 MPNN"时必须说清是优于它的哪个用途。

---

## F. 非蛋白质抗体替代物的现状（备选口径）

- **Aptamer：亲和力不是短板，设计性才是**。[Bouvier-Müller et al. 2024, *NAR*](https://doi.org/10.1093/nar/gkae544) 拿到 2′F-Py RNA aptamer，对 F 型 α-syn 纤维 **Kd = 6.6±0.3 nM / 8.8±1.3 nM**，能区分 PD/DLB/MSA 病人来源的纤维多形体；相比之下已发表的 DNA aptamer 约 10 nM，但只能给出 2–10 倍的多形体偏好。**能做到什么**：nM 级、构象层面区分、无需动物免疫。**做不到什么**：SELEX 需要真实靶标材料，**不能从一段无序序列"设计"出 aptamer**；也没有从结构出发的可编程界面设计。**对 DisorderFlow 意味着**：如果项目的核心资产是"从序列/结构生成候选"，aptamer 路线在可设计性上完全不匹配，只能当"如果实验做不了蛋白"的替代读出。

- **MIP / "塑料抗体"：便宜、稳定，但模板化而非设计**。[Resina et al. 2023, *Biotechnol Adv*](https://doi.org/10.1016/j.biotechadv.2023.108220) 与 [Teixeira et al. 2025](https://doi.org/10.1007/s40883-025-00478-x) 综述了蛋白印迹与**表位印迹**纳米颗粒（EINP）作为"合成抗体"。性能数字来自传感器而非亲和力：[Akgönüllü et al. 2023, *Polymers*](https://doi.org/10.3390/polym15030629) 报告表位印迹 BSA 的 LOD 0.02 ng/mL、HSA 16.6 ng/mL、CRP 0.41×10⁻⁵ ng/mL。**能力差**：给的是"检测限"而不是"Kd"；结合位点异质、多克隆；对同一模板的不同构象几乎没有选择性；且**必须先有模板分子**，也就是"印迹"，不是"设计"。**对 DisorderFlow 意味着**：MIP 只能在"我需要在无抗体条件下检测某个肽"这种场景里作为备选，不能作为"柔性表位结合体设计"的对标对象——它没有设计环节。

- **大环肽 / peptoid：通量极高，但仍是"筛选"路线**。[Lee et al. 2025, *Sci Adv*](https://doi.org/10.1126/sciadv.adr1018) 用 **1 亿成员**大环肽库 + 亲和选择-MS，得到 cadherin-2 的 CBP（表观 Kd 53 nM），再做 SAR 优化到 NCBP-4（29 nM）；[Habault et al. 2025, *Prostate*](https://doi.org/10.1002/pros.24868) 的大环 peptoid–peptide 杂合体在体内调控 β-catenin/TCF。**对 DisorderFlow 意味着**：nM 级与非蛋白化学是兼容的，但这条路的"设计"含义是"库设计 + 选择"，需要合成与质谱基础设施；它不能替代结构驱动的 de novo 生成。

- **能力差总结（三者 vs 蛋白 binder）**
  | 维度 | de novo 蛋白 binder | Aptamer | 大环肽/peptoid | MIP |
  |---|---|---|---|---|
  | 亲和力 | 3–100 nM（无序靶点）(Liu 2025) | 6.6–8.8 nM (Bouvier-Müller 2024) | 29–53 nM (Lee 2025) | 无 Kd 报告，仅 LOD (Akgönüllü 2023) |
  | 特异性/构象分辨 | all-by-all 反筛可做 (Liu 2025) | 可区分纤维多形体 (Bouvier-Müller 2024) | 有 SAR 热点/冷点 (Lee 2025) | 弱，位点异质 |
  | 可设计性（从序列/结构生成） | **有**（RFdiffusion/AF2 闭环） | 无（依赖 SELEX + 真实靶标） | 部分（库设计，非结构驱动） | 无（模板印迹） |
  | 遗传可编码 / 可融合 | 是（成像、胞内、AAV 改造） | 否 | 否 | 否 |
  | 主要瓶颈 | 表达、成功率、无体内数据 | 体内稳定性、递送 | 合成与筛选设施 | 定量亲和力与选择性 |

- **一句话结论**：在"从序列出发设计一个能结合柔性表位的分子"这件事上，**只有蛋白 binder 路线具备可编程性**；其余三条路线的亲和力并不差，但它们把"设计"换成了"筛选""印迹""合成"，因而**不是同一个问题的替代方案**。DisorderFlow 若改用 aptamer/peptoid 口径，等于换了一个项目。

---

## G. 对本项目的直接含义：若 DisorderFlow 要声称「柔性表位结合体设计」

> 项目内部状态（来自仓库内 `publication/flex_boundary_manuscript_draft_v1.md`，非文献）：26 个 VHH–柔性肽复合物（14 个结构、8 个抗原家族：α-synuclein、tau PHF6、SARS-CoV-2 stem helix、BCMA、Syt1、V5 等），以 ECLS 作为 judge，与 1,300 条 ProteinMPNN 竞争者比对；native 平均位于池内 76.7 百分位，**在 26 个池中从未排第一**；无实验结合数据。外部文献据此要求的清单如下。

1. **不能把"打分/排序"写成"设计"**。文献里"设计"一词的最低门槛是 L2（实测 Kd）。当前项目在 L0，且 L0 的排序主张本身已被内部证据否定（native 从未第一）。外部可接受的最强措辞是 **"boundary study / 边界研究"**——这与 [Bozkurt 2026](https://doi.org/10.64898/2026.04.13.718094) 对"当前评价不能预测实验成功"的结论一致，也符合 [Yang et al. 2026](https://doi.org/10.1038/s41586-026-10328-7) 对"成功率仍需改进"的判断。

2. **必须补齐分母**。任何"我们生成了 N 个候选"的表述都要配三行数字：**生成数 → 成功表达数 → 测到结合的数目**。文献基准：[Liu 2025](https://doi.org/10.1038/s41586-025-09248-9) 96→4（amylin 初始）、174→107（优化后）、95→最好 750 nM（VP48）、48→14 nM（prion）；[Jiang 2025](https://doi.org/10.1101/2025.02.07.636769) 6×5 设计几乎全失败。只报 best-of-N 会被直接判定为不可比。

3. **至少一次真实 Kd，且说明方法**。可接受的方法：BLI / SPR / ITC / FP（[Liu 2025](https://doi.org/10.1038/s41586-025-09248-9)、[Pacesa 2025](https://doi.org/10.1038/s41586-025-09429-6)、[Hmila 2022](https://doi.org/10.1111/febs.16376)）。目标区间：**无序表位 3–100 nM 是当前 SOTA 带**，做到 100 nM 以内才有资格与 Liu 2025 并列讨论；做到 µM 级仍可发表但必须定位为"弱结合/概念验证"（参考文献里 CP-95 初始命中 16 µM 也被如实报告）。

4. **必须做特异性反筛矩阵**。最低版本=**近缘抗原 + 组成打乱表位 + 无关蛋白**三类；文献标准是 all-by-all（11×8，[Liu 2025](https://doi.org/10.1038/s41586-025-09248-9)）。项目已有"组成打乱"这一轴（内部 ECLS 的 shuffle 对照），但那是序列层对照，**不能替代蛋白层的交叉结合实验**。

5. **必须给出结构或位点证据**。要么共晶/冷冻电镜（Liu 2025 的 0.8 Å；Sappington 2026 的 KIT 共晶），要么**定点突变破坏结合**（Liu 2025 的界面突变）。对一个 VHH–柔性肽体系，退一步也要给"关键 CDR 位点 → 结合"的因果链。

6. **功能读出是性价比最高的一步**。α-syn/tau 体系有成熟读出：纤维形成抑制/解聚、种子化聚集、细胞毒性、Lewy 体/病理染色（[Hmila 2022](https://doi.org/10.1111/febs.16376)、[Abskharon 2023](https://doi.org/10.1073/pnas.2300258120)、[McArthur 2024](https://doi.org/10.1039/d4bm00707g)、[Liu 2025](https://doi.org/10.1038/s41586-025-09248-9)）。即使没有 SPR，功能读出 + 表达验证也能支撑一个较弱的声称，但**不能替代 Kd**。

7. **与最接近的可比工作正面比较，并说明差异**。三者必答：
   - 与 [Liu et al. 2025](https://doi.org/10.1038/s41586-025-09248-9)（无序靶点 seq-input diffusion，3–100 nM）比，本项目的方法差异是什么、在哪个靶点上更好？
   - 与 [Vázquez Torres et al. 2023](https://doi.org/10.1038/s41586-023-06953-1)（柔性/螺旋肽，pM 级 + partial diffusion）比，是否处理了"熵代价"？
   - 与 [Islam 2023](https://doi.org/10.1002/pro.4875)/[Hmila 2022](https://doi.org/10.1111/febs.16376)（α-syn 纳米抗体的原子级机制：伸展主链 + β 补体）比，本项目的生成几何是否与已验证的识别模式一致？

8. **不要复用有序靶点的成功率数字**。BindCraft 的 46.3% 均值（[Pacesa 2025](https://doi.org/10.1038/s41586-025-09429-6)）与 AlphaProteo 的 3–300 倍（[Zambaldi 2024](https://arxiv.org/abs/2409.08022)）都来自结构明确的靶点；把它们当作柔性表位的期望值会被审稿人直接指出。

9. **如果最终不做湿实验**，那么唯一站得住的定位是：**一个关于"打分器在柔性表位上能/不能做什么"的边界研究**，且必须包含 sampled-competitor 池（项目已有 1,300 条 ProteinMPNN 竞争者，这恰好符合文献对"competent designs 而非随机序列"的要求）。此时对外表述应避免"设计出结合体"，改为"评估了候选排序信号的适用边界"。这与 [Bozkurt 2026](https://doi.org/10.64898/2026.04.13.718094)、[Chenna 2025](https://doi.org/10.1101/2025.06.21.660895) 的诚实口径一致。

10. **备选口径要有代价意识**。若转向 aptamer/peptoid/MIP（F 节），等于放弃"可编程设计"这一核心资产：aptamer 需要 SELEX 与真实靶标材料且 Kd 也就 6.6–8.8 nM（[Bouvier-Müller 2024](https://doi.org/10.1093/nar/gkae544)），MIP 连 Kd 都不报告（[Akgönüllü 2023](https://doi.org/10.3390/polym15030629)）。换成它们并不能让项目"更接近目标"，只是换了一个目标。

---

## 附：本文件引用清单（全部来自工具返回记录）

1. Liu C, Wu K, Choi H, … Baker D. *Diffusing protein binders to intrinsically disordered proteins.* Nature, 2025. [10.1038/s41586-025-09248-9](https://doi.org/10.1038/s41586-025-09248-9)（全文）
2. Vázquez Torres S, Leung PJY, Venkatesh P, … Baker D. *De novo design of high-affinity binders of bioactive helical peptides.* Nature, 2023. [10.1038/s41586-023-06953-1](https://doi.org/10.1038/s41586-023-06953-1)（全文）
3. Watson JL, Juergens D, Bennett NR, … Baker D. *De novo design of protein structure and function with RFdiffusion.* Nature, 2023. [10.1038/s41586-023-06415-8](https://doi.org/10.1038/s41586-023-06415-8)（abstract only）
4. Bennett NR, Coventry B, Goreshnik I, … Baker D. *Improving de novo protein binder design with deep learning.* Nat Commun, 2023. [10.1038/s41467-023-38328-5](https://doi.org/10.1038/s41467-023-38328-5)（abstract only）
5. Pacesa M, Nickel L, Schellhaas C, … Correia BE. *One-shot design of functional protein binders with BindCraft.* Nature, 2025. [10.1038/s41586-025-09429-6](https://doi.org/10.1038/s41586-025-09429-6)（全文）
6. Zambaldi V, La D, Chu AE, … Wang J. *De novo design of high-affinity protein binders with AlphaProteo.* arXiv:2409.08022, 2024. [arxiv.org/abs/2409.08022](https://arxiv.org/abs/2409.08022)（abstract only）
7. Sappington I, Toul M, Lee DS, … Baker D. *Improved protein binder design using β-pairing targeted RFdiffusion.* Nat Commun, 2026. [10.1038/s41467-025-67866-3](https://doi.org/10.1038/s41467-025-67866-3)（abstract only）
8. Jiang B, Li X, Guo A, Wei M, Wu J. *RFdiffusion Exhibits Low Success Rate in De Novo Design of Functional Protein Binders for Biochemical Detection.* bioRxiv, 2025. [10.1101/2025.02.07.636769](https://doi.org/10.1101/2025.02.07.636769)（abstract only）
9. Islam Z, Vaikath NN, Hmila I, El-Agnaf OMA, Kolatkar PR. *Structural insights into the unique recognition module between α-synuclein peptide and nanobody.* Protein Sci, 2023. [10.1002/pro.4875](https://doi.org/10.1002/pro.4875)（abstract only）
10. Hmila I, Vaikath NN, Majbour NK, … El-Agnaf OMA. *Novel engineered nanobodies specific for N-terminal region of alpha-synuclein recognize Lewy-body pathology and inhibit in-vitro seeded aggregation and toxicity.* FEBS J, 2022. [10.1111/febs.16376](https://doi.org/10.1111/febs.16376)（abstract only）
11. Jiang Y, Lin Y, Krishnaswamy S, … Sigurdsson EM. *Single-domain antibody–based noninvasive in vivo imaging of α-synuclein or tau pathology.* Sci Adv, 2023. [10.1126/sciadv.adf3775](https://doi.org/10.1126/sciadv.adf3775)（abstract only）
12. Abskharon RNN, Pan H, Sawaya MR, … Eisenberg DS. *Structure-based design of nanobodies that inhibit seeding of Alzheimer's patient–extracted tau fibrils.* PNAS, 2023. [10.1073/pnas.2300258120](https://doi.org/10.1073/pnas.2300258120)（abstract only）
13. McArthur N, Squire JD, Onyeachonam OJ, … Kane RS. *Generation of nanobodies with conformational specificity for tau oligomers that recognize tau aggregates from human Alzheimer's disease samples.* Biomater Sci, 2024. [10.1039/d4bm00707g](https://doi.org/10.1039/d4bm00707g)（abstract only）
14. McArthur N, Kang B, Rivera Moctezuma FG, … Kane RS. *Development of a pan-tau multivalent nanobody that binds tau aggregation motifs and recognizes pathological tau aggregates.* Biotechnol Prog, 2024. [10.1002/btpr.3463](https://doi.org/10.1002/btpr.3463)（abstract only）
15. Mahajan SP, Meksiriporn B, Waraho-Zhmayev D, … DeLisa MP. *Computational affinity maturation of camelid single-domain intrabodies against the nonamyloid component of alpha-synuclein.* Sci Rep, 2018. [10.1038/s41598-018-35464-7](https://doi.org/10.1038/s41598-018-35464-7)（abstract only）
16. Chenna A, Priyadarshi P, Kolluru K, Singal S, Goel G. *Protein Language Model Based Structure-guided Antibody Screening for Disordered Protein Targets.* bioRxiv, 2025. [10.1101/2025.06.21.660895](https://doi.org/10.1101/2025.06.21.660895)（abstract only）
17. Bozkurt Ç. *Beyond Structure and Affinity: Context-Dependent Signals for de novo Binder Success.* bioRxiv, 2026. [10.64898/2026.04.13.718094](https://doi.org/10.64898/2026.04.13.718094)（abstract only）
18. Sinclair M, Meigooni M, Vasan AK, … Ramanathan A. *Scalable Agentic Reasoning for Designing Biologics Targeting Intrinsically Disordered Proteins.* PASC, 2026. [10.1145/3815572.3815764](https://doi.org/10.1145/3815572.3815764)（abstract only）
19. Bouvier-Müller A, Fourmy D, Fenyi A, Bousset L, Melki R, Ducongé F. *Aptamer binding footprints discriminate α-synuclein fibrillar polymorphs from different synucleinopathies.* Nucleic Acids Res, 2024. [10.1093/nar/gkae544](https://doi.org/10.1093/nar/gkae544)（全文）
20. Resina L, Alemán C, Ferreira FC, Esteves T. *Protein-imprinted polymers: How far have "plastic antibodies" come?* Biotechnol Adv, 2023. [10.1016/j.biotechadv.2023.108220](https://doi.org/10.1016/j.biotechadv.2023.108220)（abstract only）
21. Akgönüllü S, Kılıç S, Esen C, Denizli A. *Molecularly Imprinted Polymer-Based Sensors for Protein Detection.* Polymers, 2023. [10.3390/polym15030629](https://doi.org/10.3390/polym15030629)（全文，部分）
22. Teixeira SPB, Reis RL, Domingues RMA. *Epitope Imprinted Nanoparticles as Customized Synthetic Antibodies for Biorecognition in Theranostics.* Regen Eng Transl Med, 2025. [10.1007/s40883-025-00478-x](https://doi.org/10.1007/s40883-025-00478-x)（abstract only）
23. Lee MA, Brown JS, Farquhar CE, Loas A, Pentelute BL. *Affinity selection–mass spectrometry with linearizable macrocyclic peptide libraries.* Sci Adv, 2025. [10.1126/sciadv.adr1018](https://doi.org/10.1126/sciadv.adr1018)（abstract only）
24. Habault J, Franco JL, Ha S, … Logan SK. *In Vivo Efficacy of a Macrocyclic Peptoid-Peptide Hybrid That Selectively Modulates the Beta-Catenin/TCF Interaction to Inhibit Prostate Cancer.* Prostate, 2025. [10.1002/pros.24868](https://doi.org/10.1002/pros.24868)（abstract only）
25. Yang W, Wang S, Lee GR, … Baker D. *The past, present and future of de novo protein design.* Nature, 2026. [10.1038/s41586-026-10328-7](https://doi.org/10.1038/s41586-026-10328-7)（abstract only）
26. Fox D, Taveneau C, Clement J, … Knott GJ. *Code to complex: AI-driven de novo binder design.* Structure, 2025. [10.1016/j.str.2025.08.007](https://doi.org/10.1016/j.str.2025.08.007)（abstract only）

**未纳入**：`epitope3D`、`DiscoTope-3.0`、`PAbFold` 等仅出现在检索列表里、我只看到题名与不完整摘要，未读正文，因此没有在正文中作为论据使用。检索中若出现同一条记录的 review/decision letter 条目（如 Crossref 的 `.../v1/review1`），已剔除，不用作文献。

---

**未覆盖**（时间到点、未开新检索轮次）：① 未找到「线性表位 vs 构象表位设计难度」的受控比较论文，C 节只能由案例系列支撑；② 未检索到任何已发表的、针对 α-synuclein 或 tau 的 **de novo 设计 miniprotein/VHH** 结合体（现有均为免疫筛选或计算亲和力成熟），该"未见"仅限本轮索引范围，不作不存在性断言；③ de novo 无序靶点 binder 的**体内 PK / 免疫原性**数据未检索到；④ Semantic Scholar 全程限流、arXiv 多次超时，`reference_verify` 对 26+3 条全部 `fetch failed`，故无第三方 DOI 交叉核验；⑤ 项目内部数字（26 复合物、76.7 百分位、1,300 竞争者）取自仓库 `publication/flex_boundary_manuscript_draft_v1.md`，属内部状态而非文献，已在 G 节开头标注。
