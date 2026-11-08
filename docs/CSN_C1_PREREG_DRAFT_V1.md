# CSN C1：瞬态结构富集图谱（预注册**已冻结**）

日期：2026-11-08。**状态：已冻结**——`results/csn_c1_v1/preregistration.json`
（sha256 e8b191107b5ab7990e92227d6b3e835a754501027ede864e75aa3449872b440c，
由 `scripts/csn_c1_prereg_v1.py` 在数据产物落地后、任何标签读取前冻结）。
本草案文本保留为冻结前记录；**冻结版与草案的差异（披露）**：

1. **control 定义修正（草案缺陷）**：草案写"IEDB metadata 计数 = 0 的区段"——
   IEDB 计数是蛋白级 metadata，**窗口级 IEDB 先例在 metadata-only 红线下不可
   获得**。冻结版改为：control = 同亲本、同长度、暴露匹配（结合态抗原链 CA
   SASA）、局部证据宇宙（1,146 池 + 队列）内无结构先例的窗口；IEDB 蛋白级
   计数只作协变量记录；
2. **mismatch 锚定规则**：tau（GSVQIVYKPK）与 canonical P10636 差 1 个替换
   （canonical 为 VQIVYKPVL 段），肽搜索服务降级无法独立验证 → 冻结版定义
   `anchored_with_mismatch`（RCSB entity 指名 + ≤1 替换 + 条目标题指名），
   逐窗记录；
3. **标签分层**：主层（ATLAS/NMR v2 覆盖，预数 3 亲本：α-syn 经 2kkw、
   gp160/spike 经 ATLAS）承载主终点；AFDB 降级层（预期 6 亲本）仅描述性，
   永不与主层合并；
4. **环境事实**：UniProt 肽搜索 API 轮询 ≥90 s 返回空（已知阳性对照肽
   KEDAFSKLKEKFMNEL⊂P21579 亦空）——记为服务不可用，非零证据；6i2g/8skj/
   6zwk 保持未解析并排除出 C1 宇宙。

数据产物（`results/csn_c1_v1/`）：8/14 PDB 经 E0b 只读复用锚定；tau 三行走
mismatch 规则（预注册内授权）；IEDB 计数 8 accession（metadata-only）；ATLAS
可用性 2/8（旗标行）。**下一步**：C1 执行脚本（窗口构造 + 标签读取 + 配对
bootstrap），在预注册约束下进行——本战役 v1 内另起 `csn_c1_enrich_v1`。

## 1. 问题（H1）

神经/突触 IDP 上有抗体先例的表位窗口（case），是否比同蛋白内无先例的
匹配窗口（control）更常落在**自由态瞬态结构偏置**区段（局部 RMSF 谷 /
MoRF 样段）？

## 2. 设计

- **case 窗口**：C0 有效族的 parent 肽在 parent 蛋白上的映射区段 +
  ESEL E0b 已锚定行（α-syn DYEPEA/P37840、Syt1/P21579 等 8 行）；
- **control 窗口**：同 parent 蛋白、同窗口长度、暴露度匹配（A1b 暴露
  工具复用）、**IEDB metadata 计数 = 0** 的区段；
- **标签**（按可用性阶梯）：
  1. ATLAS MD RMSF（parent 蛋白在库时；离线 API 缓存先查）；
  2. NMR v2 RMSF（`results/flex_a2_nmr_rmsf_v2`——已修复，优先于 v1）；
  3. 回退：AF2 pLDDT 轨道缓存（`flex_a2_atlas_plddt_audit_v1/tracks`，
     只作降级标注，不作主标签）；
- **终点**：parent 级配对差（case − control 的窗口 RMSF 均值差），
  parent 聚类 bootstrap 95% CI；预注册方向：case **更低**（瞬态结构富集）；
  双侧结果均如实冻结；
- **判读封闭规则**：CI 排零且方向为负 → 富集成立（development 级）；
  CI 排零且方向为正 → 反向富集（同样可发表）；CI 跨零 → 无证据。

## 3. 停止规则

- 标签覆盖的 parent 蛋白 < 3 → 「标签饥饿」：降级 AFDB pLDDT 或
  上报换方向（E0 先例）；
- IEDB metadata 查询不可用（网络/配额）→ 只用 C0 结构先例定义 case，
  控制窗口改用"同蛋白非表位暴露段"，降级声明写进 protocol。

## 4. 红线

- IEDB 访问保持 metadata 级（计数 + 查询串，零记录级序列落盘）；
- 不与 ECLS/PAE/J0 冻结结果互证；N1/N2 纪律全文适用；
- 富集是窗口属性的描述，**不是**疾病机制或设计主张。

## 5. 与 C0/C2 的关系

C0 的跨结合者构象共享（tau PHF6*）提供"被读的构象态真实存在"的正面先验；
C1 检验的是**哪些窗口**容易被读——两者合成 CSN 的计算层结论：
「免疫可读窗口 = 自由态系综中的瞬态结构岛」或其否定。
