# CSN C1：瞬态结构富集图谱（预注册草案，**未冻结**）

日期：2026-11-08。**状态：草案**——冻结前置：(a) C0/C2 已完成 ✓；
(b) 标签源决策（下表）；(c) IEDB 控制窗口 metadata 查询（网络化，
metadata-only 红线不变，E0b 阶梯复用）。冻结动作沿用先例：本草案定稿 →
sha256 记入 `results/csn_c1_v1/preregistration.json` → 之后才允许任何
case/control 窗口的标签读取。

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
