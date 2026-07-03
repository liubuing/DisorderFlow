# Disorder Head 处理决策 — V13.1 §3

**日期**: 2026-07-04 | **状态**: 决策已定
**背景**: V12 CAID 标准基准 (18 IDP, 207 residues, balanced) → AUC-ROC=0.470, separation Δ=−0.011

---

## 1. 事实

| 版本 | 设置 | AUC-ROC | Separation | 标签 |
|---|---|---|---|---|
| 红斑版 (V15) | 单一 Aβ42 42残基 vs 15327 SAbDab | 1.000 | +0.772 | `[FALSIFIED]` |
| 标准版 (V20) | 18 IDP 207残基 vs 80 SAbDab 22816残基 | 0.470 | −0.011 | `[VERIFIED]` (无效) |

**根因分析**:
- Disorder head 从结构 (CA trace) 预测 disorder
- IDP 表位在配体结合态复合物中结构有序 → 模型无法区分配体结合态 IDP vs 折叠蛋白
- 单一 Aβ42 的 AUC=1.0 是假象: Aβ42 42 残基的 RMSF 标签与 SAbDab 折叠残基的结构 pLDDT 有天然分离
- 多 IDP 均衡测试揭示真相: AUC 低于随机 (0.47 < 0.50)

---

## 2. 选项评估

### A: 重训 disorder head
- **数据**: DisProt/CAID9 现成 (需验证是否已有)
- **算力**: 单 head 重训, RTX 5060 可承担
- **目标**: AUC>0.7
- **风险**: 即使重训, 结构输入的根本限制仍在——训练数据也是 folded structures
- **时间**: 1-2 周

### B: 降级为 conditioning 输入 (推荐)
- 诚实标注 AUC=0.47, 保留作方式4的 conditioning 信号源
- 即便 disorder head 不能独立判定 order/disorder, 它仍为生成器提供梯度
- 论文中说明 limitation: "disorder head trained on folded complex structures cannot generalize to IDP disorder detection"
- **时间**: 即日完成
- **风险**: 审稿人可能质疑 conditioning 信号的有效性

### C: 放弃
- 论文不提 disorder head 作为独立贡献
- 仅留作内部 conditioning 工具
- 等同 B 但更保守

---

## 3. 决策: **B (降级)** + C (论文不提独立贡献)

**理由**:
1. 重训 (A) 需 1-2 周且根本限制 (结构输入) 无法突破 → 不追
2. 降级 (B) 诚实且务实: disorder head 作为 conditioning 工具仍然有效 (提供梯度), 但不作独立方法学贡献
3. 红斑版 AUC=1.0 保留产物 + 标注 `[FALSIFIED]`, 不删除不隐瞒, 论文不进结论

**执行**:
- [x] 标准基准已完成: `idp_design_results/caid_standard_v12_20260704_033622.json`
- [x] 红斑版已标注 [FALSIFIED] (见 V13.1 §1 资产表)
- [ ] 论文 limitation 段落中加入此 limitation

---

## 4. 资产标签

| 资产 | 标签 | 文件 |
|---|---|---|
| CAID 红斑 AUC=1.0 | `[FALSIFIED]` | 保留, 不删除 |
| CAID 标准 AUC=0.47 | `[VERIFIED]` | `idp_design_results/caid_standard_v12_20260704_033622.json` |
| Disorder head 作为 conditioning | `[VERIFIED]` | 功能成立, 独立预测不成立 |
