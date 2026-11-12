# 4HIX H3 候选 BLI 执行包 v1（2026-11-11）

状态：**可执行交付物；本环境未执行任何湿实验。**  
目标：用 BLI 初筛 50 条 4HIX H3 设计中的首批 8 条 + native + scramble，测 Aβ42/DAEFRH 结合，给出可审计的 KD 或明确阴性。

候选来源：`results/design_v1_20261101/order_list_v1.csv` 与 `order_list_v1.fasta`。  
选择依据：ProteinMPNN 自身似然 + 可开发性过滤；**没有任何自研亲和力分数参与排序**。

---

## 1. 样本

| 类别 | 样本 | 数量 | 作用 |
|---|---|---:|---|
| 阳性对照 | `4HIX-native` | 1 | 天然 H3 `VRYDHYSGSSDY`，应结合 DAEFRH |
| 设计 | `4HIX-D001` … `4HIX-D008` | 8 | 首批 8 条设计 |
| 反事实对照 | `4HIX-H3scramble` | 1 | 组成匹配 H3 shuffle；**不是实验阴性** |
| 阴性对照 | BSA | 1 | 非特异结合 |
| 阴性对照 | α-synuclein | 1 | 无关 IDP |
| 阴性对照 | tau | 1 | 无关 IDP |
| 空白 | buffer | 1 | 基线 |

**建议另购**：无关 scFv（如抗溶菌酶 D1.3）作为真正的实验阴性；scramble 只能作为反事实对照。

格式：优先 scFv（VH-GGGGS×3-VL-His6，266 aa），native 必须同批表达；Fab（VH 219 + VL 219）作为备选。

---

## 2. 仪器与耗材

- 仪器：FortéBio Octet 或等价 BLI。
- 传感器：Streptavidin (SA) tips。
- 配体：生物素-Aβ42（或生物素-DAEFRH 肽），>90% HPLC 纯度。
- 缓冲液：PBS + 0.02% Tween-20 + 0.1% BSA，pH 7.4。
- 再生：10 mM glycine-HCl pH 1.5（或 0.5% SDS 短脉冲，按仪器推荐）。
- 板：96 孔黑色平底板，300 µL/孔。

---

## 3. 96 孔板布局

文件：`plate_layout_96well.csv`。

- 列 1–10：native、D001–D008、scramble；
- 列 11：BSA；列 12：buffer；
- 行 A–C：100 nM，三复孔；
- 行 D–F：10 nM，三复孔；
- 行 G–H：buffer / regeneration。

每列一个样本，行内为浓度/复孔；上样体积 200 µL（或按仪器最小体积）。

---

## 4. 运行步骤

1. **预湿**：SA tips 在 buffer 中预湿 10 min。
2. **固化配体**：生物素-Aβ42 100 nM，加载 120 s；达到 0.5–1.0 nm 响应。
3. **基线**：buffer 60 s。
4. **结合**：样本 100 nM / 10 nM，association 120 s。
5. **解离**：buffer 180 s。
6. **再生**：glycine pH 1.5，10 s × 3；回到基线后再跑下一样本。
7. **顺序**：每板先 native、BSA、buffer，再设计；native 失败则整板作废。

**关键控制**：
- 每板至少 1 个 native、1 个 BSA、1 个 buffer；
- 同一 tip 不要跨样本复用，除非再生验证基线恢复 >95%；
- 记录温度、批号、tip 批号、配体批号。

---

## 5. 验收判据

机器可读：`acceptance_criteria.json`。

### 5.1 初筛（100 nM）

- `response(native, 100 nM) ≥ 3 × response(BSA, 100 nM)`，否则**整板无效**；
- 设计 hit：`response(design) ≥ 3 × BSA`；
- 特异性：`response(scramble) ≤ 0.20 × response(native)`；
- 复孔 CV ≤ 20%；
- 10 nM 响应应低于 100 nM（方向一致），否则标记为浓度异常。

### 5.2 定量（hit 后）

- 8 点浓度：0、3.125、6.25、12.5、25、50、100、200 nM；
- 1:1 Langmuir 拟合；
- χ² < 3；Rmax 在预测值 ±20%；
- 报告 kon、koff、KD；
- KD ≤ 100 nM 记为结合 hit；否则记为弱/未测到。

### 5.3 分母

必须记录：

```
合成数 → 表达成功数 → 可溶单体数 → 可测数 → hit 数
```

没有分母，不得写“命中率”。

---

## 6. 数据回填

模板：`data_import_template.csv`。

列：

```
sample_id, role, h3, concentration_nm, replicate, response_nm,
fit_kd_nm, fit_kon_per_ms, fit_koff_per_s, fit_chi2, fit_rmax,
pass_primary, pass_specificity, notes
```

每个样本一行；复孔三行；失败/异常在 `notes` 记录，不删除。

---

## 7. Go / No-Go

| 结果 | 决策 |
|---|---|
| native 不结合或 <3× BSA | 整板无效；检查配体固化/缓冲/构象，不解释设计 |
| native 结合，0 条设计 hit | 记录阴性；不追加合成；回到候选选择/表位 |
| ≥1 条设计 hit | 进入 SPR 定量 + 特异性反筛；追加剩余 42 条 |
| scramble 也强结合 | 结合可能是组成/非特异驱动；需加无关 scFv 与表位突变对照 |

---

## 8. 主张边界

- BLI/SPR 的 KD 是唯一 binding 证据；
- 计算分数（MPNN、ECLS、PAE）不是 binding 证据；
- scramble 是反事实对照，不是实验阴性；
- 没有 KD 之前，不得写“设计出结合体”；
- 所有失败、异常、分母都保留。

---

## 9. 文件

| 文件 | 内容 |
|---|---|
| `sample_manifest.csv` | 78 行样本×浓度×复孔 |
| `plate_layout_96well.csv` | 96 孔布局 |
| `data_import_template.csv` | 数据回填模板 |
| `acceptance_criteria.json` | 机器可读验收判据 |
| `../../results/design_v1_20261101/order_list_v1.fasta` | 合成序列 |
