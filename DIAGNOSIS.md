# BFN 抗体设计系统 — 完整诊断报告

**日期**: 2026-06-20  
**分支**: `feature/idp-aware`

---

## 一、发现总结

| # | 问题 | 严重程度 | 状态 |
|---|------|---------|------|
| 1 | BFN 置信头对所有设计输出相同分数 (std < 1e-4) | 🔴 致命 | 已确认 |
| 2 | BFN 过度自信 12-15x (BFN ipTM 0.86 vs AF2 ipTM 0.06) | 🔴 致命 | 已确认 |
| 3 | Complex 模式产出与 FixBB 几乎相同的 CDR（不同 epitope 之间也相同） | 🔴 致命 | 已确认 |
| 4 | 回收循环用无区分力的 backbone 置信度做 feedback，冲掉 epitope 信号 | 🔴 根因 | 已确认 |
| 5 | V11 SeqConf bypass 训练失败（OC ipTM 从 13.7x → 15.0x，反而退步） | 🟡 已验证 | 已确认 |
| 6 | Encoder 在 forward pass 中确实能看到 epitope（30% CDR logits 不同） | 🟢 好消息 | 已确认 |

---

## 二、完整因果链

```
训练数据:     PDB 抗体-抗原复合物
              每个 backbone 只有极少的设计变体
                    ↓
BFN encoder:  IPA 结构编码器 → backbone 特征主导
              对 epitope 有 16% attention（足够）
              但 backdone 特征在所有设计中完全相同
                    ↓
置信头:       head_plddt(features) + head_plddt_seq(2D bypass)
              backbone 路径 → 饱和 sigmoid (logit ~3.3)
              bypass 路径 → 只有 2 维输入 (entropy+max_prob)
                            entropy 对所有设计 ~2.0（近乎均匀）
                            → bypass 输出 ~0.1，不改变 sigmoid 值
                    ↓
置信度输出:   所有设计 pLDDT=0.96, ipTM=0.81 (std < 1e-4)
              FixBB 和 Complex 模式没有实质区别
                    ↓
回收循环:     prev_conf/prev_iptm/prev_pae 反馈到下一轮
              所有设计收到相同的 feedback
              → 不同 epitope 的微弱信号被相同的 feedback 淹没
                    ↓
采样输出:     不同 epitope 产出相同 CDR 序列
              Complex 模式名存实亡
```

## 三、已尝试的修复

| 阶段 | 方法 | 结果 |
|------|------|------|
| Phase 3 | Disorder head baseline | OC 13.7x |
| Phase 4 | 40% 负样本训练 | OC 12.1x（微降） |
| Phase 5 | Calibration 训练 | disorder loss 恶化 |
| IDP Combined | 冻结 backbone + encoder dropout | OC 14.7x（恶化） |
| V11 SeqConf | 序列感知 2D bypass 置信头 | **OC 15.0x（恶化）** |
| Disorder 校准 | 推理时用 BFN disorder 修正 AF2 分数 | 绕过问题，治标 |
| Ensemble scoring | 多构象鲁棒性评分 | 未实施 |

## 四、建议修复方案：大刀 A+B

### 大刀 A：替换置信头架构 (`receiver.py`)

```
现在:  pLDDT = sigmoid(head_plddt(features) + head_plddt_seq([entropy, max_prob]))
       backbone logit ~3.3              bypass logit ~0.1
       → 饱和 sigmoid，bypass 无效

改后:  pLDDT = sigmoid(head_plddt_seq_deep(probs_seq))
       probs_seq: (N, L, 20) — 完整 AA 概率分布
       网络: 20→128→64→1 (ReLU+LayerNorm+Dropout)
       → 纯序列置信头，每个设计不同
```

对 ipTM 和 PAE 同样处理。

### 大刀 B：对比训练 (`core.py`)

```python
# 新 loss：惩罚置信度方差太小
loss_variance = -torch.std(pred_iptm)

# 新 loss：PPL 排序应与置信度排序一致
loss_ranking = pairwise_ranking_loss(pred_iptm, -ppl_values)

# 总 loss
total += 0.05 * loss_variance + 0.02 * loss_ranking
```

### 改动范围

| 文件 | 改动 | 行数 |
|------|------|------|
| `receiver.py` | 替换 3 个置信头（pLDDT/ipTM/PAE） | ~50 行 |
| `core.py` | 加 variance + ranking loss | ~20 行 |
| checkpoint 加载 | `strict=False`，skip 旧的 backbone 置信头 key | 1 行参数 |

其余全部不变——训练流程、推理管线、多构象数据集、disorder head。

### 预期效果

- 置信头开始区分不同设计（PPL 41→101 对应置信度显著不同）
- 回收循环获得设计特异性 feedback → epitope 信号不再被淹没
- Complex 模式真正利用 epitope 信息
- 训练成本：1 天（从头训置信头，其余 frozen）

## 五、关键数据点

```
同一 backbone 上不同设计的区分力:
  AF2 ipTM range:  0.044 - 0.072  (delta = 0.028)  ← 几乎无区分力
  BFN PPL range:   21 - 101       (delta = 80)     ← 有区分力
  
Encoder attention (Complex 模式):
  Scaffold → Epitope: 16.0% (final layer)
  CDR → Epitope:      14.8% (final layer)
  
CDR logits 差异 (Native vs All-Ala epitope):
  Max diff:  2.52  (显著)
  Mean diff: 0.30
  不同 AA:   30.3%  ← Encoder 能看到 epitope 变化
  
OC 比率 (Abeta42, FixBB 模式):
  Phase 3:   pLDDT 3.77x  ipTM 13.66x
  Phase 4:   pLDDT 3.67x  ipTM 12.12x
  IDP Comb:  pLDDT 3.77x  ipTM 14.74x
  V11 SeqC:  pLDDT 3.66x  ipTM 15.04x  ← 最差
```

## 六、相关文件

| 文件 | 作用 | 改动 |
|------|------|------|
| `disorderflow/modules/bfn/receiver.py` | 置信头架构 + 回收 embedding | 🔴 核心 |
| `disorderflow/modules/bfn/core.py` | 训练 loss + 回收循环 | 🔴 核心 |
| `modules/bfn_loader.py` | BFN 推理加载 | 🟢 不改 |
| `modules/idp_antibody_design.py` | 复合评分 + 管线 | 🟢 不改（已有 disorder 校准） |
| `modules/closed_loop_scorer.py` | 多目标优化 + disorder 批量校准 | 🟢 不改 |
| `disorderflow/datasets/conformation_dataset.py` | 多构象数据加载 | 🟢 不改 |
| `configs/train/bfn_conformation_xpu.yml` | 多构象训练配置 | 🟡 Phase B 解冻 backbone |
| `verify_epitope_attention.py` | epitope 信号测试 | 📋 诊断用 |
| `probe_attention_v2.py` | encoder attention 探针 | 📋 诊断用 |
| `probe_epitope_signal.py` | CDR logits 对比测试 | 📋 诊断用 |
| `benchmark_scoring_v2.py` | 序列特征基准 | 📋 诊断用 |
| `oc_fixbb_vs_complex.py` | FixBB vs Complex 对比 | 📋 诊断用 |
