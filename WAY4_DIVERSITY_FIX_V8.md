# 方式 4 Loss 多样性修复 V8 — 让熵梯度传导到残基多样性

**交付执行**: DeepSeek V4 | **日期**: 2026-07-02 | **目标**: 修复熵→多样性断裂,P2 从"部分走通"推到"走通"
**依据**: `way4-loss-diversity-diagnosis` 记忆 + `core.py:451-525` 实测
**前置**: 不依赖 B1/Chai(全阻塞);复用 V7 已就绪数据(`data/idp_antigen_subset_v2/`,gate PASS)+ V7 checkpoint

---

## 0. 问题与修复概览

**现象**(V7 P2 实测,n=138):
- `ag_disorder_max vs entropy_mean`: Spearman +0.44 p<0.001 ✅(熵维度成立)
- `ag_disorder_max vs unique_aa`: Spearman −0.087 p=0.311 ❌(多样性不传导)
- `entropy_mean=2.96 std=0.0047`(整体拉到高熵端,无方差)

**三个根因**(都在 `disorderflow/modules/bfn/core.py:451-525`):
1. `disorder_align` 只 `push per_residue entropy`(MSE 到 2.5),**从不 push unique_aa** → 熵 2.5 可由"真多样(20种各5%)"或"假多样(3-4种各25%)"同样达到,模型选了假多样
2. `anti_degen` 只在 high 区罚 T/V 占比,**没真罚 6-mer 重复**,把"退化TT"换成"退化RR"
3. `high_mask = cdr_epi_d > 0.3`,但数据 mean=0.115、p90=0.22 → high 档样本太少,mid 档把全员拉平到 entropy~2.96 稀释梯度

**修复**:三处对症,见 §1-3。

---

## 1. 修复 1(最高优先):加 diversity loss,push unique_aa proxy

### 1.1 原理
entropy 只测"分布平不平",不测"残基种类多不多"。需加一个**直接惩罚 top-k AA 概率集中度**的项,逼"真多样(A路径)"。

可微 proxy:`top-3 AA 概率和` 越低越多样。用 `mean of top-3 probs`(每残基取该残基 AA 分布最高的 3 个概率之和)作 penalty——top-3 和高=少数残基集中=假多样;top-3 和低=残基分散=真多样。

### 1.2 代码改动(`core.py`,在 disorder_align 段内,约 497 行后插入)

**现状**(`core.py:488-506`):
```python
if disorder_align_w > 0:
    high_mask = cdr_epi_d > 0.3
    low_mask = cdr_epi_d < 0.1
    mid_mask = ~(high_mask | low_mask)
    loss_da = torch.zeros((), device=device)
    if high_mask.any():
        loss_da = loss_da + F.mse_loss(per_res_entropy[high_mask],
            torch.full_like(per_res_entropy[high_mask], 2.5))
    if low_mask.any():
        loss_da = loss_da + F.mse_loss(per_res_entropy[low_mask],
            torch.full_like(per_res_entropy[low_mask], 0.5))
    if mid_mask.any():
        loss_da = loss_da + F.mse_loss(per_res_entropy[mid_mask],
            torch.full_like(per_res_entropy[mid_mask], 1.5))
    losses['disorder_align'] = disorder_align_w * loss_da
```

**改为**:
```python
if disorder_align_w > 0:
    # V8 修复3: high阈值用分位数,不再用绝对>0.3(数据p90=0.22)
    # 取当前batch epitope disorder的top-30%作high, bottom-30%作low
    if cdr_epi_d.numel() >= 4:
        q70 = torch.quantile(cdr_epi_d.float(), 0.70)
        q30 = torch.quantile(cdr_epi_d.float(), 0.30)
        high_mask = cdr_epi_d >= q70
        low_mask = cdr_epi_d <= q30
    else:
        high_mask = cdr_epi_d > 0.2   # 兜底:绝对阈值降到0.2(数据p90=0.22)
        low_mask = cdr_epi_d < 0.1
    mid_mask = ~(high_mask | low_mask)

    loss_da = torch.zeros((), device=device)
    if high_mask.any():
        # 原entropy push(保留,但target降到2.7,log20=3.0)
        loss_da = loss_da + F.mse_loss(per_res_entropy[high_mask],
            torch.full_like(per_res_entropy[high_mask], 2.7))
    if low_mask.any():
        loss_da = loss_da + F.mse_loss(per_res_entropy[low_mask],
            torch.full_like(per_res_entropy[low_mask], 0.5))
    if mid_mask.any():
        loss_da = loss_da + F.mse_loss(per_res_entropy[mid_mask],
            torch.full_like(per_res_entropy[mid_mask], 1.5))
    losses['disorder_align'] = disorder_align_w * loss_da

    # V8 修复1: diversity loss — 惩罚high区top-3 AA概率集中,逼真多样
    # top-3 prob sum 越低 = 残基越分散 = unique_aa越高
    diversity_w = self.loss_weight.get('diversity', 0.0)
    if diversity_w > 0 and high_mask.any():
        cdr_probs_high = cdr_probs[high_mask]  # (n_high, 20)
        top3_probs, _ = torch.topk(cdr_probs_high, k=3, dim=-1)
        top3_sum = top3_probs.sum(dim=-1)  # (n_high,) 越低越多样
        # target top-3 sum ~0.5 (20种各5%时top3≈0.15... 实际可微调)
        # 用负值push: 想让top3_sum小, loss = mean(top3_sum)
        losses['diversity'] = diversity_w * top3_sum.mean()
```

### 1.3 关键设计点
- `top-3 sum mean` 作 loss(非 MSE)——直接 push 它变小,等价 push 残基分散;
- target 无需设:`mean(top3_sum)` 本身就是准的,越小越多样;
- 只在 high 区激活(低 disorder 要确定性,不该多样);
- 与 entropy MSE 共存:entropy 保证"平",diversity 保证"平得分散"——两者合力逼 A 路径。

---

## 2. 修复 2:真 anti-degen(全 CDR 区 + k-mer + 单 AA dominance)

### 2.1 现状(`core.py:508-513`)
```python
if anti_degen_w > 0:
    if high_mask.any():
        tv_penalty = (cdr_probs[high_mask][:, [19, 16]].sum(dim=-1)).mean()  # T=19,V=16
        losses['anti_degen'] = anti_degen_w * tv_penalty
```
问题:只 T/V、只 high、probs 求和梯度弱、没 6-mer。

### 2.2 改为
```python
if anti_degen_w > 0:
    # V8 修复2: 全CDR区, 罚任何单一AA dominance(非仅T/V), 用max prob
    # max prob高 = 某AA独占 = 退化倾向
    max_prob_per_res = cdr_probs.max(dim=-1).values  # (n_cdr,) 越低越不退化
    # 对全CDR区生效(不只high): 退化可能发生在任何区
    losses['anti_degen'] = anti_degen_w * max_prob_per_res.mean()
    
    # 额外: T/V专项(保留原意,但用logit penalty梯度更强)
    # T=19, V=16; 对logit加惩罚比probs梯度更直接
    # (可选,若max_prob已够则省略)
```

### 2.3 理由
- `max_prob.mean()` 罚"任何 AA 独占",解决"RR 退化"(R 的 max_prob 高)——比只罚 T/V 覆盖广;
- 全 CDR 区生效,防中/低区退化;
- `probs.max` 可微且梯度集中,比 `probs[:,T,V].sum` 强;
- 真要做 6-mer 需采样序列(非概率),训练中难;`max_prob` 是概率层等价物,够用。

---

## 3. 修复 3:阈值分位数化(已并入修复 1)

见 §1.2 代码:`high/low` 从绝对阈值 `>0.3/<0.1` 改为**分位数 top-30%/bottom-30%**(兜底 `>0.2`)。理由:数据 p90=0.22,绝对阈值 `>0.3` 让 high 档几乎空,分位数保证每 batch 都有 high 样本参与梯度。

---

## 4. Config 改动(`configs/train/bfn_v18_v7_per_residue_xpu.yml` 复制为 `bfn_v19_diversity_xpu.yml`)

```yaml
loss_weight:
  seq: 1.0
  disorder_align: 0.05      # 保留(entropy push)
  diversity: 0.03           # V8新增 — top-3集中度惩罚,逼真多样
  anti_degen: 0.02          # 改语义(现为max_prob,非仅T/V)
# 其余:freeze_backbone=true, lr=2e-5, max_iters=4000, resume V7 best.pt
```

**权重建议**:`diversity: 0.03` 起步,若训练后 unique_aa 仍不升 → 提到 0.05;若 seq loss 爆 → 降到 0.02。

---

## 5. 验证(V8 训练后重跑 P2 探针)

用 `probe_way4_v7_per_residue.py`(已有,改 checkpoint 路径指向 V19),重测:
```
ag_disorder_max vs unique_aa:  期望 r > +0.2 p < 0.05  (从 −0.087 翻正)
ag_disorder_max vs entropy_mean: 维持 r > +0.4 (不退步)
top20 CDR: 无 >6 连同残基(anti-degen 仍生效)
```

**V8 Go 判据**:`unique_aa` 相关从 −0.087 翻到显著正(p<0.05),且 entropy 梯度不退步,且无退化 → 方式 4 走通。

---

## 6. 风险与兜底

| 风险 | 兜底 |
|---|---|
| diversity loss 与 seq cross-entropy(主 loss 93.68)冲突 | freeze backbone;diversity_w 从 0.03 起;监控 val seq loss 不涨过 94.0 |
| top-3 sum 全压到 0(过度均匀,丧失特异性) | diversity_w 上限 0.05;监控 entropy 不超过 2.9(log20=3.0) |
| 分位数在 batch 太小时抖 | 已有兜底 `numel<4` 用绝对阈值 |
| max_prob 惩罚过强导致 logit 全缩 | anti_degen_w 保持 0.02;若 seq loss 涨则降 |

---

## 7. 执行顺序

1. 改 `core.py:451-525`(三处修复,见 §1.2 §2.2);
2. 复制 config 为 `bfn_v19_diversity_xpu.yml`,加 `diversity: 0.03`(§4);
3. resume V7 best.pt,freeze backbone,4000 iters;
4. 重跑 P2 探针(§5),检 unique_aa 相关翻正;
5. Go/No-Go。

**硬约束**:不碰 B1;保留 V7 数据与 anti-degen 思路(只改实现);每步留 JSON+commit。

---

## 附:一句话修复

> 三处对症:(1)加 diversity loss,push top-3 AA 概率集中度(可微 unique_aa proxy),逼"真多样"而非"假多样";(2)anti-degen 改 max_prob 全 CDR 区,罚任何 AA dominance(非仅 T/V);(3)high 阈值分位数化(数据 p90=0.22,绝对>0.3 让 high 档空)。V8 训练后重跑 P2,unique_aa 相关从 −0.087 翻正即方式 4 走通。约 1 天改+训+验。
