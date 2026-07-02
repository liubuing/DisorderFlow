# B1 / Chai-1 执行方案 V5 — IDP 抗体结合特异性的最终判据

**日期**: 2026-07-01 | **目标**: 跑通 B1 命门验证——native CDR 在 template-conditioned Chai-1 下 ipTM 从 0.118(blind) → >0.6,且 native vs scram 首次可分
**交接**: DeepSeek V4 / 执行者

---

## 0. 为什么是这条方案(一句话背景)

V11→至今所有纯计算度量(L3 物理 / L4-scaffold / L4-fold / B2)在 native-vs-scram 上**四路全败**,根因统一:**都不含"抗体-肽相对界面几何",而结合特异性恰存在于那个相对几何里。**

B1(template-conditioned 结构预测)是**唯一**把界面几何放进表征的路径:
- crystal 复合物作 template/constraint 注入 → 模型不必盲猜短肽构象(修 M0 失效根因);
- 预测的是"突变对既定界面的影响" → 相对几何在表征里 → 结合信号可被捕获。

**Chai-1 是 B1 的解药**:pip 装、`use_msa_server=False` + `use_templates_server=False` + `constraint_path` 注入,**零二进制依赖**,绕开了 HDOCK 和 hhsearch 的同构死结。

**这是项目最终判据**:过了 → 纯计算首次捕获结合信号,V4 全线打通;不过 → 纯计算在结合特异性上彻底达极限,诚实结案。无第四条路可退。

---

## 1. 假设与判据

### 1.1 核心假设(待验证)
> 给 4HIX/5CSZ 的 crystal 抗体-肽复合物作 Chai-1 的 template/constraint,native CDR 的预测 ipTM 应从 0.118(盲重折叠,M0)升至 >0.6;且 native CDR ipTM 显著高于 scrambled CDR(sep>0.1)。

### 1.2 三态判据(Go/No-Go)

| 结局 | 数据 | 决策 |
|---|---|---|
| **GO** | native ipTM>0.6 且 native vs scram 可分(sep>0.1) | B1 度量成立 → 用它评估 A 的 69 变体,筛选结合保持者 → 冲可用抗体 |
| **PARTIAL** | template 下 ipTM 升至>0.3 但 native-vs-scram 仍不可分 | template 帮了 fold 但未捕获特异性 → 试更严 constraint(锁全 CDR backbone)再判 |
| **NO-GO** | template 下 ipTM 仍<0.3 或 native-vs-scram 无分离 | 纯计算在结合特异性上达极限 → 诚实结案,写 limitation,转湿实验 |

---

## 2. 执行步骤(5 步,每步可独立验证)

### Step 0:Dry-run 确认 Chai-1 API 与显存(不跑真实推理)
**目的**:避免一次写错全盘崩。Chai-1 0.6.1 的 `run_inference` API 在不同小版本有微调,必须先确认签名。

- 在 WSL dry-run:加载一个极短序列(2 链各 10aa),`num_recycle=0`,确认 `run_inference` 可调用、确认 `constraint_path`/`template` 入参名、确认 ESM embeddings 可离线加载;
- 显存探测:RTX 5060 8GB,4HIX Fv(~240 残基)+8aa 肽,`num_recycle=1`,看是否 OOM;若 OOM → 降到 `num_recycle=0` 或仅用 Fv 的 VH 单链+肽试。
- **Step 0 不通过则停**,先解 API/显存,不进 Step 1。

### Step 1:制备 B1 输入(template + FASTA)
复用已生成的 `idp_benchmark_results/templates/{4HIX,5CSZ}_template.pdb`(抗体 H 链 + 配体肽)。

每个抗体制 3 个测试对:
- **native**:native CDR 序列 + 肽;
- **scram**:乱序 CDR + 同肽(对照,2 个 scramble 种子);
- **degen**:低复杂度 CDR(`TTTTTT`)+ 肽(预期低分,floor 对照)。

输出:`b1_inputs/{ab_id}_{native|scram|degen}.fasta` + 对应 `constraints/{ab_id}.constraint`(锁 template 的 Cα backbone)。

### Step 2:Chai-1 template-conditioned 推理(WSL GPU)
对每个输入:
```python
# 伪代码 — API 名以 Step 0 确认为准
from chai_lab.chai1 import run_inference
result = run_inference(
    fasta_file=..., 
    constraint_path=...,          # 注入 crystal template(界面几何来源)
    use_msa_server=False,         # 离线
    use_templates_server=False,   # 不依赖 hhsearch
    use_esm_embeddings=True,      # 单序列模式
    num_recycle=1,                # 8GB 显存约束
    device="cuda",
)
# 提取 ipTM / pTM / pLDDT + 界面 PAE
```
输出每个设计的 `{iptm, ptm, plddt, interface_pae, complex_pdb}`。

### Step 3:命门判据计算
对 4HIX、5CSZ 各自:
- native iptm vs blind(0.118) → 是否 >0.6?
- native iptm vs scram(2 种子均值)→ sep = native − scram,是否 >0.1?
- degen iptm → 应显著低于 native(地板对照,验证度量有下界区分力)。

落盘 `idp_benchmark_results/b1_m0_gate_chai_{ts}.json`,schema 同 M0 以便对比。

### Step 4:判决策(§1.2)
根据三态判据,在 `RESULTS.md` 新增 §11 记录 B1 命门结果 + Go/No-Go 决策。

---

## 3. 显存与失败兜底(RTX 5060 8GB)

Chai-1 在 8GB 上跑抗体-肽复合物是本方案最大现实风险。兜底链:

| 问题 | 兜底 |
|---|---|
| 8GB OOM | `num_recycle=0` → 仅 VH+肽(去掉 VL)→ 半精度(fp16) |
| ESM embeddings 联网 | 确认 `use_esm_embeddings=True` 用本地 ESM 权重(已装 fair-esm);离线跑 |
| constraint API 不符 | Step 0 先确认;Chai 0.6.1 用 `chai_lab.constraints` 模块构造,非路径字符串 |
| Chai 仍不可用 | 退 Boltz-1(同纯 pip,备胎,作为 §5 No-Go 前的最后尝试) |

**红线**:若 Step 0 后 Chai 在 8GB 上连 native 单例都跑不动(所有兜底无效),则环境硬约束成立,**不硬撑**——转 §5 No-Go 评估。

---

## 4. 关键设计决策与理由

### 4.1 为什么用 `constraint_path` 锁 crystal backbone,而非仅作 template hint
M0 失效是因为 AF2 **盲重折叠**短肽。仅 hint template 仍可能让模型偏离 crystal 界面。**锁 Cα backbone(constraint)** 强制模型在真实界面几何上只预测侧链/突变影响——这是把"相对界面几何"硬塞进表征的最强手段,直接对症 §0 根因。

### 4.2 为什么 native vs scram 是判据,而不是看 native 绝对 ipTM
native ipTM>0.6 只证明"template 让 fold 不再盲";**native vs scram 可分**才证明"度量捕获了结合特异性而非 scaffold-fit"。B2 的教训正是:native 比 variant 分低(B2 测 scaffold-fit),所以**必须用 sep 作主判据**,绝对 ipTM 只作辅助。

### 4.3 为什么保留 degen 地板对照
若 native/scram 都难分,需要确认度量至少能区分"合理 CDR vs 退化 `TTTTTT`"——这是度量有效性的下界。连这都分不开,则度量彻底无效,直接 No-Go,不浪费 GPU 跑变体。

---

## 5. No-Go 情境的诚实收尾

若 §1.2 判 NO-GO(B1 也无法区分 native vs scram):

1. **写 limitation**:在 RESULTS.md 明确"纯计算评估在 IDP-scaffold 抗体结合特异性上达极限,L3/L4/B2/B1 四范式同根失效(无/含界面几何皆不可分)";
2. **转向可发表资产**:
   - C 安全港:把 CAID 换标准数据集(修红斑),disorder head 作独立方法学贡献;
   - 生成器侧:V17i Δ+30pp + A template-seeded 69 变体,作为"抗原感知设计方法学"贡献(不绑定结合验证成功);
3. **结合特异性交给湿实验**:计算侧用 B2(scaffold-fit)+ A(合理起点)作"成药性过滤器",产出湿实验候选,不声称纯计算验证结合。

**No-Go 不是失败——它界定了纯计算的能力边界,本身是有效结论。**

---

## 6. 执行清单与产出物

| 步骤 | 产出 | 验收 |
|---|---|---|
| Step 0 | dry-run 日志 | `run_inference` 可调,显存不 OOM |
| Step 1 | `b1_inputs/` + `constraints/` | 3 抗体 × 3 类(native/scram/degen)= 9 输入 |
| Step 2 | `b1_raw/{input}.json` | 每个含 iptm/ptm/plddt/interface_pae |
| Step 3 | `b1_m0_gate_chai_{ts}.json` | native iptm + sep + degen 对照 |
| Step 4 | RESULTS.md §11 + Go/No-Go | 三态判据落定 |

**预估时间**:Step 0(0.5 天)+ Step 1-2(1-1.5 天,9 推理 × 显存约束)+ Step 3-4(0.5 天) = **2-2.5 天**。

---

## 7. 协作约定

| 约定 | 原因 |
|---|---|
| **sep(native−scram)是主判据,绝对 iptm 仅辅助** | B2 教训:绝对分可能测 scaffold-fit 而非结合 |
| **Step 0 不过不进 Step 1** | 避免一次写错全盘崩 |
| **degen 地板对照必跑** | 度量有效性下界,先于变体评估 |
| **No-Go 也是有效结论,不延期** | 已无第四条路,诚实止损 > 无限调参 |
| **Chai 8GB 跑不动不硬撑,转 Boltz-1 或 No-Go** | 环境硬约束需诚实承认 |

---

## 8. 与前序方案的关系

- V3(IDP_METRIC_PLAN_V3):押"给定坐标物理打分"→ 被 c28c97d 证伪;
- V4(IDP_NEXT_STAGE_V4):范式再转移——A 种子 + B template/fold 评估;
- 本 V5:B1(Chai-1)的具体执行方案,V4 方纲里 B1 路径的落地。
- **不推翻 A/B2**:A 已产出 69 变体,B2 已证伪 scaffold-fit。V5 只补 B1 这一最后缺口。B1 若 GO,立刻用 B1 评估 A 的 69 变体。

---

## 附:一句话方案

> 用 Chai-1(pip 装、零二进制)的 template/constraint 注入,把 4HIX/5CSZ crystal 界面几何强制喂入结构预测,在真实界面坐标上预测 native vs scram vs degen CDR 的 iptm。**sep(native−scram)>0.1 是唯一判据**——过了,纯计算首次捕获结合特异性、V4 全线打通;不过,纯计算在结合特异性上彻底达极限,诚实结案。无第四条路。先 Step 0 dry-run 确认 API+显存,不一次写错全盘崩。
