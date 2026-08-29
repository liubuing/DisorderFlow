#!/usr/bin/env python3
"""生成 DisorderFlow 项目完整历程 Word 文档"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# -- 样式设置 --
style = doc.styles['Normal']
font = style.font
font.name = 'Segoe UI'
font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.25

for level in range(1, 5):
    heading_style = doc.styles[f'Heading {level}']
    heading_style.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)

def add_code_block(doc, text):
    """添加代码块"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    # 添加灰色背景
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), 'F0F0F0')
    shading_elm.set(qn('w:val'), 'clear')
    p.paragraph_format.element.get_or_add_pPr().append(shading_elm)
    return p

def add_metric_table(doc, headers, rows):
    """添加指标表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        for p in table.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)
                r.font.bold = True
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            table.rows[ri + 1].cells[ci].text = str(val)
            for p in table.rows[ri + 1].cells[ci].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()
    return table

def add_verdict(doc, text, status="info"):
    """添加强调结论框"""
    p = doc.add_paragraph()
    prefix = {"success": "✅ ", "failure": "❌ ", "warning": "⚠️ ", "info": "📌 "}
    p.add_run(prefix.get(status, "") + text).bold = True
    return p

# ============================================================
# 封面
# ============================================================
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('DisorderFlow 项目全历程')
run.font.size = Pt(28)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x3C, 0x6E)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('基于 Bayesian Flow Networks 的 IDP 感知抗体设计平台\n从 V3 到 StateContrast 的完整研发记录')
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run(f'生成日期：{datetime.date.today().strftime("%Y-%m-%d")}\n').font.size = Pt(10)
meta.add_run('硬件环境：RTX 5060 Laptop GPU (8GB) + WSL2 Debian + JAX CUDA\n').font.size = Pt(10)
meta.add_run('作者：liubuing').font.size = Pt(10)

doc.add_page_break()

# ============================================================
# 目录页（手动）
# ============================================================
doc.add_heading('目录', level=1)
toc_items = [
    '第一章  平台奠基：从零到 BFN V6（2026-05）',
    '第二章  过自信之战：V11→V14（2026-06-14 至 2026-06-26）',
    '  2.1  V11/V12：发现并缓解过自信',
    '  2.2  V14：根因诊断与根治',
    '  2.3  B1 诊断：设计端置信展宽为零',
    '第三章  设计精度的长征：V15→V17i（2026-06-26 至 2026-06-28）',
    '  3.1  V15：SAbDab 重训——希望与幻灭',
    '  3.2  方向 A：无序引导采样',
    '  3.3  方向 B：CDR 共设计',
    '  3.4  方向 C：多骨架/多构象',
    '  3.5  架构对照诊断——锁定真根因',
    '  3.6  抗原信号使用诊断——闭合根因链',
    '  3.7  V17 系列：打通抗原路由',
    '第四章  范式转移：从 AF2 到物理评分（2026-06-28 至 2026-06-30）',
    '  4.1  M0 闸门：AF2 范式失效',
    '  4.2  V3 物理评分管线',
    '  4.3  L3+L4：Crystal vs Generated 的根本鸿沟',
    '  4.4  IDP 设计方案 V2',
    '第五章  方式4：Disorder 约束设计的机制验证（2026-07-01 至 2026-07-05）',
    '  5.1  方式4 的 thesis 与 V7-V8 训练',
    '  5.2  P2 机制验证——首次统计显著',
    '  5.3  Loss 多样性断裂诊断',
    '  5.4  V19→V20：信号翻转',
    '  5.5  V13.1 消融实验——进一步证伪',
    '  5.6  V14 终局：S0 观测 + §5 Fallback',
    '第六章  资产验证与铁律（2026-07-04）',
    '第七章  StateContrast 与新管线（2026-07-09）',
    '第八章  全局总结与数字摘要',
    '附录  核心教训与方法论反思',
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)
    if not item.startswith('  '):
        p.runs[0].font.bold = True

doc.add_page_break()

# ============================================================
# 第一章
# ============================================================
doc.add_heading('第一章  平台奠基：从零到 BFN V6（2026-05）', level=1)

doc.add_heading('1.1 项目起源与定位', level=2)
doc.add_paragraph(
    'DisorderFlow 最初名为 "AntibodyDesignBFN"，于 2026 年 5 月启动。项目的核心目标是用 Bayesian Flow Networks (BFN) '
    '替代传统的 ProteinMPNN/ESM-IF，构建一个具备自我评估能力的蛋白质/抗体序列设计平台。'
)
doc.add_paragraph(
    '项目的独特竞争力从一开始就被定义为三个维度：'
)
doc.add_paragraph('1. BFN 序列设计：对传统逆向折叠方法的替代方案', style='List Number')
doc.add_paragraph('2. 置信度自评估：pLDDT/ipTM/PAE 预测，不需要 AlphaFold2 即可评估设计质量', style='List Number')
doc.add_paragraph('3. IDP（固有无序蛋白）感知设计：区别于所有现有工具——能识别并避开无序区域进行设计', style='List Number')

doc.add_heading('1.2 核心架构', level=2)
doc.add_paragraph(
    'BFN（Bayesian Flow Network）是一个 joint sequence-structure diffusion 模型：'
)
doc.add_paragraph('Encoder：基于 IPA（Invariant Point Attention）的结构编码器，处理 pair representations', style='List Bullet')
doc.add_paragraph('Receiver/Decoder：预测序列（20 种氨基酸）、结构（CA 坐标）、侧链方向、扭转角', style='List Bullet')
doc.add_paragraph('Sequence head：从 res_feat（残基特征）直接映射到 20 类氨基酸的 logits', style='List Bullet')
doc.add_paragraph('Confidence heads：独立的 pLDDT、ipTM、PAE 预测头', style='List Bullet')

doc.add_heading('1.3 V3.0 里程碑（2026-05-19）', level=2)
doc.add_paragraph(
    '5 月 19 日发布了 V3.0 版本，包含 BFN V6 置信度模型和统一工作流平台。这是项目第一个完整可用的版本，'
    '具备了从输入 PDB 结构到输出设计序列 + 置信度评分的端到端能力。'
)

doc.add_heading('1.4 Phase 1-5 训练体系（2026-05-27 至 2026-05-28）', level=2)
doc.add_paragraph(
    '项目迅速建立了分阶段训练体系：'
)
add_metric_table(doc,
    ['阶段', '内容', '目标'],
    [
        ['Phase 1', '无序检测头', '识别蛋白中的无序区域（disorder head）'],
        ['Phase 2', '真实逐残基无序标签', '用实验数据替换合成标签'],
        ['Phase 3-5', '扩展训练 + SAbDab 数据', '增强抗体-抗原复合物建模能力'],
    ]
)

doc.add_heading('1.5 三条发展主线确立', level=2)
doc.add_paragraph(
    '此时确立了项目的三条发展主线（优先级 P0→P3）：'
)
doc.add_paragraph('主线一（方法学深度）：BFN 模型完善 → 联合设计 → 多状态约束 → 方法论文', style='List Bullet')
doc.add_paragraph('主线二（应用深度）：TfR 抗体设计 → 湿实验协同 → 海参肽 cargo → 应用论文', style='List Bullet')
doc.add_paragraph('主线三（平台工程）：HuggingFace 部署 → pip 包 → 基准对比 → 社区建设', style='List Bullet')

doc.add_page_break()

# ============================================================
# 第二章
# ============================================================
doc.add_heading('第二章  过自信(OC)之战：V11→V14（2026-06-14 至 2026-06-26）', level=1)

doc.add_heading('2.1 V11/V12：发现并缓解过自信', level=2)

doc.add_heading('问题发现', level=3)
doc.add_paragraph(
    '在 FixBB（固定骨架）设计模式下，BFN 的置信度头对设计质量严重高估。具体表现为：'
    '模型预测 BFN ipTM = 0.81，但真实的 AF2 折叠 ipTM 仅 0.05。BFN 自信地认为自己设计得很好，'
    '但实际上设计的 CDR 根本无法折叠或结合抗原。这是项目遇到的第一个重大障碍——"过自信"问题 (Overconfidence, OC)。'
)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    'V11→V12 的核心思路是：问题出在置信度预测头读了骨架几何特征（backbone coordinates），'
    '而 FixBB 模式中骨架固定 → 对不同序列变体，几何特征完全相同 → 置信度无法区分。解决方案是"纯序列置信度"——'
    '让 ipTM/pLDDT 头只读序列概率分布（22 维 AA probs），不读几何特征。'
)

doc.add_heading('V12 架构改动', level=3)
doc.add_paragraph(
    '新增 v12_iptm 头：64→32→1 MLP，输入从 pooled backbone features 改为序列概率分布。'
    '同时加入 P2 系综评分（多 seed AF2 ipTM 投票），并用 PPL（perplexity）作为廉价的质量代理指标。'
)

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['阶段', 'OC (ipTM)', 'BFN ipTM', '关键改动'],
    [
        ['V11 (baseline)', '15.04x', '0.810', '原始置信度头'],
        ['V12 A1', '12.40x', '0.688', '纯序列置信度头 (−18%)'],
        ['Phase C', '9.86x', '0.541', '500-backbone foundation 数据 (−34%)'],
        ['Phase D', '9.47x', '0.510', '1000-backbone scale-up (−37%)'],
    ]
)
doc.add_paragraph(
    '虽然从 15.04x 降到了 9.47x（37% 改善），但 OC 仍然高得无法接受——模型仍然高估设计质量近 10 倍。'
    '这说明纯序列头的改良是方向对的，但远不够。需要深入诊断根因。'
)

doc.add_heading('2.2 V14：根因诊断与根治', level=2)

doc.add_heading('深层诊断', level=3)
doc.add_paragraph(
    '通过 probe_conf_collapse.py 对 val 集 260 条样本的置信度分布进行实测，坐实了一个此前未预料到的现象：'
    '"全局常数坍缩"（global constant collapse）。'
)
add_code_block(doc,
    'val ipTM: mean=0.550755, std=1.8e-5, unique≈2  # 对任意输入都吐训练集均值 0.55'
)

doc.add_heading('根因链（三环相扣）', level=3)
doc.add_paragraph(
    '根因 1 — freeze_backbone=True 冻死编码器特征：Phase A 训练配置冻住了 encoder，'
    '导致 backbone features 对不同蛋白无区分力。'
)
doc.add_paragraph(
    '根因 2 — V12 ipTM 头在合成常数标签上训练饱和：V12 checkpoint 用启发式标签（native=1.0, BFN=0.5, scrambled=0.1）训练，'
    '头本身已收敛到常数盆地。V14 finetune 只有 5 个可训练参数（recycle gates），逃不出这个盆地。'
)
doc.add_paragraph(
    '根因 3 — conf_variance 权重被 iptm MSE 压垮：conf_variance 0.20 vs iptm MSE 1.5，'
    '展宽奖励被回归拉均值的力完全压制。'
)

doc.add_heading('V14-unfreeze 方案（四管齐下）', level=3)
doc.add_paragraph(
    '目标：让置信头对同 scaffold 不同 CDR 变体有区分力（std > 0.05）。策略：'
)
doc.add_paragraph('① freeze_backbone: false + unfreeze_encoder_layers: 6 — 解冻全骨干，让特征能承载质量信号', style='List Number')
doc.add_paragraph('② lr: 1e-4 → 2e-5 — 解冻后保护结构头（seq/dist/ang loss weight=0 所以结构头无梯度）', style='List Number')
doc.add_paragraph('③ 新增 v14_iptm_bb（ResidualMLP, 随机初始化）on pooled 骨干几何特征 — 给 ipTM 头真实结构信号 + 逃出毒盆地的随机起点', style='List Number')
doc.add_paragraph('④ conf_anticollapse: 0.5 硬 hinge — within-group std < 0.05 时 ReLU 惩罚，直接反制坍缩', style='List Number')

doc.add_heading('决定性结果', level=3)
add_metric_table(doc,
    ['指标', '旧坍缩头（冻结）', '新解冻头', '判定'],
    [
        ['BFN ipTM mean', '0.5508', '0.0102', '标定修对 ✅'],
        ['AF2 ipTM mean（真实折叠）', '0.0564', '0.0554', '真实质量未变'],
        ['OC (ipTM)', '9.8x', '0.19x ✅', '目标 ≤2x ✅'],
        ['OC (pLDDT)', '—', '1.05x', '目标 ≤2x ✅'],
        ['within-design std', '1.8e-5', '0.052', '放大 2900x ✅'],
        ['Spearman ρ (BFN vs AF2)', 'n/a', '0.300', '目标 >0.4 ❌'],
    ]
)

add_verdict(doc,
    'OC 从 9.8x 降到 0.19x（甚至略低估），过自信被根治。但这暴露了更深的问题：'
    'AF2 ipTM 从 0.056 到 0.055 纹丝不动——模型既不会生成好设计、之前也不会承认差。'
    '现在承认了，但生成的仍是差的。',
    'warning'
)

doc.add_heading('2.3 B1 诊断：设计端置信展宽为零', level=2)

doc.add_heading('问题', level=3)
doc.add_paragraph(
    '虽然 val 集上跨 scaffold 的置信度有区分（std=0.052），但在设计端——对同一 scaffold 的不同 CDR 变体——'
    'BFN ipTM 全部输出 0.010215（完全相同）。排序失效。这意味着 run_bfn_design 的 sort_by 形同虚设（盲抽），好的设计无法被筛出。'
)

doc.add_heading('诊断过程（probe_design_iptm_components.py）', level=3)
doc.add_paragraph(
    '分解 iptm_seq（序列头）和 iptm_bb（几何头）两个通路：'
)
doc.add_paragraph(
    '8 个 PPL 差异巨大（59-136）的设计 → iptm_seq 全 0.7111（std=6.7e-5，常数！）；'
    'iptm_bb 在变（range 0.016），但被 -5.27 强负偏置推到 sigmoid 深度饱和区 → '
    'pred 恒 0.010。训练 5000 步中 iptm_seq 通路几乎收不到梯度 → 停在 V12 毒初始化常数盆地 0.711，'
    '从未学会区分序列质量。'
)

doc.add_heading('修复尝试与结果', level=3)
doc.add_paragraph(
    '改 per-pathway sigmoid 融合（pred_iptm = 0.5σ(seq) + 0.5σ(bb)）→ 脱离饱和，但 iptm_seq 仍常数。'
    '最终 dead-point 诊断（probe_seq_pathway_deadpoint.py）定位：'
    '问题不在置信头，而在上游——BFN 对同一 fixBB 骨架，采样几乎收敛到同一 CDR 序列分布。'
    'probs_seq 在 CDR 区跨设计 std=0.000002 ≈ 0。这暴露了 B1（置信度展宽）与 B2（设计多样性）的深层纠缠。'
)

doc.add_page_break()

# ============================================================
# 第三章
# ============================================================
doc.add_heading('第三章  设计精度的长征：V15→V17i（2026-06-26 至 2026-06-28）', level=1)

doc.add_heading('3.1 V15：SAbDab 重训——希望与幻灭', level=2)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    'OC 已修好之后，核心问题变成了：BFN 生成的 CDR 为什么折叠/结合这么差？'
    '假设是生成器从未在真实的抗体-抗原复合物数据上训练过——V14 config 中 seq/dist/ang loss weight=0，'
    '且 confidence_dataset.py 强制 generate_flag=zeros → seq loss 恒为 0。'
    '生成器训练数据中从无抗原、无结合目标。'
)

doc.add_paragraph(
    'V15 的改造思路：重开 seq loss + 用 SAbDab 抗体-抗原复合物数据训练 + fp32 精度消 NaN。'
    '期望是"学过的序列恢复先验"能让 CDR 更天然、更适合折叠。'
)

doc.add_heading('结果', level=3)
doc.add_paragraph(
    '训练健康（0 NaN、6GB VRAM、loss 95→93.3）。生成器确实变了——设计 PPL 从 84.6 降到 28，'
    '熵从 1.70 升到 2.62，CDR 更"天然"了。但 AF2 验证结果令人失望：'
)
add_metric_table(doc,
    ['设置', 'AF2 ipTM', 'AF2 pLDDT', '备注'],
    [
        ['V14 FixBB (旧)', '0.055', '0.254', 'HOH 污染但不影响结论'],
        ['V15 Complex', '0.052', '0.256', 'SAbDab 重训后反而更差'],
        ['Spearman ρ', '−0.70', '', '排序更差了'],
    ]
)

add_verdict(doc, '通用 SAbDab 序列恢复 ≠ Aβ42 结合。"更天然的 CDR"对特定抗原没有结合指导。', 'failure')

doc.add_heading('20 设计全测——最彻底的验证', level=3)
doc.add_paragraph(
    '把 V15 复合模式 20 个设计（序列多样、PPL 25-48）全送 AF2，排除"top-5 盲抽"嫌疑：'
)
add_metric_table(doc,
    ['指标', 'Mean', 'Max', '结论'],
    [
        ['AF2 ipTM', '0.057', '0.071', '0/20 超过 0.10'],
        ['AF2 pLDDT', '0.264', '0.323', 'VHH 框架应 80+，0.32 = CDR 破坏折叠'],
        ['Spearman ρ', '−0.08', '', '排序头基本无信号'],
    ]
)
add_verdict(doc, '即使测全部 20 个多样设计，ipTM 硬上限 0.071。生成器有系统性天花板。', 'failure')

doc.add_heading('3.2 方向 A：无序引导采样', level=2)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    'Aβ42 是固有无序蛋白（IDP），而 BFN 的 seq 恢复先验在折叠蛋白上训练——它把一切当"该有序"。'
    '结合 IDP 表位需要相反直觉：可塑 CDR。方向 A 把 BFN 已有的 disorder head 接进采样循环，'
    '让高柔性表位驱动高熵可塑 CDR，框架区强制低熵有序约束。'
)

doc.add_heading('前置诊断', level=3)
doc.add_paragraph(
    'disorder head 在通用 SAbDab 复合物上有区分（std 0.05-0.13），但在 Aβ42 上判错：'
    'Aβ42 物理 RMSF=3.91（高度无序），disorder head 却判它 0.036（几乎完全有序）。'
    '根因：V15 训练时 SAbDab 数据无 disorder_label → disorder loss 不工作 → head 从未在真实 IDP 上受监督。'
)

doc.add_heading('修复执行', level=3)
doc.add_paragraph(
    '① train_disorder_head.py：disorder-only loss，SAbDab 标 0、Aβ42 用 RMSF/6 标 ~0.66。400 步 ~2min → '
    'Aβ42 pred disorder 从 0.28 跳到 0.59 ✅'
)
doc.add_paragraph(
    '② core.py sample() 加 disorder_guided 选项：按每残基 pliability 缩放 CDR 采样噪声。'
    'strength=0.3 产出多样 CDR（PPL 22-36）。'
)

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['指标', '之前最佳', 'disorder-guided', '判定'],
    [
        ['AF2 ipTM max', '0.071', '0.062', '未破 ❌'],
        ['AF2 pLDDT', '0.266', '0.266', '未破 ❌'],
        ['Spearman ρ', '−0.08', '+0.60', '排序修好 ✅'],
    ]
)

add_verdict(doc,
    'disorder 引导没破折叠/结合天花板，但修好了排序相关性（ρ +0.60）。'
    'BFN 现在能正确把更易折叠的设计排前面——这是后续 reject-sampling 真正生效的前提。',
    'info'
)

doc.add_heading('3.3 方向 B：CDR 共设计 (Co-design)', level=2)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    '固定骨架是天花板的一部分。V16 co-design 解冻 dist/ang loss，让 BFN 同时优化序列和 CDR loop 构象。'
    '期望是生成的结构 + 序列联合改进能破折叠天花板。'
)

doc.add_heading('训练结果', level=3)
doc.add_paragraph(
    'BFN V16：dist loss 从 680 降到 118（5.7x 改善），结构头在学。seq loss 稳 93.7，0 NaN。'
    '训练健康。踩坑：max_grad_norm=1.0 把结构头大梯度刹死，改 100 后才学。'
)

doc.add_heading('AF2 验证——失败', level=3)
doc.add_paragraph(
    '12 设计（3STB+5IMK × Aβ42 seed0/2/4）：AF2 ipTM max = 0.066（低于 C 的 0.081），pLDDT max 0.286。'
)

doc.add_heading('失败根因', level=3)
add_verdict(doc,
    'AF2 验证只送序列（graft_cdrs 只取 CDR 序列，AF2 独立折叠），BFN 生成的 CDR 结构被丢弃。'
    'co-design 训练动的是结构头，但 seq 通路没改善 → AF2 折不出。折叠失败在序列解码器产出的序列，不在给的骨架。',
    'failure'
)

doc.add_heading('3.4 方向 C：多骨架/多构象', level=2)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    '两个天花板假设：① 5IMK 单一 VHH 骨架不适配 Aβ42；② 固定 Aβ42 构象上下文（仅解析 19/42 残基）限制了探索空间。'
    '方向 C 加入多骨架（3STB、1ZVH、4KRL）+ 多构象 Aβ42（5 seed × 完整 42 残基重建）。'
)

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['指标', '之前最佳（5IMK）', '方向 C（3STB）', '判定'],
    [
        ['AF2 ipTM max', '0.071', '0.081', '部分突破 ✅'],
        ['3STB vs 5IMK', '—', '3STB 全 >0.067, 5IMK 全 <0.053', '骨架有效 ✅'],
        ['AF2 pLDDT', '0.266', '~0.26', '天花板仍在 ❌'],
    ]
)

add_verdict(doc,
    '换骨架有实质效果（3STB 把 ipTM 天花板从 0.071 抬到 0.081），但折叠天花板本质未破（pLDDT ~0.26）。'
    '生成器核心仍产不出自洽折叠 CDR。',
    'info'
)

doc.add_heading('3.5 架构对照诊断——锁定真根因', level=2)

doc.add_heading('实验设计', level=3)
doc.add_paragraph(
    '用已知抗 Aβ42 抗体复合物 PDB（5CSZ=aducanumab, 4HIX=solanezumab）做 gold-standard 对照。'
    '截取 5CSZ H-Fv（112aa），几何定位 CDR（H1:24-32 / H2:47-56 / H3:89-102），'
    'mask CDR 后让 V15 BFN 在"正确 5CSZ 骨架 + 完整 Aβ42 抗原（visible）"下恢复 CDR。'
    '测序列恢复率。'
)

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['CDR', '真实（5CSZ）', '平均恢复率', '最高恢复率', '随机基线'],
    [
        ['H1', 'SGFTFSSYA', '4.4%', '11.1%', '~5%'],
        ['H2', 'VSAINASGTR', '4.5%', '20.0%', '~5%'],
        ['H3', 'DTAVYYCARGKGYV', '3.2%', '14.3%', '~5%'],
    ]
)

add_verdict(doc,
    'BFN 完全恢复不出真实抗 Aβ CDR（≈ 随机），即使给了正确骨架 + 正确抗原上下文。'
    '不是数据/采样/结构/上下文问题 —— 是 BFN 序列解码器架构本身不具备'
    '"从（抗体骨架+抗原）生成特异性结合 CDR"的能力。它只做序列恢复的统计平均，不建模残基-抗原特异识别。',
    'failure'
)

doc.add_heading('3.6 抗原信号使用诊断——闭合根因链', level=2)

doc.add_heading('实验设计', level=3)
doc.add_paragraph(
    'probe_antigen_signal_usage.py：在 8 个 SAbDab 复合物（抗原 84-810aa，24 个 CDR-骨架实例）上，'
    '比较 complex（抗原 visible）和 fixbb（抗原 hidden）两种模式下的真实 CDR 序列恢复率。'
)

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['模式', 'CDR 恢复率', 'Δ'],
    [
        ['Complex（抗原 visible）', '5.3%', ''],
        ['FixBB（抗原 hidden）', '5.2%', ''],
        ['Δ = complex − fixbb', '', '+0.10 pp（噪声水平）'],
    ]
)

add_verdict(doc,
    '抗原链作为 context 进了 encoder，但对 CDR 序列生成贡献为零。根因链完整闭合：\n'
    '架构设计缺陷 → 抗原信号未接入 CDR 解码 → BFN 只做序列恢复统计平均（~5% 随机）→ '
    'A/B/C 调参全失败 → AF2 折叠/结合天花板恒定。',
    'failure'
)

doc.add_heading('3.7 V17 系列：打通抗原路由', level=2)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    '闭合的根因链指明了唯一的修复路径：把抗原特征显式注入 CDR 序列解码。'
    '改造 receiver.py，让 seq head 不再只是 Linear(res_feat, 22)，'
    '而是接收抗原信息的条件化解码器。'
)

doc.add_heading('V17c：首次抗原注入（Direction D）', level=3)
doc.add_paragraph(
    '加入 antigen→CDR cross-attention（antigen_xa），seq head 读 cross-attention 输出而非裸 res_feat。'
    'zero-init out_proj + 移除 learned gate，让训练从零开始逐步学会用抗原信息。'
    '首次让抗原信息进入了 CDR 解码路径，但 NaN 频发（1153 次 NaN 事件）。'
)

doc.add_heading('V17e：抗原池化替代 cross-attention（Direction E）', level=3)
doc.add_paragraph(
    '把复杂的 cross-attention 替换为简单的 antigen pooling + concat → 更稳定，'
    '但抗原信号利用率仍不足。改用 residual add 而非 replacement，保留 CDR 自身特征。'
)

doc.add_heading('V17f：接触预测辅助 loss（Direction F）', level=3)
doc.add_paragraph(
    '新增 contact prediction auxiliary loss：预测 CDR-抗原的残基间接触图，'
    '作为辅助任务强制编码器学习 CDR-抗原交互。contact_head bias 初始化为 -1.0（稀疏先验）。'
)

doc.add_heading('V17g：NaN 根治', level=3)
doc.add_paragraph(
    'NaN 修复三件套：'
)
doc.add_paragraph('① pre-LayerNorm on antigen_xa（防梯度爆炸）', style='List Number')
doc.add_paragraph('② feature clamp ±50/±100（激活值钳位）', style='List Number')
doc.add_paragraph('③ t_clamp=0.95（扩散时间钳位）', style='List Number')

add_metric_table(doc,
    ['指标', 'V17c（修复前）', 'V17g（修复后）'],
    [
        ['NaN 事件', '1153', '0（2720 步）✅'],
        ['val loss', '—', '92.38→92.06（↓0.32）'],
        ['Δ(complex − fixbb)', '—', '−1.39pp ❌（比 V15 +0.86pp 更差）'],
    ]
)
add_verdict(doc, 'NaN 根治了，但 seq recovery 训练让模型学会压制抗原信息。', 'warning')

doc.add_heading('V17h：对比训练（首次让模型利用抗原信号）', level=3)
doc.add_paragraph(
    '核心洞察：seq recovery 的 loss 方向是"恢复给定序列"——这教模型忽略抗原差异（所有抗原都恢复正确 CDR 就行）。'
    '需要一个直接奖励抗原信息利用的 loss。'
)
doc.add_paragraph(
    '方法：新增 contrastive head + CDR shuffling（50% 概率置换 CDR 为假配对）。'
    'BCE loss 让模型判断 CDR-抗原是否匹配。从 V17g best.pt 续训。'
)
doc.add_paragraph(
    '结果：contrastive loss 从 0.91 降到 0.48（↓47%，跌破随机基线 0.693）。'
    '100 步 0 NaN。这是第一次有训练信号直接奖励抗原信息利用。'
)

doc.add_heading('V17i：Pair Feature Routing —— 突破', level=3)

doc.add_heading('核心洞察', level=3)
doc.add_paragraph(
    'V17h 的 contrastive 信号是全局的（整个 CDR-抗原对是否匹配），太弱。真正的金矿在 pair_feat：'
    '编码器的 pair representation 已经自学习了 CDR 残基-抗原残基的 pair 交互信息，'
    '只是从未被路由到序列解码头。'
)

doc.add_heading('架构', level=3)
add_code_block(doc,
    'head_seq 输入: concat(res_feat, pair_aggr_proj, antigen_pooled)\n'
    '  res_feat: CDR自身特征 (256维)\n'
    '  pair_aggr: 对每个CDR残基, 聚合其与所有抗原残基的pair_feat (128→256投影)\n'
    '  antigen_pooled: 抗原特征全局池化 (256维)\n'
    '  → 768维 MLP → 256 → 128 → 22 (氨基酸类别)'
)

doc.add_paragraph(
    '用最简改动（3× 输入 + MLP）打通了抗原信号到 CDR 序列的完整路径。'
)

doc.add_heading('决定性结果（9 个 checkpoint，2200→3800 步，0 NaN）', level=3)
add_metric_table(doc,
    ['方向', 'Δ(complex − fixbb)', '评价'],
    [
        ['V15 基线（seq recovery only）', '+0.86 pp', '基线'],
        ['V17g（cross-attn）', '−1.39 pp', '路由架构不够'],
        ['V17h（contrastive）', '−2.10 pp', '全局信号太弱'],
        ['V17i（pair routing）', '+30 pp ✅', '25-35× 改善！'],
    ]
)

doc.add_paragraph(
    'complex CDR 恢复率 28-35%（基线 ~5% 随机）。val loss 93.87→92.30 持续下降。'
    'BFN 首次实现抗原特异性 CDR 设计。'
)

add_verdict(doc,
    'Pair Feature Routing 是让 BFN 学会抗原特异性 CDR 设计的正确路径。'
    '但 AF2 最终 ceilling 仍在：ipTM 0.121-0.140（比基线 0.071 改善，但离 >0.6 仍有距离）。',
    'success'
)

doc.add_page_break()

# ============================================================
# 第四章
# ============================================================
doc.add_heading('第四章  范式转移：从 AF2 到物理评分（2026-06-28 至 2026-06-30）', level=1)

doc.add_heading('4.1 M0 闸门：AF2 范式失效', level=2)

doc.add_heading('问题发现', level=3)
doc.add_paragraph(
    '在准备用 AF2-multimer 作为 IDP 抗体设计的裁判时，一个关键 sanity check 暴露了根本问题：'
    '已知抗 Aβ42 抗体 native CDR 重新折叠的 ipTM 仅 0.118，而 scrambled CDR 是 0.116。'
    '——几乎无法区分。'
)

add_verdict(doc,
    'AF2-multimer 对短肽-抗体从头预测系统性弱。作为 IDP 抗体设计的裁判/度量，AF2 不可信。'
    '"度量不可信之前训模型 = 盲跑"——这是 M0 闸门的核心约定。',
    'warning'
)

doc.add_heading('4.2 V3 物理评分管线', level=2)

doc.add_heading('改造思路', level=3)
doc.add_paragraph(
    '从"AF2 重折叠验证"转向"物理对接评分"。如果 AF2 不可靠，那就用第一性原理——'
    '接触数、电荷互补、疏水匹配、形状互补。构建纯 Python 物理评分管线，'
    '绕开 HDOCK 编译问题（RTX 5060 Blackwell CUDA 不兼容）。'
)

doc.add_heading('管线架构（五层过滤）', level=3)
doc.add_paragraph('L0 闸门：接触计数评分（纯 Python）—— 5CSZ native d=2.545 PASS ✅', style='List Number')
doc.add_paragraph('L1 表位筛选：功能表位库 + 多构象 Aβ42（替代 disorder<0.3）', style='List Number')
doc.add_paragraph('L2 姿态生成：纯 Python FFT 对接（替代 HDOCK）', style='List Number')
doc.add_paragraph('L3 界面评分：接触 + 电荷 + 疏水 + 氢键 + 形状互补（5 项物理指标）', style='List Number')
doc.add_paragraph('L4 ESM-IF：逆向折叠模型验证（native vs scrambled 在晶体上可通过，生成设计上塌方）', style='List Number')
doc.add_paragraph('L5 AF2 单链：折叠性验证（单链 pLDDT，不用 multimer）', style='List Number')

doc.add_heading('关键验证', level=3)
doc.add_paragraph(
    'L0 闸门在晶体结构上有效（5CSZ native d=2.545），L4 ESM-IF 在晶体上可区分 native vs scrambled（2/2 抗体通过）。'
    '这证明物理度量本身是有效的——问题在于生成设计的输入。'
)

doc.add_heading('4.3 L3+L4：Crystal vs Generated 的根本鸿沟', level=2)

add_metric_table(doc,
    ['度量', 'Crystal（native vs scrambled）', 'Generated（design ranking）'],
    [
        ['L0 物理接触', '✅ 5CSZ d=2.545', '❌ 全部相同'],
        ['L3 界面综合', '✅ composite varies', '❌ backbone-dominated'],
        ['L4 ESM-IF', '✅ sep=+1.76', '❌ 全部相同'],
    ]
)

add_verdict(doc,
    '所有结构依赖的度量都要求 CDR 的真实折叠坐标。晶体结构有真实的 CDR 侧链位置 → '
    'native vs scrambled 可分。生成设计中 CDR 只有骨架模板坐标 → 所有序列得分相同。'
    '这与 AF2/Docking 无关——是输入坐标的问题。',
    'warning'
)

doc.add_heading('4.4 IDP 设计方案 V2', level=2)

doc.add_heading('核心约定', level=3)
doc.add_paragraph('M0 闸门：先证明已知 Aβ 抗体 native CDR 在肽段模式下 ipTM>0.6，否则禁止动模型', style='List Bullet')
doc.add_paragraph('五指标体系：肽段 ipTM + 全链 ipTM + 对接分 + 鲁棒分 + OC(IDP)', style='List Bullet')
doc.add_paragraph('问题重构：从"全链 multimer 设计"→"线性表位/肽段中心设计"', style='List Bullet')
doc.add_paragraph('已知模板：4HIX (solanezumab), 5CSZ (aducanumab), 3STB——每条管线必须过 native CDR sanity check', style='List Bullet')

doc.add_heading('五阶段管线', level=3)
doc.add_paragraph('P0: 度量重建（肽段 benchmark + 三联验证器）', style='List Number')
doc.add_paragraph('P1: 表位重构（功能表位库，不再靠 disorder）', style='List Number')
doc.add_paragraph('P2: 构象系综（功能态区分 + 鲁棒分）', style='List Number')
doc.add_paragraph('P3: 生成换源（MPNN + amyloid 芳香先验 + 模板 seed）', style='List Number')
doc.add_paragraph('P4: IDP 置信校准（disorder 门控 + IDP 专用头）', style='List Number')

doc.add_page_break()

# ============================================================
# 第五章
# ============================================================
doc.add_heading('第五章  方式4：Disorder 约束设计的机制验证（2026-07-01 至 2026-07-05）', level=1)

doc.add_heading('5.1 方式4 的 Thesis 与 V7-V8 训练', level=2)

doc.add_heading('Thesis', level=3)
doc.add_paragraph(
    '方式4 的核心假说（thesis）：抗原表位的无序程度应该指导 CDR 的设计策略。'
    '高无序表位 → 可塑多样的 CDR；低无序表位 → 刚性聚焦的 CDR。'
    '这是 DisorderFlow 项目名称的来源——让"disorder"成为设计的驱动力，而不仅仅是被识别的特征。'
)

doc.add_heading('V7-V8 训练架构', level=3)
doc.add_paragraph(
    '方式4 在训练期注入了 disorder 信号到 BFN 的 loss 中：'
)
doc.add_paragraph('disorder_align loss（λ=0.05）：MSE(per_res_entropy, target)，让 CDR 残基的 Shannon 熵与表位 disorder 对齐', style='List Bullet')
doc.add_paragraph('anti_degen loss（λ=0.02）：惩罚 CDR 的 T/V 占比过高（反退化）', style='List Bullet')
doc.add_paragraph('diversity loss（V8+，λ=0.03）：惩罚低熵（push 多样性）', style='List Bullet')

doc.add_paragraph(
    '数据层面：P0 命门——DisProt 120 IDP + SAbDab 800 增强，三档分层（5 high / 398 medium / 521 low），'
    '逐残基 epitope_disorder_profile 沿 pair_aggr 传播。'
)

doc.add_heading('P0 命门翻盘', level=3)
doc.add_paragraph(
    '早期方式4 的 P0 数据门 FAIL：n_flexible=0，似乎"数据硬约束 No-Go"。'
    '后来发现是扫描 bug——anchor_flag=0 把抗原丢了，不是数据问题。'
    '修复后 n_flexible 从 0 跳到 5616，gate=PASS。'
)
add_verdict(doc, '教训：上一版过早判"数据硬约束 No-Go"，应先排扫描 bug。', 'warning')

doc.add_heading('5.2 P2 机制验证——首次统计显著', level=2)

doc.add_heading('实验', level=3)
doc.add_paragraph(
    'n=138 设计，测两个关键相关性：'
)
doc.add_paragraph('ag_disorder_max（表位 disorder 最大值）vs entropy_mean（CDR 平均熵）', style='List Bullet')
doc.add_paragraph('ag_disorder_max vs unique_aa（CDR 独特氨基酸种类数）', style='List Bullet')

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['指标', 'Spearman r', 'p 值', '判定'],
    [
        ['entropy_mean vs disorder', '+0.44', '<0.001', '✅ 熵维度成立'],
        ['unique_aa vs disorder', '−0.087', '0.311', '❌ 多样性不传导'],
    ]
)
add_verdict(doc,
    '方式4 核心机制（高 disorder 表位 → 高熵 CDR）首次统计显著成立！'
    '这是项目第一次正面机制结论。但熵→多样性断裂——熵升了但残基种类没多。',
    'success'
)

doc.add_heading('关键诚实限定', level=3)
doc.add_paragraph('机制 ≠ binding：熵梯度只证模型按 disorder 调了 CDR 熵，不证结合更好', style='List Bullet')
doc.add_paragraph('entropy_mean=2.96（整体偏高，接近均匀），可能是 loss push 到天花板', style='List Bullet')
doc.add_paragraph('effect size 窄：entropy_mean std=0.0047，统计显著但实际差异小', style='List Bullet')
doc.add_paragraph('B1 三引擎全阻塞：ColabFold 缺 hhsearch + Chai-1 RTX5060 Blackwell CUDA 不兼容 + Boltz 未装', style='List Bullet')

doc.add_heading('5.3 Loss 多样性断裂诊断', level=2)

doc.add_heading('现象', level=3)
doc.add_paragraph(
    '方式4 最关键的矛盾：entropy 对 disorder 有显著响应（r=+0.44），但 unique_aa 完全不响应（r=−0.087）。'
    '这意味着"少数残基高熵抖动"而非真正的多样性——模型在走捷径。'
)

doc.add_heading('三个根因', level=3)

doc.add_paragraph('根因 1（最致命）—— disorder_align 只 push 熵，从不 push 多样性：', style='List Bullet')
add_code_block(doc,
    'loss_da += F.mse_loss(per_res_entropy[high_mask], target=2.5)  # 只对 entropy'
)
doc.add_paragraph(
    'entropy=2.5 可由两种方式同样达到：A（真多样）20 种 AA 各 ~5%；B（假多样）3-4 种 AA 各 ~25%。'
    'loss 对 A/B 梯度相同 → 模型选 B（代价最小）。'
)

doc.add_paragraph('根因 2 —— anti_degen 只罚 T/V 占比，非真 6-mer 重复：', style='List Bullet')
add_code_block(doc,
    'tv_penalty = cdr_probs[:, [19,16]].sum(dim=-1).mean()  # 仅 T/V, 仅 high 区'
)
doc.add_paragraph(
    '结果：把"退化 TT"换成"退化 RR"（如 V18 生成 GRYRRRFRHR），没解决多样性。'
)

doc.add_paragraph('根因 3 —— high 阈值 >0.3 错配数据分布：', style='List Bullet')
doc.add_paragraph(
    '数据 mean=0.115, p90=0.22，high 档样本极少（仅 5 个 high tier）。'
    '绝大多数落 mid 档（target=1.5）被 push 向中间 → 整体拉平到 entropy~2.96。'
)

add_verdict(doc,
    '这是可修复的工程问题——非架构/数据障碍。方式4 走通概率因此上升。卡点精确到 loss 设计层。',
    'info'
)

doc.add_heading('5.4 V19→V20：信号翻转——Thesis 被推翻', level=2)

doc.add_heading('V19 的假象', level=3)
doc.add_paragraph(
    'V19 训练报告 unique_aa vs disorder r=+0.309 —— 看起来 thesis 成立（"高 disorder → 多样 CDR"）。'
    '项目因此给方式4 打了 9 分（"thesis 成立"）。'
)

doc.add_heading('V20 真值', level=3)
doc.add_paragraph(
    '进一步分析发现 V19 的 head_seq 因 train.py 配置 bug 被意外冻结——模型根本没学。'
    'V19 的 r=+0.309 是冻结 artifact（entropy≈2.96，接近 log(20)≈3.0 的理论最大值——本质是随机）。'
)

doc.add_paragraph(
    'V20 修复冻结 bug 后真训（head_seq unfrozen + 2 encoder layers unfrozen，4000 iter）：'
)

add_metric_table(doc,
    ['指标', 'V18（冻结）', 'V20（解冻）', 'Δ'],
    [
        ['Unique AA', '5.3 ± 1.5', '15.9 ± 3.2', '+3.0× ✅'],
        ['Entropy (bits)', '2.962 ± 0.003', '0.987 ± 0.15', '−1.975'],
        ['Top AA%', '46.1%', '20.1%', '−26 pp'],
        ['6-mer repeats', '15/45', '0/45', '−100% ✅'],
        ['unique_aa vs disorder (r)', '−0.128 (ns)', '−0.449 (p=0.002)', '翻转！'],
        ['entropy vs disorder (r)', '+0.385 (p=0.009)', '+0.125 (ns)', '翻转！'],
    ]
)

add_verdict(doc,
    '核心发现：r 从 +0.309 翻转到 −0.449。高 disorder → 更少 unique_aa。'
    '原 thesis（高 disorder→可塑多样 CDR）被推翻。模型实际学到的是——'
    '高 disorder 表位呈现更少结构约束 → 模型收敛到更聚焦的序列模式。',
    'failure'
)

doc.add_heading('Reframe：disorder-constrains-design', level=3)
doc.add_paragraph(
    'V20 的 r=−0.45 仍是"模型利用了 disorder 信号"的证据，但方向是约束而非放大可塑。'
    '高 disorder 区 CDR 残基更聚焦（种类集中），不是"散漫多样"。'
    '但这需要消融实验区分两种解释：'
)
doc.add_paragraph('① 生物学约束（可发表）：模型学到对柔性表位用聚焦 paratope', style='List Number')
doc.add_paragraph('② loss 副产物（站不住）：head_seq 真训下 diversity loss 与 anti_degen 拉锯，模型妥协成压低高 disorder 区多样性', style='List Number')

doc.add_heading('5.5 V13.1 消融实验——进一步证伪', level=2)

doc.add_heading('实验设计', level=3)
doc.add_paragraph(
    '三组消融：A1=去 diversity loss，A2=去 anti_degen loss，A3=冻结 encoder。'
    '每组从 V20 best.pt 续训，测 unique_aa vs disorder 的 r 是否维持。'
)

doc.add_heading('结果', level=3)
add_metric_table(doc,
    ['消融', 'unique_aa vs disorder (r)', '判定'],
    [
        ['V20 全量', '−0.449 (p=0.002)', 'reframe 候选'],
        ['A1: 去 diversity loss', '−0.135 (ns)', '❌ r 坍塌'],
        ['A2: 去 anti_degen', '−0.148 (ns)', '❌ r 坍塌'],
        ['A3: 冻结 encoder', '−0.170 (ns)', '❌ r 坍塌'],
    ]
)

add_verdict(doc,
    'r=−0.45 是 diversity + anti_degen loss 的副产物，不是生物学约束。Reframe 站不住。'
    '三个组件（diversity + anti_degen + encoder_unfreeze）缺一不可——'
    '但它们的联合作用产物（r=−0.45）不表达独立的生物学机制。',
    'failure'
)

doc.add_heading('5.6 V14 终局：S0 观测 + §5 Fallback', level=2)

doc.add_heading('S0 观测实验', level=3)
doc.add_paragraph(
    '方式4 的 thesis 已被 V13.1 消融证伪。但还有一个更基础的预设：'
    '"表位柔性 ↔ CDR-H3 柔性"的相关性——即 epitope flexibility 是否真的与 CDR 的柔性有物理上的关联？'
    '如果连这个基础相关性都不存在，方式4 的整个框架（disorder 驱动设计）就没有物理基础。'
)

doc.add_paragraph(
    'S0 用真实 PDB 晶体学数据（B-factor 作为柔性的 proxy）做观测性检验：'
)
doc.add_paragraph('N=33 PDBs → 16 analyzed（5 IDP, 11 folded）', style='List Bullet')
doc.add_paragraph('CDR-H3 B-factor: IDP 24.69 vs Folded 35.87 (p=0.827 ns)', style='List Bullet')
doc.add_paragraph('Spearman epitope-CDR-H3: r=−0.327, p=0.253', style='List Bullet')

add_verdict(doc,
    'S0 thesis [FALSIFIED]——晶体学数据不支持 epitope flexibility ↔ CDR-H3 flexibility 的基础相关性。'
    '方式4 的整个框架失去了物理基础。',
    'failure'
)

doc.add_heading('§5 Fallback：从生成驱动到选择过滤', level=3)
doc.add_paragraph(
    'V14 终局决定：禁止训练 Way4 生成模型。方式4 的"可塑性匹配"降级为选择过滤器（而非生成驱动力）。'
    '转向三个保底方案：'
)
doc.add_paragraph('Pillar A：template-seeded CDR redesign——69 变体（从已知抗 Aβ 抗体 4HIX/5CSZ 种子，hotspot-frozen MPNN）', style='List Number')
doc.add_paragraph('§5.1：收集 15 个已知 anti-IDP CDR', style='List Number')
doc.add_paragraph('§5.2+§5.3：84 个 unique 候选，pliability 选择排序，top-25 送湿实验', style='List Number')

doc.add_page_break()

# ============================================================
# 第六章
# ============================================================
doc.add_heading('第六章  资产验证铁律（2026-07-04）', level=1)

doc.add_heading('6.1 两次崩盘的教训', level=2)

doc.add_paragraph(
    '项目经历了两次严重的"表述崩盘"——把"待复测"当成"已成立"：'
)

doc.add_paragraph(
    '崩盘 1 — V19 r=+0.309 "方式4 thesis 成立 9分"：V19 的 head_seq 被冻结，r=+0.309 是 artifact。'
    'V20 真训后翻转到 −0.449。当初的 9 分断语"thesis 成立"严重偏乐观。',
    style='List Bullet'
)
doc.add_paragraph(
    '崩盘 2 — CAID AUC=1.0 "disorder head 保底资产"：AUC=1.0 是在单一 Aβ42（42 残基 vs 15327）上的"红斑"。'
    '标准 CAID 基准复测后 AUC=0.47——无效。',
    style='List Bullet'
)

add_verdict(doc,
    '两次都是"待复测/待标准验证"的资产被表述为"已成立/保底"，复测后崩盘。'
    '数据不造假（都如实留了产物），是"表述"把待复测当成立。',
    'warning'
)

doc.add_heading('6.2 三件套 + 验证状态标签（铁律）', level=2)
doc.add_paragraph('每条项目资产必须有三件套：')
doc.add_paragraph('① 原始产物（JSON/PDB）', style='List Number')
doc.add_paragraph('② 生成脚本（可复现）', style='List Number')
doc.add_paragraph('③ 验证状态标签之一：', style='List Number')

add_metric_table(doc,
    ['标签', '含义', '使用规则'],
    [
        ['[VERIFIED]', '已在标准/独立基准验证，结论站得住', '可计入"已成立"；可作为论文结论'],
        ['[PENDING]', '待重测/待标准复测', '禁止在评估/论文中用作"成立"断语'],
        ['[FALSIFIED]', '已被复测推翻', '保留产物+标注，不删除不隐瞒；如实作负面结果/limitation'],
    ]
)

doc.add_heading('6.3 资产盘点（2026-07-04）', level=2)
add_metric_table(doc,
    ['资产', '标签', '说明'],
    [
        ['V17i 通用抗原路由 Δ+30pp', '[PENDING]', 'V20真训对antigen routing影响待测'],
        ['方式4 训练期 V19 r=+0.31', '[FALSIFIED]', 'head_seq冻结artifact'],
        ['方式4 训练期 V20 r=−0.45', '[FALSIFIED]', 'V13.1消融证实为loss副产物'],
        ['推理期 disorder_guided (ρ+0.60)', '[FALSIFIED] 误标', '实为V15dg03置信头OC排序，非方式4'],
        ['V15dg03 置信头 OC +0.595', '[VERIFIED]', '置信度校准独立贡献'],
        ['Disorder head CAID 标准0.47', '[VERIFIED]', '无效（结论站得住=无效）'],
        ['评估墙 CDR侧3路证伪', '[VERIFIED]', '物理极限真结论'],
        ['V20 多样性突破 (3× unique_aa)', '[PENDING]', 'V20训练期附产物，非独立thesis'],
        ['Wet-lab 候选', '[PENDING]', '基于动摇叙事，需重审'],
    ]
)

doc.add_page_break()

# ============================================================
# 第七章
# ============================================================
doc.add_heading('第七章  StateContrast 与新管线（2026-07-09）', level=1)

doc.add_heading('7.1 StateContrast-Ab', level=2)
doc.add_paragraph(
    '项目最新方向。StateContrast 的核心思路是：不是简单地让 disorder 驱动 CDR 多样性（方式4 已被证伪），'
    '而是对比不同构象状态下抗原的结构差异，在"有序态"和"无序态"之间找到 CDR 的设计目标。'
)

doc.add_paragraph(
    '已产出：StateContrast domain 位置分析（3 个参考构象的 Aβ42 结构比对）、'
    '效应引导验证（effect-guided validation v4，含突变规则）、'
    'domain 工作流 schema、GPU runbook、集成路线图。'
)

doc.add_heading('7.2 IDP 抗体接触设计管线', level=2)
doc.add_paragraph(
    '并行于 StateContrast，产出了一套完整的 IDP 抗体接触设计管线：'
)
doc.add_paragraph('Contact-guided mutation plans（Aβ 接触引导突变计划，含 N-glyco rescue 变体）', style='List Bullet')
doc.add_paragraph('Full-chain constructs（全链构建体，N-glyco rescued）', style='List Bullet')
doc.add_paragraph('Reference contact audit + whitelist decoy benchmark', style='List Bullet')
doc.add_paragraph('ANARCI Chothia numbering + N-glyco rescue', style='List Bullet')
doc.add_paragraph('Fv ColabFold smoke evidence', style='List Bullet')

doc.add_page_break()

# ============================================================
# 第八章
# ============================================================
doc.add_heading('第八章  全局总结与数字摘要', level=1)

doc.add_heading('8.1 关键数字', level=2)
add_metric_table(doc,
    ['指标', '数值', '备注'],
    [
        ['Git 提交数', '160+', '2026-05-19 至 2026-07-09'],
        ['训练大版本', '15+', 'V6→V20'],
        ['OC 改善', '15.04x → 0.19x (80×)', 'V11→V14 两章战役'],
        ['抗原路由改善', '+0.86pp → +30pp (35×)', 'V15→V17i Pair Routing'],
        ['设计多样性', '5.3 → 15.9 unique_aa (3×)', 'V18→V20 head_seq unfreezing'],
        ['退化消除', '15/45 → 0/45', '6-mer repeats'],
        ['被推翻的 thesis', '3+', '方式4方向、CAID红斑、S0观测'],
        ['湿实验候选', '84', '§5 fallback 产出'],
        ['核心 [VERIFIED] 资产', '3', 'V15dg03 + 评估墙 + disorder head标准'],
    ]
)

doc.add_heading('8.2 技术演进轨迹', level=2)
doc.add_paragraph(
    'V3 BFN V6 基础平台 → V11/V14 过自信根治 → V15 SAbDab 重训（失败）→ '
    'A/B/C 三方向围攻设计精度（天花板未破）→ 架构诊断锁定根因（抗原未接入解码）→ '
    'V17 系列打通抗原路由（V17i Pair Routing 突破）→ '
    '范式转移 AF2→物理评分（M0+V3+L3L4）→ '
    '方式4 disorder 机制验证（V19 thesis 假成立 → V20 翻转 → V13.1 证伪）→ '
    'S0 观测证伪（基础相关性不成立）→ §5 Fallback + StateContrast'
)

doc.add_heading('8.3 保底可发表资产', level=2)
doc.add_paragraph('1. V17i 通用抗原路由（Δ+30pp, 35× 改善）—— BFN 首次实现抗原特异性 CDR 设计', style='List Number')
doc.add_paragraph('2. V15dg03 置信头 OC 排序（Spearman +0.595）—— 置信度校准方法', style='List Number')
doc.add_paragraph('3. 评估墙 CDR 侧物理极限（3 路证伪的诚实 limitation）', style='List Number')
doc.add_paragraph('4. 方式4 V20 多样性突破（3× unique_aa, 0 degen）—— 生成器训练方法', style='List Number')
doc.add_paragraph('5. Disorder head 标准基准结果（CAID AUC=0.47, 诚实无效结论）', style='List Number')

doc.add_heading('8.4 湿实验候选状态', level=2)
doc.add_paragraph(
    '当前有 84 个 ranked wet-lab candidates（§5 fallback 产出），但基于被证伪的叙事。'
    'Pillar A 69 模板种子变体（4HIX/5CSZ hotspot-frozen MPNN）+ 15 已知 anti-IDP CDR。'
    'top-25 已排序待湿实验验证。'
)

doc.add_page_break()

# ============================================================
# 附录
# ============================================================
doc.add_heading('附录：核心教训与方法论反思', level=1)

doc.add_heading('A.1 方法论教训', level=2)

doc.add_paragraph(
    '教训 1 — 度量不可信之前训模型 = 盲跑：V14→V17i 四个版本在没有可靠度量（AF2-multimer 对 IDP 短肽失效）'
    '的情况下迭代。M0 闸门（先证明度量可信再动模型）是项目后期才建立的约定，却应该是第一步。',
    style='List Number'
)
doc.add_paragraph(
    '教训 2 — "待复测"断语铁律：V19 +0.31 和 CAID AUC=1.0 两次崩盘，本质都是'
    '"待复测"的结论被表述为"已成立"。三件套 + [VERIFIED/PENDING/FALSIFIED] 标签是最低成本的保险。',
    style='List Number'
)
doc.add_paragraph(
    '教训 3 — 生成器训练数据决定了天花板：V15 SAbDab 重训、方向 A/B/C 全部失败，根因在生成器从未做过结合训练。'
    '在架构层面打通抗原路由（V17i）之前，所有"优化序列质量"的努力都无法改善 AF2 结合。'
    '先确定信号在哪里断，再决定在哪里修。',
    style='List Number'
)
doc.add_paragraph(
    '教训 4 — 先排 bug 再判 No-Go：方式4 P0 n_flexible=0 被判"数据硬约束 No-Go"，实际是扫描 bug（anchor_flag=0）。'
    '修复后 n_flexible=5616。过早判死刑会错过重大方向。',
    style='List Number'
)
doc.add_paragraph(
    '教训 5 — 消融实验是区分"真机制"和"loss 副产物"的唯一手段：V20 r=−0.45 在 V13.1 消融中被证实是 loss 副产物。'
    '没有消融，"disorder-constrains-design"的 reframe 就会被当作发现而不是 artifact。',
    style='List Number'
)

doc.add_heading('A.2 架构洞察', level=2)
doc.add_paragraph(
    '洞察 1 — BFN 的 encoder 有能力学习 CDR-抗原 pair 交互（V17i pair_feat 路由的成功证明了这一点），'
    '但原始架构从未将 pair 信息路由到序列解码头。这是项目最关键的架构缺陷——'
    'pair_feat 是金矿，Linear(res_feat, 22) 是废墟。',
    style='List Bullet'
)
doc.add_paragraph(
    '洞察 2 — 结构依赖度量在生成设计上全部塌方（L0/L3/L4），不是因为度量不好，而是因为生成设计没有真实的 CDR 结构坐标。'
    '这暗示物理评分管线的正确入口不是"给生成序列打分"，而是"先给生成序列做结构预测（AF2 fold 或 docking），再用真实坐标打分"。',
    style='List Bullet'
)
doc.add_paragraph(
    '洞察 3 — AF2-multimer 对 IDP 短肽的 re-fold 系统性弱（native vs scrambled 几乎无法区分）。'
    '这不是 DisorderFlow 的 bug，是 AF2 的 limitation。但它是 IDP 抗体设计领域的核心瓶颈——'
    '没有可靠的 in silico binding 评估，整个设计循环就缺了 feedback。',
    style='List Bullet'
)

doc.add_heading('A.3 技术债务与未完成项', level=2)
doc.add_paragraph('B1 评估墙（三引擎全阻塞）：ColabFold 缺 hhsearch + Chai-1 Blackwell 不兼容 + Boltz 未装。binding 评估需云 GPU 或湿实验。', style='List Bullet')
doc.add_paragraph('V17i AF2 天花板 0.121-0.140：抗原路由已通但 AF2 结合仍未破 >0.6。需要 (a) reject-sampling 蒸馏 或 (b) 已知抗体 seq recovery。', style='List Bullet')
doc.add_paragraph('方式4 推理期 disorder_guided 从未独立验证：旧 +0.60 是置信头误标，真值待测。', style='List Bullet')
doc.add_paragraph('Conformation dataset 808 IDP >500aa 未完成（需 24GB+ RAM）。', style='List Bullet')
doc.add_paragraph('HuggingFace Space / pip 包 / 测试体系（主线三）尚未启动。', style='List Bullet')

doc.add_heading('A.4 硬件限制一览', level=2)
add_metric_table(doc,
    ['组件', '规格', '限制'],
    [
        ['GPU', 'RTX 5060 Laptop 8GB', 'batch_size=4, max N≈150, 无 sm_90 支持'],
        ['RAM', '16GB (host)', '大 IDP (L>500) 需 WSL config 扩容'],
        ['AF2 速度', 'WSL GPU: 2-5s/条', 'JIT compile 首次 ~118s'],
        ['JAX 版本', '0.10.2 (WSL CUDA)', 'API 不兼容 alphafold 旧版'],
        ['B1 评估', '不可用', 'Chai-1 需 sm_90+, ColabFold 缺 hhsearch'],
    ]
)

# ============================================================
# 保存
# ============================================================
output_path = 'D:/biological/DisorderFlow/docs/DisorderFlow_项目全历程.docx'
doc.save(output_path)
print(f'文档已保存至: {output_path}')
