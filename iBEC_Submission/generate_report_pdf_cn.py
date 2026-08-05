# -*- coding: utf-8 -*-
"""
iBEC 项目报告 PDF 生成器（中文版）
带术语中英对照注释 + 脚注补充说明
"""
import sys
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Register CJK fonts ──
pdfmetrics.registerFont(TTFont('MSYH', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('MSYHBD', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))

PAGE_W, PAGE_H = A4

# ── Footnote flowable ──
class FootnoteBlock(Flowable):
    """A horizontal rule + footnote text block."""
    def __init__(self, text, width=None):
        Flowable.__init__(self)
        self.text = text
        self._width = width or (PAGE_W - 5*cm)
        self.height = 28

    def wrap(self, availWidth, availHeight):
        self._width = min(self._width, availWidth)
        return (self._width, self.height)

    def draw(self):
        c = self.canv
        # Horizontal rule
        c.setStrokeColor(HexColor('#CCCCCC'))
        c.setLineWidth(0.5)
        c.line(0, self.height - 2, self._width * 0.3, self.height - 2)
        # Footnote text
        c.setFont('MSYH', 8)
        c.setFillColor(HexColor('#777777'))
        # Word-wrap manually
        max_chars = int(self._width / 4.2)  # ~4.2pt per CJK char at 8pt
        lines = []
        remaining = self.text
        while remaining:
            if len(remaining) <= max_chars:
                lines.append(remaining)
                break
            # Find a good break point
            cut = max_chars
            lines.append(remaining[:cut])
            remaining = remaining[cut:]
        y = self.height - 16
        for line in lines:
            c.drawString(0, y, line)
            y -= 11


def build_report():
    output_path = "C:\\biological\\DisorderFlow\\iBEC_Submission\\28-解析无序-项目报告_中文版.pdf"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ──
    styles.add(ParagraphStyle(
        name='CNTitle',
        fontName='MSYHBD',
        fontSize=18,
        leading=26,
        spaceAfter=6,
        textColor=HexColor('#1a1a2e'),
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='CNSubtitle',
        fontName='MSYH',
        fontSize=12,
        leading=18,
        spaceAfter=4,
        textColor=HexColor('#444444'),
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='CNSection',
        fontName='MSYHBD',
        fontSize=14,
        leading=20,
        spaceBefore=18,
        spaceAfter=8,
        textColor=HexColor('#1a1a2e'),
    ))
    styles.add(ParagraphStyle(
        name='CNSubSection',
        fontName='MSYHBD',
        fontSize=12,
        leading=17,
        spaceBefore=12,
        spaceAfter=6,
        textColor=HexColor('#2d3436'),
    ))
    styles.add(ParagraphStyle(
        name='CNBody',
        fontName='MSYH',
        fontSize=10,
        leading=16,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name='CNFormula',
        fontName='Courier',
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=6,
        textColor=HexColor('#2d3436'),
    ))
    styles.add(ParagraphStyle(
        name='CNTableCell',
        fontName='MSYH',
        fontSize=9,
        leading=13,
    ))
    styles.add(ParagraphStyle(
        name='CNFootnote',
        fontName='MSYH',
        fontSize=8,
        leading=11,
        textColor=HexColor('#777777'),
        spaceAfter=2,
    ))
    # Annotation style: for inline English terms
    styles.add(ParagraphStyle(
        name='CNAnnotation',
        fontName='MSYH',
        fontSize=9,
        leading=13,
        textColor=HexColor('#087F6D'),
        leftIndent=12,
        spaceAfter=4,
    ))

    elements = []

    # Helper: add footnote
    def add_footnote(text):
        elements.append(FootnoteBlock(text))
        elements.append(Spacer(1, 4))

    # ==================== 封面 ====================
    elements.append(Spacer(1, 3*cm))
    elements.append(Paragraph(
        "DisorderFlow：面向无序蛋白表位的<br/>贝叶斯流网络抗体 CDR-H3 设计平台",
        styles['CNTitle']
    ))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph(
        "DisorderFlow: Bayesian Flow Networks for<br/>Disorder-Aware Antibody CDR-H3 Design",
        ParagraphStyle('ENSubtitle', fontName='MSYH', fontSize=11, leading=15,
                       textColor=HexColor('#888888'), alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 1*cm))
    elements.append(Paragraph("第 28 号队伍 — 解析无序", styles['CNSubtitle']))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("成员：陈昊阳、胡静、余家瑞", styles['CNSubtitle']))
    elements.append(Paragraph("单位：齐鲁工业大学", styles['CNSubtitle']))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("iBEC 2026 — 国际生物信息学工程竞赛", styles['CNSubtitle']))
    elements.append(Paragraph("2026 年 8 月", styles['CNSubtitle']))
    elements.append(PageBreak())

    # ==================== 第1节：执行摘要 ====================
    elements.append(Paragraph("1. 执行摘要", styles['CNSection']))
    elements.append(Paragraph(
        "DisorderFlow 是一个面向<strong>内在无序蛋白（Intrinsically Disordered Protein, IDP）</strong>"
        "表位的抗体 CDR-H3 序列设计计算平台。该平台将<strong>贝叶斯流网络（Bayesian Flow Network, BFN）</strong>"
        "与<strong>表位条件似然评分（Epitope-Conditioned Likelihood Scoring）</strong>相结合，"
        "在骨架几何和表位无序谱双重条件下生成并排序 CDR-H3 候选序列。",
        styles['CNBody']
    ))
    elements.append(Paragraph(
        "本项目解决了计算抗体设计中的一个关键空白：现有方法大多假设表位为刚性、结构良好的构象，"
        "然而许多具有治疗意义靶标（如淀粉样蛋白-β / Amyloid-beta、tau 蛋白、α-突触核蛋白 / alpha-synuclein）"
        "均为内在无序蛋白。DisorderFlow 引入了一种<strong>无序感知评分框架</strong>，"
        "通过<strong>位置敏感路由（Position-Sensitive Routing）</strong>机制将抗原残基无序值"
        "耦合到 CDR-抗原配对特征中，使模型能够学习到无序表位不同区域对 CDR-H3 序列选择施加不同约束。",
        styles['CNBody']
    ))
    elements.append(Paragraph(
        "主要成果包括：（1）严格验证的<strong>表位条件似然位移（Epitope-Conditioned Likelihood Shift, ECLS）</strong>"
        "指标在 46 个抗原簇上展示了阳性原生序列 vs. 打乱序列判别能力（均值优势 0.217，80.4% 阳性簇）；"
        "（2）确定性扭转扰动恢复基准（T2.1 v2）达到 96.2% 有效结构比例，100% 阳性接触恢复；"
        "（3）在 4HIX 支架上的前瞻性 IDP 抗体设计验证中，20 个 CDR-H3 设计序列全部通过 "
        "<strong>AlphaFold 2 多聚体（AF2 Multimer）</strong>验证，最优设计的界面质量超过天然序列"
        "（ipTM 0.462 vs. 天然 0.449）。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注①：ipTM（interface predicted Template Model）为 AlphaFold 预测的界面质量指标，"
        "范围 0–1，值越高表示界面预测越可靠。pLDDT 为每残基预测的局部置信度。"
    )

    # ==================== 第2节：背景与动机 ====================
    elements.append(Paragraph("2. 背景与动机", styles['CNSection']))

    elements.append(Paragraph("2.1 IDP 抗体设计挑战", styles['CNSubSection']))
    elements.append(Paragraph(
        "<strong>内在无序蛋白（IDP）</strong>在生理条件下缺乏稳定的三维结构，"
        "却在细胞信号传导、调控和疾病中发挥关键作用。"
        "许多神经退行性疾病靶标，包括淀粉样蛋白-β（阿尔茨海默病）、tau 蛋白和"
        "α-突触核蛋白（帕金森病），均为 IDP 或含有无序区域。",
        styles['CNBody']
    ))
    elements.append(Paragraph(
        "针对 IDP 表位的抗体疗法面临根本性挑战：表位可采取多种构象，"
        "成功的 CDR-H3 必须适应这种构象异质性。"
        "传统计算抗体设计管线假设单一的刚性表位构象，对无序靶标根本不适用。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注②：CDR-H3（互补决定区 H3）是抗体可变区中多样性最高的环区，"
        "是决定抗原特异性的主要因素。其长度和序列变异直接影响结合亲和力和特异性。"
    )

    elements.append(Paragraph("2.2 贝叶斯流网络（BFN）用于序列设计", styles['CNSubSection']))
    elements.append(Paragraph(
        "<strong>贝叶斯流网络（Bayesian Flow Network, BFN）</strong>提供了一种蛋白质序列设计的"
        "生成框架，通过迭代细化氨基酸序列上的概率分布来工作。"
        "与自回归或扩散方法不同，BFN 在每个位置维持离散类别分布，"
        "并通过基于骨架几何和结构条件化的<strong>接收网络（Receiver Network）</strong>进行更新。",
        styles['CNBody']
    ))
    elements.append(Paragraph(
        "DisorderFlow 扩展了 BFN 框架，引入了预测每残基无序概率的<strong>无序头（Disorder Head）</strong>"
        "和将这些预测耦合到 CDR-抗原交互特征的<strong>位置敏感路由机制</strong>。",
        styles['CNBody']
    ))

    elements.append(Paragraph("2.3 表位条件似然位移（ECLS）", styles['CNSubSection']))
    elements.append(Paragraph(
        "ECLS 指标衡量 CDR-H3 序列是否优先被肽表位坐标的存在所支持。其定义为：",
        styles['CNBody']
    ))
    elements.append(Paragraph(
        "ECLS(s) = NLL<sub>complex</sub>(s) &minus; NLL<sub>peptide-stripped</sub>(s)",
        styles['CNFormula']
    ))
    elements.append(Paragraph(
        "其中 NLL 为 <strong>ProteinMPNN</strong> 下的平均每残基负对数似然。"
        "正的 ECLS 优势（打乱序列 ECLS 均值减去天然序列 ECLS）表明天然序列从表位上下文中获得的收益"
        "大于成分匹配的对照序列。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注③：ProteinMPNN 是一种基于消息传递的蛋白质序列设计模型（Dauparas et al., Science 2022），"
        "此处用作评估 CDR-H3 序列似然的打分器。200 个成分匹配打乱作为反事实对照。"
    )

    # ==================== 第3节：技术方案 ====================
    elements.append(Paragraph("3. 技术方案", styles['CNSection']))

    elements.append(Paragraph("3.1 架构概览", styles['CNSubSection']))
    elements.append(Paragraph(
        "DisorderFlow 由四大核心组件构成："
        "（1）<strong>BFN 核心模块</strong>——实现 20 种氨基酸类别分布的贝叶斯流网络；"
        "（2）<strong>无序头</strong>——预测每残基无序概率，使用焦点加权训练（focal weighting, gamma=4.0）；"
        "（3）<strong>位置敏感路由</strong>——将抗原残基无序值耦合到 CDR-抗原配对特征；"
        "（4）<strong>对比排序损失</strong>——教导模型真实无序谱比打乱或错配谱产生更低的 NLL。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注④：焦点损失（Focal Loss, Lin et al., ICCV 2017）通过降低易分类样本权重来应对类别不平衡。"
        "无序残基占比仅约 0.74%，gamma=4.0 的焦点加权有效缓解了这一极端不平衡。"
    )

    elements.append(Paragraph("3.2 训练管线", styles['CNSubSection']))
    elements.append(Paragraph(
        "训练管线使用 <strong>SAbDab2 结构数据库</strong>，采用同源性不相交划分"
        "（6 轴：PDB 编号、配对抗体、VH、VL、H3、抗原）。"
        "每个训练批次包含骨架坐标、每残基无序标签、负训练谱（打乱和错配）、"
        "焦点加权无序损失和对比排序损失。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注⑤：SAbDab2（Dunbar et al., Nucleic Acids Res 2014, 2018）是抗体结构数据库，"
        "6 轴同源性不相交划分确保训练集与测试集在各维度上无序列泄漏。"
    )

    elements.append(Paragraph("3.3 IDP 抗体设计管线", styles['CNSubSection']))
    elements.append(Paragraph(
        "IDP 靶标的设计管线包含五个阶段："
        "（1）靶标分析——识别 IDP 靶标并提取表位区域；"
        "（2）无序谱分析——预测每残基无序概率；"
        "（3）支架选择——选择合适的抗体支架；"
        "（4）CDR-H3 设计——使用 ProteinMPNN（T=0.5, 20 个样本）生成候选；"
        "（5）AF2 验证——使用 AlphaFold 2 多聚体验证每个设计，"
        "抗体链使用冒号分隔符传递（VH:VL）。",
        styles['CNBody']
    ))

    # ==================== 第4节：结果 ====================
    elements.append(Paragraph("4. 结果", styles['CNSection']))

    elements.append(Paragraph("4.1 ECLS 检测与时间迁移", styles['CNSubSection']))

    # 适配集表格
    elements.append(Paragraph("<b>适配集（46 个抗原簇）：</b>", styles['CNBody']))
    t1_data = [
        ['指标', '数值'],
        ['ECLS 均值优势', '0.217'],
        ['ECLS 中位数优势', '0.203'],
        ['95% Bootstrap 置信区间', '[0.146, 0.290]'],
        ['阳性簇比例', '80.4%'],
    ]
    t1 = Table(t1_data, colWidths=[140, 120])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'MSYHBD'),
        ('FONTNAME', (0, 1), (-1, -1), 'MSYH'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f9fa')]),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 6))

    # 时间终验表格
    elements.append(Paragraph("<b>时间终验集（31 个结构, 15 个簇, 均为 2021 年后）：</b>", styles['CNBody']))
    t2_data = [
        ['指标', '数值'],
        ['ECLS 均值优势', '0.172'],
        ['ECLS 中位数优势', '0.119'],
        ['95% Bootstrap 置信区间', '[0.059, 0.292]'],
        ['阳性簇比例', '80.0%'],
    ]
    t2 = Table(t2_data, colWidths=[140, 120])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'MSYHBD'),
        ('FONTNAME', (0, 1), (-1, -1), 'MSYH'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f9fa')]),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "两个冻结门控集均通过验证，证实了已沉积的 CDR-H3 序列携带可测量的表位构象特异性信号，"
        "且该信号可迁移至独立沉积的结构上。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注⑥：时间终验集使用 2021 年后沉积的结构，代表模型训练期间不可见的独立数据。"
        "Bootstrap 置信区间基于 10,000 次重采样，Sign-flip 检验 P = 0.014。"
    )

    elements.append(Paragraph("4.2 T2.1 v2 确定性扭转恢复基准", styles['CNSubSection']))
    elements.append(Paragraph(
        "T2.1 v2 基准对 phi/psi 扭转角施加确定性扰动以达到 1.5–3.0 Å 的肽 RMSD 层级，"
        "然后测试天然残基接触能否引导恢复。",
        styles['CNBody']
    ))

    t3_data = [
        ['指标', '数值'],
        ['总结构数', '31'],
        ['扭转命中（1.5–3.0 Å）', '26/31（83.9%）'],
        ['恢复后有效结构', '25/26（96.2%）'],
        ['平均留出接触恢复', '0.581'],
        ['95% 置信区间', '[0.533, 0.635]'],
        ['阳性恢复比例', '100%'],
        ['门控通过（>= 0.70）', '是'],
    ]
    t3 = Table(t3_data, colWidths=[150, 120])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'MSYHBD'),
        ('FONTNAME', (0, 1), (-1, -1), 'MSYH'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f9fa')]),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "这代表了对初始 T2 方案（3/7 有效，-0.017 恢复）的决定性改进，"
        "证明此前的失败源于参数校准问题而非方法的根本局限。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注⑦：肽约束力从 25 kJ/mol/nm² 提升至 100 kJ/mol/nm² 后，有效结构比例从 43% 提升至 96.2%。"
        "T2.1 v2 使用确定性扭转扰动（非高温扰动），确保结果可精确复现。"
    )

    elements.append(Paragraph("4.3 4HIX IDP 设计验证", styles['CNSubSection']))
    elements.append(Paragraph(
        "使用 4HIX 支架（人源化 3D6 Fab，Aβ 1-6 表位 DAEFRH），"
        "ProteinMPNN 生成了 20 个 CDR-H3 候选。全部 20 个通过 AF2 多聚体验证。",
        styles['CNBody']
    ))

    t4_data = [
        ['排名', 'H3 序列', 'ipTM', 'pLDDT', 'iPAE (Å)'],
        ['天然', 'VRYDHYSGSSDY', '0.449', '0.194', '24.7'],
        ['1', 'LYDESKDAESE', '0.462', '0.200', '24.5'],
        ['2', 'LYDAHHGAHSL', '0.461', '0.192', '24.4'],
        ['3', 'LYDGSIGAESQ', '0.460', '0.195', '24.5'],
        ['4', 'LYDSSVDASGH', '0.459', '0.199', '24.7'],
        ['5', 'LFNEANCAESY', '0.458', '0.199', '24.3'],
    ]
    t4 = Table(t4_data, colWidths=[40, 110, 50, 50, 55])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'MSYHBD'),
        ('FONTNAME', (0, 1), (-1, -1), 'MSYH'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f9fa')]),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#d4edda')),
    ]))
    elements.append(t4)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "前 3 个设计的 ipTM 超过天然序列，证明 BFN 衍生的框架可以指导 CDR-H3 设计，"
        "在计算界面质量上达到或超过沉积的天然序列。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注⑧：4HIX 支架于 2012 年沉积，处于 AlphaFold 训练集中。"
        "6 残基表位（DAEFRH）较短，AF2 界面指标对短肽的区分力有限。"
        "此结果为计算验证，尚无湿实验数据支持。"
    )

    elements.append(Paragraph("4.4 关键方法学发现：AF2 链分隔符", styles['CNSubSection']))
    elements.append(Paragraph(
        "在 AF2 验证过程中，我们发现抗体链必须使用冒号分隔符（VH:VL）传递而非直接拼接（VH+VL）。"
        "缺少链断裂标记时，ipTM 从约 0.45 降至约 0.10，界面质量被低估 4.5 倍。"
        "此修复已纳入设计管线，对准确评估抗体-抗原复合物的 AF2 结果至关重要。",
        styles['CNBody']
    ))

    # ==================== 第5节：证据层级 ====================
    elements.append(Paragraph("5. 证据层级", styles['CNSection']))
    elements.append(Paragraph(
        "从最强到最弱排列："
        "（1）ECLS 时间终验（n=15 簇, 2021 年后, 密封）——独立结构上的阳性原生 vs. 打乱判别；"
        "（2）ECLS 适配集（n=46 簇）——在留出数据上复现时间终验模式；"
        "（3）T2.1 v2 扭转恢复（n=31 结构, 96.2% 有效）——确定性扰动后肽接触的物理恢复；"
        "（4）4HIX 设计验证（n=20 设计, 20/20 AF2 通过）——前瞻性设计，最优 ipTM 超过天然；"
        "（5）T1 系综（n=7 簇）——确立单构象过度自信问题；"
        "（6）生成器校准（n=7 簇）——探索性实验；校准重排序器有效，通用重排序器被拒绝。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注⑨：所有评估均使用冻结门控、密封数据上的一次性评估和 6 轴同源性不相交划分。"
        "证据层级反映了从独立数据验证到探索性分析的可靠性递减。"
    )

    # ==================== 第6节：软件实现 ====================
    elements.append(Paragraph("6. 软件实现", styles['CNSection']))
    elements.append(Paragraph(
        "代码库组织为模块化组件：具有类别分布和焦点加权损失的 BFN 核心模块；"
        "标签注入和对比负谱的无序增强数据集；位置敏感路由的接收网络；"
        "五阶段 IDP 抗体设计管线；以及正确处理链分隔符的 AF2 验证基础设施。"
        "所有代码均提供冻结的 YAML 配置契约和 SHA256 校验清单以确保可复现性。",
        styles['CNBody']
    ))

    # ==================== 第7节：可复现性 ====================
    elements.append(Paragraph("7. 可复现性", styles['CNSection']))
    elements.append(Paragraph(
        "所有结果均可从冻结配置和校验数据复现。"
        "T2.1 v2 每结构结果提供于 results_t2.1_v2_final.json（31 条含扭转轨迹的逐结构记录）。"
        "4HIX 设计结果提供于 idp_design_results/4hix_final_validation/final_report.json"
        "（20 个含 AF2 指标的设计）。"
        "整合分析文档（INTEGRATIVE_ANALYSIS.md）总结了完整证据链。"
        "所有结果文件均提供 SHA256 校验清单。",
        styles['CNBody']
    ))

    # ==================== 第8节：局限性 ====================
    elements.append(Paragraph("8. 局限性", styles['CNSection']))
    elements.append(Paragraph(
        "（1）无湿实验验证：所有结果均为计算结果，无实验结合、亲和力或特异性数据。"
        "（2）短表位：4HIX 验证使用 6 残基表位（DAEFRH）；AF2 界面指标对极短肽区分力有限。"
        "（3）训练集重叠：4HIX Fab 于 2012 年沉积，处于 AF2 训练集中。"
        "（4）无序头评估：重新训练的无序头尚未在 CAID 基准上评估。"
        "（5）单一 IDP 靶标：仅验证了 4HIX；向其他 IDP 靶标的泛化有待证明。"
        "（6）成分匹配打乱为反事实对照，非实验非结合物。",
        styles['CNBody']
    ))
    add_footnote(
        "脚注⑩：CAID（Critical Assessment of IDP prediction）是无序预测领域的标准基准。"
        "计划在后续工作中完成重训练无序头的 CAID 评估。"
    )

    # ==================== 第9节：未来工作 ====================
    elements.append(Paragraph("9. 未来工作", styles['CNSection']))
    elements.append(Paragraph(
        "计划扩展包括："
        "（1）重训练无序头的 CAID 基准评估；"
        "（2）使用修复模型重新设计 Aβ 抗体；"
        "（3）扩展 IDP 靶标（tau、α-突触核蛋白）；"
        "（4）通过 SPR、ELISA 或生物膜干涉技术的实验验证；"
        "（5）向 Bioinformatics 期刊投稿；"
        "（6）4HIX T1 系综分子动力学（MD）模拟以表征构象异质性。",
        styles['CNBody']
    ))

    # ==================== 第10节：数据可用性 ====================
    elements.append(Paragraph("10. 已发表结果与数据可用性", styles['CNSection']))
    elements.append(Paragraph(
        "截至 2026 年 8 月，尚无同行评审论文发表。稿件草案正在准备中，"
        "拟投 Bioinformatics。所有代码、冻结配置、结果文件和可复现性文档"
        "均可在 DisorderFlow 仓库中获取。第三方数据集和模型权重通过校验和分发。",
        styles['CNBody']
    ))

    # ==================== 第11节：团队 ====================
    elements.append(Paragraph("11. 团队与贡献", styles['CNSection']))
    t5_data = [
        ['成员', '角色', '贡献'],
        ['陈昊阳', '队长', '项目设计、BFN 架构、ECLS 方法论、稿件撰写'],
        ['胡静', '计算生物学', 'IDP 管线、AF2 验证、无序头训练'],
        ['余家瑞', '软件工程', '训练基础设施、基准测试、可复现性'],
    ]
    t5 = Table(t5_data, colWidths=[60, 70, 230])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3436')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'MSYHBD'),
        ('FONTNAME', (0, 1), (-1, -1), 'MSYH'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f9fa')]),
    ]))
    elements.append(t5)

    # ==================== 第12节：参考文献 ====================
    elements.append(Paragraph("12. 参考文献", styles['CNSection']))
    refs = [
        "1. Dunbar et al. Nucleic Acids Res (2014, 2018) — SAbDab/SAbDab2 抗体结构数据库",
        "2. Dauparas et al. Science (2022) — ProteinMPNN 蛋白质序列设计",
        "3. Hsu et al. bioRxiv (2022) — ESM-IF 蛋白质语言模型",
        "4. Jumper et al. Nature (2021) — AlphaFold 2 蛋白质结构预测",
        "5. Evans et al. bioRxiv (2021) — AlphaFold Multimer 复合物预测",
        "6. Grimstead et al. arXiv (2023) — Bayesian Flow Networks 贝叶斯流网络",
        "7. Eastman et al. PLoS Comput Biol (2017) — OpenMM 分子动力学引擎",
        "8. Maier et al. Nat Methods (2015) — Amber14 力场",
        "9. Lin et al. ICCV (2017) — Focal Loss 焦点损失",
        "10. Benjamini & Hochberg, J R Stat Soc (1995) — BH 多重检验校正",
    ]
    for ref in refs:
        elements.append(Paragraph(ref, ParagraphStyle('CNRef', parent=styles['Normal'],
                                                       fontName='MSYH',
                                                       fontSize=9, leading=13, spaceAfter=3,
                                                       leftIndent=20)))

    # Build
    doc.build(elements)
    print(f"中文版项目报告已生成: {output_path}")
    return output_path

if __name__ == '__main__':
    build_report()
