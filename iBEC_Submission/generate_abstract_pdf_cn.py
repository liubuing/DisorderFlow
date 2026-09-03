# -*- coding: utf-8 -*-
"""
iBEC 技术摘要 PDF 生成器（中文版）
横版 A4，10 页，幻灯片风格布局
带术语中英对照注释 + 脚注补充说明
"""
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Register CJK fonts ──
pdfmetrics.registerFont(TTFont('MSYH', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('MSYHBD', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))

PAGE_W, PAGE_H = landscape(A4)

# ── Colors ──
BG_WHITE = HexColor('#F7F8F6')
DARK_TEXT = HexColor('#1A1A2E')
ACCENT = HexColor('#087F6D')
GRAY = HexColor('#555555')
LIGHT_GRAY = HexColor('#999999')
TABLE_HEADER_BG = HexColor('#2D3436')
TABLE_ALT_BG = HexColor('#F0F3F5')
GREEN_HL = HexColor('#D4EDDA')
GREEN_TEXT = HexColor('#155724')
YELLOW_BG = HexColor('#FFF3CD')
YELLOW_BORDER = HexColor('#FFC107')
YELLOW_TEXT = HexColor('#856A04')
BOX_BG = HexColor('#E8F5F3')
BOX_BORDER = HexColor('#087F6D')
FORMULA_BG = HexColor('#F0F3F5')
FOOTNOTE_COLOR = HexColor('#888888')


class SlideCanvas(canvas.Canvas):
    """自定义画布，绘制幻灯片风格背景。"""
    def draw_slide_bg(self):
        self.setFillColor(BG_WHITE)
        self.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    def draw_footer(self, page_num):
        self.setFont('MSYH', 8)
        self.setFillColor(LIGHT_GRAY)
        self.drawRightString(PAGE_W - 40, 25, f"{page_num} / 10")
        self.drawString(40, 25, "第 28 号队伍 — 解析无序  |  iBEC 2026")

    def draw_accent_bar(self, x, y, w=4, h=30):
        self.setFillColor(ACCENT)
        self.rect(x, y, w, h, fill=1, stroke=0)

    def draw_section_title(self, number, title, y_offset=None):
        y = y_offset or (PAGE_H - 55)
        self.draw_accent_bar(38, y - 5, 4, 28)
        self.setFont('MSYHBD', 18)
        self.setFillColor(DARK_TEXT)
        self.drawString(50, y, f"{number}. {title}")

    def draw_footnote(self, text, y=None):
        """在页面底部绘制脚注（自动折行）。"""
        if y is None:
            y = 42
        self.setFont('MSYH', 7)
        self.setFillColor(FOOTNOTE_COLOR)
        # Short line above
        self.setStrokeColor(HexColor('#CCCCCC'))
        self.setLineWidth(0.4)
        self.line(40, y + 10, 160, y + 10)
        # Word wrap for CJK: ~85 chars per line at 7pt on landscape A4
        max_chars = 95
        lines = []
        remaining = text
        while remaining:
            if len(remaining) <= max_chars:
                lines.append(remaining)
                break
            lines.append(remaining[:max_chars])
            remaining = remaining[max_chars:]
        ly = y
        for line in lines:
            self.drawString(40, ly, line)
            ly -= 10


def _draw_table(c, x, y, data, col_widths=None, highlight_row=None, cn_header=False):
    """在画布上绘制简单表格。"""
    if col_widths is None:
        col_widths = [120] * len(data[0])
    row_h = 20
    total_w = sum(col_widths)

    for r, row in enumerate(data):
        ry = y - r * row_h
        cx = x
        if r == 0:
            c.setFillColor(TABLE_HEADER_BG)
            c.rect(x, ry - 4, total_w, row_h, fill=1, stroke=0)
        elif r == highlight_row:
            c.setFillColor(GREEN_HL)
            c.rect(x, ry - 4, total_w, row_h, fill=1, stroke=0)
        elif r % 2 == 0:
            c.setFillColor(TABLE_ALT_BG)
            c.rect(x, ry - 4, total_w, row_h, fill=1, stroke=0)
        else:
            c.setFillColor(white)
            c.rect(x, ry - 4, total_w, row_h, fill=1, stroke=0)

        for ci, cell_val in enumerate(row):
            if r == 0:
                c.setFillColor(white)
                c.setFont('MSYHBD', 10)
            else:
                c.setFillColor(DARK_TEXT)
                c.setFont('MSYH', 10)
            c.drawString(cx + 5, ry + 3, str(cell_val))
            cx += col_widths[ci]

    c.setStrokeColor(HexColor('#DEE2E6'))
    c.setLineWidth(0.5)
    total_h = len(data) * row_h
    c.rect(x, y - (len(data) - 1) * row_h - 4, total_w, total_h, fill=0, stroke=1)
    for r in range(1, len(data)):
        ly = y - r * row_h + row_h - 4
        c.line(x, ly, x + total_w, ly)
    cx = x
    for cw in col_widths[:-1]:
        cx += cw
        c.line(cx, y + row_h - 4, cx, y - (len(data) - 1) * row_h - 4)


def build_abstract_pdf():
    output_path = "C:\\biological\\DisorderFlow\\iBEC_Submission\\28-解析无序-技术摘要_中文版.pdf"
    c = SlideCanvas(output_path, pagesize=landscape(A4))

    # ==================== 第1页：封面 ====================
    c.draw_slide_bg()
    c.setFillColor(ACCENT)
    c.rect(80, PAGE_H - 140, PAGE_W - 160, 2, fill=1, stroke=0)

    c.setFont('MSYHBD', 30)
    c.setFillColor(DARK_TEXT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 180, "DisorderFlow")

    c.setFont('MSYH', 14)
    c.setFillColor(GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 210,
                        "面向无序蛋白表位的贝叶斯流网络抗体 CDR-H3 设计平台")

    c.setFont('MSYH', 10)
    c.setFillColor(LIGHT_GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 232,
                        "Bayesian Flow Networks for Disorder-Aware Antibody CDR-H3 Design")

    c.setFont('MSYHBD', 13)
    c.setFillColor(ACCENT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 275, "第 28 号队伍 — 解析无序")

    c.setFont('MSYH', 11)
    c.setFillColor(GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 300,
                        "陈昊阳（队长）、胡静、戴谭宇、孙宇超  |  齐鲁工业大学")
    c.setFont('MSYH', 10)
    c.setFillColor(LIGHT_GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 325,
                        "iBEC 2026 — 国际生物信息学工程竞赛")

    c.showPage()

    # ==================== 第2页：问题 ====================
    c.draw_slide_bg()
    c.draw_section_title(1, "IDP 抗体设计挑战")
    c.draw_footer(2)

    bullets = [
        ("内在无序蛋白（IDP, Intrinsically Disordered Protein）缺乏稳定三维结构，", False),
        ("  却在神经退行性疾病中发挥关键作用（阿尔茨海默病、帕金森病）", True),
        ("", False),
        ("治疗靶标：淀粉样蛋白-β（Aβ）、tau、α-突触核蛋白 — 均为无序蛋白", False),
        ("", False),
        ("现有抗体设计方法假设刚性、结构良好的表位 —", False),
        ("  对 IDP 靶标根本不适用", True),
        ("", False),
        ("CDR-H3（互补决定区 H3）是抗原特异性的主要决定因素，", False),
        ("  也是抗体结构中多样性最高的环区", True),
        ("", False),
        ("关键空白：没有计算框架将表位无序谱耦合到 CDR-H3 序列选择", False),
    ]
    y = PAGE_H - 95
    for text, is_sub in bullets:
        if text == "":
            y -= 6
            continue
        if is_sub:
            c.setFont('MSYH', 11)
            c.setFillColor(GRAY)
            c.drawString(75, y, text)
        else:
            c.setFont('MSYH', 12)
            c.setFillColor(DARK_TEXT)
            c.drawString(55, y, "• " + text)
        y -= 18

    c.setFillColor(ACCENT)
    c.setFont('MSYHBD', 11)
    c.drawString(55, y - 15, "目标：构建无序感知评分框架，在骨架几何和每残基表位无序谱")
    c.drawString(55, y - 30, "双重条件下指导 CDR-H3 设计。")

    c.draw_footnote(
        "注释：CDR-H3 位于抗体可变区重链上，长度通常为 3-35 个残基，"
        "其序列和构象多样性直接决定了抗体对抗原的识别特异性和结合亲和力。"
    )
    c.showPage()

    # ==================== 第3页：平台架构 ====================
    c.draw_slide_bg()
    c.draw_section_title(2, "DisorderFlow 平台架构")
    c.draw_footer(3)

    components = [
        ("BFN 核心", "20 种氨基酸类别分布；", "通过接收网络迭代细化", "概率分布"),
        ("无序头", "每残基无序概率预测；", "焦点加权损失（γ=4.0）；", "应对 0.74% 类别不平衡"),
        ("位置路由", "抗原残基无序值直接", "耦合到 CDR-抗原", "配对特征"),
        ("对比损失", "真实 vs. 打乱/错配", "无序谱；排序感知", "训练信号"),
    ]
    box_w = 155
    box_h = 100
    start_x = 45
    gap = 12
    box_y = PAGE_H - 200

    for i, (title, *lines) in enumerate(components):
        x = start_x + i * (box_w + gap)
        c.setFillColor(BOX_BG)
        c.setStrokeColor(BOX_BORDER)
        c.setLineWidth(1.2)
        c.roundRect(x, box_y, box_w, box_h, 6, fill=1, stroke=1)

        c.setFont('MSYHBD', 11)
        c.setFillColor(ACCENT)
        c.drawCentredString(x + box_w / 2, box_y + box_h - 20, title)

        c.setFont('MSYH', 9)
        c.setFillColor(GRAY)
        for j, line in enumerate(lines):
            c.drawCentredString(x + box_w / 2, box_y + box_h - 40 - j * 14, line)

    c.setFont('MSYHBD', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, box_y - 35, "训练管线")

    c.setFont('MSYH', 10)
    c.setFillColor(GRAY)
    c.drawString(55, box_y - 55,
                 "SAbDab2 结构数据库  |  6 轴同源性不相交划分（PDB、配对抗体、VH、VL、H3、抗原）")
    c.drawString(55, box_y - 72,
                 "骨架坐标 + 无序标签 + 负训练谱  |  ProteinMPNN v_48_020 用于评估")

    c.draw_footnote(
        "注释：BFN（Bayesian Flow Network，贝叶斯流网络）是一种离散序列生成框架（Grimstead et al., 2023），"
        "与扩散方法不同，它在每个位置维持离散类别分布而非连续高斯分布。"
        "焦点损失（Focal Loss）通过降低易分类样本权重来应对极端类别不平衡。"
    )
    c.showPage()

    # ==================== 第4页：ECLS 指标 ====================
    c.draw_slide_bg()
    c.draw_section_title(3, "ECLS：表位条件似然位移")
    c.draw_footer(4)

    # 公式框
    c.setFillColor(FORMULA_BG)
    c.setStrokeColor(HexColor('#DEE2E6'))
    c.setLineWidth(0.8)
    c.roundRect(80, PAGE_H - 160, PAGE_W - 160, 40, 6, fill=1, stroke=1)
    c.setFont('Courier', 15)
    c.setFillColor(DARK_TEXT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 145,
                        "ECLS(s) = NLL_complex(s) - NLL_peptide-stripped(s)")

    c.setFont('MSYH', 10)
    c.setFillColor(GRAY)
    y = PAGE_H - 190
    c.drawString(55, y, "衡量 CDR-H3 序列是否优先被肽表位坐标的存在所支持。")
    c.drawString(55, y - 16, "正优势 = 天然序列从表位上下文中获得的收益大于成分匹配打乱序列（每条记录 200 个打乱）。")

    c.setFont('MSYHBD', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, y - 50, "方法学要点")

    methods = [
        "仅评分重链；固定非 H3 位置；保留轻链 + 肽上下文",
        "每条记录 200 个成分匹配打乱作为反事实对照",
        "官方 SAbDab2 CDR-H3 注释；重原子距离 ≤ 4.5 Å 接触门控",
        "官方抗原簇作为推断单元；10,000 次 Bootstrap 重采样计算置信区间",
        "在密封时间数据上一次性评估；评估前设定冻结门控",
    ]
    c.setFont('MSYH', 10)
    c.setFillColor(DARK_TEXT)
    my = y - 73
    for m in methods:
        c.drawString(65, my, "•  " + m)
        my -= 18

    c.draw_footnote(
        "注释：NLL（Negative Log-Likelihood，负对数似然）由 ProteinMPNN 计算。"
        "ECLS 的核心思想是：如果天然 CDR-H3 序列与表位存在特异性耦合，"
        "则移除表位坐标后其 NLL 应显著升高（似然降低），而打乱序列不受影响。"
    )
    c.showPage()

    # ==================== 第5页：ECLS 结果 ====================
    c.draw_slide_bg()
    c.draw_section_title(4, "结果：ECLS 检测与时间迁移")
    c.draw_footer(5)

    c.setFont('MSYHBD', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, PAGE_H - 85, "适配集（46 个抗原簇）")

    data_left = [
        ('指标', '数值'),
        ('ECLS 均值优势', '0.217'),
        ('中位数', '0.203'),
        ('95% Bootstrap CI', '[0.146, 0.290]'),
        ('阳性簇比例', '80.4%'),
    ]
    _draw_table(c, 55, PAGE_H - 105, data_left, col_widths=[150, 100])

    c.setFont('MSYHBD', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(PAGE_W // 2 + 20, PAGE_H - 85, "时间终验集（15 个簇, 2021 年后）")

    data_right = [
        ('指标', '数值'),
        ('ECLS 均值优势', '0.172'),
        ('中位数', '0.119'),
        ('95% Bootstrap CI', '[0.059, 0.292]'),
        ('阳性簇比例', '80.0%'),
    ]
    _draw_table(c, PAGE_W // 2 + 20, PAGE_H - 105, data_right, col_widths=[150, 100])

    y = PAGE_H - 270
    c.setFont('MSYH', 10)
    c.setFillColor(GRAY)
    c.drawString(55, y, "两个冻结门控集均通过。已沉积 CDR-H3 序列携带可测量的表位构象特异性信号，")
    c.drawString(55, y - 16, "且该信号可迁移至训练后独立沉积的结构上。Sign-flip P < 1e-6（适配集）和 P = 0.014（时间终验）。")

    c.setFillColor(GREEN_HL)
    c.roundRect(55, y - 55, PAGE_W - 110, 28, 4, fill=1, stroke=0)
    c.setFont('MSYH', 10)
    c.setFillColor(GREEN_TEXT)
    c.drawString(70, y - 45, "关键：密封时间数据上 80%+ 阳性簇证实了非平凡的表位-H3 耦合")

    c.draw_footnote(
        "注释：时间终验集使用 2021 年后沉积的结构（训练后新数据），代表真正的独立验证。"
        "Bootstrap 置信区间基于 10,000 次重采样，确保统计稳健性。"
    )
    c.showPage()

    # ==================== 第6页：T2.1 v2 ====================
    c.draw_slide_bg()
    c.draw_section_title(5, "T2.1 v2：确定性扭转恢复基准")
    c.draw_footer(6)

    c.setFont('MSYH', 10)
    c.setFillColor(GRAY)
    c.drawString(55, PAGE_H - 85,
                 "对 phi/psi 扭转角施加确定性扰动至 1.5–3.0 Å RMSD 层级，然后测试天然残基")
    c.drawString(55, PAGE_H - 101,
                 "接触能否引导恢复。肽约束力：100 kJ/mol/nm²。")

    data_t2 = [
        ('指标', '数值'),
        ('总结构数', '31'),
        ('扭转命中（目标层级）', '26/31（83.9%）'),
        ('总体有效结构', '25/31（80.6%）'),
        ('平均留出接触恢复', '0.581'),
        ('95% 置信区间', '[0.533, 0.635]'),
        ('阳性恢复比例', '100%'),
    ]
    _draw_table(c, 55, PAGE_H - 130, data_t2, col_widths=[180, 120])

    comp_x = PAGE_W // 2 + 30
    c.setFillColor(FORMULA_BG)
    c.setStrokeColor(HexColor('#DEE2E6'))
    c.setLineWidth(0.8)
    c.roundRect(comp_x, PAGE_H - 275, 230, 145, 6, fill=1, stroke=1)

    c.setFont('MSYHBD', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(comp_x + 10, PAGE_H - 145, "T2 vs T2.1 v2（重校准）")

    c.setFont('MSYH', 10)
    c.setFillColor(GRAY)
    lines = [
        "T2:  3/7 有效, 恢复 = −0.017",
        "   （高温扰动, 校准差）",
        "",
        "T2.1 v2:  25/31 有效, 恢复 = 0.581",
        "   （确定性扭转, 100 kJ/mol/nm²）",
        "",
        "随机约束恢复 = 0.619,",
        "不支持接触特异机制。",
    ]
    ly = PAGE_H - 165
    for line in lines:
        c.drawString(comp_x + 10, ly, line)
        ly -= 14

    c.setFillColor(GREEN_HL)
    c.roundRect(55, PAGE_H - 305, PAGE_W - 110, 24, 4, fill=1, stroke=0)
    c.setFont('MSYH', 10)
    c.setFillColor(GREEN_TEXT)
    c.drawString(70, PAGE_H - 298, "总体有效率 80.6%；随机约束高于 supplied contacts（0.619 对 0.581）")

    c.draw_footnote(
        "注释：T2.1 v2 使用确定性扭转扰动替代 T2 的高温扰动，确保结果可精确复现。"
        "总体有效比例为 25/31（80.6%）；25/26（96.2%）是命中扰动层级后的条件比例。"
    )
    c.showPage()

    # ==================== 第7页：4HIX 设计 ====================
    c.draw_slide_bg()
    c.draw_section_title(6, "4HIX ProteinMPNN 计算案例")
    c.draw_footer(7)

    c.setFont('MSYH', 10)
    c.setFillColor(GRAY)
    c.drawString(55, PAGE_H - 85,
                 "支架：4HIX（人源化 3D6 Fab，Aβ 1-6 表位 DAEFRH）。ProteinMPNN T=0.5, 20 个样本。")
    c.drawString(55, PAGE_H - 101,
                 "AF2 多聚体 V3，VH:VL 链分隔符。全部 20 个设计通过验证。")

    data_4hix = [
        ('排名', 'H3 序列', 'ipTM', 'pLDDT', 'iPAE'),
        ('天然', 'VRYDHYSGSSDY', '0.449', '0.194', '24.7'),
        ('#1', 'LYDESKDAESE', '0.462', '0.200', '24.5'),
        ('#2', 'LYDAHHGAHSL', '0.461', '0.192', '24.4'),
        ('#3', 'LYDGSIGAESQ', '0.460', '0.195', '24.5'),
        ('#4', 'LYDSSVDASGH', '0.459', '0.199', '24.7'),
        ('#5', 'LFNEANCAESY', '0.458', '0.199', '24.3'),
    ]
    tbl_y = PAGE_H - 125
    _draw_table(c, 55, tbl_y, data_4hix, col_widths=[45, 120, 45, 50, 45],
                highlight_row=2)

    kx = PAGE_W // 2 + 40
    c.setFont('MSYHBD', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(kx, tbl_y + 10, "关键发现")

    findings = [
        "20/20 设计通过 AF2 验证",
        "前 3 名超过天然 ipTM（0.449）",
        "最优设计：ipTM = 0.462（+2.8%，描述性差异）",
        "链分隔符修复：VH:VL 防止",
        "  4.5 倍 ipTM 低估",
    ]
    fy = tbl_y - 12
    for f in findings:
        if f.startswith("  "):
            c.setFont('MSYH', 10)
            c.setFillColor(GRAY)
            c.drawString(kx + 10, fy, f.strip())
        else:
            c.setFont('MSYH', 11)
            c.setFillColor(DARK_TEXT)
            c.drawString(kx, fy, "•  " + f)
        fy -= 17

    c.setFont('MSYH', 8)
    c.setFillColor(LIGHT_GRAY)
    c.drawString(55, 55, "注：6 残基表位；AF2 对短肽区分力有限。4HIX 在 AF2 训练集中（2012）。无湿实验验证。")

    c.draw_footnote(
        "注释：ipTM（interface predicted Template Model）为 AF2 预测的界面质量指标（0–1）。"
        "冒号分隔符（VH:VL）告知 AF2 在抗体轻重链间存在链断裂，避免将两条链误判为连续多肽。"
    )
    c.showPage()

    # ==================== 第8页：证据层级 ====================
    c.draw_slide_bg()
    c.draw_section_title(7, "证据层级与方法学严谨性")
    c.draw_footer(8)

    evidence = [
        ("1. ECLS 时间终验", "n=15 簇, 2021 后, 密封", "阳性判别（0.172, 80%+）"),
        ("2. ECLS 适配集", "n=46 簇, 暴露数据", "复现时间终验模式"),
        ("3. T2.1 v2 恢复", "n=31 结构, 80.6% 总体有效", "随机约束高于 supplied"),
        ("4. 4HIX 设计", "n=20 设计, 20/20 AF2 通过", "前瞻性设计（最优 ipTM 0.462）"),
        ("5. T1 系综", "n=7 簇", "单构象过度自信（−0.200）"),
        ("6. 生成器校准", "n=7 簇, 探索性", "校准重排序器有效"),
    ]

    ey = PAGE_H - 100
    for i, (title, detail, result) in enumerate(evidence):
        intensity = max(0.3, 1.0 - i * 0.12)
        bar_color = Color(0.031 * intensity, 0.498 * intensity, 0.427 * intensity)
        c.setFillColor(bar_color)
        c.rect(55, ey - 3, 4, 22, fill=1, stroke=0)

        c.setFont('MSYHBD', 12)
        c.setFillColor(DARK_TEXT)
        c.drawString(68, ey + 5, title)

        c.setFont('MSYH', 9)
        c.setFillColor(LIGHT_GRAY)
        c.drawString(68, ey - 8, detail)

        c.setFont('MSYH', 10)
        c.setFillColor(GRAY)
        c.drawString(PAGE_W // 2 + 20, ey, result)

        ey -= 42

    c.draw_footnote(
        "注释：所有评估使用冻结门控、密封数据一次性评估和 6 轴同源性不相交划分。"
        "证据层级从独立数据验证（最强）到探索性分析（最弱）递减，反映可靠性梯度。"
    )
    c.showPage()

    # ==================== 第9页：软件与可复现性 ====================
    c.draw_slide_bg()
    c.draw_section_title(8, "软件实现与可复现性")
    c.draw_footer(9)

    c.setFont('MSYHBD', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, PAGE_H - 85, "软件组件")

    sw_items = [
        "BFN 核心：类别分布、焦点加权损失",
        "无序增强数据集：标签注入 + 对比谱",
        "接收网络：位置敏感路由",
        "IDP 设计管线：5 阶段靶标到验证",
        "AF2 基础设施：正确链分隔符处理",
    ]
    c.setFont('MSYH', 10)
    c.setFillColor(DARK_TEXT)
    sy = PAGE_H - 108
    for item in sw_items:
        c.drawString(65, sy, "•  " + item)
        sy -= 17

    rx = PAGE_W // 2 + 20
    c.setFont('MSYHBD', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(rx, PAGE_H - 85, "可复现性")

    rep_items = [
        "冻结 YAML 配置契约",
        "所有结果文件 SHA256 校验清单",
        "results_t2.1_v2_final.json（31 条记录）",
        "4hix_final_validation/final_report.json（20 设计）",
        "INTEGRATIVE_ANALYSIS.md 证据链",
    ]
    c.setFont('MSYH', 10)
    c.setFillColor(DARK_TEXT)
    ry = PAGE_H - 108
    for item in rep_items:
        c.drawString(rx + 10, ry, "•  " + item)
        ry -= 17

    bug_y = PAGE_H - 310
    c.setFillColor(YELLOW_BG)
    c.setStrokeColor(YELLOW_BORDER)
    c.setLineWidth(1)
    c.roundRect(55, bug_y, PAGE_W - 110, 105, 6, fill=1, stroke=1)

    c.setFont('MSYHBD', 11)
    c.setFillColor(YELLOW_TEXT)
    c.drawString(70, bug_y + 82, "开发过程中修复的关键缺陷")

    c.setFont('MSYH', 9)
    bugs = [
        "1. 无序头：训练批次从未包含 disorder_label → PaddingCollate 丢弃了该字段",
        "   → 无序损失从未触发。修复：显式标签注入。",
        "2. AF2 链分隔符：抗体链必须使用 VH:VL（非拼接），否则 ipTM 下降 4.5 倍。",
        "3. T2.1 扰动：重校准方案总体有效率 80.6%，但机制未获对照支持。",
    ]
    by = bug_y + 62
    for b in bugs:
        if b.startswith("   "):
            c.drawString(80, by, b.strip())
        else:
            c.drawString(70, by, b)
        by -= 14

    c.draw_footnote(
        "注释：SHA256 校验清单确保数据完整性——任何文件篡改都会被检测到。"
        "冻结 YAML 配置确保训练和评估超参数可精确复现。"
    )
    c.showPage()

    # ==================== 第10页：结论 ====================
    c.draw_slide_bg()
    c.draw_section_title(9, "结论与未来工作")
    c.draw_footer(10)

    c.setFont('MSYHBD', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, PAGE_H - 85, "结论")

    conclusions = [
        "ECLS 确立了表位条件化 CDR-H3 信号（46 簇, 均值 0.217, 80.4% 阳性）",
        "时间迁移在密封 2021 后数据上得到证实（均值 0.172, 80% 阳性, P=0.014）",
        "T2.1 v2：总体有效率 80.6%；随机约束 0.619 高于 supplied 0.581",
        "4HIX 前瞻性设计：20/20 通过 AF2, 前 3 名超过天然 ipTM",
        "无序感知框架为 IDP 抗体设计提供了原则性方法",
    ]
    c.setFont('MSYH', 11)
    c.setFillColor(DARK_TEXT)
    cy = PAGE_H - 108
    for item in conclusions:
        c.drawString(65, cy, "•  " + item)
        cy -= 18

    c.setFont('MSYHBD', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, cy - 15, "局限性")

    c.setFont('MSYH', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(65, cy - 35, "•  无湿实验验证；所有结果均为计算结果")
    c.drawString(65, cy - 53, "•  短表位（6 残基）；AF2 区分力有限；4HIX 在 AF2 训练集中")

    c.setFont('MSYHBD', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, cy - 85, "未来工作")

    c.setFont('MSYH', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(65, cy - 105, "•  重训练无序头的 CAID 基准评估")
    c.drawString(65, cy - 123, "•  扩展 IDP 靶标（tau、α-突触核蛋白）及实验验证（SPR/ELISA）")

    c.draw_footnote(
        "注释：CAID（Critical Assessment of IDP prediction）是无序预测领域的标准基准测试。"
        "SPR（表面等离子共振）和 ELISA（酶联免疫吸附试验）是验证抗体结合亲和力的标准湿实验方法。"
    )

    c.save()
    print(f"中文版技术摘要已生成: {output_path}")
    return output_path


if __name__ == '__main__':
    build_abstract_pdf()
