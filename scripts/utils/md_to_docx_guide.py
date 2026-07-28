#!/usr/bin/env python3
"""Convert DISORDERFLOW_COMPLETE_GUIDE.md to a professionally formatted Word document.

Produces a Word .docx with:
  - Proper heading hierarchy (Title → H1 → H2 → H3)
  - Monospace code blocks with gray background
  - Formatted tables with header rows
  - ASCII-art diagrams in monospace
  - Consistent fonts and spacing for Chinese readability
  - Page breaks at major section boundaries
"""

import re
import sys
import os
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Configuration ──────────────────────────────────────────────────────────

MARKDOWN_PATH = 'DISORDERFLOW_COMPLETE_GUIDE.md'
OUTPUT_PATH = 'DisOrderFlow_完整说明书.docx'

# Font settings for Chinese readability
BODY_FONT = '微软雅黑'
BODY_SIZE = Pt(10.5)
CODE_FONT = 'Consolas'
CODE_SIZE = Pt(9)
HEADING_FONT = '微软雅黑'
TITLE_SIZE = Pt(22)
H1_SIZE = Pt(16)
H2_SIZE = Pt(13)
H3_SIZE = Pt(11.5)

# ── Document Setup ─────────────────────────────────────────────────────────

def setup_styles(doc):
    """Configure document styles for Chinese readability."""
    style = doc.styles['Normal']
    font = style.font
    font.name = BODY_FONT
    font.size = BODY_SIZE
    font.color.rgb = RGBColor(0x1a, 0x1a, 0x1a)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
    pf = style.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.35

    # Heading styles
    for level, (size, color) in enumerate([
        (TITLE_SIZE, RGBColor(0x0d, 0x2b, 0x4e)),
        (H1_SIZE,    RGBColor(0x0d, 0x2b, 0x4e)),
        (H2_SIZE,    RGBColor(0x1a, 0x47, 0x6f)),
        (H3_SIZE,    RGBColor(0x2c, 0x5f, 0x8a)),
    ], start=0):
        if level == 0:
            style_name = 'Title'
        else:
            style_name = f'Heading {level}'
        try:
            hs = doc.styles[style_name]
        except KeyError:
            continue
        hs.font.name = HEADING_FONT
        hs.font.size = size
        hs.font.color.rgb = color
        hs.font.bold = True
        hs.element.rPr.rFonts.set(qn('w:eastAsia'), HEADING_FONT)
        if level >= 1:
            hs.paragraph_format.space_before = Pt(18 if level == 1 else 14)
            hs.paragraph_format.space_after = Pt(8)

    # Inline code style (will be applied via runs)
    # Code block paragraphs use a custom approach

    # Page setup
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)


def add_code_block(doc, code_text):
    """Add a code block with gray background, monospace font."""
    for line in code_text.strip().split('\n'):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.left_indent = Cm(0.5)

        # Gray background shading
        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), 'F0F0F0')
        shd.set(qn('w:val'), 'clear')
        pPr.append(shd)

        run = p.add_run(line if line else ' ')
        run.font.name = CODE_FONT
        run.font.size = CODE_SIZE
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        run.element.rPr.rFonts.set(qn('w:eastAsia'), CODE_FONT)

    # Small gap after code block
    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(2)
    gap.paragraph_format.space_after = Pt(2)


def add_ascii_diagram(doc, text):
    """Add an ASCII art diagram in monospace (like the antibody sketch)."""
    for line in text.strip().split('\n'):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.left_indent = Cm(1.0)

        run = p.add_run(line)
        run.font.name = CODE_FONT
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        run.element.rPr.rFonts.set(qn('w:eastAsia'), CODE_FONT)

    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(2)
    gap.paragraph_format.space_after = Pt(2)


def add_table_from_md(doc, lines):
    """Parse markdown table lines and add a formatted Word table."""
    if len(lines) < 2:
        return

    # Parse header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    # Skip separator line (line[1] with |---|)
    data_start = 2
    rows = []
    for line in lines[data_start:]:
        if line.startswith('|') and '|' in line[1:]:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells:
                rows.append(cells)

    if not headers:
        return

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.autofit = True

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(9)
        run.font.name = BODY_FONT
        run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

    # Data rows
    for r, row_data in enumerate(rows):
        for c, val in enumerate(row_data):
            if c < len(headers):
                cell = table.rows[r + 1].cells[c]
                cell.text = ''
                p = cell.paragraphs[0]
                run = p.add_run(val)
                run.font.size = Pt(9)
                run.font.name = BODY_FONT
                run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

    doc.add_paragraph()  # space after table


def add_callout_box(doc, text, box_type='info'):
    """Add a highlighted callout box for important notes.

    box_type: 'info' (blue), 'warning' (amber), 'danger' (red)
    """
    colors = {
        'info':    ('E8F0FE', '1a476f'),
        'warning': ('FFF8E1', '8b6914'),
        'danger':  ('FFEBEE', 'a31515'),
    }
    bg, fg = colors.get(box_type, colors['info'])

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.3)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)

    # Background
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), bg)
    shd.set(qn('w:val'), 'clear')
    pPr.append(shd)

    # Border (left bar)
    pBdr = OxmlElement('w:pBdr')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '12')
    left.set(qn('w:space'), '8')
    left.set(qn('w:color'), fg)
    pBdr.append(left)
    pPr.append(pBdr)

    run = p.add_run(text)
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(*tuple(int(fg[i:i+2], 16) for i in (0, 2, 4)))
    run.font.name = BODY_FONT
    run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)


def add_image_placeholder(doc, caption):
    """Add a placeholder for an image/diagram."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)

    # Gray placeholder box
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), 'F5F5F5')
    shd.set(qn('w:val'), 'clear')
    pPr.append(shd)

    run = p.add_run(f'  [ 图示 ]  ')
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    run.font.name = BODY_FONT

    # Caption
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.font.size = Pt(8.5)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    run.font.name = BODY_FONT
    run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)


# ── Markdown Parser ────────────────────────────────────────────────────────

def parse_and_render(doc, md_text):
    """Parse markdown text and render into the Word document, section by section."""

    lines = md_text.split('\n')
    i = 0
    table_buffer = []
    code_buffer = []
    ascii_buffer = []
    in_code_block = False
    in_ascii_art = False
    in_table = False

    def flush_paragraph():
        nonlocal in_ascii_art
        if in_ascii_art:
            add_ascii_diagram(doc, '\n'.join(ascii_buffer))
            ascii_buffer.clear()
            in_ascii_art = False

    def flush_table():
        nonlocal in_table
        if in_table and table_buffer:
            add_table_from_md(doc, table_buffer)
            table_buffer.clear()
            in_table = False

    def flush_code():
        nonlocal in_code_block
        if in_code_block and code_buffer:
            add_code_block(doc, '\n'.join(code_buffer))
            code_buffer.clear()
            in_code_block = False

    while i < len(lines):
        line = lines[i]

        # Code block toggle
        if line.startswith('```'):
            if in_code_block:
                flush_code()
            else:
                flush_paragraph()
                flush_table()
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        # Table detection
        if line.startswith('|') and '|' in line[1:]:
            if not in_table:
                flush_paragraph()
                flush_code()
                in_table = True
            table_buffer.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # ASCII art detection (lines starting with spaces/special chars for diagrams)
        if (line.startswith('    ') or line.startswith('   ')) and not line.strip().startswith('*') and not line.strip().startswith('-') and not line.strip()[0].isalpha() if line.strip() else False:
            if not in_ascii_art:
                flush_code()
                flush_table()
                in_ascii_art = True
            ascii_buffer.append(line)
            i += 1
            continue
        elif in_ascii_art and line.strip() and not line.startswith('    '):
            flush_paragraph()

        # Heading detection
        heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading_match:
            flush_paragraph()
            flush_code()
            flush_table()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()

            if level == 1 and text.startswith('# '):
                text = text[2:]
                level = 1
            elif level == 1:
                pass

            if level == 1:
                doc.add_heading(text, level=1)
            elif level == 2:
                doc.add_heading(text, level=2)
            elif level == 3:
                doc.add_heading(text, level=3)
            elif level == 4:
                doc.add_heading(text, level=4)
            i += 1
            continue

        # Horizontal rule → subtle separator
        if line.strip() in ('---', '***', '___'):
            flush_paragraph()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            # Add a thin line via bottom border
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '4')
            bottom.set(qn('w:space'), '4')
            bottom.set(qn('w:color'), 'CCCCCC')
            pBdr.append(bottom)
            pPr.append(pBdr)
            i += 1
            continue

        # Blockquote / callout
        blockquote_match = re.match(r'^>\s*\*?\*?(.+)$', line)
        if blockquote_match:
            flush_paragraph()
            flush_code()
            # Collect multi-line blockquotes
            bq_lines = [blockquote_match.group(1)]
            j = i + 1
            while j < len(lines):
                bqm = re.match(r'^>\s*\*?\*?(.+)$', lines[j])
                if bqm:
                    bq_lines.append(bqm.group(1))
                    j += 1
                else:
                    break
            bq_text = ' '.join(bq_lines)
            # Determine box type
            if any(w in bq_text for w in ['重要', '关键', '铁律', '必须', 'NOTE']):
                box_type = 'warning'
            elif any(w in bq_text for w in ['风险', '不能', '禁止', '失败']):
                box_type = 'danger'
            else:
                box_type = 'info'
            add_callout_box(doc, bq_text, box_type)
            i = j
            continue

        # Regular paragraph
        if line.strip():
            flush_code()
            # Bold (**text**) and inline code (`text`)
            p = doc.add_paragraph()
            parse_inline_formatting(p, line)
        else:
            # Blank line
            pass

        i += 1

    # Flush remaining buffers
    flush_code()
    flush_table()
    flush_paragraph()


def parse_inline_formatting(paragraph, text):
    """Parse inline markdown: **bold**, `code`, *italic*."""
    # Split on inline code first
    parts = re.split(r'(`[^`]+`)', text)

    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = CODE_FONT
            run.font.size = CODE_SIZE
            run.font.color.rgb = RGBColor(0x88, 0x33, 0x22)
            run.element.rPr.rFonts.set(qn('w:eastAsia'), CODE_FONT)
        else:
            # Handle **bold** and *italic*
            sub_parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', part)
            for sp in sub_parts:
                if sp.startswith('**') and sp.endswith('**'):
                    run = paragraph.add_run(sp[2:-2])
                    run.font.bold = True
                    run.font.name = BODY_FONT
                    run.font.size = BODY_SIZE
                    run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
                elif sp.startswith('*') and sp.endswith('*') and not sp.startswith('**'):
                    run = paragraph.add_run(sp[1:-1])
                    run.font.italic = True
                    run.font.name = BODY_FONT
                    run.font.size = BODY_SIZE
                    run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
                else:
                    run = paragraph.add_run(sp)
                    run.font.name = BODY_FONT
                    run.font.size = BODY_SIZE
                    run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)


# ── Post-processing ────────────────────────────────────────────────────────

def add_header_footer(doc):
    """Add document header and footer."""
    for section in doc.sections:
        # Header
        header = section.header
        hp = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = hp.add_run('DisOrderFlow 完整说明书')
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
        run.font.name = BODY_FONT
        run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

        # Footer with page number
        footer = section.footer
        fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run('— ')
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

        # Page number field
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(qn('w:fldCharType'), 'begin')
        run._r.append(fldChar1)

        instrText = OxmlElement('w:instrText')
        instrText.set(qn('xml:space'), 'preserve')
        instrText.text = ' PAGE '
        run._r.append(instrText)

        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(qn('w:fldCharType'), 'end')
        run._r.append(fldChar2)

        run2 = fp.add_run(' —')
        run2.font.size = Pt(8)
        run2.font.color.rgb = RGBColor(0x99, 0x99, 0x99)


def add_toc_page(doc):
    """Insert a simple manual table of contents after the title."""
    # We'll add the TOC as a structured list of the main sections
    # The actual Word TOC field is unreliable across Word versions,
    # so we use a manual TOC for reliability

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(24)
    run = p.add_run('目  录')
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.name = HEADING_FONT
    run.element.rPr.rFonts.set(qn('w:eastAsia'), HEADING_FONT)

    toc_entries = [
        ('1.', '生物学基础（最简版）', '抗体、CDR、抗原、氨基酸、蛋白质折叠、IDP'),
        ('2.', 'Bayesian Flow Network (BFN)', '生成模型原理、与Diffusion的区别、三条"流"'),
        ('3.', 'DisOrderFlow 架构全景', '整体结构、GAEncoder、交叉注意力、Contrastive Head'),
        ('4.', '训练流程', '数据、损失函数、V18→V20突破、训练命令'),
        ('5.', '方式4 的叙事弧线', '核心假设、三次失败、四重根因'),
        ('6.', 'S0 观测性预检', '设计理念、CDR-H3识别、运行方法、判决结果'),
        ('7.', '§5 降级管线', 'Pillar A、已知文库、柔度匹配筛选'),
        ('8.', '完整运行指南', '环境安装、硬件要求、操作步骤、候选解读'),
        ('9.', '结果解读与论文方向', '评分维度、好候选标准、可发表内容'),
        ('10.', '常见问题', '6个FAQ'),
        ('附录A', '关键文件索引', ''),
        ('附录B', '术语表', '中英文对照'),
    ]

    for num, title, desc in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.5)

        run = p.add_run(f'{num}  ')
        run.font.bold = True
        run.font.size = Pt(10)
        run.font.name = BODY_FONT
        run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

        run = p.add_run(title)
        run.font.size = Pt(10)
        run.font.name = BODY_FONT
        run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

        if desc:
            run = p.add_run(f'  — {desc}')
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            run.font.name = BODY_FONT
            run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

    doc.add_page_break()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Read markdown
    with open(MARKDOWN_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = Document()
    setup_styles(doc)

    # ── Title Page ──
    # Add some spacing before title
    for _ in range(6):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run('DisOrderFlow 完整说明书')
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x0d, 0x2b, 0x4e)
    run.font.name = HEADING_FONT
    run.element.rPr.rFonts.set(qn('w:eastAsia'), HEADING_FONT)

    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle_p.add_run('从零理解抗体设计 AI 的底层原理与使用方法')
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x55, 0x6b, 0x82)
    run.font.name = HEADING_FONT
    run.element.rPr.rFonts.set(qn('w:eastAsia'), HEADING_FONT)

    # Metadata
    for _ in range(4):
        doc.add_paragraph()

    meta_items = [
        ('适用读者', '计算机背景，无生物学或蛋白质结构经验'),
        ('目标', '理解 BFN 如何生成抗体、为什么 disorder conditioning 失败、如何自己运行整个管线'),
        ('版本', '2026-07-05 · V14 Plan Complete'),
        ('预计阅读', '45–60 分钟'),
    ]
    for label, value in meta_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f'{label}：')
        run.font.bold = True
        run.font.size = Pt(10)
        run.font.name = BODY_FONT
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
        run = p.add_run(value)
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        run.font.name = BODY_FONT
        run.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)

    doc.add_page_break()

    # ── Table of Contents ──
    add_toc_page(doc)

    # ── Main Content ──
    # Find where the actual content starts (after frontmatter divider)
    content_start = 0
    content_lines = md_text.split('\n')
    for i, line in enumerate(content_lines):
        if line.strip() == '---' and i > 5:
            # Second horizontal rule = end of frontmatter
            content_start = i + 1
            break

    main_md = '\n'.join(content_lines[content_start:]) if content_start > 0 else md_text
    parse_and_render(doc, main_md)

    # ── Header/Footer ──
    add_header_footer(doc)

    # ── Save ──
    doc.save(OUTPUT_PATH)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f'Done: {OUTPUT_PATH}  ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
