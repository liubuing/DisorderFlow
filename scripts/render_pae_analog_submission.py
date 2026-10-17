"""Render the PAE analogue-transfer manuscript and PLOS cover letter to
author-review PDFs. Self-contained twin of the render pipeline in
render_ecls_submission.py; reads only the publication/ working drafts and the
figure produced by plot_pae_analog_figure1_v30.py. No model inference.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    ("publication/pae_analog_manuscript_draft_v1.md",
     "publication/PAE_ANALOG_MANUSCRIPT.pdf",
     "PAE analogue transfer | PLOS Computational Biology target"),
    ("publication/pae_analog_cover_letter_ploscb_v1.md",
     "publication/PAE_ANALOG_COVER_LETTER.pdf",
     "PAE cover letter | PLOS Computational Biology"),
]


class NumberedParagraph(Paragraph):
    def draw(self):
        super().draw()
        canvas = self.canv
        canvas.saveState()
        canvas.setFont("Times-Roman", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        for i in range(len(self.blPara.lines)):
            canvas._pae_line = getattr(canvas, "_pae_line", 0) + 1
            canvas.drawRightString(-16, self.height - self.style.fontSize
                                   - i * self.style.leading, str(canvas._pae_line))
        canvas.restoreState()


def inline(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="10">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"([\u3000-\u9fff\uff00-\uffef]+)", r'<font name="CJK">\1</font>', text)
    return text


def render(source, output, label, stamp=True):
    if "CJK" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("CJK", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
    styles = {
        "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=12, leading=24,
                               spaceAfter=8, allowWidows=0, allowOrphans=0),
        "front": ParagraphStyle("front", fontName="Times-Roman", fontSize=11, leading=15,
                                spaceAfter=8),
        "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=17, leading=21,
                             spaceAfter=15, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=13, leading=18,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName="Times-Bold", fontSize=12, leading=18,
                             spaceBefore=8, spaceAfter=6, keepWithNext=True),
        "caption": ParagraphStyle("caption", fontName="Times-Roman", fontSize=11,
                                  leading=15, spaceAfter=10),
    }
    source = Path(source)
    source_text = source.read_text(encoding="utf-8")
    lines = source_text.splitlines()
    story = []
    i = 0
    revision = re.search(r"(?:Author-review version, |Revision: )([^\.]+)\.", source_text)
    review_date = revision.group(1) if revision else "Undated"
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            path = source.parent / match.group(2)
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                w, h = im.size
            img = Image(str(path), width=466, height=466 * h / w)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            block = [img, Spacer(1, 8)]
            if j < len(lines) and lines[j].startswith("Figure "):
                block.append(Paragraph(inline(lines[j]), styles["caption"]))
                i = j
            story.append(KeepTogether(block))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["h3"]))
            i += 1
            continue
        para = [line]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(("#", "![", "|"))
               and not re.match(r"^\d+\.\s", lines[i])):
            para.append(lines[i].strip())
            i += 1
        story.append(NumberedParagraph(inline(" ".join(para)), styles["body"]))

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.setFillColor(colors.HexColor("#58636a"))
        if stamp:
            canvas.drawString(67, A4[1] - 34, label + " | Author-review draft | " + review_date)
        canvas.drawRightString(A4[0] - 62, 30, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=62, leftMargin=67,
                            topMargin=57, bottomMargin=52,
                            title=lines[0].lstrip("# "), author="", pageCompression=1)
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return len(re.findall(r"\b\S+\b", source_text))


def main():
    outputs = []
    for src, out, label in PAIRS:
        words = render(ROOT / src, ROOT / out, label)
        outputs.append({"file": out, "source_word_count_including_references": words})
    import json
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
