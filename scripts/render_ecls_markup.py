"""Render a four-category AI-flavour markup copy of the ECLS manuscript.

Source (reproduced verbatim; only highlights added):
  publication/MANUSCRIPT_DRAFT.md
Output:
  publication/ECLS_SUBMISSION_MARKUP.docx

Categories (aligned with scripts/render_pae_plos_markup.py and
iBEC_Submission/AI_FLAVOR_REWRITE_WORKSHEET.md):
  A yellow       boundary-negation sentences ("X, not Y" family)
  D bright green repeated hedge/limitation formulas
  B turquoise    signpost / meta paragraphs
  C pink         aphoristic short sentences

The 2026-11-03 boundary recast should leave A at zero here; any A sentence the
renderer finds is a residual the author may still want to fold into a positive
statement (see publication/ECLS_BOUNDARY_RECAST_20261103.md).

The script self-verifies highlight counts and source coverage before exiting.
"""

# 目的：把 ECLS 主稿（Bioinformatics 版）渲染为带四类 AI 痕迹高亮的 DOCX 审阅副本：
# A 边界否定句/黄、D 限定公式/亮绿、B 路标元话语/青、C 警句/粉；退出前自检
# 高亮计数与源覆盖。仅用于稿件语言审阅，不改动科学内容。

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "publication/MANUSCRIPT_DRAFT.md"
OUT = ROOT / "publication/ECLS_SUBMISSION_MARKUP.docx"

COLORS = {
    "A": WD_COLOR_INDEX.YELLOW,
    "D": WD_COLOR_INDEX.BRIGHT_GREEN,
    "B": WD_COLOR_INDEX.TURQUOISE,
    "C": WD_COLOR_INDEX.PINK,
}

LEGEND = {
    "A": "黄色 = 边界否定句（“X, not Y”家族）：多为科学主张本身，只可换句式，不可删语义",
    "D": "亮绿 = 重复的限定/保守措辞公式：换词汇，保留限定含义",
    "B": "青色 = 路标句/元话语段落：可放心改写",
    "C": "粉色 = 警句式短句：保留一两处即可",
}

A_SUBSTR = [", not ", " is not ", " are not ", " was not ", " not evidence ",
            " no binding, affinity", " cannot ", " rather than ",
            " no detectable ", " not binding rates", " ruling out",
            " rules out", " no selection by effect size", " was not enlarged",
            " not a validation", " are not reused", " no multiple-comparison",
            " no wall-clock", " carries no ", " with no implication",
            " not measured affinity", " no templates", " no claim"]

D_SUBSTR = ["does not establish", "claims are limited", "explicitly",
            "descriptive", "quantized by construction", "coarse by design",
            "met exactly", "relative to the teacher", "bounded", "by construction",
            "preregistered minimums"]

B_STARTS = ["the study asks two things", "the study establishes a boundary",
            "there are further limits", "the useful outcome is",
            "this provided development evidence", "the practical question",
            "the result answers"]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\*\[\"])")

# 前后附件（年份、DOI、声明）不是行文，不参与高亮。
SKIP_HIGHLIGHT_PREFIXES = ("Source:", "Planned Zenodo", "Pending author",
                           "Author-review version", "No AI tool",
                           "License", "Preprint")


def categorize_sentence(sentence):
    low = " " + sentence.lower()
    if any(sub in low for sub in A_SUBSTR):
        return "A"
    if any(sub in low for sub in D_SUBSTR):
        return "D"
    words = re.findall(r"[A-Za-z\u00c0-\u024f]+", sentence)
    if 0 < len(words) <= 6 and sentence.rstrip().endswith("."):
        return "C"
    return None


def paragraph_category(text):
    low = text.strip().lower()
    if any(low.startswith(start) for start in B_STARTS):
        return "B"
    return None


FIG_SENTINEL = "\u0001"


def protect_figure_numbers(text):
    # "Figure 1." would otherwise be split off as a one-word sentence and
    # counted as an aphorism (C); keep it inside its caption sentence.
    return re.sub(r"\b(Figure \d+[A-Z]?)\.(?=\s)", r"\1" + FIG_SENTINEL, text)


def add_runs(paragraph, text):
    sentences = SENT_SPLIT.split(protect_figure_numbers(text))
    counts = {"A": 0, "D": 0, "B": 0, "C": 0}
    for sentence in sentences:
        if not sentence:
            continue
        sentence = sentence.replace(FIG_SENTINEL, ".")
        cat = categorize_sentence(sentence)
        if cat:
            run = paragraph.add_run(sentence + " ")
            run.font.highlight_color = COLORS[cat]
            run.font.size = Pt(11)
            counts[cat] += 1
        else:
            run = paragraph.add_run(sentence + " ")
            run.font.size = Pt(11)
    return counts


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    doc = Document()
    doc.add_heading("ECLS manuscript — AI-flavour markup (Bioinformatics submission)", level=0)
    doc.add_paragraph(
        "源文件（逐字对应，仅加高亮）：publication/MANUSCRIPT_DRAFT.md。"
        "直接在本 Word 文件中修改；改完交回移植到 markdown 源并重渲染。"
    )
    doc.add_paragraph("颜色图例：")
    for key in ("A", "D", "B", "C"):
        p = doc.add_paragraph()
        run = p.add_run("■ ")
        run.font.highlight_color = COLORS[key]
        p.add_run(f" {key} 类。{LEGEND[key]}")
    doc.add_paragraph(
        "注：2026-11-03 边界句改写后 A 类应为 0；若仍见黄色，即为残留，"
        "可参照 publication/ECLS_BOUNDARY_RECAST_20261103.md 的句式改写。"
    )

    total = {"A": 0, "D": 0, "B": 0, "C": 0}
    src_paragraphs = 0
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# ") and not line.startswith("## "):
            doc.add_heading(line[2:], level=1)
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:], level=2)
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
            continue
        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            p = doc.add_paragraph()
            run = p.add_run(f"［图位置：{match.group(2)}（题注：{match.group(1)}）］")
            run.font.size = Pt(11)
            continue
        if re.match(r"^\d+\.\s", line.strip()):
            p = doc.add_paragraph()
            run = p.add_run(line)
            run.font.size = Pt(11)
            continue
        src_paragraphs += 1
        p = doc.add_paragraph()
        if line.strip().startswith(SKIP_HIGHLIGHT_PREFIXES):
            run = p.add_run(line)
            run.font.size = Pt(11)
            continue
        pcat = paragraph_category(line)
        if pcat:
            run = p.add_run(line + " ")
            run.font.highlight_color = COLORS[pcat]
            run.font.size = Pt(11)
            total[pcat] += 1
        else:
            counts = add_runs(p, line)
            for k in total:
                total[k] += counts[k]

    doc.save(OUT)

    check = Document(OUT)
    hl = 0
    for para in check.paragraphs:
        for run in para.runs:
            if run.font.highlight_color is not None:
                hl += 1
    body_text = "\n".join(p.text for p in check.paragraphs)
    missing = [ln for ln in lines
               if ln.strip() and not ln.startswith(("#", "!["))
               and not re.match(r"^\d+\.\s", ln.strip())
               and ln.strip()[:40] not in body_text]
    if missing or hl == 0:
        print("VERIFY FAILED: missing paragraphs:", missing[:5], "highlights:", hl)
        return 1
    print(f"verify ok: source paragraphs {src_paragraphs}; highlighted runs {hl}; "
          f"category totals {total}")
    print(f"output: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
