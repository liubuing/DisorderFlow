"""
iBEC Technical Abstract PDF Generator
Creates a landscape PDF mimicking the PPTX slides (10 pages, slide-style layout).
Uses reportlab with landscape A4 to approximate 16:9 presentation format.
"""
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm, inch
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Frame, PageTemplate, BaseDocTemplate
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Page dimensions - landscape A4
PAGE_W, PAGE_H = landscape(A4)  # 297mm x 210mm

# Colors (matching PPTX)
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

# Try to register MSYH for any CJK needs
try:
    pdfmetrics.registerFont(TTFont('MSYH', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
    HAS_CJK = True
except:
    HAS_CJK = False


class SlideCanvas(canvas.Canvas):
    """Custom canvas that draws slide-like backgrounds."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw_slide_bg(self):
        self.setFillColor(BG_WHITE)
        self.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    def draw_footer(self, page_num):
        self.setFont('Helvetica', 8)
        self.setFillColor(LIGHT_GRAY)
        self.drawRightString(PAGE_W - 40, 25, f"{page_num} / 10")
        self.drawString(40, 25, "Team 28 \u2014 Analyzing Disorder  |  iBEC 2026")

    def draw_accent_bar(self, x, y, w=4, h=30):
        self.setFillColor(ACCENT)
        self.rect(x, y, w, h, fill=1, stroke=0)

    def draw_section_title(self, number, title, y_offset=None):
        y = y_offset or (PAGE_H - 55)
        self.draw_accent_bar(38, y - 5, 4, 28)
        self.setFont('Helvetica', 20)
        self.setFillColor(DARK_TEXT)
        self.drawString(50, y, f"{number}. {title}")


def build_abstract_pdf():
    output_path = "C:\\biological\\DisorderFlow\\iBEC_Submission\\28-\u89e3\u6790\u65e0\u5e8f-\u6280\u672f\u6458\u8981.pdf"

    c = SlideCanvas(output_path, pagesize=landscape(A4))

    # ==================== SLIDE 1: TITLE ====================
    c.draw_slide_bg()

    # Accent line
    c.setFillColor(ACCENT)
    c.rect(80, PAGE_H - 140, PAGE_W - 160, 2, fill=1, stroke=0)

    # Title
    c.setFont('Helvetica', 32)
    c.setFillColor(DARK_TEXT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 185, "DisorderFlow")

    c.setFont('Helvetica', 15)
    c.setFillColor(GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 215,
                        "Bayesian Flow Networks for Disorder-Aware Antibody CDR-H3 Design")

    # Team
    c.setFont('Helvetica', 13)
    c.setFillColor(ACCENT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 265, "Team 28 \u2014 Analyzing Disorder")

    c.setFont('Helvetica', 11)
    c.setFillColor(GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 290,
                        "Chen Haoyang, Hu Jing, Yu Jiarui  |  Qilu University of Technology")
    c.setFont('Helvetica', 10)
    c.setFillColor(LIGHT_GRAY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 315,
                        "iBEC 2026 \u2014 International Bioinformatics Engineering Competition")

    c.showPage()

    # ==================== SLIDE 2: PROBLEM ====================
    c.draw_slide_bg()
    c.draw_section_title(1, "The IDP Antibody Design Challenge")
    c.draw_footer(2)

    bullets = [
        "Intrinsically disordered proteins (IDPs) lack stable 3D structure yet drive",
        "  neurodegenerative diseases (Alzheimer's, Parkinson's)",
        "",
        "Therapeutic targets: Amyloid-beta, tau, alpha-synuclein \u2014 all disordered",
        "",
        "Most antibody design methods assume rigid, well-ordered epitopes",
        "  \u2014 fundamentally inadequate for IDP targets",
        "",
        "CDR-H3 is the primary determinant of antigen specificity and the most",
        "  diverse loop in the antibody structure",
        "",
        "Key gap: no computational framework couples epitope disorder profiles",
        "  to CDR-H3 sequence selection",
    ]
    y = PAGE_H - 95
    c.setFont('Helvetica', 13)
    c.setFillColor(DARK_TEXT)
    for line in bullets:
        if line == "":
            y -= 6
            continue
        if line.startswith("  "):
            c.setFont('Helvetica', 12)
            c.setFillColor(GRAY)
            c.drawString(75, y, line.strip())
        else:
            c.setFont('Helvetica', 13)
            c.setFillColor(DARK_TEXT)
            c.drawString(55, y, "\u2022 " + line)
        y -= 18

    # Callout
    c.setFillColor(ACCENT)
    c.setFont('Helvetica', 11)
    c.drawString(55, y - 15, "Goal: Build a disorder-aware scoring framework that conditions CDR-H3")
    c.drawString(55, y - 30, "design on both backbone geometry and per-residue epitope disorder profiles.")

    c.showPage()

    # ==================== SLIDE 3: PLATFORM ====================
    c.draw_slide_bg()
    c.draw_section_title(2, "DisorderFlow Platform Architecture")
    c.draw_footer(3)

    # Four component boxes
    components = [
        ("BFN Core", "Categorical distributions over", "20 amino acids; iterative", "refinement via receiver net"),
        ("Disorder Head", "Per-residue disorder prediction;", "focal-weighted loss (gamma=4.0);", "handles 0.74% class imbalance"),
        ("Position Routing", "Direct coupling of antigen", "residue disorder values to", "CDR-antigen pair features"),
        ("Contrastive Loss", "Factual vs shuffled/mismatched", "disorder profiles; rank-aware", "training signal"),
    ]
    box_w = 155
    box_h = 100
    start_x = 45
    gap = 12
    box_y = PAGE_H - 200

    for i, (title, *lines) in enumerate(components):
        x = start_x + i * (box_w + gap)
        # Box background
        c.setFillColor(BOX_BG)
        c.setStrokeColor(BOX_BORDER)
        c.setLineWidth(1.2)
        c.roundRect(x, box_y, box_w, box_h, 6, fill=1, stroke=1)

        # Title
        c.setFont('Helvetica', 12)
        c.setFillColor(ACCENT)
        c.drawCentredString(x + box_w / 2, box_y + box_h - 20, title)

        # Description
        c.setFont('Helvetica', 9)
        c.setFillColor(GRAY)
        for j, line in enumerate(lines):
            c.drawCentredString(x + box_w / 2, box_y + box_h - 40 - j * 14, line)

    # Training pipeline
    c.setFont('Helvetica', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, box_y - 35, "Training Pipeline")

    c.setFont('Helvetica', 10)
    c.setFillColor(GRAY)
    c.drawString(55, box_y - 55,
                 "SAbDab2 structural database  |  6-axis homology-disjoint split (PDB, paired Ab, VH, VL, H3, antigen)")
    c.drawString(55, box_y - 72,
                 "Backbone coordinates + disorder labels + negative profiles  |  ProteinMPNN v_48_020 for evaluation")

    c.showPage()

    # ==================== SLIDE 4: ECLS METRIC ====================
    c.draw_slide_bg()
    c.draw_section_title(3, "ECLS: Epitope-Conditioned Likelihood Shift")
    c.draw_footer(4)

    # Formula box
    c.setFillColor(FORMULA_BG)
    c.setStrokeColor(HexColor('#DEE2E6'))
    c.setLineWidth(0.8)
    c.roundRect(80, PAGE_H - 165, PAGE_W - 160, 40, 6, fill=1, stroke=1)
    c.setFont('Courier', 16)
    c.setFillColor(DARK_TEXT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 150,
                        "ECLS(s) = NLL_complex(s) \u2212 NLL_peptide-stripped(s)")

    # Description
    c.setFont('Helvetica', 11)
    c.setFillColor(GRAY)
    y = PAGE_H - 195
    c.drawString(55, y, "Measures whether a CDR-H3 sequence is preferentially supported by the presence of")
    c.drawString(55, y - 16, "peptide epitope coordinates. Positive advantage = native sequence benefits more")
    c.drawString(55, y - 32, "from epitope context than composition-matched shuffles (200 per record).")

    # Methodology
    c.setFont('Helvetica', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, y - 65, "Methodology")

    methods = [
        "Score only heavy chain; fix non-H3 positions; retain light chain + peptide context",
        "200 composition-matched shuffles per record as counterfactual controls",
        "Official SAbDab2 CDR-H3 annotations; heavy-atom distance \u2264 4.5 \u00c5 contact gate",
        "Official antigen cluster as inference unit; 10,000 bootstrap resamples for CIs",
        "One-shot evaluation on sealed temporal data; frozen gates set before evaluation",
    ]
    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    my = y - 88
    for m in methods:
        c.drawString(65, my, "\u2022  " + m)
        my -= 18

    c.showPage()

    # ==================== SLIDE 5: ECLS RESULTS ====================
    c.draw_slide_bg()
    c.draw_section_title(4, "Results: ECLS Detection and Temporal Transfer")
    c.draw_footer(5)

    # Left table: Adaptation
    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, PAGE_H - 85, "Adaptation Set (46 clusters)")

    data_left = [
        ('Metric', 'Value'),
        ('Mean ECLS advantage', '0.217'),
        ('Median', '0.203'),
        ('95% Bootstrap CI', '[0.146, 0.290]'),
        ('Positive clusters', '80.4%'),
    ]
    _draw_table(c, 55, PAGE_H - 105, data_left, col_widths=[150, 100])

    # Right table: Temporal final
    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(PAGE_W / 2 + 20, PAGE_H - 85, "Temporal Final (15 clusters, post-2021)")

    data_right = [
        ('Metric', 'Value'),
        ('Mean ECLS advantage', '0.172'),
        ('Median', '0.119'),
        ('95% Bootstrap CI', '[0.059, 0.292]'),
        ('Positive clusters', '80.0%'),
    ]
    _draw_table(c, PAGE_W // 2 + 20, PAGE_H - 105, data_right, col_widths=[150, 100])

    # Summary
    y = PAGE_H - 270
    c.setFont('Helvetica', 11)
    c.setFillColor(GRAY)
    c.drawString(55, y, "Both frozen gate sets passed. Deposited CDR-H3 sequences carry measurable")
    c.drawString(55, y - 16, "epitope-conformation-specific signal that transfers to post-training structures.")
    c.drawString(55, y - 32, "Sign-flip P < 1e-6 (adaptation) and P = 0.014 (temporal final).")

    # Green highlight box
    c.setFillColor(GREEN_HL)
    c.roundRect(55, y - 70, PAGE_W - 110, 28, 4, fill=1, stroke=0)
    c.setFont('Helvetica', 10)
    c.setFillColor(GREEN_TEXT)
    c.drawString(70, y - 60, "Key: 80%+ positive clusters on sealed temporal data confirms non-trivial epitope-H3 coupling")

    c.showPage()

    # ==================== SLIDE 6: T2.1 v2 ====================
    c.draw_slide_bg()
    c.draw_section_title(5, "T2.1 v2: Deterministic Torsion Recovery Benchmark")
    c.draw_footer(6)

    c.setFont('Helvetica', 11)
    c.setFillColor(GRAY)
    c.drawString(55, PAGE_H - 85,
                 "Deterministic phi/psi torsion perturbation to 1.5\u20133.0 \u00c5 RMSD tier, then test if native")
    c.drawString(55, PAGE_H - 101,
                 "residue contacts guide recovery. Peptide restraint: 100 kJ/mol/nm\u00b2.")

    # Results table
    data_t2 = [
        ('Metric', 'Value'),
        ('Total structures', '31'),
        ('Torsion hits (target tier)', '26/31 (83.9%)'),
        ('Valid structures overall', '25/31 (80.6%)'),
        ('Mean held-out contact recovery', '0.581'),
        ('95% CI', '[0.533, 0.635]'),
        ('Positive recovery fraction', '100%'),
    ]
    _draw_table(c, 55, PAGE_H - 130, data_t2, col_widths=[180, 120])

    # Comparison box
    comp_x = PAGE_W // 2 + 30
    c.setFillColor(FORMULA_BG)
    c.setStrokeColor(HexColor('#DEE2E6'))
    c.setLineWidth(0.8)
    c.roundRect(comp_x, PAGE_H - 280, 230, 150, 6, fill=1, stroke=1)

    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(comp_x + 10, PAGE_H - 145, "T2 vs T2.1 v2 (recalibrated)")

    c.setFont('Helvetica', 10)
    c.setFillColor(GRAY)
    lines = [
        "T2:  3/7 valid, recovery = \u22120.017",
        "   (high-temp perturbation, poor calibration)",
        "",
        "T2.1 v2:  25/31 valid, recovery = 0.581",
        "   (deterministic torsion, 100 kJ/mol/nm\u00b2)",
        "",
        "Random restraint recovery = 0.619;",
        "contact-specific mechanism unsupported.",
    ]
    ly = PAGE_H - 165
    for line in lines:
        c.drawString(comp_x + 10, ly, line)
        ly -= 14

    # Green highlight
    c.setFillColor(GREEN_HL)
    c.roundRect(55, PAGE_H - 310, PAGE_W - 110, 24, 4, fill=1, stroke=0)
    c.setFont('Helvetica', 10)
    c.setFillColor(GREEN_TEXT)
    c.drawString(70, PAGE_H - 303, "80.6% valid overall; random restraints outperform supplied contacts (0.619 vs 0.581)")

    c.showPage()

    # ==================== SLIDE 7: 4HIX DESIGN ====================
    c.draw_slide_bg()
    c.draw_section_title(6, "4HIX ProteinMPNN Computational Case Study")
    c.draw_footer(7)

    c.setFont('Helvetica', 11)
    c.setFillColor(GRAY)
    c.drawString(55, PAGE_H - 85,
                 "Scaffold: 4HIX (humanized 3D6 Fab, Abeta 1-6 DAEFRH). ProteinMPNN T=0.5, 20 samples.")
    c.drawString(55, PAGE_H - 101,
                 "AF2 multimer V3 with VH:VL chain separator. All 20 designs passed validation.")

    # Design table
    data_4hix = [
        ('Rank', 'H3 Sequence', 'ipTM', 'pLDDT', 'iPAE'),
        ('Native', 'VRYDHYSGSSDY', '0.449', '0.194', '24.7'),
        ('#1', 'LYDESKDAESE', '0.462', '0.200', '24.5'),
        ('#2', 'LYDAHHGAHSL', '0.461', '0.192', '24.4'),
        ('#3', 'LYDGSIGAESQ', '0.460', '0.195', '24.5'),
        ('#4', 'LYDSSVDASGH', '0.459', '0.199', '24.7'),
        ('#5', 'LFNEANCAESY', '0.458', '0.199', '24.3'),
    ]
    tbl_y = PAGE_H - 125
    _draw_table(c, 55, tbl_y, data_4hix, col_widths=[45, 120, 45, 50, 45],
                highlight_row=2)

    # Key findings on the right
    kx = PAGE_W // 2 + 40
    c.setFont('Helvetica', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(kx, tbl_y + 10, "Key Findings")

    findings = [
        "20/20 designs pass AF2 validation",
        "Top 3 exceed native ipTM (0.449)",
        "Best design: ipTM = 0.462 (+2.8%, descriptive difference)",
        "Chain separator fix: VH:VL prevents",
        "  4.5x ipTM underestimation",
    ]
    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    fy = tbl_y - 12
    for f in findings:
        if f.startswith("  "):
            c.setFont('Helvetica', 10)
            c.setFillColor(GRAY)
            c.drawString(kx + 10, fy, f.strip())
        else:
            c.setFont('Helvetica', 11)
            c.setFillColor(DARK_TEXT)
            c.drawString(kx, fy, "\u2022  " + f)
        fy -= 17

    # Note
    c.setFont('Helvetica', 9)
    c.setFillColor(LIGHT_GRAY)
    c.drawString(55, 50, "Note: 6-residue epitope; AF2 less discriminative for short peptides. 4HIX in AF2 training set (2012). No wet-lab validation.")

    c.showPage()

    # ==================== SLIDE 8: EVIDENCE HIERARCHY ====================
    c.draw_slide_bg()
    c.draw_section_title(7, "Evidence Hierarchy and Methodological Rigor")
    c.draw_footer(8)

    evidence = [
        ("1. ECLS Temporal Final", "n=15 clusters, post-2021, sealed", "Positive discrimination (0.172, 80%+)"),
        ("2. ECLS Adaptation", "n=46 clusters, exposed", "Replicates temporal final pattern"),
        ("3. T2.1 v2 Recovery", "n=31 structures, 80.6% valid", "Random control higher than supplied"),
        ("4. 4HIX Design", "n=20 designs, 20/20 AF2 pass", "Prospective design (top ipTM 0.462)"),
        ("5. T1 Ensemble", "n=7 clusters", "Single-pose overconfidence (\u22120.200)"),
        ("6. Generator Calibration", "n=7 clusters, exploratory", "Calibrated reranker works"),
    ]

    ey = PAGE_H - 100
    for i, (title, detail, result) in enumerate(evidence):
        # Strength bar
        intensity = max(0.3, 1.0 - i * 0.12)
        bar_color = Color(0.031 * intensity, 0.498 * intensity, 0.427 * intensity)
        c.setFillColor(bar_color)
        c.rect(55, ey - 3, 4, 22, fill=1, stroke=0)

        c.setFont('Helvetica', 12)
        c.setFillColor(DARK_TEXT)
        c.drawString(68, ey + 5, title)

        c.setFont('Helvetica', 9)
        c.setFillColor(LIGHT_GRAY)
        c.drawString(68, ey - 8, detail)

        c.setFont('Helvetica', 11)
        c.setFillColor(GRAY)
        c.drawString(PAGE_W // 2 + 20, ey, result)

        ey -= 42

    # Rigor note
    c.setFont('Helvetica', 9)
    c.setFillColor(LIGHT_GRAY)
    c.drawString(55, 50,
                 "All evaluations use frozen gates, one-shot evaluation on sealed data, and homology-disjoint")
    c.drawString(55, 38,
                 "splitting across 6 axes. Bootstrap CIs with 10,000 resamples.")

    c.showPage()

    # ==================== SLIDE 9: SOFTWARE & REPRODUCIBILITY ====================
    c.draw_slide_bg()
    c.draw_section_title(8, "Software Implementation and Reproducibility")
    c.draw_footer(9)

    # Left column: Software
    c.setFont('Helvetica', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, PAGE_H - 85, "Software Components")

    sw_items = [
        "BFN core: categorical distributions, focal-weighted losses",
        "Disorder-augmented dataset: label injection + contrastive profiles",
        "Receiver network: position-sensitive routing",
        "IDP design pipeline: 5-stage target-to-validation",
        "AF2 infrastructure: proper chain separator handling",
    ]
    c.setFont('Helvetica', 10)
    c.setFillColor(DARK_TEXT)
    sy = PAGE_H - 108
    for item in sw_items:
        c.drawString(65, sy, "\u2022  " + item)
        sy -= 17

    # Right column: Reproducibility
    rx = PAGE_W // 2 + 20
    c.setFont('Helvetica', 12)
    c.setFillColor(DARK_TEXT)
    c.drawString(rx, PAGE_H - 85, "Reproducibility")

    rep_items = [
        "Frozen YAML configuration contracts",
        "SHA256 manifests for all result files",
        "results_t2.1_v2_final.json (31 records)",
        "4hix_final_validation/final_report.json (20 designs)",
        "INTEGRATIVE_ANALYSIS.md evidence chain",
    ]
    c.setFont('Helvetica', 10)
    c.setFillColor(DARK_TEXT)
    ry = PAGE_H - 108
    for item in rep_items:
        c.drawString(rx + 10, ry, "\u2022  " + item)
        ry -= 17

    # Yellow box: Critical bugs
    bug_y = PAGE_H - 310
    c.setFillColor(YELLOW_BG)
    c.setStrokeColor(YELLOW_BORDER)
    c.setLineWidth(1)
    c.roundRect(55, bug_y, PAGE_W - 110, 105, 6, fill=1, stroke=1)

    c.setFont('Helvetica', 11)
    c.setFillColor(YELLOW_TEXT)
    c.drawString(70, bug_y + 82, "Critical Bugs Fixed During Development")

    c.setFont('Helvetica', 9)
    bugs = [
        "1. Disorder head: training batches never contained disorder_label \u2192 PaddingCollate dropped",
        "   the field \u2192 disorder loss never fired. Fixed with explicit label injection.",
        "2. AF2 chain separator: antibody chains must use VH:VL (not concatenation) or ipTM drops 4.5x.",
        "3. T2.1 perturbation: recalibrated protocol reached 80.6% overall validity; mechanism unsupported.",
    ]
    by = bug_y + 62
    for b in bugs:
        if b.startswith("   "):
            c.drawString(80, by, b.strip())
        else:
            c.drawString(70, by, b)
        by -= 14

    c.showPage()

    # ==================== SLIDE 10: CONCLUSIONS ====================
    c.draw_slide_bg()
    c.draw_section_title(9, "Conclusions and Future Work")
    c.draw_footer(10)

    # Conclusions
    c.setFont('Helvetica', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, PAGE_H - 85, "Conclusions")

    conclusions = [
        "ECLS establishes epitope-conditioned CDR-H3 signal (46 clusters, mean 0.217, 80.4% positive)",
        "Temporal transfer confirmed on sealed post-2021 data (mean 0.172, 80% positive, P=0.014)",
        "T2.1 v2: 80.6% valid overall; random control 0.619 vs supplied 0.581",
        "4HIX prospective design: 20/20 pass AF2, top 3 exceed native ipTM",
        "Disorder-aware framework provides principled approach to IDP antibody design",
    ]
    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    cy = PAGE_H - 108
    for item in conclusions:
        c.drawString(65, cy, "\u2022  " + item)
        cy -= 18

    # Limitations
    c.setFont('Helvetica', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, cy - 15, "Limitations")

    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(65, cy - 35, "\u2022  No wet-lab validation; all results computational")
    c.drawString(65, cy - 53, "\u2022  Short epitope (6 residues); AF2 less discriminative; 4HIX in AF2 training set")

    # Future work
    c.setFont('Helvetica', 13)
    c.setFillColor(DARK_TEXT)
    c.drawString(55, cy - 85, "Future Work")

    c.setFont('Helvetica', 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(65, cy - 105, "\u2022  CAID benchmark evaluation of retrained disorder head")
    c.drawString(65, cy - 123, "\u2022  Extended IDP targets (tau, alpha-synuclein) and experimental validation (SPR/ELISA)")

    c.save()
    print(f"Abstract PDF saved: {output_path}")
    return output_path


def _draw_table(c, x, y, data, col_widths=None, highlight_row=None):
    """Draw a simple table on the canvas."""
    if col_widths is None:
        col_widths = [120] * len(data[0])

    row_h = 20
    total_w = sum(col_widths)

    for r, row in enumerate(data):
        ry = y - r * row_h
        cx = x

        # Row background
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

        # Cell text
        for ci, cell_val in enumerate(row):
            if r == 0:
                c.setFillColor(white)
                c.setFont('Helvetica-Bold', 10)
            else:
                c.setFillColor(DARK_TEXT)
                c.setFont('Helvetica', 10)
            c.drawString(cx + 5, ry + 3, str(cell_val))
            cx += col_widths[ci]

    # Grid lines
    c.setStrokeColor(HexColor('#DEE2E6'))
    c.setLineWidth(0.5)
    total_h = len(data) * row_h
    c.rect(x, y - (len(data) - 1) * row_h - 4, total_w, total_h, fill=0, stroke=1)
    # Horizontal lines
    for r in range(1, len(data)):
        ly = y - r * row_h + row_h - 4
        c.line(x, ly, x + total_w, ly)
    # Vertical lines
    cx = x
    for cw in col_widths[:-1]:
        cx += cw
        c.line(cx, y + row_h - 4, cx, y - (len(data) - 1) * row_h - 4)


if __name__ == '__main__':
    build_abstract_pdf()
