"""Render the rewritten ECLS review manuscript using saved evidence only."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from plot_ecls_narrative import build as build_figures
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "publication/figures"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def figures_and_table():
    build_figures(ROOT)
    a = read("results/publication/h3_ecls_adaptation_v1/results.json")["aggregate"]
    t = read("results/publication/h3_ecls_temporal_final_v1/results.json")["aggregate"]
    u = read("results/publication/h3_candidate_reranking_dev_v1/results.json")["aggregate"]
    c = read("results/publication/h3_generator_calibration_v1/analysis.json")["development"]
    rows = [
        ("Exposed adaptation",46,a["mean_ecls_advantage"],a["ecls_advantage_mean_ci95"],"exposed_adaptation"),
        ("Temporal final",15,t["mean_ecls_advantage"],t["ecls_advantage_mean_ci95"],"primary_final"),
        ("Universal ECLS vs complex NLL",7,u["overall_mean_gain"],u["overall_gain_ci95"],"development_gate_failed"),
        ("Calibrated vs random",7,c["mean_generator_aware_nnr"]-.5,c["generator_aware_over_random_ci95"],"exploratory"),
        ("Calibrated vs complex NLL",7,c["mean_gain_over_complex_nll"],c["gain_over_complex_nll_ci95"],"superiority_unestablished")]
    with (ROOT/'release/ecls_v1/primary_results.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['analysis','clusters','mean','ci95_low','ci95_high','evidence_class'])
        for name,n,value,ci,status in rows:writer.writerow([name,n,value,*ci,status])


class NumberedParagraph(Paragraph):
    def draw(self):
        super().draw()
        canvas=self.canv
        canvas.saveState();canvas.setFont('Times-Roman',7);canvas.setFillColor(colors.HexColor('#888888'))
        for i in range(len(self.blPara.lines)):
            canvas._ecls_line=getattr(canvas,'_ecls_line',0)+1
            canvas.drawRightString(-16,self.height-self.style.fontSize-i*self.style.leading,str(canvas._ecls_line))
        canvas.restoreState()


def inline(text):
    text=escape(text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier" size="10">\1</font>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'([\u3000-\u9fff\uff00-\uffef]+)',r'<font name="CJK">\1</font>',text)
    return text


def render(source, output, label, stamp=True):
    if 'CJK' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('CJK',r'C:\Windows\Fonts\simsun.ttc',subfontIndex=0))
    styles={
        'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=12,leading=24,spaceAfter=8,allowWidows=0,allowOrphans=0),
        'front':ParagraphStyle('front',fontName='Times-Roman',fontSize=11,leading=15,spaceAfter=8),
        'h1':ParagraphStyle('h1',fontName='Times-Bold',fontSize=17,leading=21,spaceAfter=15,keepWithNext=True),
        'h2':ParagraphStyle('h2',fontName='Times-Bold',fontSize=13,leading=18,spaceBefore=14,spaceAfter=8,keepWithNext=True),
        'h3':ParagraphStyle('h3',fontName='Times-Bold',fontSize=12,leading=18,spaceBefore=8,spaceAfter=6,keepWithNext=True),
        'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=11,leading=15,spaceAfter=10),
        'table':ParagraphStyle('table',fontName='Times-Roman',fontSize=10,leading=13),
    }
    source=Path(source);lines=source.read_text(encoding='utf-8').splitlines();story=[];i=0
    front = source.name == 'MANUSCRIPT_DRAFT.md'
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line);path=source.parent/match.group(2)
            from PIL import Image as PILImage
            with PILImage.open(path) as im:w,h=im.size
            img=Image(str(path),width=466,height=466*h/w)
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            block=[img,Spacer(1,8)]
            if j<len(lines) and lines[j].startswith('Figure '):
                block.append(Paragraph(inline(lines[j]),styles['caption']));i=j
            story.append(KeepTogether(block));i+=1;continue
        if line.startswith('|'):
            tab=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.match(r'^:?-+:?$',c) for c in cells):
                    tab.append([Paragraph(inline(c),styles['table']) for c in cells])
                i+=1
            table=Table(tab,colWidths=[118,45,63,135,105],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eeef')),
                ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#405862')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
                ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#cccccc'))]))
            story.extend([table,Spacer(1,12)]);continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['h1']));i+=1;continue
        if line.startswith('## '):
            front=False
            story.append(Paragraph(inline(line[3:]),styles['h2']));i+=1;continue
        if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['h3']));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','![','|')) and not re.match(r'^\d+\.\s', lines[i]):
            para.append(lines[i].strip());i+=1
        story.append((Paragraph if front else NumberedParagraph)(inline(' '.join(para)),styles['front' if front else 'body']))
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('Times-Roman',9);canvas.setFillColor(colors.HexColor('#58636a'))
        if stamp:
            canvas.drawString(67,A4[1]-34,label+' | Author-review draft | 29 September 2026')
        canvas.drawRightString(A4[0]-62,30,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=62,leftMargin=67,topMargin=57,bottomMargin=52,
                          title=lines[0].lstrip('# '),author='',
                          pageCompression=1)
    doc.build(story,onFirstPage=page,onLaterPages=page)
    return len(re.findall(r'\b\S+\b',source.read_text(encoding='utf-8')))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean',action='store_true',
                        help='render without the Author-review-draft page header, '
                             'to *clean-named* outputs; the stamped review PDFs are kept')
    args=parser.parse_args()
    figures_and_table()
    outputs=[]
    pairs=[('MANUSCRIPT_DRAFT.md','ECLS_MANUSCRIPT.pdf','ECLS main manuscript'),
           ('ECLS_SUPPLEMENT.md','ECLS_SUPPLEMENT.pdf','ECLS supplementary methods')]
    for md,pdf,label in pairs:
        if args.clean:
            pdf=pdf.replace('.pdf','_clean.pdf')
            words=render(ROOT/'publication'/md,ROOT/'publication'/pdf,label,stamp=False)
        else:
            words=render(ROOT/'publication'/md,ROOT/'publication'/pdf,label)
        outputs.append({'file':'publication/'+pdf,'source_word_count_including_references':words})
    print(json.dumps(outputs,indent=2))
