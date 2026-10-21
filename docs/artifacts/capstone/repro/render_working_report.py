"""Render the current capstone manuscript and retain page-level inspection images."""
import argparse
from datetime import date
import hashlib
from html import escape
import json
from importlib.metadata import version
from pathlib import Path
import re

from pypdf import PdfReader
import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

FONT_FILES = [('CapstoneSerif', 'DejaVuSerif.ttf'), ('CapstoneSerifBold', 'DejaVuSerif-Bold.ttf'),
              ('CapstoneSerifItalic', 'DejaVuSerif-Italic.ttf'), ('CapstoneSerifBoldItalic', 'DejaVuSerif-BoldItalic.ttf')]


def normalized(text):
    return text.translate(str.maketrans({'\u2010':'-','\u2011':'-','\u2012':'-','\u2013':'-','\u2014':'-','\u2212':'-','\u00a0':' '}))


def inline(text):
    text=escape(normalized(text))
    def link(m):
        label,url=m.groups()
        return f'<a href="{url}" color="#145670">{label}</a>' if url.startswith(('https://','http://')) else label
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier" size="8.7">\1</font>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    return text


def main(source,output,pages,font_dir,report_date):
    source=Path(source).resolve();output=Path(output);pages=Path(pages)
    font_dir=Path(font_dir).resolve();report_date=date.fromisoformat(report_date).strftime('%d %B %Y')
    for name,file in FONT_FILES:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    pdfmetrics.registerFontFamily('CapstoneSerif',normal='CapstoneSerif',bold='CapstoneSerifBold',italic='CapstoneSerifItalic',boldItalic='CapstoneSerifBoldItalic')
    if output.exists() or pages.exists():raise FileExistsError('use a fresh report/layout output')
    output.parent.mkdir(parents=True,exist_ok=True);pages.mkdir(parents=True)
    source_bytes=source.read_bytes();lines=source_bytes.decode().splitlines()
    width=letter[0]-2*.65*72
    body=ParagraphStyle('body',fontName='CapstoneSerif',fontSize=10.3,leading=14.1,spaceAfter=6,allowWidows=0,allowOrphans=0)
    heading=ParagraphStyle('heading',parent=body,fontName='CapstoneSerifBold',fontSize=11.4,leading=14.3,spaceBefore=10,spaceAfter=5,keepWithNext=True)
    title=ParagraphStyle('title',parent=body,fontName='CapstoneSerifBold',fontSize=17,leading=21,alignment=TA_CENTER,spaceAfter=8)
    small=ParagraphStyle('small',parent=body,fontSize=8.4,leading=11,spaceAfter=5)
    cell=ParagraphStyle('cell',parent=body,fontSize=8.1,leading=10.5,spaceAfter=0)
    story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):
            story.extend([Paragraph(inline(line[2:]),title),Paragraph('Anurag Lnu | MS Computer Science, Rochester Institute of Technology',small),Spacer(1,3)])
            i+=1;continue
        if line=='## Working evidence links':break
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]),heading));i+=1;continue
        if line.startswith('!['):
            match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
            if not match:raise ValueError('unrecognized figure')
            image=Image(str(source.parent/match.group(2)));ratio=image.imageHeight/image.imageWidth
            image.drawWidth=width;image.drawHeight=width*ratio
            story.extend([Spacer(1,4),image,Spacer(1,7)]);i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                r=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',v.replace(' ','')) for v in r):rows.append(r)
                i+=1
            weights=[max(10,min(25,max(len(r[c]) for r in rows))) for c in range(len(rows[0]))]
            colwidth=[width*w/sum(weights) for w in weights]
            data=[[Paragraph(inline(v),cell) for v in r] for r in rows]
            table=Table(data,colWidths=colwidth,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f3')),('LINEABOVE',(0,0),(-1,0),.6,colors.HexColor('#33434b')),('LINEBELOW',(0,0),(-1,0),.35,colors.HexColor('#8b989e')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#33434b')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([Spacer(1,4),table,Spacer(1,8)]);continue
        if line.startswith('- '):
            content=line[2:];i+=1
            while i<len(lines) and lines[i].startswith('  '):content+=' '+lines[i].strip();i+=1
            story.append(Paragraph(inline(content),ParagraphStyle('bullet',parent=body,leftIndent=11,firstLineIndent=-8),bulletText='-'));continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','- ')):
            paragraph.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(re.sub(r'(?<=[A-Za-z])- (?=[a-z])','-',' '.join(paragraph))),body))
    references=[
        ('Wang et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. Revised version with ChestX-ray14 appendix.','https://arxiv.org/html/1705.02315v5'),
        ('Yin et al. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML.','https://arxiv.org/html/1803.01498v2'),
        ('Mironov (2017). Renyi Differential Privacy. IEEE Computer Security Foundations Symposium.','https://arxiv.org/html/1702.07476'),
        ('Guerraoui et al. (2021). Differential Privacy and Byzantine Resilience in SGD: Do They Add Up? PODC.','https://arxiv.org/pdf/2102.08166v3'),
        ('Allouah et al. (2023). On the Privacy-Robustness-Utility Trilemma in Distributed Learning. ICML.','https://proceedings.mlr.press/v202/allouah23a.html'),
        ('Gu, Li and Xiong (2025). DP-BREM: Differentially-Private and Byzantine-Robust Federated Learning with Client Momentum. USENIX Security.','https://www.usenix.org/conference/usenixsecurity25/presentation/gu-xiaolan'),
        ('Allouah, Guerraoui and Stephan (2025). Towards Trustworthy Federated Learning with Untrusted Participants. ICML. Proceedings abstract used for positioning only.','https://proceedings.mlr.press/v267/allouah25a.html'),
    ]
    reference_block=[Paragraph('References',heading)]
    for index,(label,url) in enumerate(references,1):reference_block.append(Paragraph(f'[{index}] <a href="{url}" color="#145670">{escape(label)}</a>',small))
    story.append(KeepTogether(reference_block))
    def page(canvas,doc):
        canvas.setAuthor('Anurag Lnu');canvas.setTitle('Measuring the Cost of Privacy and Robustness in Federated Learning - Working Report')
        canvas.setCreator('');canvas._doc.info.producer=''
        canvas.saveState();canvas.setFont('CapstoneSerif',7.6);canvas.setFillColor(colors.HexColor('#53616a'))
        canvas.drawString(.65*72,.34*72,f'MSCS Capstone | Working report | {report_date}')
        canvas.drawRightString(letter[0]-.65*72,.34*72,str(doc.page));canvas.restoreState()
    document=SimpleDocTemplate(str(output),invariant=1,pagesize=letter,leftMargin=.65*72,rightMargin=.65*72,topMargin=.62*72,bottomMargin=.65*72)
    document.build(story,onFirstPage=page,onLaterPages=page)
    reader=PdfReader(output);text='\n'.join(p.extract_text() or '' for p in reader.pages)
    for token in ['0.002266150','0.000704398','0.003827901','360','0.618250','0.627389']:
        if token not in text:raise ValueError('critical result missing from extracted PDF: '+token)
    if '\ufffd' in text or '\u25a0' in text:raise ValueError('unreadable replacement glyph in extracted PDF')
    pdf=pdfium.PdfDocument(str(output));rendered=[]
    for n in range(len(pdf)):
        path=pages/f'page-{n+1:02}.png';pdf[n].render(scale=1.4).to_pil().save(path);rendered.append(path.name)
    result={'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dependencies':{name:version(name) for name in ['reportlab','pypdf','pypdfium2']},'figure_sha256':{target:hashlib.sha256((source.parent/target).read_bytes()).hexdigest() for target in re.findall(r'!\[[^]]*\]\(([^)]+)\)',source_bytes.decode())},'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'pdf_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'pages':len(reader.pages),'text_characters':len(text),'rendered_pages':rendered,'report_date':report_date,'font_sha256':{file:hashlib.sha256((font_dir/file).read_bytes()).hexdigest() for _,file in FONT_FILES},'scope':'working historical-study manuscript; simulator workflow verified separately; disease follow-up, advisor requirements, final presentation/rehearsal and visual page acceptance remain separate'}
    (pages/'render-receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pages',type=Path,required=True)
    parser.add_argument('--font-dir',type=Path,required=True,help='Directory containing the four DejaVuSerif TTF files')
    parser.add_argument('--report-date',required=True,help='Explicit report date, YYYY-MM-DD')
    args=parser.parse_args();main(args.source,args.output,args.pages,args.font_dir,args.report_date)
