import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import re, os
import sys

def build_docx():
    print("Building research_paper.docx...")
    doc = Document()

    # Set standard 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    def set_cell_shading(cell, color_hex):
        shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
        cell._tc.get_or_add_tcPr().append(shading_elm)

    # Set document font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    with open('research_paper/research_paper.md', 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        if line.startswith('# '):
            p = doc.add_paragraph()
            run = p.add_run(line[2:].strip())
            run.bold = True
            run.font.size = Pt(18)
            run.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            i += 1
        elif line.startswith('## '):
            p = doc.add_paragraph()
            run = p.add_run(line[3:].strip())
            run.bold = True
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            i += 1
        elif line.startswith('### '):
            p = doc.add_paragraph()
            run = p.add_run(line[4:].strip())
            run.bold = True
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0x2b, 0x5c, 0x8f)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(2)
            i += 1
        elif line.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if m:
                caption_text = m.group(1)
                img_path = m.group(2)
                if img_path.startswith('file:///'):
                    img_path = img_path.replace('file:///', '')
                img_path = os.path.normpath(img_path)
                if os.path.exists(img_path):
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before = Pt(8)
                    p.paragraph_format.space_after = Pt(2)
                    run = p.add_run()
                    run.add_picture(img_path, width=Inches(6.2))
            i += 1
        elif line.startswith('|') and '|' in line[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            rows_data = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.split('|')[1:-1]]
                if cells and not all(re.match(r'^:?-+:?$', c) for c in cells):
                    rows_data.append(cells)
            
            if rows_data:
                num_rows = len(rows_data)
                num_cols = max(len(r) for r in rows_data)
                table = doc.add_table(rows=num_rows, cols=num_cols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.autofit = True
                
                for r_idx, row in enumerate(rows_data):
                    for c_idx, val in enumerate(row):
                        if c_idx < num_cols:
                            cell = table.cell(r_idx, c_idx)
                            clean_val = val.replace('**', '').replace('$', '')
                            cell.text = clean_val
                            p = cell.paragraphs[0]
                            p.paragraph_format.space_after = Pt(2)
                            p.paragraph_format.space_before = Pt(2)
                            if r_idx == 0:
                                set_cell_shading(cell, '1a365d')
                                for r in p.runs:
                                    r.font.bold = True
                                    r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
                                    r.font.size = Pt(9.5)
                            else:
                                for r in p.runs:
                                    r.font.size = Pt(9)
                                if r_idx % 2 == 1:
                                    set_cell_shading(cell, 'f7fafc')
                doc.add_paragraph()
        else:
            clean_line = line
            is_bullet = False
            if clean_line.startswith('- ') or clean_line.startswith('* '):
                is_bullet = True
                clean_line = clean_line[2:].strip()
            elif re.match(r'^\d+\.\s', clean_line):
                is_bullet = True
                clean_line = re.sub(r'^\d+\.\s', '', clean_line).strip()
                
            p = doc.add_paragraph(style='List Bullet' if is_bullet else 'Normal')
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            
            clean_line = clean_line.replace('$', '')
            parts = re.split(r'(\*\*.*?\*\*)', clean_line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2])
                    r.bold = True
                else:
                    p.add_run(part)
            i += 1

    out_docx = 'research_paper/research_paper.docx'
    doc.save(out_docx)
    print(f"DOCX successfully generated: {out_docx}")

def build_pdf():
    print("Building research_paper.pdf with ReportLab...")
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Register Devanagari capable TTF font if available
    mangal_font_path = "C:\\Windows\\Fonts\\mangal.ttf"
    mangal_bold_path = "C:\\Windows\\Fonts\\mangalb.ttf"
    arial_font_path = "C:\\Windows\\Fonts\\arial.ttf"
    arial_bold_path = "C:\\Windows\\Fonts\\arialbd.ttf"
    
    body_font = "Helvetica"
    body_bold = "Helvetica-Bold"
    
    if os.path.exists(mangal_font_path):
        try:
            pdfmetrics.registerFont(TTFont('Mangal', mangal_font_path))
            if os.path.exists(mangal_bold_path):
                pdfmetrics.registerFont(TTFont('Mangal-Bold', mangal_bold_path))
            body_font = 'Mangal'
            body_bold = 'Mangal-Bold' if os.path.exists(mangal_bold_path) else 'Mangal'
            print("Registered Mangal font for Devanagari support in PDF.")
        except Exception as e:
            print(f"Could not register Mangal: {e}")

    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_number(num_pages)
                super().showPage()
            super().save()

        def draw_page_number(self, page_count):
            self.saveState()
            self.setFont("Helvetica", 9)
            self.setFillColor(colors.HexColor('#666666'))
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * inch - 0.75 * inch, 0.45 * inch, page_text)
            header_text = "Beyond Tokenization: Devanagari Hindi Model Behaviour & Explainability"
            self.drawString(0.75 * inch, 11 * inch - 0.45 * inch, header_text)
            self.setStrokeColor(colors.HexColor('#cccccc'))
            self.setLineWidth(0.5)
            self.line(0.75 * inch, 11 * inch - 0.5 * inch, 8.5 * inch - 0.75 * inch, 11 * inch - 0.5 * inch)
            self.line(0.75 * inch, 0.6 * inch, 8.5 * inch - 0.75 * inch, 0.6 * inch)
            self.restoreState()

    pdf_filename = "research_paper/research_paper.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        fontName=body_bold,
        fontSize=17,
        leading=22,
        alignment=1, # Center
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'DocH1',
        fontName=body_bold,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'DocH2',
        fontName=body_bold,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2b5c8f'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'DocBody',
        fontName=body_font,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        fontName=body_font,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        leftIndent=15,
        spaceAfter=3
    )
    table_cell_style = ParagraphStyle(
        'DocTableCell',
        fontName=body_font,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#222222')
    )
    table_header_style = ParagraphStyle(
        'DocTableHeader',
        fontName=body_bold,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )
    caption_style = ParagraphStyle(
        'DocCaption',
        fontName=body_font,
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.HexColor('#444444'),
        spaceBefore=4,
        spaceAfter=8
    )

    with open('research_paper/research_paper.md', 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    story = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith('# '):
            story.append(Paragraph(line[2:].strip(), title_style))
            story.append(Spacer(1, 4))
            i += 1
        elif line.startswith('## '):
            story.append(Paragraph(line[3:].strip(), h1_style))
            i += 1
        elif line.startswith('### '):
            story.append(Paragraph(line[4:].strip(), h2_style))
            i += 1
        elif line.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if m:
                caption_text = m.group(1)
                img_path = m.group(2)
                if img_path.startswith('file:///'):
                    img_path = img_path.replace('file:///', '')
                img_path = os.path.normpath(img_path)
                if os.path.exists(img_path):
                    # Aspect ratio preserving scaling to max width 6.8 inch
                    story.append(Spacer(1, 6))
                    story.append(RLImage(img_path, width=6.5 * inch, height=2.8 * inch))
                    story.append(Paragraph(caption_text, caption_style))
            i += 1
        elif line.startswith('|') and '|' in line[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            rows_data = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.split('|')[1:-1]]
                if cells and not all(re.match(r'^:?-+:?$', c) for c in cells):
                    rows_data.append(cells)
            
            if rows_data:
                num_cols = max(len(r) for r in rows_data)
                col_width = (7.0 * inch) / num_cols
                
                table_flowable_data = []
                for r_idx, row in enumerate(rows_data):
                    row_cells = []
                    for c_idx in range(num_cols):
                        val = row[c_idx] if c_idx < len(row) else ""
                        val = val.replace('**', '').replace('$', '')
                        st = table_header_style if r_idx == 0 else table_cell_style
                        row_cells.append(Paragraph(val, st))
                    table_flowable_data.append(row_cells)
                
                t = Table(table_flowable_data, colWidths=[col_width]*num_cols)
                t_style = [
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a365d')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('RIGHTPADDING', (0,0), (-1,-1), 2),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0'))
                ]
                for r_idx in range(1, len(rows_data)):
                    if r_idx % 2 == 1:
                        t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor('#f8fafc')))
                t.setStyle(TableStyle(t_style))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
        else:
            clean_line = line
            is_bullet = False
            if clean_line.startswith('- ') or clean_line.startswith('* '):
                is_bullet = True
                clean_line = clean_line[2:].strip()
            elif re.match(r'^\d+\.\s', clean_line):
                is_bullet = True
                clean_line = re.sub(r'^\d+\.\s', '', clean_line).strip()
            
            clean_line = clean_line.replace('$', '')
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            clean_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_line)
            
            # HTML entity encode ampersands not part of tags
            clean_line = clean_line.replace('& ', '&amp; ')
            
            try:
                story.append(Paragraph(clean_line, bullet_style if is_bullet else body_style))
            except Exception as e:
                # Fallback to plain text if XML tag error occurs
                plain = re.sub(r'<.*?>', '', clean_line)
                story.append(Paragraph(plain, bullet_style if is_bullet else body_style))
            i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {pdf_filename}")

if __name__ == '__main__':
    build_docx()
    build_pdf()
