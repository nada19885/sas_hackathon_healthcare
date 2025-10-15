from langchain.tools import BaseTool
from typing import Optional, Type, ClassVar
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import re, os

class PdfInput(BaseModel):
    filled_form_text: str = Field(..., description="The filled SOMA form as text (already cleaned for PDF).")
    output_filename: str = Field(..., description="Output PDF filename, e.g., 'filled_soma_form.pdf'.")
    anomaly_info: list[dict] = Field(default=None, description="Optional list of anomaly metadata dicts with file, loss, result, and plot_file.")

class GeneratePdfTool(BaseTool):
    name: ClassVar[str] = "generate_pdf"
    description: ClassVar[str] = "Generate a styled PDF from a filled SOMA form text using ReportLab, with optional anomaly ECG section."
    args_schema: ClassVar[Optional[Type[BaseModel]]] = PdfInput

    def _run(self, filled_form_text: str, output_filename: str, anomaly_info: list[dict] = None) -> str:
        """
        Synchronous execution: builds a PDF from the filled_form_text.
        Returns 'OK' if successful, otherwise 'Error: ...'.
        """
        try:
            doc = SimpleDocTemplate(output_filename, pagesize=A4,
                                    leftMargin=0.75*inch, rightMargin=0.75*inch,
                                    topMargin=0.75*inch, bottomMargin=0.75*inch)

            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'Title', parent=styles['Heading1'],
                fontSize=18, textColor=colors.HexColor('#1a365d'),
                alignment=1, spaceAfter=20, fontName='Helvetica-Bold'
            )

            section_style = ParagraphStyle(
                'Section', parent=styles['Heading2'],
                fontSize=12, textColor=colors.HexColor('#2c5282'),
                spaceBefore=12, spaceAfter=8, fontName='Helvetica-Bold'
            )

            normal_style = ParagraphStyle(
                'Normal', parent=styles['Normal'],
                fontSize=10, leading=14, fontName='Helvetica'
            )

            bullet_style = ParagraphStyle(
                'Bullet', parent=styles['Normal'],
                fontSize=10, leading=14, leftIndent=20, fontName='Helvetica'
            )

            content = []
            content.append(Paragraph("HEART SOMA MEDICAL FORM", title_style))
            content.append(Spacer(1, 0.2*inch))

            # Render the filled form text
            lines = filled_form_text.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Section headers: lines like "# 1. Patient Information"
                if re.match(r'^#\s*\d+\.', line):
                    section_title = re.sub(r'^#\s*\d+\.\s*', '', line)
                    content.append(Paragraph(section_title, section_style))
                    content.append(Spacer(1, 0.1*inch))
                    i += 1
                    continue

                # Markdown-like tables (| col1 | col2 | ...)
                if '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
                    table_lines = []
                    while i < len(lines) and '|' in lines[i]:
                        if not re.match(r'^[\|\-\s:]+$', lines[i]):
                            table_lines.append(lines[i].strip())
                        i += 1

                    if table_lines:
                        table_data = []
                        for tline in table_lines:
                            cells = [cell.strip() for cell in re.split(r'\s*\|\s*', tline.strip('|'))]
                            if any(cells):
                                table_data.append(cells)

                        if table_data:
                            num_cols = len(table_data[0])
                            col_widths = [doc.width / num_cols] * num_cols
                            table = Table(table_data, colWidths=col_widths, hAlign='LEFT')
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 9),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                                ('TOPPADDING', (0, 0), (-1, 0), 10),
                                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ]))
                            content.append(table)
                            content.append(Spacer(1, 0.15*inch))
                    continue

                # Bullets and checkbox lines
                if line.startswith('*') or line.startswith('-') or line.startswith('[X]') or line.startswith('[ ]'):
                    content.append(Paragraph(line, bullet_style))
                else:
                    content.append(Paragraph(line, normal_style))

                content.append(Spacer(1, 0.05*inch))
                i += 1

            # --- Extra section: Anomaly ECG ---
            if anomaly_info:
                content.append(Spacer(1, 0.3*inch))
                content.append(Paragraph("Anomaly ECG Analysis", section_style))
                content.append(Spacer(1, 0.1*inch))

                for idx, anomaly in enumerate(anomaly_info, start=1):
                    file_name = anomaly.get("file", "N/A")
                    loss = anomaly.get("loss", "N/A")
                    result = anomaly.get("result", "N/A")
                    plot_path = anomaly.get("plot_file")

                    content.append(Paragraph(f"{idx}. Input file: {file_name}", normal_style))
                    content.append(Paragraph(f"   Reconstruction loss: {loss}", normal_style))
                    content.append(Paragraph(f"   Result: {result}", normal_style))
                    content.append(Spacer(1, 0.1*inch))

                    if plot_path and os.path.exists(plot_path):
                        img = Image(plot_path, width=5*inch, height=3*inch)
                        content.append(img)
                        content.append(Spacer(1, 0.2*inch))

            # Build the PDF
            doc.build(content)
            return "OK"
        except Exception as e:
            return f"Error: PDF Generation Error: {e}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented")
