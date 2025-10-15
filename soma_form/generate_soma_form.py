

import os
import json
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from openai import OpenAI

# ----------------------------
# Set your OpenAI API key
# ----------------------------
os.environ["OPENAI_API_KEY"] = 
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Load JSON data
# ----------------------------
def load_patient_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading patient data: {e}")
        return None

# ----------------------------
# Load SOMA template
# ----------------------------
def load_soma_template(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error loading SOMA template: {e}")
        return None

# ----------------------------
# Fill SOMA form using OpenAI
# ----------------------------
def fill_soma_form(patient_data, soma_template):
    print("⏳ Sending request to OpenAI API...\n")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": (
                     "You are a medical assistant. Fill SOMA forms accurately based on patient data. "
                     "Use ☑ for checked/selected options and ☐ for unchecked options. "
                     "Maintain exact formatting and structure of the template."
                 )},
                {"role": "user",
                 "content": f"PATIENT DATA:\n{json.dumps(patient_data, indent=2)}\n\n"
                            f"SOMA TEMPLATE:\n{soma_template}\n\nFill checkboxes and blanks accordingly."}
            ],
            temperature=0.2
        )
        filled_form = response.choices[0].message.content
        print("✅ Form filled successfully!\n")
        return filled_form
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return None

# ----------------------------
# Clean text for PDF
# ----------------------------
def clean_text_for_pdf(text):
    # Remove markdown and replace checkbox symbols
    text = text.replace("###", "")
    text = text.replace("**", "")
    text = text.replace("☑", "[X]")
    text = text.replace("☐", "[ ]")
    text = text.replace("■", "[ ]")
    text = text.replace("□", "[ ]")
    return text

# ----------------------------
# Generate PDF
# ----------------------------
def generate_pdf(filled_form, output_filename="filled_soma_form.pdf"):
    doc = SimpleDocTemplate(output_filename, pagesize=A4,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                 fontSize=18, textColor=colors.HexColor('#1a365d'),
                                 alignment=1, spaceAfter=20, fontName='Helvetica-Bold')
    
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                   fontSize=12, textColor=colors.HexColor('#2c5282'),
                                   spaceBefore=12, spaceAfter=8, fontName='Helvetica-Bold')
    
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
                                  fontSize=10, leading=14, fontName='Helvetica')
    
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
                                  fontSize=10, leading=14, leftIndent=20, fontName='Helvetica')
    
    content = []
    content.append(Paragraph("HEART SOMA MEDICAL FORM", title_style))
    content.append(Spacer(1, 0.2*inch))
    
    cleaned_form = clean_text_for_pdf(filled_form)
    lines = cleaned_form.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Sections
        if re.match(r'^#\s*\d+\.', line):
            section_title = re.sub(r'^#\s*\d+\.\s*', '', line)
            content.append(Paragraph(section_title, section_style))
            content.append(Spacer(1, 0.1*inch))
            i += 1
            continue
        
        # Tables
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
        
        # Bullet points
        if line.startswith('*') or line.startswith('-') or line.startswith('[X]') or line.startswith('[ ]'):
            content.append(Paragraph(line, bullet_style))
        else:
            content.append(Paragraph(line, normal_style))
        content.append(Spacer(1, 0.05*inch))
        i += 1
    
    try:
        doc.build(content)
        print(f"📄 PDF saved successfully as: {output_filename}")
    except Exception as e:
        print(f"❌ PDF Generation Error: {e}")

# ----------------------------
# Main execution
# ----------------------------
def main():
    patient_json_path = "sample_patient.json"
    soma_template_path = "soma_form_template.txt"
    output_pdf = "filled_soma_form.pdf"
    
    patient_data = load_patient_data(patient_json_path)
    if not patient_data:
        return
    
    soma_template = load_soma_template(soma_template_path)
    if not soma_template:
        return
    
    filled_form = fill_soma_form(patient_data, soma_template)
    if not filled_form:
        return
    
    # Preview in terminal
    print("\n" + "="*60)
    print("AI-FILLED SOMA FORM PREVIEW")
    print("="*60)
    print(filled_form)
    print("="*60 + "\n")
    
    generate_pdf(filled_form, output_pdf)

if __name__ == "__main__":
    main()
