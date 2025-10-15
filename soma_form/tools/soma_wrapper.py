# tools/soma_wrapper.py

from tools.fill_soma_form_tool import FillSomaFormTool
from tools.generate_pdf_tool import GeneratePdfTool
from generate_soma_form import load_patient_data, load_soma_template, clean_text_for_pdf


class SomaFormWrapper:
    """
    Wrapper that orchestrates the SOMA workflow:
    1. Fill the SOMA form using patient JSON + template.
    2. Clean the filled form text for PDF readability.
    3. Generate a styled PDF from the cleaned form, with optional anomaly info.
    """

    def __init__(self):
        self.fill_tool = FillSomaFormTool()
        self.pdf_tool = GeneratePdfTool()

    def process(self, patient_file: str, template_file: str, output_pdf: str, anomaly_info=None) -> str:
        """
        Run the full workflow:
        - Load patient data and template
        - Fill the form
        - Clean the text
        - Generate the PDF (with anomaly info if provided)

        Returns 'OK' if successful, or an error message string.
        """
        # Load inputs
        patient_data = load_patient_data(patient_file)
        soma_template = load_soma_template(template_file)

        # Step 1: Fill the form
        filled = self.fill_tool.run({
            "patient_data": patient_data,
            "soma_template": soma_template
        })
        if not filled or "Error:" in filled:
            return f"Error filling form: {filled}"

        # Step 2: Clean text for PDF
        cleaned = clean_text_for_pdf(filled)

        # Step 3: Generate PDF
        result = self.pdf_tool.run({
            "filled_form_text": cleaned,
            "output_filename": output_pdf,
            "anomaly_info": anomaly_info
        })
        return result


if __name__ == "__main__":
    # Example usage
    wrapper = SomaFormWrapper()

    anomaly_info = [
        {
            "file": "ecg_input1.csv",
            "loss": 0.0345,
            "result": "Normal",
            "plot_file": None
        },
        {
            "file": "ecg_input2.csv",
            "loss": 0.12,
            "result": "Anomalous",
            "plot_file": "img1.png"  # make sure this path exists
        }
    ]

    status = wrapper.process(
        patient_file="sample_patient.json",
        template_file="soma_form_template.txt",
        output_pdf="filled_soma_form.pdf",
        anomaly_info=anomaly_info
    )
    print(status)
