from langchain.tools import BaseTool
from typing import ClassVar, Optional, Type
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import os
import re

class SomaInput(BaseModel):
    patient_data: dict = Field(..., description="Patient JSON data")
    soma_template: str = Field(..., description="SOMA template text")

class FillSomaFormTool(BaseTool):
    name: ClassVar[str] = "fill_soma_form"
    description: ClassVar[str] = "Fill a SOMA medical form based on patient data and description"
    args_schema: ClassVar[Type[BaseModel]] = SomaInput
    def _run(self, patient_data: dict, soma_template: str) -> str:
        """
        Synchronous execution: calls OpenAI chat completion to fill the form.
        Returns filled form (string) or error message (string starting with 'Error:').
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."

        client = OpenAI(api_key=api_key)

        system_prompt = (
            "You are a medical assistant. Fill SOMA forms accurately based on patient data. "
            "Use ☑ for checked/selected options and ☐ for unchecked options. "
            "Maintain exact formatting and structure of the template. "
            "Do not add sections or omit any; only fill checkboxes and blanks."
        )

        user_content = (
            f"PATIENT DATA:\n{json.dumps(patient_data, indent=2, ensure_ascii=False)}\n\n"
            f"SOMA TEMPLATE:\n{soma_template}\n\n"
            "Fill checkboxes and blanks accordingly."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2
            )
            filled_form = response.choices[0].message.content

            # Defensive cleanup if model adds code fences
            if filled_form.strip().startswith("```"):
                filled_form = re.sub(r"^```(json|text)?\s*|\s*```$", "", filled_form.strip(), flags=re.MULTILINE)

            return filled_form
        except Exception as e:
            return f"Error: OpenAI API Error: {e}"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented")
