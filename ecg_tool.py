from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
import torch
import torch.nn as nn
import numpy as np
from model import RecurrentAutoencoder
from typing import Optional
import json as jsonlib

class ECGAnomalyTool(BaseTool):
    name: str = "ecg_anomaly_detector"
    description: str = "Detects anomalies in ECG sequences using a trained autoencoder."

    model_path: str = "model.pth"
    seq_len: int = 140
    n_features: int = 1
    embedding_dim: int = 128
    device: Optional[torch.device] = None

    model: Optional[torch.nn.Module] = None
    criterion: Optional[torch.nn.Module] = None
    threshold: float = 0.05

    def __init__(self, **data):
        super().__init__(**data)
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = RecurrentAutoencoder(self.seq_len, self.n_features, self.embedding_dim)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Loss function
        self.criterion = nn.L1Loss(reduction="mean").to(self.device)

    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Synchronous run"""
        try:
            values = np.array([float(x) for x in tool_input.split(",")])
            seq_true = torch.tensor(values, dtype=torch.float32).reshape(
                1, self.seq_len, self.n_features
            ).to(self.device)

            with torch.no_grad():
                seq_pred = self.model(seq_true)
                loss = self.criterion(seq_pred, seq_true).item()

            is_normal = loss <= self.threshold
            result = {
                "input_row": tool_input,
                "loss": round(loss, 6),
                "result": "Normal" if is_normal else "Anomalous"
            }
            return jsonlib.dumps(result)

        except Exception as e:
            return jsonlib.dumps({"error": str(e)})

    async def _arun(self, tool_input: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async run not implemented"""
        raise NotImplementedError("Async not supported for this tool.")


if __name__ == "__main__":
    tool = ECGAnomalyTool()
    row = ",".join(str(x) for x in np.random.randn(140))
    print(tool._run(row))
