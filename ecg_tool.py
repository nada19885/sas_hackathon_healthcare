from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
import torch
import torch.nn as nn
import numpy as np
from model import RecurrentAutoencoder
from typing import Optional
import json as jsonlib
import matplotlib.pyplot as plt
import sys

import datetime

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

    def _run(self, tool_input: str, run_manager=None) -> str:
        """
        tool_input is expected to be a file path (e.g. 'ecg_input.csv')
        """
        try:
            # Read the file
            with open(tool_input, "r") as f:
                content = f.read().strip()

            # Assume the file contains one row of comma-separated values
            values = np.array([float(x) for x in content.split(",")])

            seq_true = torch.tensor(values, dtype=torch.float32).reshape(
                1, self.seq_len, self.n_features
            ).to(self.device)

            with torch.no_grad():
                seq_pred = self.model(seq_true)
                loss = self.criterion(seq_pred, seq_true).item()

            is_normal = loss <= self.threshold

            result = {
                "file": tool_input,
                "loss": round(loss, 6),
                "result": "Normal" if is_normal else "Anomalous"
            }

            if not is_normal:
                seq_true_np = seq_true.cpu().numpy().flatten()
                seq_pred_np = seq_pred.cpu().numpy().flatten()

                plt.figure(figsize=(8,4))
                plt.plot(seq_true_np, label="Original", color="blue")
                plt.plot(seq_pred_np, label="Reconstruction", color="red", linestyle="--")
                plt.title("ECG Anomaly Detected")
                plt.legend()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_file = f"anomaly_plot_{timestamp}.png"

                # Save to file
                plt.savefig(plot_file)
                plt.close()

                result["plot_file"] = plot_file

            return jsonlib.dumps(result)

        except Exception as e:
            return jsonlib.dumps({"error": str(e)})

    async def _arun(self, tool_input: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async run not implemented"""
        raise NotImplementedError("Async not supported for this tool.")


if __name__ == "__main__":
    tool = ECGAnomalyTool()
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(tool._run(filepath))
    else:
        print("Usage: python ecg_tool.py <path_to_ecg_file>")
