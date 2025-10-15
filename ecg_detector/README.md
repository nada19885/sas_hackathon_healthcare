
# ECG Anomaly Detection Tool (LangChain Wrapper)

This project provides an ECG anomaly detection tool built on a recurrent autoencoder in PyTorch, wrapped as a [LangChain](https://www.langchain.com/) `BaseTool`.  
It can be used standalone or plugged into a LangChain agent.

---

## 📂 Project Structure

```
ecg_detector/
│── ecg_tool.py        # LangChain tool wrapper
│── model.py           # Model architecture (encoder/decoder/autoencoder)
│── model.pth          # Trained model weights
│── requirements.txt   # Python dependencies
│── README.md          # This file
```

---

## ⚙️ Setup

1. **Clone or copy** the project folder.
2. Create and activate a virtual environment (Python 3.11 recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Standalone test
Run the tool directly to test with a dummy ECG row:
```bash
python ecg_tool.py
```

Example output:
```json
{
  "input_row": "0.12,-0.45,0.33,...,0.87",
  "loss": 0.052341,
  "result": "Normal"
}
```

### As a LangChain Tool
You can import and register the tool in a LangChain agent:

```python
from ecg_tool import ECGAnomalyTool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

# Initialize tool
ecg_tool = ECGAnomalyTool()

# Register with an agent
llm = ChatOpenAI()
agent = initialize_agent([ecg_tool], llm, agent="zero-shot-react-description", verbose=True)

# Query in natural language
agent.run("Check if this ECG is normal: 0.1,0.2,0.3,...")
```

---

## 📌 Notes
- `model.pth` must be present in the project folder.  
- The tool expects **140 comma‑separated values** per ECG row.  
- Threshold for anomaly detection is currently set to `0.05` (tune as needed).  
- Output is returned as JSON for easy parsing.

