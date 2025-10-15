import torch
import torch.nn as nn
from model import RecurrentAutoencoder

# -------------------
# 1. Setup
# -------------------
seq_len = 140        # adjust to your ECG sequence length
n_features = 1       # ECG is usually 1D
embedding_dim = 128  # from your training setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate architecture
model = RecurrentAutoencoder(seq_len, n_features, embedding_dim)
state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("✅ Model loaded successfully!")

# -------------------
# 2. Prediction function
# -------------------
def classify_ecg_row(model, ecg_row, threshold, device):
    criterion = nn.L1Loss(reduction='mean').to(device)

    # Convert to tensor of shape (1, seq_len, n_features) to match model output
    seq_true = torch.tensor(ecg_row, dtype=torch.float32).reshape(1, seq_len, n_features).to(device)

    model.eval()
    with torch.no_grad():
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true).item()

    is_normal = loss <= threshold
    return is_normal, loss

# -------------------
# 3. Example usage
# -------------------
# Example: one dummy ECG row
dummy_ecg = torch.randn(seq_len).numpy()

THRESHOLD = 0.05

is_normal, loss = classify_ecg_row(model, dummy_ecg, THRESHOLD, device)
print(f"Loss: {loss:.4f} | Normal: {is_normal}")
