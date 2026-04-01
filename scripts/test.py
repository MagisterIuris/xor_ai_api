import torch
from ml.model import XORNet
from config import MODEL_PATH
from datasets import test_data

model = XORNet()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    predictions = model(test_data)
    predictions_bin = (predictions > 0.5).float()

print("🧪 XOR Model Predictions:")
for i in range(len(test_data)):
    print(f"Input : {test_data[i].tolist()} → Predicted : {predictions[i].item():.4f} → Class : {int(predictions_bin[i].item())}")
