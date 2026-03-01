import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_lstm import DrivingTimeSeriesDataset
from model_lstm import HandlingLSTM
import numpy as np

# ============================================================
# SETTINGS
# ============================================================
WET_CSV      = "data/wet_laps.csv"   # your wet test data
MODEL_PATH   = "handling_lstm.pth"   # trained on dry
SEQUENCE_LEN = 30
BATCH_SIZE   = 32
# ============================================================

print("Loading wet test data...")
wet_dataset = DrivingTimeSeriesDataset(WET_CSV, sequence_length=SEQUENCE_LEN)
wet_loader  = DataLoader(wet_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the model trained on dry
model = HandlingLSTM(input_size=12, hidden_size=64, num_layers=2, output_size=3)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print(f"Loaded model from {MODEL_PATH}")
print("Testing on wet condition data (model has never seen this)...\n")

criterion = nn.CrossEntropyLoss()

total    = 0
correct  = 0
test_loss = 0

class_correct = [0, 0, 0]
class_total   = [0, 0, 0]

# Confusion matrix — rows = actual, cols = predicted
confusion = np.zeros((3, 3), dtype=int)

with torch.no_grad():
    for X_batch, y_batch in wet_loader:
        outputs  = model(X_batch)
        loss     = criterion(outputs, y_batch)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total   += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        for t, p in zip(y_batch.numpy(), predicted.numpy()):
            confusion[t][p] += 1

        for c in range(3):
            mask = (y_batch == c)
            class_correct[c] += (predicted[mask] == y_batch[mask]).sum().item()
            class_total[c]   += mask.sum().item()

overall_acc = correct / total

print("=" * 50)
print(f"WET CONDITION RESULTS")
print("=" * 50)
print(f"Overall Accuracy : {overall_acc:.4f} ({overall_acc*100:.1f}%)")
print(f"Test Loss        : {test_loss:.4f}")
print()

label_names = ["Neutral", "Understeer", "Oversteer"]
for i, name in enumerate(label_names):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"{name:12s}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

print()
print("Confusion Matrix (rows=Actual, cols=Predicted)")
print(f"{'':15s} {'Neutral':>10s} {'Understeer':>12s} {'Oversteer':>10s}")
for i, name in enumerate(label_names):
    row = confusion[i]
    print(f"{name:15s} {row[0]:>10d} {row[1]:>12d} {row[2]:>10d}")