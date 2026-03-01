import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_lstm import DrivingTimeSeriesDataset
from model_lstm import HandlingLSTM
import numpy as np

# ============================================================
# SETTINGS — change these if needed
# ============================================================
DRY_CSV        = "data/dry_laps.csv"   # your dry training data
SEQUENCE_LEN   = 30                    # 30 frames = 0.5 seconds at 60fps
BATCH_SIZE     = 32
EPOCHS         = 30
LEARNING_RATE  = 0.001
TRAIN_SPLIT    = 0.8                   # 80% train, 20% validation
MODEL_SAVE     = "handling_lstm.pth"
# ============================================================

# === LOAD DATASET ===
print("Loading dry training data...")
dataset = DrivingTimeSeriesDataset(DRY_CSV, sequence_length=SEQUENCE_LEN)

# Split into train and validation
train_size = int(TRAIN_SPLIT * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain samples: {train_size} | Val samples: {val_size}")

# === HANDLE CLASS IMBALANCE ===
# Neutral will dominate the dataset (most frames are neutral driving)
# Weighted loss gives more penalty for getting understeer/oversteer wrong
all_labels = dataset.labels.numpy()
class_counts = np.bincount(all_labels)
class_weights = 1.0 / (class_counts + 1e-8)
class_weights = class_weights / class_weights.sum()
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"\nClass weights: Neutral={class_weights[0]:.3f}, Understeer={class_weights[1]:.3f}, Oversteer={class_weights[2]:.3f}")

# === MODEL ===
model     = HandlingLSTM(input_size=12, hidden_size=64, num_layers=2, output_size=3, dropout=0.3)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training for {EPOCHS} epochs...\n")

# === TRAINING LOOP ===
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --- Validation ---
    model.eval()
    val_loss = 0
    correct  = 0
    total    = 0

    # Track per-class accuracy
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]

    with torch.no_grad():
        for X_val, y_val in val_loader:
            outputs   = model(X_val)
            loss_val  = criterion(outputs, y_val)
            val_loss += loss_val.item()

            _, predicted = torch.max(outputs, 1)
            total   += y_val.size(0)
            correct += (predicted == y_val).sum().item()

            for c in range(3):
                mask = (y_val == c)
                class_correct[c] += (predicted[mask] == y_val[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    val_acc = correct / total

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE)

    # Print every epoch
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {total_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # Print per-class breakdown every 5 epochs
    if (epoch + 1) % 5 == 0:
        for i, name in enumerate(["Neutral", "Understeer", "Oversteer"]):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"   {name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        print()

print(f"\nBest validation accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {MODEL_SAVE}")