import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Label mapping — matches what the AC logger writes
LABEL_MAP = {"Neutral": 0, "Understeer": 1, "Oversteer": 2}

# Only physics-based features — no position, no track, no time
# These match your actual CSV column names exactly
FEATURES = [
    "YawRate",
    "LateralAccel",
    "LongitudinalAccel",
    "SteerAngle",
    "Speed",
    "WheelSlipFL",
    "WheelSlipFR",
    "WheelSlipRL",
    "WheelSlipRR",
    "Throttle",
    "Brake",
    "SlipDiff"
]

# 12 features total — update input_size in model and train to match

class DrivingTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, sequence_length=30):
        """
        csv_path: path to your log.csv
        sequence_length: number of frames per input window (30 frames = 0.5s at 60fps)
        """
        # CSV uses semicolon delimiter
        df = pd.read_csv(csv_path, delimiter=";")

        # Convert string labels to integers
        df["Label"] = df["Label"].map(LABEL_MAP)

        # Drop any rows where label didn't map (shouldn't happen but just in case)
        df = df.dropna(subset=["Label"])
        df["Label"] = df["Label"].astype(int)

        # Normalize features — zero mean, unit variance
        # This is important so yaw rate and wheel slip are on the same scale
        self.X = df[FEATURES].values.astype(np.float32)
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-8)

        self.y = df["Label"].values

        self.sequence_length = sequence_length

        # Sliding window — each sample is sequence_length consecutive frames
        # Label is taken from the last frame in the window (what is happening NOW)
        self.sequences = []
        self.labels = []

        for i in range(len(self.X) - sequence_length):
            seq = self.X[i:i + sequence_length]
            label = self.y[i + sequence_length - 1]
            self.sequences.append(seq)
            self.labels.append(label)

        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # Print class distribution so you can see if data is balanced
        unique, counts = np.unique(self.y, return_counts=True)
        label_names = {v: k for k, v in LABEL_MAP.items()}
        print("Class distribution:")
        for u, c in zip(unique, counts):
            print(f"  {label_names[u]}: {c} ({100*c/len(self.y):.1f}%)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]