import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from preprocessing.dataset import MemoryAccessWindowDataset
from models.autoencoder import Autoencoder
from tqdm import tqdm

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WINDOW_SIZE = 100
FEATURE_DIM = 6
EMBEDDING_DIM = 64
DATA_PATH = "data/labeled_traces.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "models/autoencoder.pt"

print("Loading dataset...")
full_dataset = MemoryAccessWindowDataset(DATA_PATH, window_size=WINDOW_SIZE)
scaler = full_dataset.get_scaler

if scaler is not None:
    import numpy as np
    np.save("models/input_mean.npy", scaler.mean_)
    np.save("models/input_std.npy", scaler.scale_)
    print("✅ Saved input normalization stats to models/")

train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

input_dim = WINDOW_SIZE * FEATURE_DIM
model = Autoencoder(input_dim=input_dim, embedding_dim=EMBEDDING_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        x = x.to(DEVICE)
        x = x.view(x.size(0), -1)

        recon, _ = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(DEVICE)
            x = x.view(x.size(0), -1)
            recon, _ = model(x)
            loss = criterion(recon, x)
            val_loss += loss.item() * x.size(0)

    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")