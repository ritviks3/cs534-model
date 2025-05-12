import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import glob

from preprocessing.dataset import DRAMBankWindowPreprocessor, CachedDRAMDataset
from models.autoencoder import Autoencoder
from models.joint_model import JointAutoencoderClassifier

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WINDOW_SIZE = 10
FEATURE_DIM = 6
EMBEDDING_DIM = 32
ALPHA = 1.0
BETA = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILEPATHS = glob.glob("data_new/**/*.csv", recursive=True)
SAVE_DIR = "models"
MODEL_PATH = os.path.join(SAVE_DIR, "joint_model.pt")
MEAN_STD_PATH = os.path.join(SAVE_DIR, "addr_norm_stats.npz")

# print("Processing dataset...")
# preprocessor = DRAMBankWindowPreprocessor(FILEPATHS, window_size=WINDOW_SIZE)
# preprocessor.save("data_new/all.pt")

print("Loading dataset...")
full_dataset = CachedDRAMDataset("data_new/all.pt")

print("Splitting and normalizing address data...")
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)

addr_all = torch.stack([full_dataset[i][0] for i in train_idx])
addr_flat = addr_all.view(-1, FEATURE_DIM)
addr_mean = addr_flat.mean(dim=0)
addr_std = addr_flat.std(dim=0) + 1e-8

os.makedirs(SAVE_DIR, exist_ok=True)
np.savez(MEAN_STD_PATH, mean=addr_mean.cpu().numpy(), std=addr_std.cpu().numpy())
print(f"Saved normalization stats to {MEAN_STD_PATH}")

def normalize(dataset, mean, std):
    for i in range(len(dataset)):
        addr, stats_window, label = dataset[i]
        addr_norm = (addr - mean) / std
        dataset.samples[i] = (
            addr_norm.numpy(), stats_window, label
        )

print("Normalizing training and dev sets...")
normalize(full_dataset, addr_mean.view(1, 1, -1), addr_std.view(1, 1, -1))

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Initializing model...")
input_dim = WINDOW_SIZE * FEATURE_DIM
ae = Autoencoder(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)
model = JointAutoencoderClassifier(ae, embedding_dim=EMBEDDING_DIM).to(DEVICE)
recon_loss_fn = nn.MSELoss()
clf_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_recon_loss = 0
    total_clf_loss = 0

    for addr_window, stats_window, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        addr_window = addr_window.to(DEVICE)
        stats_window = stats_window.to(DEVICE)
        label = label.to(DEVICE)

        recon, logits = model(addr_window, stats_window)

        x_flat = addr_window.view(addr_window.size(0), -1)
        loss_recon = recon_loss_fn(recon, x_flat)
        loss_clf = clf_loss_fn(logits, label)
        loss = ALPHA * loss_recon + BETA * loss_clf

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon_loss += loss_recon.item() * addr_window.size(0)
        total_clf_loss += loss_clf.item() * addr_window.size(0)

    avg_recon = total_recon_loss / len(train_dataset)
    avg_clf = total_clf_loss / len(train_dataset)
    print(f"Epoch {epoch+1}: Recon Loss = {avg_recon:.4f}, Clf Loss = {avg_clf:.4f}")

print("\nEvaluating on dev set...")
model.eval()
val_recon_loss = 0
val_clf_loss = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for addr_window, stats_window, label in val_loader:
        addr_window = addr_window.to(DEVICE)
        stats_window = stats_window.to(DEVICE)
        label = label.to(DEVICE)

        recon, logits = model(addr_window, stats_window)
        
        x_flat = addr_window.view(addr_window.size(0), -1)
        loss_recon = recon_loss_fn(recon, x_flat)
        loss_clf = clf_loss_fn(logits, label)

        val_recon_loss += loss_recon.item() * addr_window.size(0)
        val_clf_loss += loss_clf.item() * addr_window.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

val_recon_loss /= len(val_dataset)
val_clf_loss /= len(val_dataset)
val_accuracy = accuracy_score(all_labels, all_preds)

print("Dev Set Results:")
print(f"  Reconstruction Loss: {val_recon_loss:.4f}")
print(f"  Classification Loss: {val_clf_loss:.4f}")
print(f"  Accuracy: {val_accuracy:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved joint model to {MODEL_PATH}")