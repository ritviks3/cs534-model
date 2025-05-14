import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
from tqdm import tqdm
import glob

from preprocessing.dataset import DRAMBankWindowPreprocessor, CachedDRAMDataset
from models.vae import VAE
from models.joint_model import JointVAEClassifier

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WINDOW_SIZE = 10
FEATURE_DIM = 6
EMBEDDING_DIM = 32
ALPHA = 0.5
BETA = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILEPATHS = glob.glob("labeled_data_v4/**/*labeled_output.csv", recursive=True)
SAVE_DIR = "models"
MODEL_PATH = os.path.join(SAVE_DIR, "joint_model.pt")
MEAN_STD_PATH = os.path.join(SAVE_DIR, "addr_norm_stats.npz")

print("Processing dataset...")
preprocessor = DRAMBankWindowPreprocessor(FILEPATHS, window_size=10, append_from="labeled_data_v4/all.pt")
preprocessor.save("labeled_data_v4/all.pt")

print("Loading dataset...")
full_dataset = CachedDRAMDataset("labeled_data_v4/all.pt")

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
        addr_norm = ((addr - mean) / std).to(dtype=torch.float32)
        dataset.samples[i] = (
            addr_norm, stats_window, label
        )

print("Normalizing training and dev sets...")
normalize(full_dataset, addr_mean.view(1, 1, -1), addr_std.view(1, 1, -1))

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_labels = [full_dataset[i][2].item() for i in train_idx]
label_counts = Counter(train_labels)
count_0 = label_counts.get(0, 1)
count_1 = label_counts.get(1, 1)
print(count_0, count_1)
total = count_0 + count_1
w_neg = total / (2.0 * count_0)
w_pos = total / (2.0 * count_1)
weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(DEVICE)

print("Initializing model...")
input_dim = WINDOW_SIZE * FEATURE_DIM
vae = VAE(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)
model = JointVAEClassifier(vae, embedding_dim=EMBEDDING_DIM).to(DEVICE)
recon_loss_fn = nn.MSELoss()
clf_loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

def vae_loss_fn(recon_x, x_flat, mu, logvar, kl_weight):
    recon_loss = nn.MSELoss()(recon_x, x_flat)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_flat.size(0)
    return recon_loss + kl_weight * kl_div, recon_loss, kl_div

VAE_WARMUP_EPOCHS = 5

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_recon_loss = 0
    total_kl_loss = 0
    total_clf_loss = 0
    if epoch < 5:
      kl_weight = 0.0
    else:
        kl_weight = (epoch - 5) / 15

    if epoch == VAE_WARMUP_EPOCHS:
        print("→ Unfreezing classifier...")
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif epoch < VAE_WARMUP_EPOCHS:
        print(f"→ Epoch {epoch+1}: Freezing classifier")
        for param in model.classifier.parameters():
            param.requires_grad = False

    for addr_window, stats_window, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        addr_window = addr_window.to(DEVICE)
        stats_window = stats_window.to(DEVICE)
        label = label.to(DEVICE)

        recon, mu, logvar, logits = model(addr_window, stats_window)
        x_flat = addr_window.view(addr_window.size(0), -1)
        
        vae_loss, loss_recon, loss_kl = vae_loss_fn(recon, x_flat, mu, logvar, kl_weight)
        loss_clf = clf_loss_fn(logits, label)

        if epoch < VAE_WARMUP_EPOCHS:
            loss = ALPHA * vae_loss
        else:
            loss = ALPHA * vae_loss + BETA * loss_clf

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon_loss += loss_recon.item() * addr_window.size(0)
        total_kl_loss += loss_kl.item() * addr_window.size(0)
        if epoch >= VAE_WARMUP_EPOCHS:
            total_clf_loss += loss_clf.item() * addr_window.size(0)

    scheduler.step()

    avg_recon = total_recon_loss / len(train_dataset)
    avg_kl = total_kl_loss / len(train_dataset)
    avg_clf = total_clf_loss / len(train_dataset)
    print(f"Epoch {epoch+1}: Recon = {avg_recon:.4f}, KL = {avg_kl:.4f}, Clf = {avg_clf:.4f}")

print("\nFinetuning on val set...")
for param in model.vae.parameters():
    param.requires_grad = False

finetune_optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE / 5)

model.train()
finetune_epochs = 10
for epoch in range(finetune_epochs):
    total_clf_loss = 0
    all_preds, all_labels = [], []

    for addr_window, stats_window, label in val_loader:
        addr_window = addr_window.to(DEVICE)
        stats_window = stats_window.to(DEVICE)
        label = label.to(DEVICE)

        finetune_optimizer.zero_grad()

        recon, mu, logvar, logits = model(addr_window, stats_window)

        clf_loss = clf_loss_fn(logits, label)
        loss = clf_loss

        loss.backward()
        finetune_optimizer.step()

        total_clf_loss += loss_clf.item() * addr_window.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    avg_clf = total_clf_loss / len(val_dataset)
    val_accuracy = accuracy_score(all_labels, all_preds)

    print(f"[Fine-tune Epoch {epoch+1}] Clf: {avg_clf:.4f} | Acc: {val_accuracy:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Saved joint model to {MODEL_PATH}")