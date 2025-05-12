import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.mlp import MLPClassifier

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
DATA_PATH = "data/embeddings.pt"
SAVE_PATH = "models/mlp_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading embeddings...")
data = torch.load(DATA_PATH)
X = data["embeddings"]
y = data["labels"]

dataset = TensorDataset(X, y)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

torch.save({
    "embeddings": test_set[:][0],
    "labels": test_set[:][1]
}, "data/test_embeddings.pt")
print("Saved test set to data/test_embeddings.pt")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

model = MLPClassifier(embedding_dim=EMBEDDING_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training MLP classifier...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / train_size

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = torch.argmax(model(X_batch), dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ MLP model saved to {SAVE_PATH}")