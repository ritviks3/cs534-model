import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import MemoryAccessWindowDataset
from models.autoencoder import Autoencoder
from tqdm import tqdm
import os

WINDOW_SIZE = 100
FEATURE_DIM = 6
EMBEDDING_DIM = 64
BATCH_SIZE = 64
DATA_PATH = "data/labeled_traces.csv"
MODEL_PATH = "models/autoencoder.pt"
OUTPUT_PATH = "data/embeddings.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MemoryAccessWindowDataset(DATA_PATH, window_size=WINDOW_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = WINDOW_SIZE * FEATURE_DIM
model = Autoencoder(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

all_embeddings = []
all_labels = []

with torch.no_grad():
    for x, y in tqdm(loader, desc="Extracting embeddings"):
        x = x.to(DEVICE)
        x = x.view(x.size(0), -1)
        _, embedding = model(x)
        all_embeddings.append(embedding.cpu())
        all_labels.append(y)

embeddings_tensor = torch.cat(all_embeddings, dim=0)
labels_tensor = torch.cat(all_labels, dim=0)

os.makedirs("data", exist_ok=True)
torch.save({
    "embeddings": embeddings_tensor,
    "labels": labels_tensor
}, OUTPUT_PATH)

print(f"Saved embeddings to {OUTPUT_PATH}")