import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from models.autoencoder import Autoencoder
from models.mlp import MLPClassifier
from models.end_to_end import AEMLPWrapper

WINDOW_SIZE = 100
FEATURE_DIM = 6
EMBEDDING_DIM = 64
input_dim = WINDOW_SIZE * FEATURE_DIM

mean = np.load("models/input_mean.npy")
std = np.load("models/input_std.npy")

ae = Autoencoder(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)
ae.load_state_dict(torch.load("models/autoencoder.pt"))
ae.eval()

mlp = MLPClassifier(embedding_dim=EMBEDDING_DIM)
mlp.load_state_dict(torch.load("models/mlp_classifier.pt"))
mlp.eval()

model = AEMLPWrapper(ae, mlp, mean, std)
model.eval()

# Trace with dummy input
example_input = torch.randn(1, WINDOW_SIZE, FEATURE_DIM)  # (B, 100, 6)
traced = torch.jit.trace(model, example_input)

# Save
torch.jit.save(traced, "models/ae_mlp_end_to_end.pt")
print("✅ Exported TorchScript model with built-in normalization to models/end_to_end.pt")