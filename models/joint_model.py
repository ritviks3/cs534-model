import torch
import torch.nn as nn
from models.autoencoder import Autoencoder

class JointAutoencoderClassifier(nn.Module):
    def __init__(self, ae: Autoencoder, embedding_dim, hidden_dims=(64, 32), num_classes=2):
        super().__init__()
        self.ae = ae
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + 40, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, addr_window, stats_window):
        B = addr_window.size(0)
        recon, embedding = self.ae(addr_window)
        x_stats_flat = stats_window.view(B, -1)
        x = torch.cat([embedding, x_stats_flat], dim=1)
        logits = self.classifier(x)
        return recon, logits