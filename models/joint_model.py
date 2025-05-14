import torch
import torch.nn as nn
from models.vae import VAE

class JointVAEClassifier(nn.Module):
    def __init__(self, vae: VAE, embedding_dim, hidden_dims=(64, 32), num_classes=2):
        super().__init__()
        self.vae = vae

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + 40, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, num_classes)
        )

    def forward(self, addr_window, stats_window):
        B = addr_window.size(0)

        x_flat = addr_window.view(B, -1)

        recon, mu, logvar = self.vae(x_flat)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_proj = self.projection(z)

        x_stats_flat = stats_window.view(B, -1)
        x = torch.cat([z_proj, x_stats_flat], dim=1)

        logits = self.classifier(x)
        return recon, mu, logvar, logits