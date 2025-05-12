import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        """
        input_dim: 10 * 6  (flattened window)
        embedding_dim: size of the compressed representation
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        """
        x shape: (batch_size, 10, 6)
        returns: (reconstructed_x, embedding)
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        embedding = self.encoder(x_flat)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding