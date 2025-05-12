import torch
import torch.nn as nn
from models.autoencoder import Autoencoder
from models.mlp import MLPClassifier

class AEMLPWrapper(nn.Module):
    def __init__(self, ae: Autoencoder, mlp: MLPClassifier, mean, std):
        super().__init__()
        self.encoder = ae.encoder
        self.mlp = mlp

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 1, -1))


    def forward(self, x):
        x = (x - self.mean) / self.std
        x = x.reshape(x.size(0), -1)
        embedding = self.encoder(x)
        output = self.mlp(embedding)
        return output
