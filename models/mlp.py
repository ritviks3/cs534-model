import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dims=(128, 64), num_classes=2):
        super(MLPClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x):
        return self.model(x)