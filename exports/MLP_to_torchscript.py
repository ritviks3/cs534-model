import torch
from models.mlp import MLPClassifier

model = MLPClassifier(embedding_dim=64)
model.load_state_dict(torch.load("models/mlp_classifier.pt"))
model.eval()

example = torch.randn(1, 64)
traced_mlp = torch.jit.trace(model, example)
traced_mlp.save("models/mlp_classifier_traced.pt")