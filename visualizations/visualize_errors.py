import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from models.mlp import MLPClassifier

data = torch.load("data/test_embeddings.pt")
X = data["embeddings"]
y_true = data["labels"]

model = MLPClassifier(embedding_dim=64)
model.load_state_dict(torch.load("models/mlp_classifier.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(X)
    y_pred = torch.argmax(logits, dim=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Closed", "Open"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: MLP on Test Set")
plt.grid(False)
plt.show()

X_np = X.numpy()
y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()

tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_np)

correct_idx = np.where(y_true_np == y_pred_np)[0]
incorrect_idx = np.where(y_true_np != y_pred_np)[0]

plt.figure(figsize=(10, 7))

correct_colors = np.array(['darkblue' if cls == 0 else 'deeppink' for cls in y_true_np[correct_idx]])

plt.scatter(X_2d[correct_idx, 0], X_2d[correct_idx, 1],
            c=correct_colors, alpha=0.3, label=None)
plt.scatter(X_2d[incorrect_idx, 0], X_2d[incorrect_idx, 1],
            c='black', marker='x', label="Misclassified")

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Correct (Closed)', markerfacecolor='darkblue', markersize=8, alpha=0.6),
    Line2D([0], [0], marker='o', color='w', label='Correct (Open)', markerfacecolor='lightpink', markersize=8, alpha=0.6),
    Line2D([0], [0], marker='x', color='black', label='Misclassified',
           linestyle='None', markersize=8, markeredgewidth=1.5)
]

plt.legend(handles=legend_elements)
plt.title("t-SNE of Test Embeddings: Correct vs. Misclassified")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)
plt.show()