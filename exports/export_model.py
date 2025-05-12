import torch
import os
from models.autoencoder import Autoencoder
from models.joint_model import JointAutoencoderClassifier

# Inference wrapper
class RowPolicyInferenceWrapper(torch.nn.Module):
    def __init__(self, joint_model):
        super().__init__()
        self.joint_model = joint_model

    def forward(self, addr_window, stats_window):
        _, logits = self.joint_model(addr_window, stats_window)
        return torch.argmax(logits, dim=1)

MODEL_PATH = "models/joint_model.pt"
EXPORT_PATH = "models/row_policy_model.pt"
INPUT_DIM = 10 * 6
EMBEDDING_DIM = 32
HIDDEN_DIMS = (64, 32)

print("Loading trained model...")
ae = Autoencoder(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)
model = JointAutoencoderClassifier(ae, embedding_dim=EMBEDDING_DIM, hidden_dims=HIDDEN_DIMS)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

inference_model = RowPolicyInferenceWrapper(model).to("cpu")

example_addr = torch.randn(1, 10, 6)   # (B, 10, 6)
example_stats = torch.randn(1, 10, 4)  # (B, 10, 4)

# Trace and export
print("Tracing and exporting TorchScript model...")
traced_model = torch.jit.trace(inference_model, (example_addr, example_stats))
os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
traced_model.save(EXPORT_PATH)

print(f"TorchScript model saved to: {EXPORT_PATH}")