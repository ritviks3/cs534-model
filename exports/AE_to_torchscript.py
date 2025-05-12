import torch
from models.autoencoder import Autoencoder

input_dim = 100 * 6
embedding_dim = 64

model = Autoencoder(input_dim, embedding_dim)
model.load_state_dict(torch.load("models/autoencoder.pt"))
model.eval()

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.encoder

    def forward(self, x):
        return self.encoder(x)

wrapper = EncoderWrapper(model)
example = torch.randn(1, input_dim)
traced_encoder = torch.jit.trace(wrapper, example)
traced_encoder.save("models/encoder_traced.pt")