import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.block(x) + x)

class VAE(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, min_logvar=-6.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.min_logvar = min_logvar

        self.enc_fc1 = nn.Linear(input_dim, 512)
        self.enc_ln1 = nn.LayerNorm(512)
        self.enc_act1 = nn.LeakyReLU(0.2)

        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_ln2 = nn.LayerNorm(256)
        self.enc_act2 = nn.LeakyReLU(0.2)

        self.res_enc = ResidualBlock(256)

        self.enc_fc3 = nn.Linear(256, 128)
        self.enc_ln3 = nn.LayerNorm(128)
        self.enc_act3 = nn.LeakyReLU(0.2)

        self.fc_mu = nn.Linear(128, embedding_dim)
        self.fc_logvar = nn.Linear(128, embedding_dim)

        self.dec_fc1 = nn.Linear(embedding_dim, 128)
        self.dec_act1 = nn.ReLU()

        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_act2 = nn.ReLU()

        self.output_layer = nn.Linear(256, input_dim)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=self.min_logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        h = self.enc_act1(self.enc_ln1(self.enc_fc1(x_flat)))
        h = self.enc_act2(self.enc_ln2(self.enc_fc2(h)))
        h = self.res_enc(h)
        h = self.enc_act3(self.enc_ln3(self.enc_fc3(h)))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        h = self.dec_act1(self.dec_fc1(z))
        h = self.dec_act2(self.dec_fc2(h))
        reconstruction = self.output_layer(h)

        return reconstruction, mu, logvar