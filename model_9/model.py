import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
    A simple residual block: Conv3x3 -> ReLU -> Conv3x3 + identity skip.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class ConvAutoencoder(nn.Module):
    """
    U-Net–style convolutional autoencoder with residual blocks and skip connection.
    Produces both reconstruction and latent vector.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),  # now preserves 84x84
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)  # -> 64 x 42 x 42
        self.res_e1 = ResidualBlock(64)
        self.res_e2 = ResidualBlock(64)

        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc_enc  = nn.Linear(64 * 42 * 42, latent_dim)
        self.fc_dec  = nn.Linear(latent_dim, 64 * 42 * 42)

        # Decoder
        self.res_d1 = ResidualBlock(64)
        self.res_d2 = ResidualBlock(64)
        self.up     = nn.Upsample(scale_factor=2, mode='nearest')  # -> 64 x 84 x 84
        self.dec2   = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final  = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # 32 x 84 x 84
        e2 = self.enc2(e1)     # 64 x 84 x 84
        p  = self.pool(e2)     # 64 x 42 x 42
        r  = self.res_e1(p)
        r  = self.res_e2(r)

        # Bottleneck
        flat = self.flatten(r)
        z    = self.fc_enc(flat)
        d    = self.fc_dec(z)
        d    = d.view(-1, 64, 42, 42)

        # Decoder
        d  = self.res_d1(d)
        d  = self.res_d2(d)
        u  = self.up(d)         # 64 x 84 x 84
        u  = u + e2             # skip connection
        u  = self.dec2(u)       # 32 x 84 x 84
        out = torch.sigmoid(self.final(u))
        return out, z