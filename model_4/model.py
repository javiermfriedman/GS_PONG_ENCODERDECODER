import torch

# -------- Autoencoder Definition --------
class ConvAutoencoder(torch.nn.Module):
    """
    Convolutional Autoencoder for grayscale images of size 84x84.
    Encoder compresses the image; decoder reconstructs it.
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder: progressively reduces spatial dimensions and increases channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 1x84x84 -> 64x84x84
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),  # 64x84x84 -> 64x42x42
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1), # 64x42x42 -> 32x42x42
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),  # 32x42x42 -> 32x21x21
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1), # 32x21x21 -> 16x21x21
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2)   # 16x21x21 -> 16x10x10
        )
        # Decoder: reconstructs the image using transposed convolutions
        self.decoder = torch.nn.Sequential(
            # Input: 16 x 10 x 10
            torch.nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2), # -> 32 x 21 x 21
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2), # -> 64 x 42 x 42
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),  # -> 1 x 84 x 84
            torch.nn.Sigmoid()  # Output in [0,1] for image
        )

    def forward(self, x):
        # Pass input through encoder then decoder
        return self.decoder(self.encoder(x))