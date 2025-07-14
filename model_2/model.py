import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (N, 1, 84, 84) - N is batch size, 1 is channels, 84x84 is image size
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # First convolutional layer: finds 16 patterns, reduces image to 42x42
            nn.ReLU(), # Activation function
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # Second convolutional layer: finds 32 patterns, reduces image to 21x21
            nn.ReLU(), # Activation function
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # Third convolutional layer: finds 64 patterns, reduces image to 11x11
            nn.ReLU(), # Activation function
            nn.Flatten(), # Flattens the image into a single vector
            nn.Linear(64 * 11 * 11, 6) # Fully connected layer to create the 6-dimensional latent vector
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(6, 64 * 11 * 11),
            nn.ReLU(),
            nn.Unflatten(1, (64, 11, 11)), # -> (N, 64, 11, 11)
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0), # -> (N, 32, 21, 21)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # -> (N, 16, 42, 42)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1), # -> (N, 1, 84, 84)
            nn.Sigmoid()
        )

    def forward(self, x):
        # Assuming x is flattened (N, 84*84)
        x = x.view(-1, 1, 84, 84)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 84 * 84) # Flatten the output
        return decoded
