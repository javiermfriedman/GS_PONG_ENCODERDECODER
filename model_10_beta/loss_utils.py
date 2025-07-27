# loss_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedPixelwiseMSE(nn.Module):
    """
    Weighted Mean Squared Error loss that emphasizes sparse foreground features
    in grayscale Pong images. Background pixels (value = 87) are assigned low weight,
    while foreground (value > 110) are assigned high weight.
    """
    def __init__(self, background_val=87, foreground_thresh=110, foreground_weight=5.0, background_weight=1.0):
        super(WeightedPixelwiseMSE, self).__init__()
        self.background_val = background_val
        self.foreground_thresh = foreground_thresh
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight

    def forward(self, recon, target):
        """
        recon: (B, 1, H, W) - reconstructed image from the autoencoder
        target: (B, 1, H, W) - original image
        """
        # Compute pixel-wise squared error
        pixel_loss = (recon - target) ** 2

        # Create a binary mask for foreground pixels
        # Foreground: target pixel > foreground_thresh
        foreground_mask = (target > self.foreground_thresh).float()

        # Assign weights: foreground gets higher weight, background gets lower
        weight_map = foreground_mask * self.foreground_weight + (1 - foreground_mask) * self.background_weight

        # Apply the weights to the loss
        weighted_loss = weight_map * pixel_loss

        # Average over all pixels and batch
        return weighted_loss.mean()
