🕹️ GS_PONG_ENCODERDECODER
This project explores various autoencoder architectures for learning compressed representations of grayscale 84×84 Atari Pong frames. The primary objective is to reconstruct these images with high fidelity, preserving critical visual features — namely, the paddles, score, and the 2×2 pixel ball — and then export the resulting latent vectors for downstream tasks such as clustering.

🎯 Problem Statement
While the majority of a Pong frame consists of uniform gray background, the essential information lies in sparse, high-contrast foreground elements. These include:

Two vertical paddles,

The numerical score at the top,

And most importantly, the ball — a tiny 2×2 pixel object.

Most architectures aggressively downsample the image, inadvertently discarding these crucial fine-grained features.

🧪 Model Overview and Results

🟦 Model 1 — Fully Connected Autoencoder
- Fully connected encoder-decoder with no spatial priors.
- Performs poorly on reconstruction due to lack of convolutional structure.
- Serves as a baseline reference only.

🟨 Model 2 — Basic Convolutional Autoencoder
- First convolutional attempt with stride-2 convolutions and large kernels.
- Encodes into a 6-dimensional latent space, resulting in extreme compression.
- Reconstruction quality is poor due to lack of normalization and over-aggressive downsampling.
Issue: Too much information is lost before the latent representation is formed.

🟩 Model 5 — Baseline Convolutional Autoencoder
- Symmetric encoder-decoder using max pooling and transposed convolutions.
- Latent vector size increased to 64 dimensions for more representational capacity.
- No dropout or normalization; simple but more effective than Model 2.
Issue: Still suffers from information loss due to excessive downsampling before bottleneck.

🟥 Model 6.2 — Regularized Autoencoder
- Adds BatchNorm for training stability and Dropout to regularize the latent space.
- Uses LeakyReLU activations and larger decoder kernels for smoother reconstruction.
- Latent dimension remains 64.
Issue: Although reconstruction improves, fine detail such as the ball is often lost due to spatial resolution bottlenecks.

🟪 Model 7 — Variational Autoencoder (VAE)
- Probabilistic encoder produces a latent distribution (mu, logvar) for sampling.
- Useful for generative tasks and modeling uncertainty.
- Implements a β-VAE loss to balance reconstruction and latent regularization.
Issue: The stochastic nature of the VAE makes it unsuitable for precise reconstruction of small features like the 2×2 pixel ball.

🟧 Model 9 — U-Net Style Autoencoder
- Skip connections enable sharp reconstructions and retain high-frequency information.
- Performs well visually, including on fine features.
Issue: Latent information bypasses the bottleneck via skip connections, making it ill-suited for downstream tasks that rely solely on the latent vector (e.g., clustering).

✅ Model 10 — Weighted Loss Autoencoder (Final Architecture)
Introduces a custom pixel-wise weighted loss that penalizes reconstruction errors more severely on foreground features (pixels > threshold).

Uses only one level of max pooling to retain as much spatial resolution as possible.

Demonstrates the best balance of reconstruction quality and usable latent representation.

Key Insight: Prior models failed to retain detail due to either architectural bottlenecks or unweighted loss functions. Model 10 overcomes this by directly addressing both problems.

extract latent.py