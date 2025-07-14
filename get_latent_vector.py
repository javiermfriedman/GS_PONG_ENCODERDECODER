# """
# Script to extract and save latent vectors from images using a trained autoencoder.
# """
# import os
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import numpy as np
# from model import Autoencoder

# # Settings
# image_dir_path = "/Users/javierfriedman/Desktop/Research/Autoencoder_data"
# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_autoencoder.pth')
# output_latents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_latents.npy')
# batch_size = 32

# def get_latents():
#     # Transform: just ToTensor (assuming images are 84x84)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     dataset = datasets.ImageFolder(root=image_dir_path, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Load model
#     model = Autoencoder()
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # Collect all latent vectors and image paths
#     all_latents = []
#     all_image_paths = []

#     with torch.no_grad():
#         for batch_idx, (img, _) in enumerate(dataloader):
#             img = img.reshape(-1, 84*84)
#             # Forward pass only through encoder
#             latents = model.encoder(img)
#             all_latents.append(latents.numpy())
#             # Get file paths for this batch
#             start_idx = batch_idx * batch_size
#             end_idx = start_idx + img.shape[0]
#             batch_paths = [dataset.samples[i][0] for i in range(start_idx, end_idx)]
#             all_image_paths.extend(batch_paths)

#     # Stack all latent vectors
#     all_latents = np.vstack(all_latents)
#     # Save as npy file and also save image paths for reference
#     np.save(output_latents_path, all_latents)
#     with open(output_latents_path.replace('.npy', '_paths.txt'), 'w') as f:
#         for path in all_image_paths:
#             f.write(f"{path}\n")
#     print(f"✓ Saved latent vectors to {output_latents_path}")
#     print(f"✓ Saved image paths to {output_latents_path.replace('.npy', '_paths.txt')}")

# if __name__ == "__main__":
#     get_latents() 