import os
import torch
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from model_7.model import ConvAutoencoder7

def get_single_image_latent(image_path, model_path=None):
    """
    Get latent vector for a single image
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model file
    
    Returns:
        numpy.ndarray: 6-dimensional latent vector
    """
    # Set default model path if none provided
    if model_path is None:
        model_path = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_7/trained_model.pth"
    
    # Define the same image transformation used during training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((84, 84)),                  # Resize to 84x84
        transforms.ToTensor()                         # Convert to tensor
    ])
    
    # Load the trained model
    model = ConvAutoencoder7()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # Set to evaluation mode
    
    # Load and process the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, 1, 84, 84)
    
    # Extract latent vector (no gradient computation needed for inference)
    with torch.no_grad():
        # Pass through encoder to get compressed representation
        encoded = model.encoder(image_tensor)
        
        # Flatten the encoded features
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        # Pass through first fully connected layer with batch norm and dropout
        h = model.fc1_dropout(torch.relu(model.fc1_bn(model.fc1(encoded_flat))))
        
        # Get the final 6-dimensional latent vector
        latent_vector = model.fc2(h)
    
    # Remove batch dimension and convert to numpy array
    return latent_vector.squeeze().numpy()

def get_all_image_paths(data_dir):
    """
    Get all image paths from the dataset directory
    
    Args:
        data_dir (str): Path to the dataset directory
    
    Returns:
        list: List of tuples containing (image_path, class_label)
    """
    # Use ImageFolder to automatically find all images and their labels
    dataset = datasets.ImageFolder(data_dir)
    
    # Extract all image paths and their corresponding labels
    image_paths = []
    for i in range(len(dataset)):
        image_path, label = dataset.imgs[i]
        image_paths.append((image_path, label))
    
    return image_paths

if __name__ == "__main__":
    print("Starting latent vector extraction...")
 
    # Define paths
    data_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/Autoencoder_data/reconstructed_images_greyscale_reduced"
    model_path = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_7/trained_model.pth"
    output_dir = "/Users/javierfriedman/Desktop/Research/EncoderDecoder_pong_greyscale/model_7/"
    
    print("=" * 50)
    print("STARTING LATENT VECTOR EXTRACTION")
    print("=" * 50)
    
    # Get all image paths from the dataset
    print("info] loading images")

    dataset = datasets.ImageFolder(data_dir)

    # Initialize lists to store results
    all_latent_vectors = []
    all_labels = []
    all_image_paths = []
    
    for filename, label in dataset:
        image_path = os.path.join(data_dir, filename)
        print(f"Processing {filename} with label {label}")
        try:
            # Extract latent vector for this single image
            latent_vector = get_single_image_latent(image_path, model_path)
            
            # Store the results
            all_latent_vectors.append(latent_vector)
            all_labels.append(label)
            all_image_paths.append(image_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
  
    
    # Process each image one at a time
    print("\nProcessing images one by one...")
    print("-" * 50)
        
    
    # Convert lists to numpy arrays
    print("\nConverting results to numpy arrays...")
    all_latent_vectors = np.array(all_latent_vectors)
    all_labels = np.array(all_labels)
    
    print(f"Successfully processed {len(all_latent_vectors)} images")
    print(f"Latent vector shape: {all_latent_vectors.shape}")
    
    # Save the results in multiple formats
    print("\nSaving results...")
    
    # Save as numpy binary files (efficient for loading later)
    latent_save_path = os.path.join(output_dir, "latent_vectors.npy")
    labels_save_path = os.path.join(output_dir, "labels.npy")
    paths_save_path = os.path.join(output_dir, "image_paths.npy")
    
    np.save(latent_save_path, all_latent_vectors)
    np.save(labels_save_path, all_labels)
    np.save(paths_save_path, all_image_paths)
    
    # Save as human-readable text file
    text_save_path = os.path.join(output_dir, "latent_vectors.txt")
    with open(text_save_path, 'w') as f:
        f.write("# Latent vectors (6 dimensions per image)\n")
        f.write("# Format: [dim1, dim2, dim3, dim4, dim5, dim6]\n")
        f.write("#" + "=" * 50 + "\n")
        for i, latent in enumerate(all_latent_vectors):
            f.write(f"Image {i:4d}: [{latent[0]:8.6f}, {latent[1]:8.6f}, {latent[2]:8.6f}, "
                   f"{latent[3]:8.6f}, {latent[4]:8.6f}, {latent[5]:8.6f}]\n")
    
    print(f"✓ Latent vectors saved to: {latent_save_path}")
    print(f"✓ Labels saved to: {labels_save_path}")
    print(f"✓ Image paths saved to: {paths_save_path}")
    print(f"✓ Human-readable format saved to: {text_save_path}")
    
    # Display statistics about the latent space
    print("\n" + "=" * 50)
    print("LATENT SPACE STATISTICS")
    print("=" * 50)
    
    print(f"Dataset size: {all_latent_vectors.shape[0]} images")
    print(f"Latent dimension: {all_latent_vectors.shape[1]}")
    print(f"Unique classes: {len(np.unique(all_labels))}")
    
    # Statistics for each latent dimension
    print("\nPer-dimension statistics:")
    for dim in range(6):
        mean_val = np.mean(all_latent_vectors[:, dim])
        std_val = np.std(all_latent_vectors[:, dim])
        min_val = np.min(all_latent_vectors[:, dim])
        max_val = np.max(all_latent_vectors[:, dim])
        print(f"  Dimension {dim}: mean={mean_val:7.4f}, std={std_val:7.4f}, "
              f"range=[{min_val:7.4f}, {max_val:7.4f}]")
    
    # Show first few examples
    print(f"\nFirst 5 latent vectors (as examples):")
    for i in range(min(5, len(all_latent_vectors))):
        print(f"  Image {i}: {all_latent_vectors[i]}")
        print(f"            (from: {os.path.basename(all_image_paths[i])})")
    
    return all_latent_vectors, all_labels, all_image_paths

