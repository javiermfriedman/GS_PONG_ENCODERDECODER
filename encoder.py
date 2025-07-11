import os
from PIL import Image
import numpy as np

def read_reconstructed_images():
    """
    Read all images from the reconstructed_images_greyscale directory
    """
    # Path to the directory containing reconstructed images
    image_dir_path = "/Users/javierfriedman/Desktop/Research/reconstructed_images_greyscale"

    image_dir = os.listdir(image_dir_path)


    

    
    # Loop through each image file
    counter = 0
    for filename in image_dir:
        print(filename)
        if counter > 10:
            break
        file_path = os.path.join(image_dir_path, filename)

        print(file_path)
        
        try:
            # Open and read the image
            with Image.open(file_path) as img:

                # # Convert to numpy array
                img_array = np.array(img)
                print(img_array)
                
       
                # Here you can add your processing logic for each image
                # For example, you might want to:
                # - Resize the image
                # - Normalize pixel values
                # - Apply encoding/decoding operations
                # - Save processed images
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            break

        counter += 1

if __name__ == "__main__":
    read_reconstructed_images()
