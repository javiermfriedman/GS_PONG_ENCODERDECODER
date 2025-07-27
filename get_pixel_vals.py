import cv2
import numpy as np

def display_and_save_image_pixels(image_path):
    """
    Reads a grayscale image, prints its pixel values, and saves the full
    array to a text file.

    Args:
        image_path (str): The full path to the image file.
    """
    try:
        # Read the image from the specified path in grayscale mode
        gray_image = cv2.imread(image_path, 0)

        if gray_image is None:
            print(f"Error: Unable to load image at '{image_path}'.")
            return

        # Define the output file path
        output_file = "pixel_values.txt"

        # --- SAVE TO FILE ---
        # Use numpy.savetxt to save the entire array to a text file.
        # We set a high threshold to prevent NumPy from abbreviating the array.
        np.set_printoptions(threshold=np.inf)
        
        with open(output_file, "w") as f:
            f.write(np.array2string(gray_image))
            
        print(f"\nSuccessfully saved all pixel values to '{output_file}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    path = input("Enter the full path to your image: ")
    display_and_save_image_pixels(path)