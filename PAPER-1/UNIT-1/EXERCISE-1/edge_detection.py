
"""
Paper 1: Industrial Vision
Unit I: Computer Vision Fundamentals
Exercise 1: Sobel and Canny Edge Detection

Description:
    Processes industrial images using Sobel and Canny edge detection.
    Saves the results and a side-by-side comparison for each image.
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"

# Create the output directory if it does not exist.
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Sobel configuration.
SOBEL_KERNEL_SIZE = 3

# Canny lower and upper thresholds.
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Gaussian smoothing kernel.
GAUSSIAN_KERNEL_SIZE = (5, 5)


# --------------------------------------------------
# 2. SOBEL EDGE DETECTION
# --------------------------------------------------

def apply_sobel(gray_image):
    """
    Calculate Sobel gradients and return a normalized
    gradient-magnitude image.
    """

    # Calculate horizontal gradient.
    gradient_x = cv2.Sobel(
        gray_image,
        cv2.CV_32F,
        1,
        0,
        ksize=SOBEL_KERNEL_SIZE
    )

    # Calculate vertical gradient.
    gradient_y = cv2.Sobel(
        gray_image,
        cv2.CV_32F,
        0,
        1,
        ksize=SOBEL_KERNEL_SIZE
    )

    # Calculate the gradient magnitude.
    magnitude = cv2.magnitude(gradient_x, gradient_y)

    # Normalize the result to the 0-255 range.
    sobel_image = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return sobel_image.astype(np.uint8)


# --------------------------------------------------
# 3. CANNY EDGE DETECTION
# --------------------------------------------------

def apply_canny(gray_image):
    """
    Apply Canny edge detection to a grayscale image.
    """

    canny_image = cv2.Canny(
        gray_image,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD
    )

    return canny_image


# --------------------------------------------------
# 4. PROCESS A SINGLE IMAGE
# --------------------------------------------------

def process_image(image_path):
    """
    Load an image, perform preprocessing, apply both
    algorithms, and save the outputs.
    """

    print(f"\nProcessing: {image_path.name}")

    # Read the input image.
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Could not read {image_path.name}")
        return

    # Convert BGR to grayscale.
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise before edge detection.
    blurred_image = cv2.GaussianBlur(
        gray_image,
        GAUSSIAN_KERNEL_SIZE,
        0
    )

    # Apply Sobel.
    sobel_image = apply_sobel(blurred_image)

    # Apply Canny.
    canny_image = apply_canny(blurred_image)

    # Build output filenames.
    image_name = image_path.stem

    sobel_path = RESULT_DIR / f"{image_name}_sobel.png"
    canny_path = RESULT_DIR / f"{image_name}_canny.png"
    comparison_path = RESULT_DIR / f"{image_name}_comparison.png"

    # Save individual results.
    cv2.imwrite(str(sobel_path), sobel_image)
    cv2.imwrite(str(canny_path), canny_image)

    # Create a side-by-side comparison.
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gray_image, cmap="gray")
    axes[0].set_title("Original Grayscale Image")

    axes[1].imshow(sobel_image, cmap="gray")
    axes[1].set_title("Sobel Gradient Magnitude")

    axes[2].imshow(canny_image, cmap="gray")
    axes[2].set_title("Canny Edge Detection")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(image_path.name)
    figure.tight_layout()

    # Save the comparison figure.
    figure.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"Sobel output:      {sobel_path.name}")
    print(f"Canny output:      {canny_path.name}")
    print(f"Comparison image:  {comparison_path.name}")


# --------------------------------------------------
# 5. MAIN PROGRAM
# --------------------------------------------------

def main():
    """Process every supported image in the images folder."""

    supported_extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
    }

    if not IMAGE_DIR.exists():
        print(f"Error: Image folder not found: {IMAGE_DIR}")
        return

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    if not image_paths:
        print("No supported images found in the images folder.")
        print("Add PNG, JPG, JPEG, BMP, or TIFF images and try again.")
        return

    print("=" * 55)
    print("SOBEL AND CANNY EDGE DETECTION")
    print("=" * 55)
    print(f"Images found: {len(image_paths)}")

    for image_path in image_paths:
        process_image(image_path)

    print("\nProcessing completed.")
    print(f"Check your results here: {RESULT_DIR}")


if __name__ == "__main__":
    main()
