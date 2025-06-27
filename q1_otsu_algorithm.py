import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import os


def load_input_image(image_path):
    if os.path.exists(image_path):
        # Load color image first
        color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_image is not None:
            # Convert BGR to RGB for matplotlib display
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # Convert to grayscale for processing
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            return color_image_rgb, gray_image
    return None, None


def preprocess_image_for_otsu(image):
    # Resize if too large for processing
    if image.shape[0] > 500 or image.shape[1] > 500:
        scale = min(500 / image.shape[0], 500 / image.shape[1])
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))

    return image


def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    # Clip values to valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def calculate_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    return hist


def otsu_threshold(image):
    # Calculate histogram
    hist = calculate_histogram(image)

    # Total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Initialize variables
    max_variance = 0
    optimal_threshold = 0

    # Try all possible threshold values
    for threshold in range(1, 256):
        # Calculate weights
        w0 = np.sum(hist[:threshold]) / total_pixels  # Background weight
        w1 = np.sum(hist[threshold:]) / total_pixels  # Foreground weight

        # Skip if one of the weights is zero
        if w0 == 0 or w1 == 0:
            continue

        # Calculate means
        mu0 = np.sum([i * hist[i] for i in range(threshold)]) / (w0 * total_pixels)
        mu1 = np.sum([i * hist[i] for i in range(threshold, 256)]) / (w1 * total_pixels)

        # Calculate between-class variance
        between_class_variance = w0 * w1 * (mu0 - mu1) ** 2

        # Update optimal threshold if current variance is maximum
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold

    return optimal_threshold, max_variance


def apply_threshold(image, threshold):
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold] = 255
    return binary_image


def main():
    input_image_path = "inputs/input_image_q1.jpg"

    color_image, gray_image = load_input_image(input_image_path)

    if color_image is None or gray_image is None:
        return

    # Preprocess the grayscale image for Otsu's algorithm
    processed_gray = preprocess_image_for_otsu(gray_image)

    noisy_image = add_gaussian_noise(processed_gray, mean=0, std=10)

    optimal_threshold, max_variance = otsu_threshold(noisy_image)

    binary_image = apply_threshold(noisy_image, optimal_threshold)

    cv_threshold, cv_binary = cv2.threshold(
        noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    hist = calculate_histogram(noisy_image)

    create_visualization(
        color_image,
        processed_gray,
        noisy_image,
        binary_image,
        cv_binary,
        hist,
        optimal_threshold,
        int(cv_threshold),
        max_variance,
        "Input Image Q1",
        input_image_path,
    )


def create_visualization(
    color_image,
    gray_image,
    noisy_image,
    binary_image,
    cv_binary,
    hist,
    optimal_threshold,
    cv_threshold,
    max_variance,
    image_name,
    image_path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Otsu's Algorithm - {image_name}", fontsize=16)

    # Original color image
    axes[0, 0].imshow(color_image)
    axes[0, 0].set_title(f"Original Color Image\n{os.path.basename(image_path)}")
    axes[0, 0].axis("off")

    # Grayscale version
    axes[0, 1].imshow(gray_image, cmap="gray")
    axes[0, 1].set_title("Grayscale Version")
    axes[0, 1].axis("off")

    # Noisy grayscale image
    axes[0, 2].imshow(noisy_image, cmap="gray")
    axes[0, 2].set_title("Grayscale with Gaussian Noise")
    axes[0, 2].axis("off")

    # Histogram
    axes[1, 0].plot(hist)
    axes[1, 0].axvline(
        x=optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Our Otsu: {optimal_threshold}",
    )
    axes[1, 0].axvline(
        x=cv_threshold, color="blue", linestyle=":", label=f"OpenCV: {cv_threshold}"
    )
    axes[1, 0].set_title("Histogram with Thresholds")
    axes[1, 0].set_xlabel("Pixel Intensity")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Our Otsu result
    axes[1, 1].imshow(binary_image, cmap="gray")
    axes[1, 1].set_title(f"Our Otsu Result\n(Threshold: {optimal_threshold})")
    axes[1, 1].axis("off")

    # OpenCV Otsu result
    axes[1, 2].imshow(cv_binary, cmap="gray")
    axes[1, 2].set_title(f"OpenCV Otsu Result\n(Threshold: {cv_threshold})")
    axes[1, 2].axis("off")

    plt.tight_layout()

    output_filename = "outputs/q1_otsu_results.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
