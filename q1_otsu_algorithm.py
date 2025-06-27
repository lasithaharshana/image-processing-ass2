import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def load_input_image(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    return None


def preprocess_image_for_otsu(image):
    if image.shape[0] > 500 or image.shape[1] > 500:
        scale = min(500 / image.shape[0], 500 / image.shape[1])
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image


def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def calculate_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    return hist


def otsu_threshold(image):
    hist = calculate_histogram(image)
    total_pixels = image.shape[0] * image.shape[1]
    max_variance = 0
    optimal_threshold = 0

    for threshold in range(1, 256):
        w0 = np.sum(hist[:threshold]) / total_pixels
        w1 = np.sum(hist[threshold:]) / total_pixels

        if w0 == 0 or w1 == 0:
            continue

        mu0 = np.sum([i * hist[i] for i in range(threshold)]) / (w0 * total_pixels)
        mu1 = np.sum([i * hist[i] for i in range(threshold, 256)]) / (w1 * total_pixels)
        between_class_variance = w0 * w1 * (mu0 - mu1) ** 2

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
    original_image = load_input_image(input_image_path)
    if original_image is None:
        return

    original_image = preprocess_image_for_otsu(original_image)
    noisy_image = add_gaussian_noise(original_image, mean=0, std=10)
    optimal_threshold, max_variance = otsu_threshold(noisy_image)
    binary_image = apply_threshold(noisy_image, optimal_threshold)
    cv_threshold, cv_binary = cv2.threshold(
        noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    hist = calculate_histogram(noisy_image)

    save_individual_images(
        original_image,
        noisy_image,
        binary_image,
        cv_binary,
        hist,
        optimal_threshold,
        int(cv_threshold),
        input_image_path,
    )


def save_individual_images(
    original_image,
    noisy_image,
    binary_image,
    cv_binary,
    hist,
    optimal_threshold,
    cv_threshold,
    image_path,
):

    plt.figure(figsize=(8, 6))
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Original Image")
    plt.axis("off")
    plt.savefig("outputs/q1_original.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(noisy_image, cmap="gray")
    plt.title("Noisy Image")
    plt.axis("off")
    plt.savefig("outputs/q1_noisy.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(binary_image, cmap="gray")
    plt.title(f"Otsu Result (T={optimal_threshold})")
    plt.axis("off")
    plt.savefig("outputs/q1_result.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar(range(256), hist, color='blue', alpha=0.7, width=1)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Otsu Threshold = {optimal_threshold}')
    plt.title("Image Histogram with Otsu Threshold")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/q1_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
