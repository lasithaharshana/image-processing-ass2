

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def load_input_image(image_path):
    if os.path.exists(image_path):
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
        else:
            print(f"Warning: Could not load image from {image_path}")
    else:
        print(f"Warning: Image file {image_path} not found")
    return None

def preprocess_image_for_otsu(image):
    # Resize if too large for processing
    if image.shape[0] > 500 or image.shape[1] > 500:
        scale = min(500/image.shape[0], 500/image.shape[1])
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
    print("Question 1: Otsu's Algorithm Implementation")
    print("="*50)
    
    # Load only input_image_q1.jpg for Question 1
    input_image_path = "inputs/input_image_q1.jpg"
    
    print(f"Loading {input_image_path} for Otsu's algorithm...")
    original_image = load_input_image(input_image_path)
    
    if original_image is None:
        print("❌ Failed to load input_image_q1.jpg. Please check that the file exists in the inputs/ folder.")
        return
    
    # Preprocess the image
    original_image = preprocess_image_for_otsu(original_image)
    print(f"✅ Successfully loaded {input_image_path} - Shape: {original_image.shape}")
    
    print(f"Original image shape: {original_image.shape}")
    print(f"Original image value range: [{np.min(original_image)}, {np.max(original_image)}]")
    unique_vals = np.unique(original_image)
    print(f"Number of unique pixel values: {len(unique_vals)}")
    
    # Add Gaussian noise to make the problem more challenging
    print("Adding Gaussian noise to the image...")
    noisy_image = add_gaussian_noise(original_image, mean=0, std=10)
    
    # Apply Otsu's algorithm
    print("Applying Otsu's algorithm...")
    optimal_threshold, max_variance = otsu_threshold(noisy_image)
    print(f"Optimal threshold found: {optimal_threshold}")
    print(f"Maximum between-class variance: {max_variance:.2f}")
    
    # Create binary image using optimal threshold
    binary_image = apply_threshold(noisy_image, optimal_threshold)
    
    # Compare with OpenCV's Otsu implementation
    cv_threshold, cv_binary = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OpenCV Otsu threshold: {cv_threshold}")
    
    # Calculate histogram for visualization
    hist = calculate_histogram(noisy_image)
    
    # Create visualization
    create_visualization(original_image, noisy_image, binary_image, cv_binary, 
                       hist, optimal_threshold, int(cv_threshold), 
                       max_variance, "Input Image Q1", input_image_path)

def create_visualization(original_image, noisy_image, binary_image, cv_binary, 
                        hist, optimal_threshold, cv_threshold, max_variance, 
                        image_name, image_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Otsu\'s Algorithm - {image_name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title(f'Original Image\n{os.path.basename(image_path)}')
    axes[0, 0].axis('off')
    
    # Noisy image
    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Image with Gaussian Noise')
    axes[0, 1].axis('off')
    
    # Histogram
    axes[0, 2].plot(hist)
    axes[0, 2].axvline(x=optimal_threshold, color='red', linestyle='--', 
                      label=f'Our Otsu: {optimal_threshold}')
    axes[0, 2].axvline(x=cv_threshold, color='blue', linestyle=':', 
                      label=f'OpenCV: {cv_threshold}')
    axes[0, 2].set_title('Histogram with Thresholds')
    axes[0, 2].set_xlabel('Pixel Intensity')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Our Otsu result
    axes[1, 0].imshow(binary_image, cmap='gray')
    axes[1, 0].set_title(f'Our Otsu Result\n(Threshold: {optimal_threshold})')
    axes[1, 0].axis('off')
    
    # OpenCV Otsu result
    axes[1, 1].imshow(cv_binary, cmap='gray')
    axes[1, 1].set_title(f'OpenCV Otsu Result\n(Threshold: {cv_threshold})')
    axes[1, 1].axis('off')
    
    # Difference between our result and OpenCV
    diff = np.abs(binary_image.astype(float) - cv_binary.astype(float))
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Difference Map\n(Red = Different pixels)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save to outputs folder
    output_filename = 'outputs/q1_otsu_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Results saved to {output_filename}")
    plt.close()  # Close the figure to free memory
    
    # Print some statistics
    print(f"\nStatistics for {image_name}:")
    print(f"Original image shape: {original_image.shape}")
    print(f"Original image unique values count: {len(np.unique(original_image))}")
    print(f"Noisy image value range: [{np.min(noisy_image)}, {np.max(noisy_image)}]")
    print(f"Pixels classified as foreground: {np.sum(binary_image == 255)} ({100*np.sum(binary_image == 255)/binary_image.size:.1f}%)")
    print(f"Pixels classified as background: {np.sum(binary_image == 0)} ({100*np.sum(binary_image == 0)/binary_image.size:.1f}%)")
    accuracy = 100 * (1 - np.sum(diff > 0) / diff.size)
    print(f"Accuracy compared to OpenCV: {accuracy:.2f}%")
    print(f"Maximum between-class variance: {max_variance:.2f}")

if __name__ == "__main__":
    main()
