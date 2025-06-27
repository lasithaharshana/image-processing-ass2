

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import os

class RegionGrowing:
    def __init__(self, image, threshold=10):
        self.image = image.copy()
        self.threshold = threshold
        self.height, self.width = image.shape
        self.segmented = np.zeros_like(image)
        self.visited = np.zeros((self.height, self.width), dtype=bool)
        
    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # Skip the center pixel
                    continue
                    
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def region_grow(self, seed_x, seed_y, region_label=255):
        if self.visited[seed_x, seed_y]:
            return 0
            
        # Initialize queue with seed point
        queue = deque([(seed_x, seed_y)])
        seed_value = self.image[seed_x, seed_y]
        region_pixels = 0
        
        while queue:
            x, y = queue.popleft()
            
            # Skip if already visited
            if self.visited[x, y]:
                continue
                
            # Check if pixel is within threshold
            if abs(int(self.image[x, y]) - int(seed_value)) <= self.threshold:
                # Add pixel to region
                self.visited[x, y] = True
                self.segmented[x, y] = region_label
                region_pixels += 1
                
                # Add neighbors to queue
                for nx, ny in self.get_neighbors(x, y):
                    if not self.visited[nx, ny]:
                        queue.append((nx, ny))
        
        return region_pixels
    
    def multi_seed_region_grow(self, seeds, region_labels=None):
        if region_labels is None:
            region_labels = [255 - i * 50 for i in range(len(seeds))]
        
        region_stats = {}
        
        for i, (seed_x, seed_y) in enumerate(seeds):
            label = region_labels[i] if i < len(region_labels) else 255
            pixels = self.region_grow(seed_x, seed_y, label)
            region_stats[f'Region {i+1}'] = {
                'seed': (seed_x, seed_y),
                'label': label,
                'pixels': pixels,
                'seed_value': self.image[seed_x, seed_y]
            }
            
        return region_stats

def create_test_image():
    # Create a synthetic image with different regions
    image = np.zeros((200, 200), dtype=np.uint8)
    
    # Background
    image[:, :] = 50
    
    # Add some regions with different intensities
    # Region 1: Rectangle
    image[30:80, 30:80] = 120
    
    # Region 2: Circle
    center_x, center_y = 140, 60
    radius = 25
    y, x = np.ogrid[:200, :200]
    mask1 = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask1] = 180
    
    # Region 3: Another rectangle
    image[120:170, 50:120] = 90
    
    # Region 4: Small circle
    center_x2, center_y2 = 150, 150
    radius2 = 20
    mask2 = (x - center_x2)**2 + (y - center_y2)**2 <= radius2**2
    image[mask2] = 200
    
    # Add some noise
    noise = np.random.normal(0, 5, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def load_input_image(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    return None

def interactive_seed_selection(image, title="Select Seeds"):
    seeds = []
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x = int(event.ydata)
            y = int(event.xdata)
            seeds.append((x, y))
            plt.plot(event.xdata, event.ydata, 'ro', markersize=8)
            plt.draw()
            print(f"Seed selected at ({x}, {y}), value: {image[x, y]}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{title}\nClick to select seed points, then close the window')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return seeds

def process_single_image(image, image_title, image_path, seeds, thresholds):
    # Test different threshold values
    results = {}
    
    for threshold in thresholds:
        print(f"\nTesting with threshold: {threshold}")
        
        # Create region growing instance
        rg = RegionGrowing(image, threshold=threshold)
        
        # Perform region growing
        region_stats = rg.multi_seed_region_grow(seeds)
        
        results[threshold] = {
            'segmented': rg.segmented.copy(),
            'stats': region_stats
        }
        
        # Print statistics
        total_pixels = 0
        for region_name, stats in region_stats.items():
            pixels = stats['pixels']
            seed_val = stats['seed_value']
            print(f"  {region_name}: {pixels} pixels, seed value: {seed_val}")
            total_pixels += pixels
        
        coverage = (total_pixels / (image.shape[0] * image.shape[1])) * 100
        print(f"  Total coverage: {coverage:.1f}%")
    
    # Create visualizations for this image
    create_region_growing_visualizations(image, image_title, image_path, results, seeds, thresholds)
    
    return results

def create_region_growing_visualizations(image, image_title, image_path, results, seeds, thresholds):
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(5 * (len(thresholds) + 1), 10))
    if len(thresholds) == 1:
        axes = axes.reshape(2, -1)
    
    fig.suptitle(f'Region Growing - {image_title}', fontsize=16)
    
    # Original image with seeds
    axes[0, 0].imshow(image, cmap='gray')
    for i, (x, y) in enumerate(seeds):
        axes[0, 0].plot(y, x, 'ro', markersize=8)
        axes[0, 0].text(y+5, x+5, f'S{i+1}', color='red', fontweight='bold')
    axes[0, 0].set_title(f'{image_title}\nwith Seed Points')
    axes[0, 0].axis('off')
    
    # Histogram
    axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('Image Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Results for each threshold
    for i, threshold in enumerate(thresholds):
        segmented = results[threshold]['segmented']
        stats = results[threshold]['stats']
        
        # Segmented image
        axes[0, i+1].imshow(segmented, cmap='viridis')
        axes[0, i+1].set_title(f'Segmented (T={threshold})')
        axes[0, i+1].axis('off')
        
        # Overlay on original
        axes[1, i+1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, i+1].imshow(segmented, cmap='jet', alpha=0.5)
        axes[1, i+1].set_title(f'Overlay (T={threshold})')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    
    # Save to outputs folder
    output_filename = 'outputs/q2_region_growing_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Results saved to {output_filename}")
    plt.close()

def demonstrate_region_growing():
    print("Question 2: Region Growing Implementation")
    print("="*50)
    
    # Load only input_image_q2.jpg for Question 2
    input_image_path = "inputs/input_image_q2.jpg"
    
    print(f"Loading {input_image_path} for region growing...")
    image = load_input_image(input_image_path)
    
    if image is not None:
        print(f"✅ Successfully loaded {input_image_path}")
        image_title = "Input Image Q2"
        
        # Resize if too large
        if image.shape[0] > 400 or image.shape[1] > 400:
            scale = min(400/image.shape[0], 400/image.shape[1])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"Resized to: {image.shape}")
    else:
        print(f"❌ Failed to load {input_image_path}")
        print("Creating synthetic test image as fallback...")
        image = create_test_image()
        image_title = "Synthetic Test Image"
        input_image_path = "synthetic"
    
    print(f"Image shape: {image.shape}")
    print(f"Image value range: [{np.min(image)}, {np.max(image)}]")
    
    # Define different threshold values to test
    thresholds = [5, 15, 30]
    
    # Pre-defined seed points for demonstration
    h, w = image.shape
    if image.shape == (200, 200):  # Synthetic image
        seeds = [(55, 55), (60, 140), (145, 85), (150, 150)]
    else:  # Real image - use strategic points
        seeds = [
            (h//4, w//4),      # Top-left quadrant
            (h//2, w//2),      # Center
            (3*h//4, w//4),    # Bottom-left quadrant
            (h//4, 3*w//4)     # Top-right quadrant
        ]
    
    print(f"Using {len(seeds)} seed points: {seeds}")
    
    # Process this image with different thresholds
    process_single_image(image, image_title, input_image_path, seeds, thresholds)
    
def main():
    demonstrate_region_growing()
    
    print("\nRegion Growing Analysis Complete!")
    print("Generated files in outputs/ folder:")
    print("- outputs/q2_region_growing_results.png")
    
    # Print final summary
    print("\nSummary:")
    print("Region growing successfully implemented with the following features:")
    print("- Uses input_image_q2.jpg")
    print("- 8-connected neighborhood consideration")
    print("- Multiple seed point support")
    print("- Configurable threshold parameter")
    print("- Comprehensive visualization")

if __name__ == "__main__":
    main()
