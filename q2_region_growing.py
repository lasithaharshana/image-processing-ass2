

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

def load_input_image(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    return None

def process_single_image(image, image_title, image_path, seeds, thresholds):
    results = {}
    
    for threshold in thresholds:
        rg = RegionGrowing(image, threshold=threshold)
        region_stats = rg.multi_seed_region_grow(seeds)
        
        results[threshold] = {
            'segmented': rg.segmented.copy(),
            'stats': region_stats
        }
    
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
    
    output_filename = 'outputs/q2_region_growing_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

def demonstrate_region_growing():
    input_image_path = "inputs/input_image_q2.jpg"
    
    image = load_input_image(input_image_path)
    
    if image is not None:
        image_title = "Input Image Q2"
        if image.shape[0] > 400 or image.shape[1] > 400:
            scale = min(400/image.shape[0], 400/image.shape[1])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
    else:
        return
    
    thresholds = [5, 15, 30]
    
    h, w = image.shape
    seeds = [
        (h//4, w//4),
        (h//2, w//2),
        (3*h//4, w//4),
        (h//4, 3*w//4)
    ]
    
    process_single_image(image, image_title, input_image_path, seeds, thresholds)
    
def main():
    demonstrate_region_growing()

if __name__ == "__main__":
    main()
