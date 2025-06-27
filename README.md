# Image Processing Assignment 2

## Question 1: Otsu's Algorithm
**Consider an image with 2 objects and a total of 3-pixel values (1 for each object and one for the background). Add Gaussian noise to the image. Implement and test Otsu's algorithm with this image.**

### Code
- `q1_otsu_algorithm.py` - Implements Otsu's thresholding algorithm
- Uses `inputs/input_image_q1.jpg`
- Adds Gaussian noise (mean=0, std=10)
- Compares with OpenCV implementation

### Results
- Output: `outputs/q1_otsu_results.png`
- Shows original image, noisy image, histogram, and binary thresholding results
- Displays accuracy comparison with OpenCV

---

## Question 2: Region Growing Segmentation
**Implement a region-growing technique for image segmentation. The basic idea is to start from a set of points inside the object of interest (foreground), denoted as seeds, and recursively add neighboring pixels as long as they are in a pre-defined range of the pixel values of the seeds.**

### Code
- `q2_region_growing.py` - Implements region growing segmentation
- Uses `inputs/input_image_q2.jpg`
- Tests multiple threshold values (5, 15, 30)
- Uses 8-connected neighborhood and multiple seed points

### Results
- Output: `outputs/q2_region_growing_results.png`
- Shows original image with seeds, segmentation results for different thresholds
- Displays region statistics and coverage analysis

---

## Usage

```bash
# Run both questions
python main.py

# Run individual questions
python q1_otsu_algorithm.py
python q2_region_growing.py
```

## Requirements
```bash
pip install -r requirements.txt
```
