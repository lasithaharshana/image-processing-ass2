

import sys
import os

def run_question_1():
    print("Running Question 1: Otsu's Algorithm Implementation")
    print("="*60)
    
    try:
        # Import and run question 1
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from q1_otsu_algorithm import main as q1_main
        q1_main()
        
        # Check if output file was created
        if os.path.exists('q1_otsu_results.png'):
            print("✅ Question 1 completed successfully!")
            print("   Generated: q1_otsu_results.png")
        else:
            print("⚠️  Question 1 ran but no output file found")
        
    except Exception as e:
        print(f"❌ Error running Question 1: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_question_2():
    print("\n\nRunning Question 2: Region Growing Implementation")
    print("="*60)
    
    try:
        # Import and run question 2
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from q2_region_growing import main as q2_main
        q2_main()
        
        # Check if output files were created
        output_files = ['q2_region_growing_results.png', 'q2_region_growing_analysis.png']
        created_files = []
        for file in output_files:
            if os.path.exists(file):
                created_files.append(file)
        
        if created_files:
            print("✅ Question 2 completed successfully!")
            for file in created_files:
                print(f"   Generated: {file}")
        else:
            print("⚠️  Question 2 ran but no output files found")
        
    except Exception as e:
        print(f"❌ Error running Question 2: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("Image Processing Assignment 2")
    print("="*40)
    print("Author: Student")
    print("Date: June 27, 2025")
    print("="*40)
    
    # Check if required packages are available
    try:
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        print("All required packages are available.")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        return
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created outputs directory")
    
    success_count = 0
    
    # Run Question 1
    if run_question_1():
        success_count += 1
    
    # Run Question 2
    if run_question_2():
        success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("ASSIGNMENT SUMMARY")
    print("="*60)
    print(f"Questions completed successfully: {success_count}/2")
    
    if success_count == 2:
        print("✅ All assignments completed successfully!")
        print("\nGenerated files in outputs/ folder:")
        print("- outputs/q1_otsu_results.png (Otsu's algorithm results)")
        print("- outputs/q2_region_growing_results.png (Region growing results)")
    else:
        print("❌ Some assignments failed. Please check the error messages above.")
    
    print("\nProgram descriptions:")
    print("1. Otsu's Algorithm (uses input_image_q1.jpg):")
    print("   - Loads input_image_q1.jpg")
    print("   - Adds Gaussian noise")
    print("   - Implements Otsu's thresholding from scratch")
    print("   - Compares with OpenCV implementation")
    
    print("\n2. Region Growing (uses input_image_q2.jpg):")
    print("   - Loads input_image_q2.jpg")
    print("   - Implements region growing segmentation")
    print("   - Supports multiple seed points")
    print("   - Uses 8-connected neighborhood")
    print("   - Configurable threshold parameter")
    print("   - Comprehensive analysis and visualization")

if __name__ == "__main__":
    main()
