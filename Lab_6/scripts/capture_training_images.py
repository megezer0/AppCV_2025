#!/usr/bin/env python3
# lab_6/scripts/capture_training_images.py

import os
import cv2
import time
import argparse
from datetime import datetime
from pathlib import Path

def capture_images(num_images=5, delay=0.5):
    """Capture training images with specified count and delay"""
    
    # Set up directory structure
    base_dir = Path("lab_6").resolve()
    images_dir = base_dir / "captured_images"
    
    # Create directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next available image number
    existing_files = list(images_dir.glob("image_*.jpg"))
    if existing_files:
        numbers = []
        for file in existing_files:
            try:
                num_str = file.stem.split('_')[1]
                numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue
        start_count = max(numbers) + 1 if numbers else 1
    else:
        start_count = 1
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if camera is working
    ret, test_frame = camera.read()
    if not ret:
        print("ERROR: Could not access camera")
        camera.release()
        return
    
    print(f"Starting image capture...")
    print(f"Will capture {num_images} images with {delay}s delay")
    print(f"Saving to: {images_dir}")
    print(f"Starting from image_{start_count:03d}")
    print("\nGet ready! Starting in 3 seconds...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Capturing!")
    
    # Capture images
    for i in range(num_images):
        ret, frame = camera.read()
        if ret:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{start_count + i:03d}_{timestamp}.jpg"
            filepath = images_dir / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), frame)
            if success:
                print(f"✓ Captured image {i+1}/{num_images}: {filename}")
            else:
                print(f"✗ Failed to save image {i+1}/{num_images}")
        else:
            print(f"✗ Failed to capture image {i+1}/{num_images}")
        
        # Wait before next capture (except for last image)
        if i < num_images - 1:
            time.sleep(delay)
    
    # Cleanup
    camera.release()
    print(f"\nCapture complete! {num_images} images saved to {images_dir}")

def main():
    parser = argparse.ArgumentParser(description='Capture training images for ML model')
    parser.add_argument('-n', '--num_images', type=int, default=5,
                       help='Number of images to capture (default: 5)')
    parser.add_argument('-d', '--delay', type=float, default=0.5,
                       help='Delay between images in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_images <= 0:
        print("ERROR: Number of images must be positive")
        return
    if args.delay < 0:
        print("ERROR: Delay cannot be negative")
        return
    
    capture_images(args.num_images, args.delay)

if __name__ == "__main__":
    main()