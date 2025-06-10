#!/usr/bin/env python3
# lab_6/scripts/capture_training_images.py

import os
import subprocess
import argparse
import time
from pathlib import Path

def capture_images(num_images=5, delay=0.5):
    """Capture training images using libcamera-still"""
    
    # Set up directory structure
    base_dir = Path("lab_6").resolve()
    images_dir = base_dir / "captured_images"
    
    # Create directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next available image number
    existing_files = list(images_dir.glob("img_*.jpg"))
    if existing_files:
        numbers = []
        for file in existing_files:
            try:
                num_str = file.stem.split('_')[1]
                numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue
        start_count = max(numbers) + 1 if numbers else 0
    else:
        start_count = 0
    
    print(f"Starting image capture...")
    print(f"Will capture {num_images} images with {delay}s delay")
    print(f"Saving to: {images_dir}")
    print(f"Starting from img_{start_count:03d}.jpg")
    print("\nGet ready! Starting in 3 seconds...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Capturing!")
    
    # Capture images using libcamera-still
    successful_captures = 0
    for i in range(num_images):
        image_num = start_count + i
        filename = f"img_{image_num:03d}.jpg"
        filepath = images_dir / filename
        
        try:
            # Use libcamera-still to capture image
            cmd = [
                "libcamera-still",
                "-n",  # No preview
                "--width", "640",
                "--height", "480",
                "--timeout", "100",
                "-o", str(filepath)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✓ Captured image {i+1}/{num_images}: {filename}")
                successful_captures += 1
            else:
                print(f"✗ Failed to capture image {i+1}/{num_images}: {result.stderr.strip()}")
        
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout capturing image {i+1}/{num_images}")
        except FileNotFoundError:
            print("ERROR: libcamera-still not found. Make sure camera is enabled.")
            return
        except Exception as e:
            print(f"✗ Error capturing image {i+1}/{num_images}: {e}")
        
        # Wait before next capture (except for last image)
        if i < num_images - 1:
            time.sleep(delay)
    
    print(f"\nCapture complete! {successful_captures}/{num_images} images saved to {images_dir}")

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