#!/usr/bin/env python3
# lab_6/scripts/capture_training_images.py

import os
import cv2
import time
from datetime import datetime
import readchar
from pathlib import Path

class ImageCapture:
    def __init__(self, base_dir="lab_6"):
        # Set up directory structure
        self.base_dir = Path(base_dir).resolve()
        self.images_dir = self.base_dir / "captured_images"
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Image counter for unique naming
        self.image_count = self._get_next_image_number()
        
        print(f"Images will be saved to: {self.images_dir}")
        print(f"Starting image counter at: {self.image_count}")
    
    def _get_next_image_number(self):
        """Find the next available image number to avoid overwrites"""
        existing_files = list(self.images_dir.glob("image_*.jpg"))
        if not existing_files:
            return 1
        
        # Extract numbers from existing files
        numbers = []
        for file in existing_files:
            try:
                # Extract number from filename like "image_001.jpg"
                num_str = file.stem.split('_')[1]
                numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def capture_image(self):
        """Capture a single image and save it"""
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture image from camera")
            return False
        
        # Create filename with zero-padded number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{self.image_count:03d}_{timestamp}.jpg"
        filepath = self.images_dir / filename
        
        # Save image
        success = cv2.imwrite(str(filepath), frame)
        if success:
            print(f"✓ Captured: {filename}")
            self.image_count += 1
            return True
        else:
            print(f"✗ Failed to save: {filename}")
            return False
    
    def capture_batch(self, count=5, delay=1.0):
        """Capture a batch of images with delay between shots"""
        print(f"\nCapturing {count} images with {delay}s delay...")
        print("Get your object ready! Starting in 3 seconds...")
        time.sleep(3)
        
        successful_captures = 0
        for i in range(count):
            print(f"Capturing image {i+1}/{count}...")
            if self.capture_image():
                successful_captures += 1
            
            if i < count - 1:  # Don't delay after last image
                time.sleep(delay)
        
        print(f"✓ Batch complete: {successful_captures}/{count} images captured")
        return successful_captures
    
    def preview_mode(self):
        """Show live camera preview to help with positioning"""
        print("\nPreview mode - Press 'q' to exit preview")
        print("Position your object and lighting as desired...")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Add text overlay
            cv2.putText(frame, "Preview Mode - Press 'q' to exit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Next image will be: image_{self.image_count:03d}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Camera Preview", frame)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def interactive_mode(self):
        """Interactive mode for manual capture control"""
        manual = '''
Image Capture Tool - Interactive Mode

Commands:
    1: Capture single image
    5: Capture batch of 5 images (1s delay)
    b: Capture custom batch (you specify count and delay)
    p: Preview mode (position your object)
    s: Show statistics
    q: Quit

Position your object and press a command key...
        '''
        
        print(manual)
        
        while True:
            try:
                print(f"\nReady for capture (next: image_{self.image_count:03d}) > ", end='', flush=True)
                key = readchar.readkey().lower()
                print()  # New line after key press
                
                if key == '1':
                    print("Get ready... capturing in 2 seconds!")
                    time.sleep(2)
                    self.capture_image()
                
                elif key == '5':
                    self.capture_batch(5, 1.0)
                
                elif key == 'b':
                    try:
                        count = int(input("Enter number of images: "))
                        delay = float(input("Enter delay between images (seconds): "))
                        self.capture_batch(count, delay)
                    except ValueError:
                        print("Invalid input. Please enter numbers only.")
                
                elif key == 'p':
                    self.preview_mode()
                
                elif key == 's':
                    self.show_statistics()
                
                elif key == 'q':
                    print("Exiting capture tool...")
                    break
                
                else:
                    print(f"Unknown command: '{key}'. Try again.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    def show_statistics(self):
        """Show capture statistics"""
        existing_files = list(self.images_dir.glob("image_*.jpg"))
        total_images = len(existing_files)
        
        print(f"\n=== Capture Statistics ===")
        print(f"Images directory: {self.images_dir}")
        print(f"Total images captured: {total_images}")
        print(f"Next image number: {self.image_count}")
        
        if existing_files:
            # Show some recent files
            recent_files = sorted(existing_files, key=lambda x: x.stat().st_mtime)[-5:]
            print(f"Recent captures:")
            for file in recent_files:
                size_kb = file.stat().st_size // 1024
                mod_time = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"  {file.name} ({size_kb}KB, {mod_time.strftime('%H:%M:%S')})")
    
    def cleanup(self):
        """Release camera resources"""
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

def main():
    print("=== Training Image Capture Tool ===")
    print("This tool will help you capture images for model training.")
    
    try:
        # Initialize capture tool
        capture = ImageCapture()
        
        # Check if camera is working
        ret, test_frame = capture.camera.read()
        if not ret:
            print("ERROR: Could not access camera. Please check camera connection.")
            return
        
        print("Camera initialized successfully!")
        
        # Run interactive mode
        capture.interactive_mode()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        capture.cleanup()
        print("Camera resources released.")

if __name__ == "__main__":
    main()