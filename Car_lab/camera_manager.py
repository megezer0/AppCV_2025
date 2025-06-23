#!/usr/bin/env python3

import cv2
import numpy as np
import threading
import time
from vilib import Vilib
import io

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("picamera2 not available, will use fallback methods")
    PICAMERA2_AVAILABLE = False

class CameraManager:
    def __init__(self):
        """Initialize camera using picamera2 directly"""
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.picam2 = None
        
        try:
            # Try picamera2 first if available
            if PICAMERA2_AVAILABLE:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(1)  # Give camera time to initialize
                print("Camera initialized successfully using picamera2")
            else:
                raise Exception("picamera2 not available")
            
        except Exception as e:
            print(f"Picamera2 failed: {e}")
            try:
                # Fallback to OpenCV camera
                self.picam2 = None
                self.cv_camera = cv2.VideoCapture(0)
                if self.cv_camera.isOpened():
                    self.cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print("Camera initialized using OpenCV VideoCapture")
                else:
                    raise Exception("OpenCV camera failed to open")
            except Exception as e2:
                print(f"Both camera methods failed: {e2}")
                print("Using test pattern")
                self.picam2 = None
                self.cv_camera = None
                self.frame = self._create_test_pattern()
    
    def _create_test_pattern(self):
        """Create a test pattern frame"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create a simple test pattern
        frame[100:200, 100:200] = [255, 0, 0]  # Red square
        frame[200:300, 200:300] = [0, 255, 0]  # Green square
        frame[300:400, 300:400] = [0, 0, 255]  # Blue square
        
        # Add text
        cv2.putText(frame, "TEST PATTERN - Camera Not Available", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "PiCar-X Control Interface", (150, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame
    
    def _create_error_frame(self):
        """Create a black frame with error message"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Error", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return frame
    
    def start_streaming(self):
        """Start the camera streaming thread"""
        if not self.is_running:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print("Camera streaming started")
    
    def _capture_frames(self):
        """Continuously capture frames from camera"""
        while self.is_running:
            try:
                if self.picam2:
                    # Get frame from picamera2
                    frame_array = self.picam2.capture_array()
                    # Convert RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    
                elif hasattr(self, 'cv_camera') and self.cv_camera:
                    # Get frame from OpenCV camera
                    ret, frame = self.cv_camera.read()
                    if not ret:
                        raise Exception("Failed to read from OpenCV camera")
                    
                else:
                    # Use test pattern if no camera available
                    frame = self._create_test_pattern()
                
                # Clean frame - no text overlay
                with self.lock:
                    self.frame = frame
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Error capturing frame: {e}")
                with self.lock:
                    self.frame = self._create_error_frame()
                time.sleep(0.1)
    
    def get_frame(self):
        """Get the current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                return self._create_test_pattern()
    
    def get_jpeg_frame(self):
        """Get current frame as JPEG bytes for streaming"""
        frame = self.get_frame()
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return buffer.tobytes()
        else:
            # Return test pattern if encoding fails
            test_frame = self._create_test_pattern()
            ret, buffer = cv2.imencode('.jpg', test_frame)
            return buffer.tobytes()
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)
        print("Camera streaming stopped")
    
    def cleanup(self):
        """Clean shutdown of camera"""
        self.stop_streaming()
        try:
            if self.picam2:
                self.picam2.stop()
                self.picam2.close()
            elif hasattr(self, 'cv_camera') and self.cv_camera:
                self.cv_camera.release()
            print("Camera cleaned up")
        except:
            pass

# Global camera instance
camera = CameraManager()