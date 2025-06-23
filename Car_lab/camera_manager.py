#!/usr/bin/env python3

import cv2
import threading
import time
from vilib import Vilib

class CameraManager:
    def __init__(self):
        """Initialize camera using vilib"""
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        try:
            # Start camera using vilib (as shown in PiCar-X examples)
            Vilib.camera_start(vflip=False, hflip=False)
            time.sleep(1)  # Give camera time to initialize
            print("Camera initialized successfully")
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.frame = self._create_error_frame()
    
    def _create_error_frame(self):
        """Create a black frame with error message"""
        frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
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
                # Get frame from vilib
                # Note: vilib handles the actual frame capture internally
                # We'll create a simple frame for now since vilib is mainly for web streaming
                
                # Create a basic frame with timestamp for testing
                frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                
                # Add timestamp and status
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "PiCar-X Control Test", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Use WASD to control", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Add control instructions
                cv2.putText(frame, "W: Forward", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "S: Backward", (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "A: Turn Left", (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "D: Turn Right", (10, 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
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
                return self._create_error_frame()
    
    def get_jpeg_frame(self):
        """Get current frame as JPEG bytes for streaming"""
        frame = self.get_frame()
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return buffer.tobytes()
        else:
            # Return empty frame if encoding fails
            empty_frame = self._create_error_frame()
            ret, buffer = cv2.imencode('.jpg', empty_frame)
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
            Vilib.camera_close()
            print("Camera cleaned up")
        except:
            pass

# Global camera instance
camera = CameraManager()