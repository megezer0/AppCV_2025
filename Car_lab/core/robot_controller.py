#!/usr/bin/env python3

import threading
import time
import cv2
import numpy as np
from picarx import Picarx

class RobotController:
    def __init__(self):
        """Initialize the robot controller"""
        try:
            self.picar = Picarx()
            self.is_moving = False
            
            # Autonomous mode variables
            self.autonomous_mode = False
            self.frame_counter = 0
            self.previous_frame = None
            self.current_speed = 0.0
            
            # Feature modules (loaded dynamically as students implement them)
            self.line_follower = None
            self.sign_detector = None
            self.speed_estimator = None
            
            # Feature status tracking
            self.feature_status = {
                'line_following': 'Not Implemented',
                'sign_detection': 'Not Implemented', 
                'speed_estimation': 'Not Implemented'
            }
            
            print("Robot controller initialized successfully")
        except Exception as e:
            print(f"Error initializing robot: {e}")
            self.picar = None
    
    def _try_load_features(self):
        """Attempt to load student-implemented features"""
        
        # Try to load Week 1: Line Following
        try:
            from week1_line_following.line_follower import LineFollower
            self.line_follower = LineFollower()
            self.feature_status['line_following'] = 'Active'
            print("✓ Line following module loaded")
        except ImportError:
            self.feature_status['line_following'] = 'Not Implemented'
        except Exception as e:
            self.feature_status['line_following'] = f'Error: {str(e)}'
            
        # Try to load Week 2: Sign Detection  
        try:
            from week2_object_detection.sign_detector import SignDetector
            self.sign_detector = SignDetector()
            self.feature_status['sign_detection'] = 'Active'
            print("✓ Sign detection module loaded")
        except ImportError:
            self.feature_status['sign_detection'] = 'Not Implemented'
        except Exception as e:
            self.feature_status['sign_detection'] = f'Error: {str(e)}'
            
        # Try to load Week 3: Speed Estimation
        try:
            from week3_speed_estimation.speed_estimator import SpeedEstimator
            self.speed_estimator = SpeedEstimator()
            self.feature_status['speed_estimation'] = 'Active' 
            print("✓ Speed estimation module loaded")
        except ImportError:
            self.feature_status['speed_estimation'] = 'Not Implemented'
        except Exception as e:
            self.feature_status['speed_estimation'] = f'Error: {str(e)}'
    
    def start_autonomous_mode(self):
        """Start autonomous line following mode"""
        if not self.picar:
            return False, "Robot not connected"
            
        # Load/reload features in case students have updated their code
        self._try_load_features()
        
        if not self.line_follower:
            return False, "Line following not implemented yet"
            
        self.autonomous_mode = True
        self.frame_counter = 0
        print("🤖 Autonomous mode started")
        return True, "Autonomous mode started"
    
    def stop_autonomous_mode(self):
        """Stop autonomous mode and return to manual control"""
        self.autonomous_mode = False
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
        print("🛑 Autonomous mode stopped")
        return True, "Autonomous mode stopped"
    
    def process_autonomous_frame(self, frame):
        """
        Main autonomous processing pipeline - called for each camera frame
        This is where all the magic happens!
        """
        if not self.autonomous_mode or not self.picar:
            return frame
            
        try:
            self.frame_counter += 1
            display_frame = frame.copy()
            
            # Week 1: PID Line Following (runs every frame for smooth control)
            if self.line_follower:
                try:
                    steering_angle = self.line_follower.compute_steering_angle(frame)
                    
                    # Apply steering and move forward
                    self.picar.set_dir_servo_angle(steering_angle)
                    self.picar.forward(40)  # Constant forward speed
                    
                    # Add debug visualization if available
                    if hasattr(self.line_follower, 'get_debug_overlay'):
                        display_frame = self.line_follower.get_debug_overlay(display_frame)
                    
                    # Show steering angle on display
                    cv2.putText(display_frame, f"Steering: {steering_angle:.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                               
                except Exception as e:
                    print(f"Line following error: {e}")
                    self.feature_status['line_following'] = f'Runtime Error: {str(e)}'
            
            # Week 2: Sign Detection (runs every 15 frames to save CPU)
            if self.sign_detector and self.frame_counter % 15 == 0:
                try:
                    signs = self.sign_detector.detect_signs(frame)
                    
                    if signs:
                        # Draw bounding boxes on display
                        for sign in signs:
                            bbox = sign['bbox']
                            confidence = sign['confidence']
                            x, y, w, h = bbox
                            
                            # Draw bounding box
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(display_frame, f"STOP {confidence:.2f}", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Check if we should stop
                        if self.sign_detector.should_stop(signs, frame):
                            print("🛑 Stop sign detected - stopping for 3 seconds")
                            self.picar.stop()
                            
                            # Add stop indicator to display
                            cv2.putText(display_frame, "STOPPING FOR SIGN", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            # Resume after 3 seconds (using timer to avoid blocking)
                            def resume_movement():
                                if self.autonomous_mode:  # Only resume if still in autonomous mode
                                    print("✅ Resuming movement")
                            
                            timer = threading.Timer(3.0, resume_movement)
                            timer.start()
                            
                except Exception as e:
                    print(f"Sign detection error: {e}")
                    self.feature_status['sign_detection'] = f'Runtime Error: {str(e)}'
            
            # Week 3: Speed Estimation (runs every 3 frames)
            if self.speed_estimator and self.previous_frame is not None and self.frame_counter % 3 == 0:
                try:
                    speed = self.speed_estimator.estimate_speed(frame, self.previous_frame)
                    self.current_speed = speed
                    
                    # Display speed on frame
                    cv2.putText(display_frame, f"Speed: {speed:.1f} units/s", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                               
                except Exception as e:
                    print(f"Speed estimation error: {e}")
                    self.feature_status['speed_estimation'] = f'Runtime Error: {str(e)}'
            
            # Store frame for next speed calculation
            self.previous_frame = frame.copy()
            
            # Always show autonomous mode status
            cv2.putText(display_frame, "AUTO MODE", 
                       (display_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return display_frame
            
        except Exception as e:
            print(f"Autonomous processing error: {e}")
            return frame
    
    def get_feature_status(self):
        """Return current status of all features"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'features': self.feature_status.copy(),
            'current_speed': self.current_speed,
            'frame_counter': self.frame_counter
        }
    
    # =============================================================================
    # EXISTING MANUAL CONTROL METHODS (unchanged)
    # =============================================================================
    
    def _auto_stop(self):
        """Automatically stop the robot and center wheels after movement"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)  # Center the wheels
            self.is_moving = False
            print("Robot auto-stopped and wheels centered")
    
    def move_forward(self, duration=0.5, speed=50):
        """Move robot forward for specified duration"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)  # Straight ahead
            self.picar.forward(speed)
            print(f"Moving forward at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error moving forward: {e}")
            self._auto_stop()
            return False
    
    def move_backward(self, duration=0.5, speed=50):
        """Move robot backward for specified duration"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)  # Straight back
            self.picar.backward(speed)
            print(f"Moving backward at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error moving backward: {e}")
            self._auto_stop()
            return False
    
    def turn_left(self, duration=0.5, speed=50, angle=-30):
        """Turn robot left while moving forward"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)  # Turn wheels left
            self.picar.forward(speed)
            print(f"Turning left (angle {angle}) at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error turning left: {e}")
            self._auto_stop()
            return False
    
    def turn_right(self, duration=0.5, speed=50, angle=30):
        """Turn robot right while moving forward"""
        if not self.picar or self.is_moving or self.autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)  # Turn wheels right
            self.picar.forward(speed)
            print(f"Turning right (angle {angle}) at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error turning right: {e}")
            self._auto_stop()
            return False
    
    def emergency_stop(self):
        """Immediately stop the robot"""
        self.autonomous_mode = False  # Exit autonomous mode
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
            self.is_moving = False
            print("Emergency stop activated")
    
    def cleanup(self):
        """Clean shutdown of robot"""
        if self.picar:
            self.emergency_stop()
            print("Robot controller cleaned up")

# Global robot instance
robot = RobotController()