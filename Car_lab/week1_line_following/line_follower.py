#!/usr/bin/env python3

import cv2
import numpy as np
import time

class LineFollower:
    """
    Week 1 Implementation: PID Line Following with Debug Visualization
    
    Students implement computer vision + control theory to follow lines.
    Includes comprehensive debugging system for learning and parameter tuning.
    """
    
    def __init__(self):
        """Initialize PID controller and line detection parameters"""
        
        # =================================================================
        # PID Controller Parameters (Students tune these)
        # =================================================================
        self.Kp = 0.5  # Proportional gain - START HERE and tune first
        self.Ki = 0.0  # Integral gain - Add if steady-state error exists
        self.Kd = 0.1  # Derivative gain - Add to reduce oscillation
        
        # PID controller state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
        # =================================================================
        # Computer Vision Parameters (Students may need to tune these)
        # =================================================================
        self.canny_low = 70    # Lower threshold for Canny edge detection
        self.canny_high = 270  # Upper threshold for Canny edge detection
        self.crop_offset_y = 180  # How much to crop from bottom (adjust based on camera angle)
        self.crop_height = 60     # Height of the region of interest
        
        # Image center for error calculation (320px width / 2)
        self.image_center_x = 160
        
        # Steering limits (degrees) - PiCar-X servo limits
        self.max_steering_angle = 30
        
        # Debug visualization state
        self.debug_frame = None
        
        print("LineFollower initialized - Students: Implement the algorithm step by step!")
    
    def compute_steering_angle(self, camera_frame, debug_level=0):
        """
        Main function students implement for Week 1
        
        Args:
            camera_frame: numpy array of shape (240, 320, 3) - RGB image from camera
            debug_level: int 0-4, controls debugging visualization detail
            
        Returns:
            steering_angle: float between -30 and +30 degrees
                           negative = turn left, positive = turn right
        """
        
        try:
            # Initialize debug frame if debugging enabled
            if debug_level > 0:
                self.debug_frame = camera_frame.copy()
            
            # =============================================================
            # STEP 1: CROP IMAGE (Section 1.1: Region of Interest)
            # Focus on the lower portion of the image where the line is
            # =============================================================
            
            # Extract the region of interest from the bottom of the image
            roi = camera_frame[self.crop_offset_y:self.crop_offset_y + self.crop_height, 0:320]
            
            # Debug Level 1+: Show ROI boundary
            if debug_level >= 1:
                cv2.rectangle(self.debug_frame, 
                            (0, self.crop_offset_y), 
                            (320, self.crop_offset_y + self.crop_height), 
                            (0, 255, 0), 2)
                cv2.putText(self.debug_frame, "ROI", (5, self.crop_offset_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # =============================================================
            # STEP 2: EDGE DETECTION (Section 1.2: Canny Edge Detection)
            # Convert to grayscale and apply Canny edge detection
            # =============================================================
            
            # Convert ROI to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            
            # Debug Level 2+: Show edge detection result
            if debug_level >= 2:
                # Create small inset showing edge detection
                edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edge_resized = cv2.resize(edge_display, (160, 30))
                self.debug_frame[10:40, 10:170] = edge_resized
                cv2.rectangle(self.debug_frame, (10, 10), (170, 40), (255, 255, 0), 1)
                cv2.putText(self.debug_frame, "Edges", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # =============================================================
            # STEP 3: LINE DETECTION (Section 1.3: Hough Transform)
            # Use Hough Transform to detect line segments
            # =============================================================
            
            # Apply Hough Transform to detect line segments
            lines = cv2.HoughLinesP(
                edges,
                rho=1,                    # Distance resolution: 1 pixel
                theta=np.pi/180,          # Angle resolution: 1 degree  
                threshold=50,             # Minimum intersections to form line
                minLineLength=50,         # Minimum line length
                maxLineGap=10            # Maximum gap in line
            )
            
            # =============================================================
            # STEP 4: CALCULATE LINE CENTER (Section 1.4: Error Calculation)
            # Find the center point of all detected line segments
            # =============================================================
            
            line_center_x = self.image_center_x  # Default to image center if no lines found
            
            if lines is not None and len(lines) > 0:
                # Calculate the average center point of all line segments
                x_coords = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    center_x = (x1 + x2) / 2
                    x_coords.append(center_x)
                    
                    # Debug Level 3+: Draw detected line segments
                    if debug_level >= 3:
                        # Draw line segment on debug frame (adjust coordinates for ROI offset)
                        cv2.line(self.debug_frame, 
                               (x1, y1 + self.crop_offset_y), 
                               (x2, y2 + self.crop_offset_y), 
                               (255, 0, 0), 2)
                        # Draw center point of each segment
                        center_y = (y1 + y2) / 2 + self.crop_offset_y
                        cv2.circle(self.debug_frame, 
                                 (int(center_x), int(center_y)), 
                                 3, (0, 0, 255), -1)
                
                # Calculate average center of all line segments
                line_center_x = np.mean(x_coords)
            
            # =============================================================
            # STEP 5: PID CONTROL (Section 2: PID Controller)
            # Convert line position error to steering angle
            # =============================================================
            
            # Calculate error (horizontal distance from image center)
            error = line_center_x - self.image_center_x
            
            # Calculate time delta for integral and derivative terms
            current_time = time.time()
            dt = current_time - self.last_time
            
            if dt > 0:  # Avoid division by zero
                # Proportional term: current error
                P = self.Kp * error
                
                # Integral term: accumulated error over time
                self.integral += error * dt
                I = self.Ki * self.integral
                
                # Derivative term: rate of change of error
                D = self.Kd * (error - self.last_error) / dt
                
                # Combined PID output
                pid_output = P + I + D
                
                # Convert to steering angle and apply limits
                steering_angle = np.clip(pid_output, -self.max_steering_angle, self.max_steering_angle)
                
                # Update for next iteration
                self.last_error = error
                self.last_time = current_time
                
                # =============================================================
                # DEBUG VISUALIZATION
                # =============================================================
                
                if debug_level >= 1:
                    # Draw image center line
                    cv2.line(self.debug_frame, 
                           (self.image_center_x, self.crop_offset_y), 
                           (self.image_center_x, self.crop_offset_y + self.crop_height), 
                           (0, 255, 255), 2)
                    
                    # Draw detected line center
                    cv2.circle(self.debug_frame, 
                             (int(line_center_x), self.crop_offset_y + self.crop_height//2), 
                             5, (255, 0, 255), -1)
                    
                    # Draw error line
                    cv2.line(self.debug_frame, 
                           (self.image_center_x, self.crop_offset_y + self.crop_height//2), 
                           (int(line_center_x), self.crop_offset_y + self.crop_height//2), 
                           (0, 0, 255), 3)
                    
                    # Add text overlays
                    cv2.putText(self.debug_frame, f"Error: {error:.1f}px", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(self.debug_frame, f"Steering: {steering_angle:.1f}deg", 
                               (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if debug_level >= 4:
                    # Show PID component values
                    cv2.putText(self.debug_frame, f"P: {P:.2f}", 
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.debug_frame, f"I: {I:.2f}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.debug_frame, f"D: {D:.2f}", 
                               (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(self.debug_frame, f"Lines: {len(lines) if lines is not None else 0}", 
                               (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                return float(steering_angle)
            
            # Return 0 if timing calculation fails
            return 0.0
            
        except Exception as e:
            print(f"Line following error: {e}")
            return 0.0  # Safe default - go straight
    
    def get_debug_frame(self):
        """Return the debug visualization frame"""
        return self.debug_frame if self.debug_frame is not None else None
    
    def reset_integral(self):
        """Reset integral term (useful when starting or after errors)"""
        self.integral = 0.0
    
    def update_parameters(self, kp=None, ki=None, kd=None, canny_low=None, canny_high=None):
        """Update PID and vision parameters during runtime"""
        if kp is not None:
            self.Kp = kp
        if ki is not None:
            self.Ki = ki
        if kd is not None:
            self.Kd = kd
        if canny_low is not None:
            self.canny_low = canny_low
        if canny_high is not None:
            self.canny_high = canny_high