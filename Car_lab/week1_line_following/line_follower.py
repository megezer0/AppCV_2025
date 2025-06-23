#!/usr/bin/env python3

import cv2
import numpy as np
import time

class LineFollower:
    """
    Week 1 Implementation: PID Line Following
    
    Students will implement computer vision + control theory to follow lines.
    This is the core implementation following the provided academic paper.
    """
    
    def __init__(self):
        """Initialize PID controller and line detection parameters"""
        
        # =================================================================
        # STUDENTS: Configure these PID parameters through experimentation
        # Start with Kp only, then add Ki and Kd if needed
        # =================================================================
        self.Kp = 0.5  # Proportional gain - START HERE and tune first
        self.Ki = 0.0  # Integral gain - Add if steady-state error exists
        self.Kd = 0.1  # Derivative gain - Add to reduce oscillation
        
        # PID controller state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        
        # Line detection parameters (students may need to tune these)
        self.canny_low = 70    # Lower threshold for Canny edge detection
        self.canny_high = 270  # Upper threshold for Canny edge detection
        self.crop_offset_y = 180  # How much to crop from bottom (adjust based on camera angle)
        self.crop_height = 60     # Height of the region of interest
        
        # Image center for error calculation (320px width / 2)
        self.image_center_x = 160
        
        # Steering limits (degrees)
        self.max_steering_angle = 30
        
        print("LineFollower initialized - Students: Implement compute_steering_angle()!")
    
    def compute_steering_angle(self, camera_frame):
        """
        Main function students must implement for Week 1
        
        Args:
            camera_frame: numpy array of shape (240, 320, 3) - RGB image from camera
            
        Returns:
            steering_angle: float between -30 and +30 degrees
                           negative = turn left, positive = turn right
        
        Implementation Steps (following the academic paper):
        1. Crop the image to focus on the immediate line ahead
        2. Apply Canny edge detection to find edges
        3. Use Hough Transform to detect line segments
        4. Calculate the center of detected lines
        5. Compute error (line center - image center)
        6. Apply PID control to convert error to steering angle
        """
        
        try:
            # =============================================================
            # STEP 1: CROP IMAGE (following paper section 1.1)
            # Focus on the lower portion of the image where the line is
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # Extract the region of interest from the bottom of the image
            # Use: image[start_y:end_y, start_x:end_x]
            # Hint: You want roughly the bottom 60 pixels of the 240px high image
            
            roi = None  # TODO: Implement image cropping
            
            # =============================================================
            # STEP 2: EDGE DETECTION (following paper section 1.2)
            # Convert to grayscale and apply Canny edge detection
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # 1. Convert ROI to grayscale using cv2.cvtColor()
            # 2. Apply Gaussian blur to reduce noise (optional but recommended)
            # 3. Apply Canny edge detection using cv2.Canny()
            #    Use self.canny_low and self.canny_high as thresholds
            
            gray = None      # TODO: Convert to grayscale
            edges = None     # TODO: Apply Canny edge detection
            
            # =============================================================
            # STEP 3: LINE DETECTION (following paper section 1.3)
            # Use Hough Transform to detect line segments
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # Apply cv2.HoughLinesP() to detect line segments
            # Parameters from paper: rho=1, theta=CV_PI/180, threshold=50, min_length=50, max_gap=10
            # This returns a list of line segments as [x1, y1, x2, y2]
            
            lines = None     # TODO: Apply Hough Transform
            
            # =============================================================
            # STEP 4: CALCULATE LINE CENTER (following paper section 1.4)
            # Find the center point of all detected line segments
            # =============================================================
            
            line_center_x = self.image_center_x  # Default to image center if no lines found
            
            if lines is not None and len(lines) > 0:
                # STUDENTS IMPLEMENT:
                # Calculate the average center point of all line segments
                # For each line [x1, y1, x2, y2], the center is ((x1+x2)/2, (y1+y2)/2)
                # You only need the x-coordinate for steering
                
                # TODO: Calculate line_center_x from detected lines
                pass
            
            # =============================================================
            # STEP 5: PID CONTROL (following paper section 2)
            # Convert line position error to steering angle
            # =============================================================
            
            # Calculate error (horizontal distance from image center)
            error = line_center_x - self.image_center_x
            
            # STUDENTS IMPLEMENT PID CONTROLLER:
            # Calculate time delta
            current_time = time.time()
            dt = current_time - self.last_time
            
            if dt > 0:  # Avoid division by zero
                # Proportional term
                P = self.Kp * error
                
                # Integral term (accumulate error over time)
                self.integral += error * dt
                I = self.Ki * self.integral
                
                # Derivative term (rate of change of error)
                D = self.Kd * (error - self.last_error) / dt
                
                # Combined PID output
                pid_output = P + I + D
                
                # Convert to steering angle and apply limits
                steering_angle = np.clip(pid_output, -self.max_steering_angle, self.max_steering_angle)
                
                # Update for next iteration
                self.last_error = error
                self.last_time = current_time
                
                return float(steering_angle)
            
            # Return 0 if timing calculation fails
            return 0.0
            
        except Exception as e:
            print(f"Line following error: {e}")
            return 0.0  # Safe default - go straight
    
    def get_debug_overlay(self, frame):
        """
        Optional: Add debug visualization to the camera frame
        Students can implement this to help with debugging
        """
        # This is optional - students can add line detection visualization here
        # For example: draw detected lines, ROI rectangle, center point, etc.
        return frame

# =============================================================================
# HELPFUL REFERENCE CODE (for students to adapt)
# This shows the structure but students must implement the actual logic
# =============================================================================

"""
EXAMPLE IMAGE CROPPING:
roi = frame[180:240, 0:320]  # Bottom 60 pixels, full width

EXAMPLE CANNY EDGE DETECTION:
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 70, 270)

EXAMPLE HOUGH TRANSFORM:
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

EXAMPLE LINE CENTER CALCULATION:
if lines is not None:
    x_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        center_x = (x1 + x2) / 2
        x_coords.append(center_x)
    line_center_x = np.mean(x_coords)
"""