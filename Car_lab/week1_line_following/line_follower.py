
# =============================================================================
# IMPLEMENTATION STRATEGY
# =============================================================================

"""
APPROACH THIS STEP-BY-STEP:

PHASE 1 - GET BASIC DETECTION WORKING:
□ Extract ROI from bottom portion of image
□ Convert to grayscale 
□ Apply Canny edge detection
□ Verify you can see edges in debug visualization

PHASE 2 - FIND LINES:
□ Apply Hough transform to detect line segments
□ Check debug data shows lines_detected > 0
□ Visualize detected lines to verify they make sense

PHASE 3 - CALCULATE POSITION:
□ Filter lines by length and angle
□ Calculate center points of line segments
□ Combine multiple segments into single position

PHASE 4 - IMPLEMENT CONTROL:
□ Start with only P term (Ki=0, Kd=0)
□ Tune Kp until robot follows line (may oscillate)
□ Add D term to reduce oscillation
□ Add I term only if steady-state error exists

DEBUGGING TIPS:
- Use debug_level parameter to see what's happening
- Check lines_detected count in sidebar
- Verify ROI is positioned correctly
- Monitor processing time (should be <50ms)
- Print intermediate values to understand failures

COMMON PITFALLS:
- ROI positioned wrong (no lines in region)
- Canny thresholds too strict (no edges detected)
- Hough parameters too strict (no lines found)
- PID gains too high (oscillation/instability)
- Forgetting to adjust coordinates for ROI offset
"""
#!/usr/bin/env python3

import cv2
import numpy as np
import time

class LineFollower:
    """
    Week 1 Implementation: PID Line Following
    
    AVAILABLE OPENCV FUNCTIONS YOU MAY NEED:
    - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) - convert to grayscale
    - cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) - blur to reduce noise
    - cv2.Canny(image, low_threshold, high_threshold) - edge detection
    - cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap) - line detection
    - cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness) - draw rectangle
    - cv2.line(image, (x1,y1), (x2,y2), color, thickness) - draw line
    - cv2.circle(image, (center_x, center_y), radius, color, thickness) - draw circle
    
    USEFUL NUMPY FUNCTIONS:
    - np.sqrt() - square root
    - np.arctan2(y, x) - angle from coordinates
    - np.mean() - average of array
    - np.clip(value, min_val, max_val) - constrain value to range
    
    ALGORITHM OVERVIEW:
    1. Extract region of interest from camera frame
    2. Convert to grayscale and apply edge detection
    3. Find line segments using Hough transform
    4. Calculate center position of detected lines
    5. Use PID control to convert position error to steering angle
    
    IMAGE DIMENSIONS: 320px wide × 240px tall
    """
    
    def __init__(self):
        """Initialize PID controller and line detection parameters"""
        
        # PID Controller Parameters - TUNE THESE!
        self.Kp = 0.0  # Start around 0.5-1.0
        self.Ki = 0.0  # Start with 0, add later if needed
        self.Kd = 0.0  # Around 0.1 to reduce oscillation
        
        # PID controller state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.integral_limit = 100.0
        
        # Computer Vision Parameters - EXPERIMENT WITH THESE!
        self.canny_low = 50     # Try 50-100
        self.canny_high = 150   # Try 150-300
        
        # Region of Interest - DEFINE THE SEARCH AREA!
        self.crop_offset_y = 0    # Starting Y position (try 70-80% of image height)
        self.crop_height = 60     # Height of ROI (try 15-25% of image height)
        
        # Image center for error calculation
        self.image_center_x = 160  # 320px / 2
        
        # Steering limits
        self.max_steering_angle = 30
        
        # Hough Transform Parameters - TUNE FOR LINE DETECTION!
        self.hough_threshold = 30       # Min intersections (try 20-50)
        self.hough_min_line_length = 30 # Min line length (try 20-50) 
        self.hough_max_line_gap = 10    # Max gap in line (try 5-20)
        
        self.debug_frame = None
        self.debug_data = {
            'lines_detected': 0,
            'processing_time_ms': 0.0
        }
        
        print("✅ LineFollower initialized - Time to implement!")
    
    def compute_steering_angle(self, camera_frame, debug_level=0):
        """
        Main line following algorithm
        
        Your mission:
        1. Extract the bottom portion of the image where lines appear
        2. Find edges using Canny edge detection  
        3. Detect line segments with Hough transform
        4. Calculate the center position of detected lines
        5. Use PID control to generate steering commands
        """
        
        start_time = time.time()
        
        try:
            if debug_level > 0:
                self.debug_frame = camera_frame.copy()
            
            # =============================================================
            # STEP 1: REGION OF INTEREST EXTRACTION
            # TODO: Extract the region where you expect to find the line
            # Think: Where in the image does the line appear? 
            # Consider: You don't need the whole image, just the relevant part
            # =============================================================
            
            roi = camera_frame  # REPLACE: Extract proper ROI
            
            # Debug visualization for ROI
            if debug_level >= 1 and self.crop_offset_y > 0:
                # TODO: Show where your ROI is located
                pass
            
            # =============================================================
            # STEP 2: EDGE DETECTION  
            # TODO: Convert to grayscale and find edges
            # Think: What preprocessing might help edge detection?
            # Consider: Noise reduction before edge detection
            # =============================================================
            
            # TODO: Implement grayscale conversion and edge detection
            edges = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)  # REPLACE
            
            # Debug visualization for edges
            if debug_level >= 2:
                # TODO: Show edge detection result as small inset
                pass
            
            # =============================================================
            # STEP 3: LINE DETECTION
            # TODO: Use Hough transform to find line segments
            # Think: What parameters work best for your track?
            # Consider: Rho and theta values for Hough transform
            # =============================================================
            
            lines = None  # REPLACE: Implement line detection
            
            # =============================================================
            # STEP 4: LINE CENTER CALCULATION
            # TODO: Process detected lines and find the center
            # Think: How do you handle multiple line segments?
            # Consider: Filtering lines by length and angle
            # =============================================================
            
            line_center_x = self.image_center_x
            lines_found = 0
            
            if lines is not None and len(lines) > 0:
                valid_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # TODO: Calculate line properties and filter
                    # Think: What makes a "good" line for following?
                    # Consider: Length, angle, position requirements
                    
                    line_length = 0      # IMPLEMENT: Calculate line length
                    line_angle = 0       # IMPLEMENT: Calculate line angle
                    
                    # TODO: Decide if this line is worth keeping
                    if True:  # REPLACE: Add filtering conditions
                        valid_lines.append(line)
                
                if valid_lines:
                    x_coords = []
                    
                    for line in valid_lines:
                        x1, y1, x2, y2 = line[0]
                        
                        # TODO: Find center point of each line segment
                        center_x = 0  # IMPLEMENT: Calculate center
                        x_coords.append(center_x)
                        
                        # Debug visualization for detected lines
                        if debug_level >= 3:
                            # TODO: Draw the detected line segments
                            # Remember: Adjust coordinates for ROI offset
                            pass
                    
                    if x_coords:
                        # TODO: Combine multiple line centers into single position
                        line_center_x = 0  # IMPLEMENT: Calculate overall center
                        lines_found = len(valid_lines)
            
            # =============================================================
            # STEP 5: PID CONTROL
            # TODO: Convert position error to steering angle
            # Think: How does error relate to required steering?
            # Consider: Integral windup and derivative kick
            # =============================================================
            
            error = line_center_x - self.image_center_x
            current_time = time.time()
            dt = current_time - self.last_time
            
            if dt > 0:
                # TODO: Implement PID calculation
                # Think: What does each term contribute to control?
                # Consider: How to prevent integral windup
                
                P_term = 0  # IMPLEMENT: Proportional term
                I_term = 0  # IMPLEMENT: Integral term  
                D_term = 0  # IMPLEMENT: Derivative term
                
                pid_output = 0  # IMPLEMENT: Combine terms
                
                steering_angle = np.clip(pid_output, -self.max_steering_angle, self.max_steering_angle)
                
                # Update state for next iteration
                self.last_error = error
                self.last_time = current_time
                
                # =============================================================
                # DEBUG VISUALIZATION
                # TODO: Add helpful visual overlays
                # Think: What information helps debug the algorithm?
                # Consider: Error visualization, center points, detected lines
                # =============================================================
                
                if debug_level >= 1:
                    # TODO: Implement debug visualizations
                    # Ideas: image center line, detected line center, error line
                    pass
                
                # Update debug data
                processing_time = (time.time() - start_time) * 1000
                self.debug_data.update({
                    'lines_detected': lines_found,
                    'processing_time_ms': processing_time
                })
                
                return float(steering_angle)
            
            return 0.0
            
        except Exception as e:
            print(f"Line following error: {e}")
            self.debug_data.update({
                'lines_detected': 0,
                'processing_time_ms': 0.0
            })
            return 0.0
    
    def get_debug_frame(self):
        """Return the debug visualization frame"""
        return self.debug_frame if self.debug_frame is not None else None
    
    def get_debug_data(self):
        """Return debug data for sidebar"""
        return self.debug_data.copy()
    
    def reset_integral(self):
        """Reset integral term"""
        self.integral = 0.0
        print("🔄 PID integral term reset")
    
    def update_parameters(self, kp=None, ki=None, kd=None, canny_low=None, canny_high=None):
        """Update parameters during runtime"""
        if kp is not None:
            self.Kp = kp
            print(f"✅ Kp updated to {kp}")
        if ki is not None:
            self.Ki = ki
            print(f"✅ Ki updated to {ki}")
        if kd is not None:
            self.Kd = kd
            print(f"✅ Kd updated to {kd}")
        if canny_low is not None:
            self.canny_low = canny_low
            print(f"✅ Canny low threshold updated to {canny_low}")
        if canny_high is not None:
            self.canny_high = canny_high
            print(f"✅ Canny high threshold updated to {canny_high}")
