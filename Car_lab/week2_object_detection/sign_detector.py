#!/usr/bin/env python3

import cv2
import numpy as np
import os

class SignDetector:
    """
    Week 2 Implementation: ONNX YOLO Model Integration
    
    Students will integrate a pre-trained YOLOv8 model to detect stop signs
    and implement decision logic for when to stop the robot.
    """
    
    def __init__(self):
        """Initialize ONNX model and detection parameters"""
        
        # Path to the pre-trained ONNX model (provided by instructor)
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'stop_signs.onnx')
        
        # YOLO detection parameters (students may need to tune these)
        self.confidence_threshold = 0.5  # Minimum confidence to consider a detection
        self.nms_threshold = 0.4         # Non-maximum suppression threshold
        
        # Input size for YOLO model (typically 640x640 for YOLOv8)
        self.input_size = (640, 640)
        
        # Simple stopping thresholds (students will implement these)
        self.simple_area_threshold = 5000   # Stop if bounding box area > this
        self.depth_distance_threshold = 1.5  # Stop if estimated distance < this (meters)
        
        # Initialize the ONNX model
        self.net = None
        self._load_model()
        
        print("SignDetector initialized - Students: Implement detect_signs() and should_stop()!")
    
    def _load_model(self):
        """Load the ONNX model using OpenCV DNN"""
        try:
            if os.path.exists(self.model_path):
                # STUDENTS: This is provided - shows how to load ONNX with OpenCV
                self.net = cv2.dnn.readNet(self.model_path)
                
                # Set computation backend (CPU by default)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                print(f"✅ ONNX model loaded: {self.model_path}")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("   Place your stop_signs.onnx file in the models/ directory")
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            self.net = None
    
    def detect_signs(self, camera_frame):
        """
        Main detection function students must implement
        
        Args:
            camera_frame: numpy array of shape (240, 320, 3) - RGB image from camera
            
        Returns:
            List of detected signs in format:
            [
                {
                    'bbox': [x, y, w, h],          # Bounding box coordinates
                    'confidence': 0.95,            # Detection confidence (0-1)
                    'class': 'stop'                # Object class (always 'stop' for this model)
                },
                ...
            ]
        """
        
        if self.net is None:
            return []  # No model loaded
        
        try:
            # =============================================================
            # STEP 1: PREPROCESS IMAGE FOR YOLO INPUT
            # YOLO models expect specific input format
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # 1. Convert BGR to RGB (OpenCV uses BGR, YOLO expects RGB)
            # 2. Resize image to YOLO input size (e.g., 640x640)
            # 3. Normalize pixel values to 0-1 range
            # 4. Create blob using cv2.dnn.blobFromImage()
            
            # Hint: cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB)
            # Example parameters: scalefactor=1/255.0, size=self.input_size, swapRB=True
            
            blob = None  # TODO: Create blob from input image
            
            # =============================================================
            # STEP 2: RUN INFERENCE
            # Feed the blob through the neural network
            # =============================================================
            
            # STUDENTS IMPLEMENT:
            # 1. Set the input blob using self.net.setInput()
            # 2. Run forward pass using self.net.forward()
            
            self.net.setInput(blob)
            outputs = None  # TODO: Run inference
            
            # =============================================================
            # STEP 3: PARSE MODEL OUTPUTS
            # YOLO outputs need to be interpreted and filtered
            # =============================================================
            
            detections = []
            
            if outputs is not None:
                # STUDENTS IMPLEMENT:
                # YOLOv8 output format is typically [batch, 84, 8400] where:
                # - 84 = 4 (bbox coords) + 80 (class scores)
                # - 8400 = number of anchor points
                # 
                # For each detection:
                # 1. Extract bounding box coordinates (center_x, center_y, width, height)
                # 2. Extract class scores and find the maximum
                # 3. Filter by confidence threshold
                # 4. Convert coordinates back to original image size
                # 5. Apply Non-Maximum Suppression to remove duplicates
                
                # TODO: Parse YOLO outputs and extract detections
                # TODO: Apply confidence filtering
                # TODO: Convert coordinates to original image scale
                # TODO: Apply Non-Maximum Suppression using cv2.dnn.NMSBoxes()
                
                pass
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def should_stop(self, detected_signs, camera_frame):
        """
        Decision logic for when to stop the robot
        
        Args:
            detected_signs: List of detected signs from detect_signs()
            camera_frame: Current camera frame for depth estimation
            
        Returns:
            bool: True if robot should stop, False otherwise
        """
        
        if not detected_signs:
            return False  # No signs detected
        
        # Get the most confident detection
        best_sign = max(detected_signs, key=lambda x: x['confidence'])
        
        # =================================================================
        # APPROACH A: SIMPLE AREA-BASED STOPPING (implement this first)
        # Stop if the bounding box is large enough (sign is close)
        # =================================================================
        
        # STUDENTS IMPLEMENT:
        # Calculate the area of the bounding box
        # If area > threshold, return True (stop the robot)
        
        bbox = best_sign['bbox']
        x, y, w, h = bbox
        area = w * h
        
        if area > self.simple_area_threshold:
            print(f"🛑 Stopping for sign - area: {area}")
            return True
        
        # =================================================================
        # APPROACH B: DEPTH-BASED STOPPING (implement this later)
        # Use monocular depth estimation to get actual distance
        # =================================================================
        
        # STUDENTS CAN IMPLEMENT (OPTIONAL):
        # 1. Load a monocular depth estimation model
        # 2. Estimate depth for pixels within the bounding box
        # 3. Calculate average distance to the sign
        # 4. Stop if distance < threshold
        
        try:
            # TODO (Optional): Implement depth-based stopping
            # estimated_distance = self._estimate_distance_to_sign(bbox, camera_frame)
            # if estimated_distance < self.depth_distance_threshold:
            #     print(f"🛑 Stopping for sign - distance: {estimated_distance:.1f}m")
            #     return True
            pass
        except Exception as e:
            print(f"Depth estimation error: {e}")
        
        return False  # Don't stop
    
    def _estimate_distance_to_sign(self, bbox, camera_frame):
        """
        Optional: Estimate distance using monocular depth estimation
        Students can implement this for more sophisticated stopping
        """
        # This is an advanced feature - students can implement if they want
        # to go beyond the basic area-based approach
        
        # Possible approaches:
        # 1. Use a pre-trained depth estimation model (MiDaS, DPT, etc.)
        # 2. Use known sign size and perspective geometry
        # 3. Use optical flow information
        
        return 999.0  # Return large distance by default (don't stop)

# =============================================================================
# HELPFUL REFERENCE CODE (for students to adapt)
# =============================================================================

"""
EXAMPLE BLOB CREATION:
blob = cv2.dnn.blobFromImage(
    image, 
    scalefactor=1/255.0,
    size=(640, 640), 
    mean=(0, 0, 0), 
    swapRB=True, 
    crop=False
)

EXAMPLE INFERENCE:
net.setInput(blob)
outputs = net.forward()

EXAMPLE OUTPUT PARSING (YOLOv8 format):
for output in outputs:
    for detection in output.T:  # Transpose to get [8400, 84]
        scores = detection[4:]  # Class scores
        confidence = np.max(scores)
        if confidence > confidence_threshold:
            class_id = np.argmax(scores)
            # Extract bbox coordinates...

EXAMPLE NON-MAXIMUM SUPPRESSION:
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
"""