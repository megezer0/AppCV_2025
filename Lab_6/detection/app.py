#!/usr/bin/env python3
import argparse
import time
import logging
import threading
import os
from flask import Flask, render_template, Response
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
detector = None
current_frame = None
frame_lock = threading.Lock()
running = True
frame_counter = 0

class YOLODetector:
    def __init__(self, model_path=None, confidence_threshold=0.5, input_size=320):
        self.conf_threshold = confidence_threshold
        self.input_size = input_size  # Smaller input size for Pi 3
        
        if model_path and Path(model_path).exists():
            # Load custom ONNX model with optimizations
            logger.info(f"Loading custom model: {model_path}")
            
            # Optimize ONNX runtime for CPU
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2  # Limit threads for Pi 3
            sess_options.intra_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            self.model_type = "custom"
            self.class_names = ['stop_sign', 'custom_object_1', 'custom_object_2']
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        else:
            # Load YOLOv8n model with optimizations
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8n pretrained model")
                self.yolo_model = YOLO('yolov8n.pt')
                
                # Configure for lower resource usage
                self.yolo_model.overrides['imgsz'] = self.input_size
                self.yolo_model.overrides['half'] = False  # Disable FP16 on Pi
                self.yolo_model.overrides['device'] = 'cpu'
                
                self.model_type = "yolov8"
                # Reduced class list for better performance
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
                    'traffic light', 'stop sign', 'bench', 'bird', 'cat', 'dog', 
                    'backpack', 'umbrella', 'handbag', 'bottle', 'cup', 'chair', 
                    'couch', 'potted plant', 'tv', 'laptop', 'cell phone', 'book'
                ]
            except ImportError:
                logger.error("Ultralytics not available and no custom model provided")
                self.model_type = "none"
                self.class_names = []
        
        logger.info(f"Detector initialized with {len(self.class_names)} classes")
        logger.info(f"Input size: {self.input_size}x{self.input_size}")
    
    def preprocess_frame_onnx(self, frame):
        """Prepare frame for ONNX YOLO inference - optimized"""
        # Resize to smaller input size
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Normalize efficiently
        input_tensor = resized.astype(np.float32)
        input_tensor /= 255.0
        
        # Convert HWC to CHW and add batch dimension
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess_onnx_detections(self, outputs, original_shape):
        """Convert ONNX model outputs to bounding boxes - optimized"""
        predictions = outputs[0][0]  # Remove batch dimension
        
        detections = []
        h, w = original_shape[:2]
        
        # Vectorized operations for better performance
        confidences = predictions[:, 4]
        valid_indices = confidences > self.conf_threshold
        valid_predictions = predictions[valid_indices]
        
        for detection in valid_predictions:
            confidence = detection[4]
            
            # Get class scores
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id] * confidence
            
            if class_confidence > self.conf_threshold and class_id < len(self.class_names):
                # Convert from normalized coordinates
                x_center, y_center, width, height = detection[:4]
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                
                # Convert to corner coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Clamp to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': class_confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return detections
    
    def detect_objects(self, frame):
        """Run inference on frame - optimized"""
        if self.model_type == "custom":
            # Use custom ONNX model
            input_tensor = self.preprocess_frame_onnx(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            detections = self.postprocess_onnx_detections(outputs, frame.shape)
        elif self.model_type == "yolov8":
            # Use YOLOv8 model with optimizations
            results = self.yolo_model(
                frame, 
                verbose=False, 
                imgsz=self.input_size,
                conf=self.conf_threshold,
                max_det=10  # Limit max detections for performance
            )
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if confidence > self.conf_threshold and class_id < len(self.class_names):
                            detections.append({
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'confidence': float(confidence),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
        else:
            detections = []
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame - optimized"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Choose color based on class
            if 'stop' in class_name.lower():
                color = (0, 0, 255)  # Red for stop signs
            elif 'custom' in class_name.lower():
                color = (255, 0, 0)  # Blue for custom objects
            else:
                color = (0, 255, 0)  # Green for other objects
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with smaller font for performance
            label = f"{class_name}: {confidence:.2f}"
            font_scale = 0.4
            thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame

def camera_capture_thread():
    """Optimized camera capture using OpenCV VideoCapture"""
    global current_frame, running
    
    logger.info("Initializing camera with OpenCV...")
    
    # Try different camera backends for Pi
    backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    cap = None
    
    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                logger.info(f"Camera opened with backend: {backend}")
                break
        except:
            continue
    
    if cap is None or not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Optimize camera settings for Pi 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)            # Lower FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    try:
        while running:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                with frame_lock:
                    current_frame = frame.copy()
            else:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
            
            # Moderate capture rate
            time.sleep(0.05)  # ~20 FPS capture, but detection will be slower
            
    except Exception as e:
        logger.error(f"Camera thread error: {e}")
    finally:
        if cap:
            cap.release()

def generate_frames():
    """Generator yielding JPEG-encoded frames with detections - optimized"""
    global current_frame, detector, frame_counter
    
    last_detection_time = 0
    last_detections = []
    detection_interval = 0.5  # Run detection every 0.5 seconds
    
    while running:
        frame = None
        
        # Get current frame
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
        
        if frame is not None:
            try:
                current_time = time.time()
                
                # Only run detection periodically to save CPU
                if (detector and detector.model_type != "none" and 
                    current_time - last_detection_time > detection_interval):
                    
                    detections = detector.detect_objects(frame)
                    last_detections = detections
                    last_detection_time = current_time
                    
                    # Log detections less frequently
                    if detections:
                        det_summary = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections[:3]]
                        logger.info(f"Detections: {', '.join(det_summary)}")
                
                # Always draw the last detections
                if last_detections:
                    frame = detector.draw_detections(frame, last_detections)
                
                # Add status text with smaller font
                status_text = f"Model: {detector.model_type if detector else 'none'}"
                cv2.putText(frame, status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add timestamp
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (5, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                cv2.putText(frame, "Detection Error", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # Create placeholder frame
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            text = "Waiting for camera..."
            cv2.putText(frame, text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Encode frame to JPEG with lower quality for speed
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control stream rate
        time.sleep(0.1)  # ~10 FPS stream

@app.route('/')
def index():
    """Render the main page"""
    model_status = detector.model_type if detector else "none"
    return render_template('index.html', model_status=model_status)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    logger.info("Shutting down...")
    running = False
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Optimized Object Detection Camera Stream Server for Raspberry Pi'
    )
    parser.add_argument(
        '--port', type=int, default=8000,
        help='Port to run the server on (default: 8000)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to custom ONNX model (optional)'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--input-size', type=int, default=320,
        help='Input size for model (default: 320 for Pi 3, try 416 or 640 for Pi 4)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Initialize detector with optimizations
    logger.info("Initializing optimized object detector...")
    detector = YOLODetector(args.model, args.confidence, args.input_size)
    
    # Start camera thread
    logger.info("Starting optimized camera capture thread...")
    camera_thread = threading.Thread(target=camera_capture_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Wait for camera to initialize
    time.sleep(2)
    
    logger.info(f"Starting optimized detection stream server on port {args.port}")
    logger.info(f"Model type: {detector.model_type}")
    logger.info(f"Input size: {args.input_size}x{args.input_size}")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}")
    
    try:
        # Use a more efficient WSGI server for production
        app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        running = False