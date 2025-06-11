#!/usr/bin/env python3
import argparse
import time
import logging
import subprocess
import threading
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
camera_process = None
current_frame = None
frame_lock = threading.Lock()
running = True

class YOLODetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.conf_threshold = confidence_threshold
        
        if model_path and Path(model_path).exists():
            # Load custom ONNX model
            logger.info(f"Loading custom model: {model_path}")
            self.session = ort.InferenceSession(model_path)
            self.model_type = "custom"
            self.class_names = ['stop_sign', 'custom_object_1', 'custom_object_2']
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        else:
            # Load YOLOv8n model
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8n pretrained model")
                self.yolo_model = YOLO('yolov8n.pt')
                self.model_type = "yolov8"
                # Subset of COCO classes most relevant for demonstration
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
            except ImportError:
                logger.error("Ultralytics not available and no custom model provided")
                self.model_type = "none"
                self.class_names = []
        
        logger.info(f"Detector initialized with {len(self.class_names)} classes")
    
    def preprocess_frame_onnx(self, frame):
        """Prepare frame for ONNX YOLO inference"""
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        
        # Resize and normalize
        resized = cv2.resize(frame, (input_width, input_height))
        input_tensor = resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW and add batch dimension
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess_onnx_detections(self, outputs, original_shape):
        """Convert ONNX model outputs to bounding boxes"""
        predictions = outputs[0][0]  # Remove batch dimension
        
        detections = []
        h, w = original_shape[:2]
        
        for detection in predictions:
            confidence = detection[4]
            if confidence > self.conf_threshold:
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
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': class_confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return detections
    
    def detect_objects(self, frame):
        """Run inference on frame"""
        if self.model_type == "custom":
            # Use custom ONNX model
            input_tensor = self.preprocess_frame_onnx(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            detections = self.postprocess_onnx_detections(outputs, frame.shape)
        elif self.model_type == "yolov8":
            # Use YOLOv8 model
            results = self.yolo_model(frame, verbose=False)
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
        """Draw bounding boxes on frame"""
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
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

def camera_capture_thread():
    """Background thread to capture frames from libcamera"""
    global current_frame, camera_process, running
    
    # Start libcamera-vid process
    camera_cmd = [
        "libcamera-vid",
        "-t", "0",  # Run indefinitely
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "-o", "-",  # Output to stdout
        "--codec", "mjpeg"
    ]
    
    try:
        camera_process = subprocess.Popen(
            camera_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        logger.info("Camera process started successfully")
        
        # Read MJPEG stream
        bytes_buffer = b''
        while running:
            chunk = camera_process.stdout.read(1024)
            if not chunk:
                break
                
            bytes_buffer += chunk
            
            # Look for JPEG boundaries
            start = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            end = bytes_buffer.find(b'\xff\xd9')    # JPEG end
            
            if start != -1 and end != -1 and end > start:
                # Extract complete JPEG
                jpeg_data = bytes_buffer[start:end+2]
                bytes_buffer = bytes_buffer[end+2:]
                
                # Decode JPEG
                try:
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with frame_lock:
                            current_frame = frame.copy()
                except Exception as e:
                    logger.warning(f"Frame decode error: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Camera thread error: {e}")
    finally:
        if camera_process:
            camera_process.terminate()

def generate_frames():
    """Generator yielding JPEG-encoded frames with detections"""
    global current_frame, detector
    
    while running:
        frame = None
        
        # Get current frame
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
        
        if frame is not None:
            try:
                # Run detection if model is loaded
                if detector and detector.model_type != "none":
                    detections = detector.detect_objects(frame)
                    frame = detector.draw_detections(frame, detections)
                    
                    # Log detections
                    if detections:
                        det_summary = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
                        logger.info(f"Detections: {', '.join(det_summary)}")
                
                # Add status text to frame
                status_text = f"Model: {detector.model_type if detector else 'none'}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
        else:
            # Create placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "Starting camera..."
            cv2.putText(frame, text, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)  # ~10 FPS

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
    global running, camera_process
    logger.info("Shutting down...")
    running = False
    if camera_process:
        camera_process.terminate()
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Object Detection Camera Stream Server'
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
    return parser.parse_args()

if __name__ == '__main__':
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Initialize detector
    logger.info("Initializing object detector...")
    detector = YOLODetector(args.model, args.confidence)
    
    # Start camera thread
    logger.info("Starting camera capture thread...")
    camera_thread = threading.Thread(target=camera_capture_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Wait a moment for camera to initialize
    time.sleep(2)
    
    logger.info(f"Starting detection stream server on port {args.port}")
    logger.info(f"Model type: {detector.model_type}")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}")
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        running = False
        if camera_process:
            camera_process.terminate()