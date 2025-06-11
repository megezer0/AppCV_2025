#!/usr/bin/env python3
import argparse
import time
import logging
from flask import Flask, render_template, Response
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and camera
detector = None
camera_available = False
picam2 = None

class YOLODetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.conf_threshold = confidence_threshold
        
        if model_path and Path(model_path).exists():
            # Load custom ONNX model
            logger.info(f"Loading custom model: {model_path}")
            self.session = ort.InferenceSession(model_path)
            self.model_type = "custom"
            # You'll need to adjust these based on your training
            self.class_names = ['stop_sign', 'custom_object_1', 'custom_object_2']
        else:
            # Load YOLOv8n model (you'll need to download this)
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8n pretrained model")
                self.yolo_model = YOLO('yolov8n.pt')
                self.model_type = "yolov8"
                # COCO dataset classes (subset relevant to our use case)
                self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
            except ImportError:
                logger.error("Ultralytics not available and no custom model provided")
                self.model_type = "none"
        
        if self.model_type == "custom":
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Custom model input shape: {self.input_shape}")
        
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

def init_camera():
    """Initialize camera"""
    global camera_available, picam2
    
    try:
        # Try picamera2 first (Raspberry Pi OS Bookworm)
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        camera_available = True
        logger.info("Picamera2 initialized successfully")
        return "picamera2"
    except Exception as e:
        logger.warning(f"Picamera2 failed: {e}")
        
    try:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_available = True
            picam2 = cap  # Store in same variable for simplicity
            logger.info("OpenCV camera initialized successfully")
            return "opencv"
        else:
            raise Exception("OpenCV camera not available")
    except Exception as e:
        logger.error(f"All camera initialization methods failed: {e}")
        camera_available = False
        return None

def generate_frames():
    """Generator yielding JPEG-encoded frames with detections"""
    global detector, camera_available, picam2
    
    camera_type = init_camera()
    
    while True:
        if camera_available:
            try:
                if camera_type == "picamera2":
                    # Capture from picamera2
                    frame = picam2.capture_array()
                    # Convert from RGB to BGR for OpenCV
                    if frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif camera_type == "opencv":
                    # Capture from OpenCV
                    ret, frame = picam2.read()
                    if not ret:
                        raise Exception("Failed to read from camera")
                else:
                    raise Exception("No camera available")
                
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
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if detector and detector.model_type != "none":
                    det_count = len(detector.detect_objects(frame)) if detector else 0
                    det_text = f"Objects detected: {det_count}"
                    cv2.putText(frame, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                # Create error frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                text = "Camera Error"
                cv2.putText(frame, text, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Create placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "No camera detected"
            cv2.putText(frame, text, (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.warning("Frame encoding failed")
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control frame rate
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
    args = parse_args()
    
    # Initialize detector
    logger.info("Initializing object detector...")
    detector = YOLODetector(args.model, args.confidence)
    
    logger.info(f"Starting detection stream server on port {args.port}")
    logger.info(f"Model type: {detector.model_type}")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}")
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Cleanup camera
        if camera_available and picam2:
            try:
                if hasattr(picam2, 'stop'):
                    picam2.stop()
                elif hasattr(picam2, 'release'):
                    picam2.release()
            except:
                pass