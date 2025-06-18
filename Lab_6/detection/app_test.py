#!/usr/bin/env python3
import argparse
import time
import logging
import subprocess
import threading
import os
import tempfile
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

class YOLODetector:
    def __init__(self, model_path=None, confidence_threshold=0.5, input_size=320):
        self.conf_threshold = confidence_threshold
        self.requested_input_size = input_size
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading custom model: {model_path}")
            
            # Optimize ONNX runtime for CPU
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            self.model_type = "custom"
            self.class_names = ['Stop_Sign', 'TU_Logo', 'Stahp', 'Falling_Cows']
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Extract actual input size from model
            if len(self.input_shape) >= 4:
                model_input_height = self.input_shape[2]
                model_input_width = self.input_shape[3]
                self.input_size = model_input_width
                
                if self.input_size != self.requested_input_size:
                    logger.warning(f"Model requires {self.input_size}x{self.input_size} input, overriding requested {self.requested_input_size}x{self.requested_input_size}")
                    if self.input_size > 416:
                        logger.warning("Large input size detected - this may be slow on Pi 3. Consider using a model trained with smaller input size.")
                
                logger.info(f"Model input shape: {self.input_shape}")
                logger.info(f"Using input size: {self.input_size}x{self.input_size}")
            else:
                logger.error(f"Unexpected input shape: {self.input_shape}")
                self.input_size = 640
                
        else:
            # Load YOLOv8n model with optimizations
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8n pretrained model")
                self.yolo_model = YOLO('yolov8n.pt')
                self.input_size = self.requested_input_size
                
                # Configure for lower resource usage
                self.yolo_model.overrides['imgsz'] = self.input_size
                self.yolo_model.overrides['half'] = False
                self.yolo_model.overrides['device'] = 'cpu'
                
                self.model_type = "yolov8"
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
                self.input_size = self.requested_input_size
        
        logger.info(f"Detector initialized with {len(self.class_names)} classes")
        logger.info(f"Final input size: {self.input_size}x{self.input_size}")
        
        # Performance warning for Pi 3
        if self.input_size >= 640:
            logger.warning("⚠️  Large input size may cause slow performance on Pi 3!")
            logger.warning("   Consider running with lower detection frequency or smaller model")
        elif self.input_size >= 416:
            logger.warning("⚠️  Medium input size - expect moderate performance on Pi 3")
    
    def preprocess_frame_onnx(self, frame):
        """Prepare frame for ONNX YOLO inference - optimized"""
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        input_tensor = resized.astype(np.float32)
        input_tensor /= 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor
    
    def postprocess_onnx_detections(self, outputs, original_shape):
        """
        Parse the Ultralytics-exported ONNX head:
            shape  : (1, 8, 8400)  --> B x (4 + nc) x N
            layout : [cx, cy, w, h, p0..p3]  (pixel coords, no obj score)
        Returns a list of dicts in xyxy image space.
        """
        # --- 1. reshape ----------------------------------------------------------
        pred = outputs[0]                       # (1, 8, 8400)
        pred = np.transpose(pred, (0, 2, 1))[0] # (8400, 8)

        xywh   = pred[:, :4]          # pixel-space box (cx,cy,w,h)
        scores = pred[:, 4:]          # per-class confidences (already multiplied!)

        conf   = scores.max(1)
        cls_id = scores.argmax(1)

        # --- 2. threshold --------------------------------------------------------
        keep = conf > self.conf_threshold
        xywh, conf, cls_id = xywh[keep], conf[keep], cls_id[keep]

        if not len(conf):
            return []

        # --- 3. cx,cy,w,h  →  x1,y1,x2,y2 ---------------------------------------
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2

        # --- 4. scale back to actual frame size ---------------------------------
        ih, iw = self.input_size, self.input_size
        H, W   = original_shape[:2]
        boxes[:, [0, 2]] *= W / iw
        boxes[:, [1, 3]] *= H / ih

        # --- 5. (optional) NMS ---------------------------------------------------
        idxs = cv2.dnn.NMSBoxes(
            boxes.tolist(), conf.tolist(),
            self.conf_threshold, 0.45)          # IoU threshold
        idxs = idxs.flatten() if len(idxs) else []

        detections = []
        for i in idxs:
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                'class_id'   : int(cls_id[i]),
                'class_name' : self.class_names[int(cls_id[i])],
                'confidence' : float(conf[i]),
                'bbox'       : [int(x1), int(y1), int(x2), int(y2)]
            })
        return detections
    
    def detect_objects(self, frame):
        """Run inference on frame - optimized"""
        if self.model_type == "custom":
            input_tensor = self.preprocess_frame_onnx(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            detections = self.postprocess_onnx_detections(outputs, frame.shape)
        elif self.model_type == "yolov8":
            results = self.yolo_model(
                frame, 
                verbose=False, 
                imgsz=self.input_size,
                conf=self.conf_threshold,
                max_det=10
            )
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
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
            
            if 'stop' in class_name.lower():
                color = (0, 0, 255)  # Red for stop signs
            elif 'custom' in class_name.lower():
                color = (255, 0, 0)  # Blue for custom objects
            else:
                color = (0, 255, 0)  # Green for other objects
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            font_scale = 0.4
            thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame

def camera_capture_thread():
    """Camera capture using libcamera-vid streaming (the only working method)"""
    global current_frame, running
    
    logger.info("Starting libcamera-vid streaming...")
    
    try:
        # Start libcamera-vid streaming to stdout
        cmd = [
            "libcamera-vid",
            "-t", "0",  # Infinite timeout
            "--width", "320",
            "--height", "240",
            "--framerate", "15",
            "-o", "-",  # Output to stdout
            "--codec", "mjpeg",
            "--inline",
            "-n"  # No preview
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Test if the process starts successfully
        time.sleep(1)
        if process.poll() is not None:
            logger.error("libcamera-vid failed to start")
            return
        
        logger.info("libcamera-vid streaming started successfully")
        
        buffer = b""
        while running:
            try:
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Look for JPEG boundaries
                start = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9')
                
                if start != -1 and end != -1 and end > start:
                    # Extract JPEG frame
                    jpeg_data = buffer[start:end+2]
                    buffer = buffer[end+2:]
                    
                    # Decode JPEG
                    nparr = np.frombuffer(jpeg_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        with frame_lock:
                            current_frame = frame.copy()
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break
        
        process.terminate()
        
    except Exception as e:
        logger.error(f"libcamera streaming setup failed: {e}")

def generate_frames():
    """Generator yielding JPEG-encoded frames with detections - optimized"""
    global current_frame, detector
    
    last_detection_time = 0
    last_detections = []
    
    # Adjust detection interval based on input size for performance
    if detector and detector.input_size >= 640:
        detection_interval = 1.0
        logger.info("Using 1.0s detection interval for large input size")
    elif detector and detector.input_size >= 416:
        detection_interval = 0.75
        logger.info("Using 0.75s detection interval for medium input size")
    else:
        detection_interval = 0.5
        logger.info("Using 0.5s detection interval for small input size")
    
    while running:
        frame = None
        
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
                    
                    if detections:
                        det_summary = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections[:3]]
                        logger.info(f"Detections: {', '.join(det_summary)}")
                
                # Always draw the last detections
                if last_detections:
                    frame = detector.draw_detections(frame, last_detections)
                
                # Add status text
                status_text = f"Model: {detector.model_type if detector else 'none'}"
                if detector:
                    status_text += f" ({detector.input_size}x{detector.input_size})"
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
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
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
        help='Input size for YOLOv8 model (default: 320). Custom ONNX models use their trained input size automatically.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Initialize detector
    logger.info("Initializing object detector...")
    detector = YOLODetector(args.model, args.confidence, args.input_size)
    
    # Start camera thread (directly using libcamera-vid)
    logger.info("Starting camera capture thread...")
    camera_thread = threading.Thread(target=camera_capture_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Wait for camera to initialize
    time.sleep(3)
    
    logger.info(f"Starting detection stream server on port {args.port}")
    logger.info(f"Model type: {detector.model_type}")
    logger.info(f"Input size: {detector.input_size}x{detector.input_size}")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}")
    
    # Performance recommendations
    if detector.input_size >= 640:
        logger.info("💡 Performance tip: Detection runs every 1.0 second due to large input size")
    elif detector.input_size >= 416:
        logger.info("💡 Performance tip: Detection runs every 0.75 seconds due to medium input size")
    else:
        logger.info("💡 Performance tip: Detection runs every 0.5 seconds for optimal speed")
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        running = False