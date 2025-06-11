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
frame_counter = 0

class YOLODetector:
    def __init__(self, model_path=None, confidence_threshold=0.5, input_size=320):
        self.conf_threshold = confidence_threshold
        self.requested_input_size = input_size  # User requested size
        
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
            
            # Extract actual input size from model (override user setting for custom models)
            if len(self.input_shape) >= 4:
                model_input_height = self.input_shape[2]
                model_input_width = self.input_shape[3]
                
                # Use the model's required input size
                self.input_size = model_input_width  # Assuming square input
                
                if self.input_size != self.requested_input_size:
                    logger.warning(f"Model requires {self.input_size}x{self.input_size} input, overriding requested {self.requested_input_size}x{self.requested_input_size}")
                    if self.input_size > 416:
                        logger.warning("Large input size detected - this may be slow on Pi 3. Consider using a model trained with smaller input size.")
                
                logger.info(f"Model input shape: {self.input_shape}")
                logger.info(f"Using input size: {self.input_size}x{self.input_size}")
            else:
                logger.error(f"Unexpected input shape: {self.input_shape}")
                self.input_size = 640  # Default fallback
                
        else:
            # Load YOLOv8n model with optimizations
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8n pretrained model")
                self.yolo_model = YOLO('yolov8n.pt')
                
                # Use requested input size for YOLOv8 (it's flexible)
                self.input_size = self.requested_input_size
                
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
    """Multi-method camera capture with fallbacks"""
    global current_frame, running
    
    # Method 1: Try OpenCV VideoCapture (works with USB cameras)
    logger.info("Trying OpenCV VideoCapture...")
    success = try_opencv_camera()
    
    if success:
        return
    
    # Method 2: Try libcamera-vid streaming (works with Pi Camera)
    logger.info("OpenCV failed, trying libcamera-vid streaming...")
    success = try_libcamera_streaming()
    
    if success:
        return
        
    # Method 3: Fallback to optimized libcamera-still
    logger.info("Streaming failed, using optimized libcamera-still...")
    try_libcamera_still_optimized()

def try_opencv_camera():
    """Try OpenCV VideoCapture method"""
    global current_frame, running
    
    # Try different camera backends
    backends_to_try = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    cap = None
    
    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"OpenCV camera working with backend: {backend}")
                    break
                else:
                    cap.release()
                    cap = None
        except:
            if cap:
                cap.release()
            cap = None
            continue
    
    if cap is None or not cap.isOpened():
        return False
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    logger.info("Using OpenCV VideoCapture")
    
    try:
        consecutive_failures = 0
        while running:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                with frame_lock:
                    current_frame = frame.copy()
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    logger.error("Too many consecutive frame failures, OpenCV method failed")
                    cap.release()
                    return False
                time.sleep(0.1)
            
            time.sleep(0.05)
            
    except Exception as e:
        logger.error(f"OpenCV camera error: {e}")
        cap.release()
        return False
    finally:
        if cap:
            cap.release()
    
    return True

def try_libcamera_streaming():
    """Try libcamera-vid with ffmpeg streaming"""
    global current_frame, running
    
    try:
        import subprocess
        
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
            logger.warning("libcamera-vid failed to start")
            return False
        
        logger.info("Using libcamera-vid streaming")
        
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
        return True
        
    except Exception as e:
        logger.error(f"libcamera streaming setup failed: {e}")
        return False

def try_libcamera_still_optimized():
    """Optimized libcamera-still method"""
    global current_frame, running
    
    import tempfile
    import subprocess
    import os
    
    # Create persistent temp file (don't recreate each time)
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, "frame.jpg")
    
    logger.info(f"Using optimized libcamera-still with temp file: {temp_image_path}")
    
    try:
        while running:
            # Faster capture with minimal settings
            cmd = [
                "libcamera-still",
                "-n",  # No preview
                "--width", "320",
                "--height", "240",
                "--timeout", "50",  # Very quick capture
                "--immediate",  # Don't wait for auto-settings
                "-q", "80",  # Lower quality for speed
                "-o", temp_image_path
            ]
            
            try:
                # Use faster method with no output capture
                result = subprocess.run(cmd, capture_output=False, timeout=1, 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if result.returncode == 0 and os.path.exists(temp_image_path):
                    # Read the image
                    frame = cv2.imread(temp_image_path)
                    if frame is not None:
                        with frame_lock:
                            current_frame = frame.copy()
                
            except subprocess.TimeoutExpired:
                logger.warning("libcamera-still timeout")
            except Exception as e:
                logger.error(f"Capture error: {e}")
            
            # Faster cycle time
            time.sleep(0.15)  # ~6-7 FPS capture
            
    except Exception as e:
        logger.error(f"libcamera-still error: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        os.rmdir(temp_dir)

def generate_frames():
    """Generator yielding JPEG-encoded frames with detections - optimized"""
    global current_frame, detector, frame_counter
    
    last_detection_time = 0
    last_detections = []
    
    # Adjust detection interval based on input size for performance
    if detector and detector.input_size >= 640:
        detection_interval = 1.0  # 1 second for large models
        logger.info("Using 1.0s detection interval for large input size")
    elif detector and detector.input_size >= 416:
        detection_interval = 0.75  # 0.75 seconds for medium models
        logger.info("Using 0.75s detection interval for medium input size")
    else:
        detection_interval = 0.5  # 0.5 seconds for small models
        logger.info("Using 0.5s detection interval for small input size")
    
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
        help='Input size for YOLOv8 model (default: 320). Custom ONNX models use their trained input size automatically.'
    )
    return parser.parse_args()

def check_camera_availability():
    """Check what cameras are available"""
    logger.info("Checking camera availability...")
    
    # Check for video devices
    video_devices = []
    for i in range(5):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if video_devices:
        logger.info(f"Found video devices: {video_devices}")
    else:
        logger.warning("No /dev/video* devices found")
    
    # Check if libcamera is available
    try:
        result = subprocess.run(["libcamera-hello", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("libcamera is available")
        else:
            logger.warning("libcamera-hello failed")
    except:
        logger.warning("libcamera tools not found")
    
    # Check if camera module is detected
    try:
        result = subprocess.run(["vcgencmd", "get_camera"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info(f"Camera status: {result.stdout.strip()}")
        else:
            logger.warning("Could not check camera status")
    except:
        logger.warning("vcgencmd not available")

def test_camera_methods():
    """Test different camera capture methods to see which works"""
    logger.info("Testing camera methods...")
    
    # Test 1: Quick libcamera test
    try:
        result = subprocess.run([
            "libcamera-still", "-n", "--timeout", "100", 
            "--width", "320", "--height", "240", "-o", "/tmp/test_camera.jpg"
        ], capture_output=True, timeout=3)
        
        if result.returncode == 0 and os.path.exists("/tmp/test_camera.jpg"):
            logger.info("✓ libcamera-still works")
            os.remove("/tmp/test_camera.jpg")
        else:
            logger.warning("✗ libcamera-still failed")
    except:
        logger.warning("✗ libcamera-still not available or failed")
    
    # Test 2: OpenCV test
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info("✓ OpenCV VideoCapture works")
            else:
                logger.warning("✗ OpenCV VideoCapture opens but can't read frames")
            cap.release()
        else:
            logger.warning("✗ OpenCV VideoCapture failed to open")
    except:
        logger.warning("✗ OpenCV VideoCapture exception")

if __name__ == '__main__':
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # Check camera availability first
    check_camera_availability()
    test_camera_methods()
    
    # Initialize detector with optimizations
    logger.info("Initializing optimized object detector...")
    detector = YOLODetector(args.model, args.confidence, args.input_size)
    
    # Start camera thread
    logger.info("Starting optimized camera capture thread...")
    camera_thread = threading.Thread(target=camera_capture_thread)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Wait for camera to initialize
    time.sleep(5)  # Give more time for initialization
    
    logger.info(f"Starting optimized detection stream server on port {args.port}")
    logger.info(f"Model type: {detector.model_type}")
    logger.info(f"Actual input size: {detector.input_size}x{detector.input_size}")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}")
    
    # Performance recommendations
    if detector.input_size >= 640:
        logger.info("💡 Performance tip: Detection runs every 1.0 second due to large input size")
    elif detector.input_size >= 416:
        logger.info("💡 Performance tip: Detection runs every 0.75 seconds due to medium input size")
    else:
        logger.info("💡 Performance tip: Detection runs every 0.5 seconds for optimal speed")
    
    try:
        # Use a more efficient WSGI server for production
        app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        running = False