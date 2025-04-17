import argparse
import io
import time
import logging
from flask import Flask, render_template, Response

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

def get_camera_instance():
    """
    Try to get a camera instance using picamera or OpenCV as a fallback
    """
    try:
        import picamera
        from picamera.array import PiRGBArray
        logger.info("Using PiCamera module")
        
        camera = picamera.PiCamera()
        camera.resolution = (854, 480)
        camera.framerate = 20
        # Allow time for the camera to warm up
        time.sleep(2)
        return camera, "picamera"
    
    except (ImportError, ModuleNotFoundError):
        try:
            import cv2
            logger.info("PiCamera module not found, using OpenCV instead")
            
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 20)
            return camera, "opencv"
        except:
            logger.error("Failed to initialize camera with OpenCV")
            return None, None

def generate_frames():
    """
    Generator function that yields frames from the camera
    """
    camera, camera_type = get_camera_instance()
    
    if camera is None:
        logger.error("No camera available. Please check your hardware and software installation.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('static/no_camera.jpg', 'rb').read() + b'\r\n')
        return
    
    try:
        if camera_type == "picamera":
            # PiCamera streaming
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                frame = stream.read()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                # Reset stream for next frame
                stream.seek(0)
                stream.truncate()
                
        elif camera_type == "opencv":
            # OpenCV streaming
            import cv2
            
            while True:
                success, frame = camera.read()
                if not success:
                    break
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Short delay to control frame rate
                time.sleep(0.01)
    
    finally:
        # Clean up camera resources
        if camera_type == "picamera":
            camera.close()
        elif camera_type == "opencv":
            camera.release()

@app.route('/')
def index():
    """
    Render the main page with video stream
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Route for the actual video stream
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Raspberry Pi Camera Stream Server')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on (default: 8000)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting camera stream server on port {args.port}")
    logger.info(f"Access the stream by navigating to http://[YOUR_PI_IP]:{args.port}")
    
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")