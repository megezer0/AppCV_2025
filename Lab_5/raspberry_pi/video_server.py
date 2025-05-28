#!/usr/bin/env python3
"""
Simple MJPEG video streaming server for Raspberry Pi
Students connect to this to get camera feed for their laptops
"""

import cv2
from flask import Flask, Response
import threading
import time
import os

app = Flask(__name__)

class VideoCamera:
    def __init__(self):
        # Try different camera backends for RPi compatibility
        self.video = None
        
        # Method 1: Try V4L2 backend (most reliable on RPi)
        try:
            self.video = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if self.video.isOpened():
                print("Using V4L2 backend")
            else:
                raise Exception("V4L2 failed")
        except:
            # Method 2: Try default backend
            try:
                self.video = cv2.VideoCapture(0)
                if self.video.isOpened():
                    print("Using default backend")
                else:
                    raise Exception("Default backend failed")
            except:
                print("ERROR: Cannot open camera!")
                return
        
        # Set lower resolution and framerate for RPi
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.video.set(cv2.CAP_PROP_FPS, 15)
        
        # Set buffer size to prevent lag
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"Camera resolution: {int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"Camera FPS: {self.video.get(cv2.CAP_PROP_FPS)}")
        
    def __del__(self):
        if self.video:
            self.video.release()
        
    def get_frame(self):
        if not self.video or not self.video.isOpened():
            return None
            
        success, image = self.video.read()
        if not success:
            print("Failed to read frame")
            return None
        
        # Resize if needed (for consistency)
        image = cv2.resize(image, (320, 240))
        
        # Encode frame as JPEG with lower quality for better streaming
        ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return jpeg.tobytes()

camera = VideoCamera()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.067)  # ~15 FPS

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    if camera.video and camera.video.isOpened():
        return {'status': 'Camera server running', 'resolution': '320x240', 'fps': 15}
    else:
        return {'status': 'Camera error', 'error': 'Camera not accessible'}, 500

@app.route('/')
def index():
    return '''
    <h1>Raspberry Pi Camera Server</h1>
    <p>Camera feed available at: <a href="/video">/video</a></p>
    <p>Status check: <a href="/status">/status</a></p>
    <img src="/video" width="320" height="240">
    '''

if __name__ == '__main__':
    if not camera.video or not camera.video.isOpened():
        print("ERROR: Camera not available!")
        print("Try these troubleshooting steps:")
        print("1. Check camera connection: ls /dev/video*")
        print("2. Check camera permissions: sudo usermod -a -G video $USER")
        print("3. Reboot the Raspberry Pi")
        exit(1)
        
    print("Starting Raspberry Pi camera server...")
    print("Students can connect using: http://cvpi33.local:5000/video")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)