#!/usr/bin/env python3
"""
Simple MJPEG video streaming server for Raspberry Pi
Students connect to this to get camera feed for their laptops
"""

import cv2
from flask import Flask, Response
import threading
import time

app = Flask(__name__)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return jpeg.tobytes()

camera = VideoCamera()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return {'status': 'Camera server running', 'resolution': '640x480'}

@app.route('/')
def index():
    return '''
    <h1>Raspberry Pi Camera Server</h1>
    <p>Camera feed available at: <a href="/video">/video</a></p>
    <p>Status check: <a href="/status">/status</a></p>
    <img src="/video" width="640" height="480">
    '''

if __name__ == '__main__':
    print("Starting Raspberry Pi camera server...")
    print("Students can connect using: http://cvpi33.local:5000/video")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)