#!/usr/bin/env python3

from flask import Flask, render_template, Response, jsonify, request
import atexit
import signal
import sys
import socket
from robot_controller import robot
from camera_manager import camera

# Initialize Flask app
app = Flask(__name__)

# Global state for tracking commands
last_command = "none"
command_count = 0

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address (doesn't actually send data)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        try:
            # Fallback method
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip.startswith("127."):
                # If we get localhost, try getting all addresses
                import subprocess
                result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                if result.returncode == 0:
                    addresses = result.stdout.strip().split()
                    if addresses:
                        local_ip = addresses[0]
            return local_ip
        except Exception:
            return "localhost"

def cleanup_handler():
    """Clean up resources on exit"""
    print("\nShutting down...")
    robot.cleanup()
    camera.cleanup()

# Register cleanup handlers
atexit.register(cleanup_handler)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

@app.route('/')
def index():
    """Main control page"""
    return render_template('control.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with autonomous processing"""
    def generate():
        while True:
            # Get raw frame from camera
            frame = camera.get_frame()
            
            # Process frame for autonomous features if enabled
            processed_frame = robot.process_autonomous_frame(frame)
            
            # Convert to JPEG for streaming
            import cv2
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
            else:
                # Fallback to camera's built-in encoding
                frame_bytes = camera.get_jpeg_frame()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =============================================================================
# AUTONOMOUS CONTROL ROUTES (NEW)
# =============================================================================

@app.route('/start_autonomous')
def start_autonomous():
    """Start autonomous line following mode"""
    success, message = robot.start_autonomous_mode()
    global last_command
    last_command = "autonomous started" if success else "autonomous failed"
    
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/stop_autonomous')
def stop_autonomous():
    """Stop autonomous mode"""
    success, message = robot.stop_autonomous_mode()
    global last_command
    last_command = "autonomous stopped"
    
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/autonomous_status')
def get_autonomous_status():
    """Get detailed autonomous system status"""
    return jsonify(robot.get_feature_status())

# =============================================================================
# EXISTING MANUAL CONTROL ROUTES (unchanged)
# =============================================================================

@app.route('/move/<direction>')
def move_robot(direction):
    """Handle movement commands"""
    global last_command, command_count
    
    # Don't allow manual control during autonomous mode
    if robot.autonomous_mode:
        return jsonify({
            'success': False,
            'message': 'Manual control disabled during autonomous mode',
            'command_count': command_count
        })
    
    command_count += 1
    success = False
    
    try:
        if direction == 'forward':
            success = robot.move_forward(duration=0.5, speed=50)
            last_command = "forward"
            
        elif direction == 'backward':
            success = robot.move_backward(duration=0.5, speed=50)
            last_command = "backward"
            
        elif direction == 'left':
            success = robot.turn_left(duration=0.5, speed=50, angle=-30)
            last_command = "turn left"
            
        elif direction == 'right':
            success = robot.turn_right(duration=0.5, speed=50, angle=30)
            last_command = "turn right"
            
        elif direction == 'stop':
            robot.emergency_stop()
            last_command = "emergency stop"
            success = True
            
        else:
            return jsonify({
                'success': False, 
                'message': f'Unknown direction: {direction}',
                'command_count': command_count
            })
        
        return jsonify({
            'success': success,
            'direction': direction,
            'message': f'Command {direction} {"executed" if success else "failed"}',
            'command_count': command_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error executing {direction}: {str(e)}',
            'command_count': command_count
        })

@app.route('/status')
def get_status():
    """Get current robot status"""
    return jsonify({
        'last_command': last_command,
        'command_count': command_count,
        'is_moving': robot.is_moving if robot.picar else False,
        'robot_connected': robot.picar is not None,
        'autonomous_mode': robot.autonomous_mode
    })

@app.route('/test')
def test_robot():
    """Test basic robot functionality"""
    if robot.picar is None:
        return jsonify({
            'success': False,
            'message': 'Robot not connected'
        })
    
    if robot.autonomous_mode:
        return jsonify({
            'success': False,
            'message': 'Cannot test during autonomous mode'
        })
    
    try:
        # Simple test sequence
        robot.move_forward(duration=0.2, speed=30)
        return jsonify({
            'success': True,
            'message': 'Test movement completed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Test failed: {str(e)}'
        })

if __name__ == '__main__':
    try:
        # Start camera streaming
        camera.start_streaming()
        
        # Get the actual IP address
        local_ip = get_local_ip()
        
        print("Starting Flask server...")
        print("Open http://localhost:5000 in your browser")
        print(f"Or from another device: http://{local_ip}:5000")
        print("Use WASD keys or buttons to control the robot")
        print("Click 'Follow Line' to start autonomous mode")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down due to keyboard interrupt")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        cleanup_handler()