#!/usr/bin/env python3

import threading
import time
from picarx import Picarx

class RobotController:
    def __init__(self):
        """Initialize the robot controller"""
        try:
            self.picar = Picarx()
            self.is_moving = False
            print("Robot controller initialized successfully")
        except Exception as e:
            print(f"Error initializing robot: {e}")
            self.picar = None
    
    def _auto_stop(self):
        """Automatically stop the robot and center wheels after movement"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)  # Center the wheels
            self.is_moving = False
            print("Robot auto-stopped and wheels centered")
    
    def move_forward(self, duration=0.5, speed=50):
        """Move robot forward for specified duration"""
        if not self.picar or self.is_moving:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)  # Straight ahead
            self.picar.forward(speed)
            print(f"Moving forward at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error moving forward: {e}")
            self._auto_stop()
            return False
    
    def move_backward(self, duration=0.5, speed=50):
        """Move robot backward for specified duration"""
        if not self.picar or self.is_moving:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)  # Straight back
            self.picar.backward(speed)
            print(f"Moving backward at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error moving backward: {e}")
            self._auto_stop()
            return False
    
    def turn_left(self, duration=0.5, speed=50, angle=-30):
        """Turn robot left while moving forward"""
        if not self.picar or self.is_moving:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)  # Turn wheels left
            self.picar.forward(speed)
            print(f"Turning left (angle {angle}) at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error turning left: {e}")
            self._auto_stop()
            return False
    
    def turn_right(self, duration=0.5, speed=50, angle=30):
        """Turn robot right while moving forward"""
        if not self.picar or self.is_moving:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)  # Turn wheels right
            self.picar.forward(speed)
            print(f"Turning right (angle {angle}) at speed {speed} for {duration}s")
            
            # Auto-stop after duration
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
            
        except Exception as e:
            print(f"Error turning right: {e}")
            self._auto_stop()
            return False
    
    def emergency_stop(self):
        """Immediately stop the robot"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
            self.is_moving = False
            print("Emergency stop activated")
    
    def cleanup(self):
        """Clean shutdown of robot"""
        if self.picar:
            self.emergency_stop()
            print("Robot controller cleaned up")

# Global robot instance
robot = RobotController()