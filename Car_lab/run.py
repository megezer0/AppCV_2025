#!/usr/bin/env python3

"""
Simple script to run the PiCar-X control application
"""

import sys
import subprocess
import time

def check_requirements():
    """Check if required modules are available"""
    required_modules = ['flask', 'cv2', 'picarx', 'vilib']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} - OK")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing modules: {', '.join(missing_modules)}")
        print("Please install missing modules before running.")
        return False
    
    return True

def main():
    print("=== PiCar-X Control Application ===\n")
    
    print("Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("\nStarting application...")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Run the Flask application
        from flask_control import app, camera
        
        # Start camera
        camera.start_streaming()
        time.sleep(1)
        
        print("\n🚀 Server starting...")
        print("📷 Camera initialized")
        print("🌐 Web interface available at:")
        print("   - Local: http://localhost:5000")
        print("   - Network: http://<your-pi-ip>:5000")
        print("\n🎮 Controls:")
        print("   - Use WASD keys or web buttons")
        print("   - Each movement lasts 0.5 seconds")
        print("   - Red button for emergency stop")
        print("\n" + "="*50)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()