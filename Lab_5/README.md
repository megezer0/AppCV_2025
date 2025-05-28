# Lab 5: Human Pose Estimation with MediaPipe

## Overview
In this lab, you'll explore human pose estimation using computer vision. You'll work with MediaPipe, Google's framework for building perception pipelines, to detect and analyze human poses in real-time video streams.

## Learning Objectives
- Understand keypoint-based pose estimation
- Work with real-time computer vision data
- Implement gesture and pose recognition algorithms
- Handle temporal aspects of motion detection

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Webcam or camera connected to your computer

### Installation
1. Navigate to the lab directory:
   ```bash
   cd lab_5
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Lab Structure

### Phase 1: Familiarization (15-20 minutes)
Start by running the familiarization scripts to understand MediaPipe's output:

```bash
# Face detection with keypoints
python familiarization/face_detection.py

# Hand pose estimation  
python familiarization/hand_pose.py

# Full body pose estimation
python familiarization/full_body_pose.py
```

Each script will:
- Display your camera feed with keypoints overlaid
- Print keypoint coordinates to the console
- Show FPS performance in the top-left corner

**[TEACHER NOTE: Insert MediaPipe keypoint diagram here showing face landmarks numbered]**

**[TEACHER NOTE: Insert MediaPipe hand keypoint diagram showing finger joint numbering]**

**[TEACHER NOTE: Insert MediaPipe pose keypoint diagram showing body joint numbering]**

Press 'q' to quit any script.

### Phase 2: Challenges
After familiarization, move to the challenges directory and work through the pose estimation challenges. See `challenges/README.md` for detailed instructions.

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Try changing the camera index in the code (0, 1, 2, etc.)
- Check that your camera permissions are enabled

### Performance Issues
- Close other applications that might be using your camera
- Ensure adequate lighting for better detection
- Lower the camera resolution if needed

### Import Errors
- Verify all packages are installed: `pip list | grep -E "(opencv|mediapipe)"`
- Try reinstalling: `pip uninstall opencv-python mediapipe && pip install opencv-python mediapipe`

## Getting Help
- Work in pairs and discuss challenges with your partner
- Ask the instructor for help when stuck
- Pay attention to the console output for debugging information