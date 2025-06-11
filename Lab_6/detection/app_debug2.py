import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time

def load_model(model_path):
    """Load ONNX model and return session"""
    try:
        session = ort.InferenceSession(model_path)
        print(f"✓ Model loaded successfully: {model_path}")
        
        # DEBUG: Model specifications
        print(f"DEBUG - Model inputs: {[input.name for input in session.get_inputs()]}")
        print(f"DEBUG - Model input shapes: {[input.shape for input in session.get_inputs()]}")
        print(f"DEBUG - Model outputs: {[output.name for output in session.get_outputs()]}")
        print(f"DEBUG - Model output shapes: {[output.shape for output in session.get_outputs()]}")
        
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image, input_size):
    """Preprocess image for YOLO model"""
    original_height, original_width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create a square canvas and center the image
    canvas = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
    x_offset = (input_size - new_width) // 2
    y_offset = (input_size - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    
    # Convert to RGB and normalize
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = canvas.astype(np.float32) / 255.0
    
    # Add batch dimension and change to CHW format
    input_tensor = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]
    
    return input_tensor, scale, x_offset, y_offset

def detect_objects(session, image, input_size=640, confidence_threshold=0.5, nms_threshold=0.4):
    """Run object detection on image"""
    input_tensor, scale, x_offset, y_offset = preprocess_image(image, input_size)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    predictions = outputs[0]
    
    # DEBUG: Raw model output analysis
    print(f"DEBUG - Raw output shapes: {[output.shape for output in outputs]}")
    print(f"DEBUG - Raw output ranges: {[f'min={output.min():.4f}, max={output.max():.4f}' for output in outputs]}")
    print(f"DEBUG - Predictions shape: {predictions.shape}")
    
    # Check if we need to transpose for YOLOv8 format
    original_shape = predictions.shape
    if len(predictions.shape) == 3 and predictions.shape[1] < predictions.shape[2]:
        # Likely YOLOv8 format (1, classes+coords, anchors) -> transpose to (1, anchors, classes+coords)
        predictions = predictions.transpose(0, 2, 1)
        print(f"DEBUG - Transposed shape from {original_shape} to {predictions.shape}")
    
    # DEBUG: Check some raw prediction values
    if len(predictions.shape) == 3:
        print(f"DEBUG - Sample raw predictions (first 5 detections, first 8 values):")
        print(predictions[0, :5, :8])
        
        # Check if any values are above confidence threshold
        if predictions.shape[2] > 4:  # Has confidence scores
            confidence_scores = predictions[0, :, 4]
            max_conf = confidence_scores.max()
            above_threshold = (confidence_scores > confidence_threshold).sum()
            print(f"DEBUG - Max confidence score: {max_conf:.4f}")
            print(f"DEBUG - Detections above threshold {confidence_threshold}: {above_threshold}")
    
    detections = []
    
    if len(predictions.shape) != 3:
        print(f"DEBUG - Unexpected prediction shape: {predictions.shape}")
        return detections
    
    # Process predictions
    for i in range(predictions.shape[1]):
        # Extract prediction
        prediction = predictions[0, i]
        
        # Check if we have enough elements
        if len(prediction) < 5:
            continue
            
        # Get confidence score
        confidence = prediction[4]
        
        # DEBUG: Show first few predictions regardless of threshold
        if i < 5:
            print(f"DEBUG - Prediction {i}: conf={confidence:.4f}, coords=[{prediction[0]:.2f}, {prediction[1]:.2f}, {prediction[2]:.2f}, {prediction[3]:.2f}]")
            if len(prediction) > 5:
                class_scores = prediction[5:]
                max_class_idx = np.argmax(class_scores)
                max_class_score = class_scores[max_class_idx]
                print(f"DEBUG - Prediction {i}: class_id={max_class_idx}, class_score={max_class_score:.4f}")
        
        if confidence < confidence_threshold:
            continue
            
        # Get class scores
        class_scores = prediction[5:] if len(prediction) > 5 else []
        if len(class_scores) == 0:
            continue
            
        class_id = np.argmax(class_scores)
        class_confidence = class_scores[class_id]
        
        # DEBUG: Log detections that pass confidence threshold
        print(f"DEBUG - PASSED CONFIDENCE: Detection {i}, conf={confidence:.4f}, class_id={class_id}, class_conf={class_confidence:.4f}")
        
        # Get bounding box coordinates (center format)
        x_center, y_center, width, height = prediction[:4]
        
        # Convert to corner format and scale back to original image
        x1 = (x_center - width / 2 - x_offset / input_size) / scale
        y1 = (y_center - height / 2 - y_offset / input_size) / scale
        x2 = (x_center + width / 2 - x_offset / input_size) / scale
        y2 = (y_center + height / 2 - y_offset / input_size) / scale
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'class_id': class_id,
            'class_confidence': class_confidence
        })
    
    # DEBUG: Pre-NMS detection count
    print(f"DEBUG - Detections before NMS: {len(detections)}")
    
    # Apply Non-Maximum Suppression
    if len(detections) > 0:
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]
            print(f"DEBUG - Detections after NMS: {len(detections)}")
        else:
            detections = []
            print(f"DEBUG - No detections survived NMS")
    
    return detections

def draw_detections(image, detections, class_names=None):
    """Draw bounding boxes and labels on image"""
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Use class name if available, otherwise use class ID
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Class {class_id}: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--model', type=str, default='yolov8n.onnx', help='Path to ONNX model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    print(f"DEBUG - Starting with parameters:")
    print(f"DEBUG - Model: {args.model}")
    print(f"DEBUG - Confidence threshold: {args.confidence}")
    print(f"DEBUG - NMS threshold: {args.nms}")
    print(f"DEBUG - Input size: {args.input_size}")
    
    # Load model
    session = load_model(args.model)
    if session is None:
        return
    
    # Define class names (customize based on your model)
    class_names = ['Stop_Sign', 'TU_Logo', 'Stahp', 'Falling_Cows']  # Your custom classes
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # DEBUG: Camera properties
    print(f"DEBUG - Camera initialized:")
    print(f"DEBUG - Camera width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"DEBUG - Camera height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"DEBUG - Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            detections = detect_objects(session, frame, args.input_size, args.confidence, args.nms)
            inference_time = time.time() - start_time
            
            # DEBUG: Per-frame info (only for first few frames to avoid spam)
            if frame_count <= 3:
                print(f"DEBUG - Frame {frame_count}: {len(detections)} detections, inference time: {inference_time:.3f}s")
            
            # Draw detections
            frame = draw_detections(frame, detections, class_names)
            
            # Add FPS counter
            fps = 1.0 / inference_time if inference_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLO Detection', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"DEBUG - Total frames processed: {frame_count}")

if __name__ == '__main__':
    main()