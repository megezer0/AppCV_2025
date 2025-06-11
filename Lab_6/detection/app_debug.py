import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time

def preprocess_image(image, input_size):
    """Preprocess image for YOLO model"""
    print(f"[DEBUG] Input image shape: {image.shape}")
    print(f"[DEBUG] Input image dtype: {image.dtype}")
    print(f"[DEBUG] Input image range: [{image.min()}, {image.max()}]")
    
    # Get original dimensions
    h, w = image.shape[:2]
    print(f"[DEBUG] Original dimensions: {w}x{h}")
    
    # Calculate scaling to maintain aspect ratio
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"[DEBUG] Scale factor: {scale}")
    print(f"[DEBUG] Scaled dimensions: {new_w}x{new_h}")
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    print(f"[DEBUG] Padding offsets: x={pad_x}, y={pad_y}")
    
    # Place resized image in center of padded image
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # Convert BGR to RGB
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    print(f"[DEBUG] After BGR->RGB conversion, range: [{padded_rgb.min()}, {padded_rgb.max()}]")
    
    # Normalize to [0, 1]
    normalized = padded_rgb.astype(np.float32) / 255.0
    print(f"[DEBUG] After normalization, range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Transpose to CHW format and add batch dimension
    input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
    print(f"[DEBUG] Final tensor shape: {input_tensor.shape}")
    print(f"[DEBUG] Final tensor dtype: {input_tensor.dtype}")
    
    return input_tensor, scale, pad_x, pad_y

def detect_objects(session, image, input_size=640, confidence_threshold=0.5, nms_threshold=0.4):
    """Detect objects using ONNX model"""
    print(f"\n[DEBUG] ==================== DETECTION START ====================")
    print(f"[DEBUG] Using confidence threshold: {confidence_threshold}")
    print(f"[DEBUG] Using NMS threshold: {nms_threshold}")
    print(f"[DEBUG] Using input size: {input_size}")
    
    # Preprocess image
    input_tensor, scale, pad_x, pad_y = preprocess_image(image, input_size)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    print(f"[DEBUG] Model input name: {input_name}")
    
    # Run inference
    print(f"[DEBUG] Running inference...")
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_time
    print(f"[DEBUG] Inference completed in {inference_time:.4f}s")
    
    # Analyze raw outputs
    print(f"[DEBUG] Number of output tensors: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"[DEBUG] Output {i} shape: {output.shape}")
        print(f"[DEBUG] Output {i} dtype: {output.dtype}")
        print(f"[DEBUG] Output {i} range: [{output.min():.4f}, {output.max():.4f}]")
        if len(output.shape) >= 2:
            print(f"[DEBUG] Output {i} first few values: {output.flat[:10]}")
    
    predictions = outputs[0]
    print(f"[DEBUG] Working with predictions shape: {predictions.shape}")
    
    # Handle YOLOv8 format (transpose if needed)
    original_shape = predictions.shape
    if len(predictions.shape) == 3:
        if predictions.shape[1] < predictions.shape[2]:  # Likely (1, classes+coords, anchors)
            print(f"[DEBUG] Detected YOLOv8 format - transposing from {predictions.shape}")
            predictions = predictions.transpose(0, 2, 1)
            print(f"[DEBUG] After transpose: {predictions.shape}")
        
    # Remove batch dimension
    if len(predictions.shape) == 3:
        predictions = predictions[0]
        print(f"[DEBUG] After removing batch dim: {predictions.shape}")
    
    # Analyze prediction structure
    num_detections = predictions.shape[0]
    num_values = predictions.shape[1] if len(predictions.shape) > 1 else 0
    print(f"[DEBUG] Number of detections: {num_detections}")
    print(f"[DEBUG] Values per detection: {num_values}")
    
    if num_values >= 8:  # At least x,y,w,h + 4 classes
        print(f"[DEBUG] Prediction structure analysis:")
        print(f"[DEBUG] - Coordinates (0-3): x,y,w,h")
        print(f"[DEBUG] - Classes (4+): confidence scores for each class")
        
        # Sample a few predictions
        for i in range(min(5, num_detections)):
            coords = predictions[i, :4]
            class_scores = predictions[i, 4:]
            max_class_idx = np.argmax(class_scores)
            max_class_score = class_scores[max_class_idx]
            print(f"[DEBUG] Detection {i}: coords={coords}, max_class={max_class_idx}, max_score={max_class_score:.4f}")
    
    detections = []
    confidence_count = 0
    nms_input_count = 0
    
    # Process predictions
    for i in range(num_detections):
        if num_values < 8:  # Need at least 4 coords + 4 classes
            continue
            
        # Extract coordinates (center format)
        x_center, y_center, width, height = predictions[i, :4]
        
        # Extract class scores (assuming they start from index 4)
        class_scores = predictions[i, 4:]
        class_id = np.argmax(class_scores)
        class_confidence = class_scores[class_id]
        
        # Debug first few detections
        if i < 5:
            print(f"[DEBUG] Raw detection {i}: center=({x_center:.2f},{y_center:.2f}), size=({width:.2f},{height:.2f}), class={class_id}, conf={class_confidence:.4f}")
        
        # Check confidence threshold
        if class_confidence > confidence_threshold:
            confidence_count += 1
            
            # Convert to corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Scale back to original image coordinates
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            detections.append([x1, y1, x2, y2, class_confidence, class_id])
            nms_input_count += 1
            
            if confidence_count <= 5:  # Debug first few valid detections
                print(f"[DEBUG] Valid detection {confidence_count}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), class={class_id}, conf={class_confidence:.4f}")
    
    print(f"[DEBUG] Detections above confidence threshold: {confidence_count}")
    print(f"[DEBUG] Detections going to NMS: {nms_input_count}")
    
    if len(detections) == 0:
        print(f"[DEBUG] No detections above confidence threshold {confidence_threshold}")
        return []
    
    # Apply Non-Maximum Suppression
    detections = np.array(detections)
    boxes = detections[:, :4]
    scores = detections[:, 4]
    
    print(f"[DEBUG] Running NMS with {len(boxes)} boxes")
    print(f"[DEBUG] Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), 
        scores.tolist(), 
        confidence_threshold, 
        nms_threshold
    )
    
    final_detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        print(f"[DEBUG] NMS kept {len(indices)} detections")
        
        for i in indices:
            x1, y1, x2, y2, confidence, class_id = detections[i]
            final_detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class_id': int(class_id)
            })
            print(f"[DEBUG] Final detection: bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), class={int(class_id)}, conf={confidence:.4f}")
    else:
        print(f"[DEBUG] NMS removed all detections")
    
    print(f"[DEBUG] ==================== DETECTION END ====================\n")
    return final_detections

def draw_detections(image, detections, class_names):
    """Draw bounding boxes and labels on image"""
    print(f"[DEBUG] Drawing {len(detections)} detections")
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        x1, y1, x2, y2 = bbox
        
        # Get class name
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class_{class_id}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        print(f"[DEBUG] Drew: {label} at ({x1},{y1},{x2},{y2})")
    
    return image

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--model', type=str, default='yolov8n.onnx', 
                       help='Path to ONNX model file')
    parser.add_argument('--input_size', type=int, default=640, 
                       help='Input image size')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, 
                       help='NMS threshold')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] ==================== MODEL LOADING ====================")
    print(f"[DEBUG] Loading model: {args.model}")
    print(f"[DEBUG] Input size: {args.input_size}")
    print(f"[DEBUG] Confidence threshold: {args.confidence}")
    print(f"[DEBUG] NMS threshold: {args.nms}")
    
    # Load ONNX model
    try:
        session = ort.InferenceSession(args.model)
        print(f"[DEBUG] Model loaded successfully!")
    except Exception as e:
        print(f"[DEBUG] ERROR loading model: {e}")
        return
    
    # Print model information
    print(f"[DEBUG] Model inputs:")
    for i, input_info in enumerate(session.get_inputs()):
        print(f"[DEBUG]   Input {i}: name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
    
    print(f"[DEBUG] Model outputs:")
    for i, output_info in enumerate(session.get_outputs()):
        print(f"[DEBUG]   Output {i}: name='{output_info.name}', shape={output_info.shape}, type={output_info.type}")
    
    # Define class names based on your model
    if 'best' in args.model.lower():  # Your custom model
        class_names = ['Stop_Sign', 'TU_Logo', 'Stahp', 'Falling_Cows']
        print(f"[DEBUG] Using custom class names: {class_names}")
    else:  # Default COCO classes
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']  # abbreviated
        print(f"[DEBUG] Using default COCO class names (truncated): {class_names[:8]}...")
    
    print(f"[DEBUG] ==================== CAMERA SETUP ====================")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[DEBUG] ERROR: Could not open camera {args.camera}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[DEBUG] Camera resolution: {width}x{height}")
    print(f"[DEBUG] Camera FPS: {fps}")
    
    print(f"[DEBUG] ==================== STARTING DETECTION ====================")
    print(f"[DEBUG] Press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[DEBUG] Failed to read frame")
            break
        
        frame_count += 1
        
        # Run detection (with full debug output for first few frames)
        if frame_count <= 3:  # Only debug first 3 frames to avoid spam
            print(f"\n[DEBUG] ================ PROCESSING FRAME {frame_count} ================")
            detections = detect_objects(session, frame, args.input_size, 
                                      args.confidence, args.nms)
        else:
            # Minimal debug for subsequent frames
            detections = detect_objects(session, frame, args.input_size, 
                                      args.confidence, args.nms)
        
        # Draw detections
        frame = draw_detections(frame, detections, class_names)
        
        # Add info overlay
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Conf: {args.confidence}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('YOLO Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"[DEBUG] ==================== CLEANUP ====================")
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DEBUG] Processed {frame_count} frames total")

if __name__ == "__main__":
    main()