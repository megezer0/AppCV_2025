def count_fingers(hand_data_list):
    """
    Count the number of fingers being held up on each hand.
    
    Args:
        hand_data_list: List of tuples, each containing:
                       (hand_landmarks, hand_classification)
                       - hand_landmarks: 21 landmarks with .x, .y, .z coordinates
                       - hand_classification: MediaPipe classification ("Left" or "Right")
    
    Returns:
        tuple: (counts_dict, keypoints_list)
            - counts_dict: Dictionary with keys "left", "right", "total"
            - keypoints_list: List of landmark indices you're using for analysis
    
    DEBUGGING HELPER:
    The keypoints you return in the list will be highlighted on the hands with
    red circles. All other hand landmarks will show as light gray circles.
    This helps you visualize which points you're analyzing and verify they're
    in the right locations for finger detection.
    
    Hand Classification:
    MediaPipe provides hand classification information that tells you if each
    detected hand is "Left" or "Right". This is accessed through:
    results.multi_handedness[i].classification[0].label
    
    Hint: Compare fingertip positions to their respective joints
    Key landmarks for each finger:
    - Thumb: tip=4, joint=3
    - Index: tip=8, joint=6  
    - Middle: tip=12, joint=10
    - Ring: tip=16, joint=14
    - Pinky: tip=20, joint=18
    
    Example usage:
    keypoints_used = [4, 8, 12, 16, 20]  # All fingertips
    counts = {"left": 0, "right": 0, "total": 0}
    return (counts, keypoints_used)
    """
    
    # Initialize counts
    left_count = 0
    right_count = 0
    
    # Process each detected hand
    for hand_landmarks, hand_classification in hand_data_list:
        # Count fingers on this hand
        hand_finger_count = 0
        
        # Check each finger
        # Thumb (special case - check x position for left/right hands)
        if hand_classification == "Right":
            # For right hand, thumb extends to the left (lower x)
            if hand_landmarks[4].x < hand_landmarks[3].x:
                hand_finger_count += 1
        else:  # Left hand
            # For left hand, thumb extends to the right (higher x)
            if hand_landmarks[4].x > hand_landmarks[3].x:
                hand_finger_count += 1
        
        # Other fingers (check y position)
        if hand_landmarks[8].y < hand_landmarks[6].y:   # Index
            hand_finger_count += 1
        if hand_landmarks[12].y < hand_landmarks[10].y: # Middle
            hand_finger_count += 1
        if hand_landmarks[16].y < hand_landmarks[14].y: # Ring
            hand_finger_count += 1
        if hand_landmarks[20].y < hand_landmarks[18].y: # Pinky
            hand_finger_count += 1
        
        # Add to appropriate hand count
        if hand_classification == "Left":
            left_count += hand_finger_count
        else:  # Right
            right_count += hand_finger_count
    
    # Prepare return values
    counts = {
        "left": left_count,
        "right": right_count, 
        "total": left_count + right_count
    }
    
    keypoints_used = [4, 8, 3, 6, 12, 10, 16, 14, 20, 18]  # Tips and joints
    
    print(f"Left: {left_count}, Right: {right_count}, Total: {left_count + right_count}")
    
    return (counts, keypoints_used)