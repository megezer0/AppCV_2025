#!/usr/bin/env python3

"""
Feature Control Configuration
Students enable features as they complete each week's implementation
"""

# =============================================================================
# STUDENT FEATURE CONTROL
# Set to True when you've completed the implementation for each week
# =============================================================================

FEATURES_ENABLED = {
    'line_following': True,    # Week 1: Set to True when line following is ready
    'sign_detection': False,   # Week 2: Set to True when sign detection is ready  
    'speed_estimation': False  # Week 3: Set to True when speed estimation is ready
}

# =============================================================================
# FEATURE DESCRIPTIONS (for reference)
# =============================================================================

FEATURE_DESCRIPTIONS = {
    'line_following': 'Computer vision + PID control for following lines',
    'sign_detection': 'ONNX model integration for detecting stop signs',
    'speed_estimation': 'Optical flow analysis for estimating robot speed'
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Minimum methods required for each feature to be considered "implemented"
REQUIRED_METHODS = {
    'line_following': ['compute_steering_angle'],
    'sign_detection': ['detect_signs', 'should_stop'],
    'speed_estimation': ['estimate_speed']
}

def is_feature_enabled(feature_name):
    """Check if a feature is enabled"""
    return FEATURES_ENABLED.get(feature_name, False)

def get_enabled_features():
    """Get list of currently enabled features"""
    return [name for name, enabled in FEATURES_ENABLED.items() if enabled]

def get_feature_description(feature_name):
    """Get description of a feature"""
    return FEATURE_DESCRIPTIONS.get(feature_name, "No description available")