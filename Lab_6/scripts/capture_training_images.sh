#!/usr/bin/env bash
# lab_6/scripts/capture_training_images.sh

set -e

# Parse command line arguments
NUM_IMAGES=${1:-5}
DELAY=${2:-0.5}

# Set up directory structure
BASE_DIR="lab_6"
IMAGES_DIR="$BASE_DIR/captured_images"

# Create directory if it doesn't exist
mkdir -p "$IMAGES_DIR"

# Find next available image number
if ls "$IMAGES_DIR"/img_*.jpg 1> /dev/null 2>&1; then
    # Get highest existing number
    HIGHEST=$(ls "$IMAGES_DIR"/img_*.jpg | sed 's/.*img_\([0-9]*\)\.jpg/\1/' | sort -n | tail -1)
    # Remove leading zeros and add 1
    START_COUNT=$((10#$HIGHEST + 1))
else
    START_COUNT=0
fi

echo "Starting image capture..."
echo "Will capture $NUM_IMAGES images with ${DELAY}s delay"
echo "Saving to: $IMAGES_DIR"
echo "Starting from img_$(printf "%03d" $START_COUNT).jpg"
echo ""
echo "Get ready! Starting in 3 seconds..."

# Countdown
for i in 3 2 1; do
    echo "$i..."
    sleep 1
done

echo "Capturing!"

# Capture images
SUCCESSFUL=0
for i in $(seq 0 $((NUM_IMAGES-1))); do
    IMAGE_NUM=$((START_COUNT + i))
    FILENAME="img_$(printf "%03d" $IMAGE_NUM).jpg"
    FILEPATH="$IMAGES_DIR/$FILENAME"
    
    echo "Capturing image $((i+1))/$NUM_IMAGES: $FILENAME"
    
    if libcamera-still -n \
        --width 640 \
        --height 480 \
        --timeout 100 \
        -o "$FILEPATH" 2>/dev/null; then
        echo "✓ Captured: $FILENAME"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "✗ Failed to capture: $FILENAME"
    fi
    
    # Wait before next capture (except for last image)
    if [ $i -lt $((NUM_IMAGES-1)) ]; then
        sleep "$DELAY"
    fi
done

echo ""
echo "Capture complete! $SUCCESSFUL/$NUM_IMAGES images saved to $IMAGES_DIR"