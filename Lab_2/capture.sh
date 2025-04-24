#!/usr/bin/env bash
set -e

# clear out any old images
mkdir -p data
rm -f data/img_*.jpg

# Capture 20 frames over 2 s → interval = 2000 ms / 19 ≈ 105 ms
libcamera-still \
    --timeout 2000 \          # total duration in ms
    --timelapse 100 \         # ms between frames
    --width 640 --height 480 \
    -n \                       # no preview
    -o data/img_%02d.jpg

count=$(ls data/img_*.jpg 2>/dev/null | wc -l)
echo "✅ Captured ${count} images into data/"