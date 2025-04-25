#!/usr/bin/env bash
set -e

# Ensure data/ exists and is empty
mkdir -p data
rm -f data/img_*.jpg

# Capture 20 frames one-by-one
for i in $(seq -w 0 19); do
  echo "Capturing frame $i…"
  libcamera-still -n \
    --width 640 \
    --height 480 \
    --timeout 100 \
    -o data/img_${i}.jpg

  # brief pause for the sensor pipeline to re-arm
  sleep 0.1
done

# Count & report
count=$(ls data/img_*.jpg 2>/dev/null | wc -l)
echo "✅ Captured $count images into data/"