#!/bin/bash
mkdir -p data
libcamera-still \
    --timeout 800 \
    --width 640 --height 480 \
    -n -r \
    -o data/img_%02d.jpg \
    --frames 20
echo "✅ Images saved to data/"