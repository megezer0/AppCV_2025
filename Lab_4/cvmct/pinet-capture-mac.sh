#!/bin/bash
#
# Modified for macOS - pinet-capture-mac.sh
#

# Load configuration
if [ -f $(dirname $0)/pinet.cfg ]; then 
    . $(dirname $0)/pinet.cfg
else
    echo $(dirname $0)/pinet.cfg not found
    exit 1
fi

# Commandline camera file
if [ -n "$1" ]; then
    PICAMERAFILE=$1
fi

# Read CAMERA config file
if [ -f $(dirname $0)/$PICAMERAFILE ]; then 
    . $(dirname $0)/$PICAMERAFILE
else
    echo "Camera config file not found: $PICAMERAFILE"
    exit 1
fi

# Output filename
CAMSET2="-o $CAMDIR\$HOSTNAME-%d.jpg --timeout $TIMEOUT"

# Show camera settings
echo "Camera-cfg: $PICAMERAFILE"
echo "raspistill $CAMSET1 $CAMSET2"

# Capture pictures
echo "raspistill $CAMSET1 $CAMSET2" | ncat -v -4 --send-only localhost $BROKERPORT