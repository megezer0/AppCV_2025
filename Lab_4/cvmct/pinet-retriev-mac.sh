#!/bin/bash
#
# Modified for macOS - pinet-retrieve.sh
#

# Check if pinet is initialized by checking for ncat processes
# Use pgrep instead of pidof for macOS
if ! pgrep -q ncat; then 
    echo "pinet is not initialized. Run pinet-init-mac.sh first"
    exit 1
fi

if [ $(pgrep ncat | wc -l) -gt 1 ]; then 
    echo "Transfer is not ready, check and try again"
    exit 1
fi

# Load configuration
if [ -f $(dirname $0)/pinet.cfg ]; then 
    . $(dirname $0)/pinet.cfg
else
    echo $(dirname $0)/pinet.cfg not found
    exit 1
fi

# Call pinet-sub-nc-pool.sh 
# defines the common local retrieve folder and client listening ports for each pi
$(dirname $0)/pinet-sub-nc-pool.sh

# File-wildcards will not work with pathoption "-C", trial without trailing path
CAMDIRPATH=$(dirname "$CAMDIR")
FILEDIR=$(basename "$CAMDIR")"/"

# Construct the retrieve command
RETRIEVECMD="tar vcH posix -C $CAMDIRPATH $FILEDIR | ncat -4 --send-only $CLIENTIP \$((\${HOSTNAME:$ALPHALENGTH}+$CLIENTPORTBASE))"

# Trigger retrieve
echo "$RETRIEVECMD" | ncat -4 --send-only localhost $BROKERPORT

echo "Retrieve triggered."
echo "Transfer from remote $CAMDIR to local $OUTPUTDIR"