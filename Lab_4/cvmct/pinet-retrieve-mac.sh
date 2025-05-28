# #!/bin/bash
# #
# # Modified for macOS - pinet-retrieve.sh
# #

# # Check if pinet is initialized by checking for ncat processes
# # Use pgrep instead of pidof for macOS
# if ! pgrep -q ncat; then 
#     echo "pinet is not initialized. Run pinet-init-mac.sh first"
#     exit 1
# fi

# # Load configuration
# if [ -f $(dirname $0)/pinet-mac.cfg ]; then 
#     . $(dirname $0)/pinet-mac.cfg
# else
#     echo $(dirname $0)/pinet-mac.cfg not found
#     exit 1
# fi

# # if [ $(pgrep ncat | wc -l) -gt 1 ]; then 
# BROKER_NC=$(pgrep -f "ncat .* -l .* $BROKERPORT" | wc -l)
#     echo "Transfer is not ready; broker still busy (found $BROKER_NC listeners on $BROKERPORT)."
#     exit 1
# fi

#!/bin/bash
# pinet-retrieve-mac.sh – repaired

CFG="$(dirname "$0")/pinet-mac.cfg"
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
. "$CFG"

# 1  make sure the broker process exists at all
if ! pgrep -f "ncat .*-l[[:space:]]*$BROKERPORT([^0-9]|$)" >/dev/null; then
    echo "pinet is not initialised – run pinet-init-mac.sh first."
    exit 1
fi

# 2  guard against concurrent use of the broker port
# BROKER_NC=$(pgrep -f "ncat .*-l[[:space:]]*$BROKERPORT([^0-9]|$)" | wc -l)
# if [ "$BROKER_NC" -ne 1 ]; then
#     echo "Transfer is not ready; broker still busy (found $BROKER_NC listeners on $BROKERPORT)."
#     exit 1
# fi
BROKER_PIDS=$(pgrep -f "ncat[[:space:]].*-l[[:space:]]*${BROKERPORT}[[:space:]].*--broker")
BROKER_NC=$(echo "$BROKER_PIDS" | wc -w)
if [ "$BROKER_NC" -ne 1 ]; then
    echo "Transfer is not ready; broker still busy (found $BROKER_NC listeners on $BROKERPORT)."
    exit 1
fi

# Call pinet-sub-nc-pool.sh 
# defines the common local retrieve folder and client listening ports for each pi
$(dirname $0)/pinet-sub-nc-pool-mac.sh

# File-wildcards will not work with pathoption "-C", trial without trailing path
CAMDIRPATH=$(dirname "$CAMDIR")
FILEDIR=$(basename "$CAMDIR")"/"

# Construct the retrieve command
RETRIEVECMD="tar vcH posix -C $CAMDIRPATH $FILEDIR | ncat -4 --send-only $CLIENTIP \$((\${HOSTNAME:$ALPHALENGTH}+$CLIENTPORTBASE))"

# Trigger retrieve
echo "$RETRIEVECMD" | ncat -4 --send-only localhost $BROKERPORT

echo "Retrieve triggered."
echo "Transfer from remote $CAMDIR to local $OUTPUTDIR"