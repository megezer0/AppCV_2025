#!/bin/bash
#
# pinet-init-mac.sh - macOS specific initialization for CVMCT
#

# Load configuration
if [ -f $(dirname $0)/pinet.cfg ]; then 
    . $(dirname $0)/pinet.cfg
else
    echo $(dirname $0)/pinet.cfg not found
    exit 1
fi

echo "Detected macOS, will use .local suffix for hostnames"

# Get the broker PC hostname
BROKERPC=$(hostname)
echo $BROKERPC

# Kill any existing ncat processes that might be using our ports
echo "Cleaning up any existing ncat processes..."
pkill -f "ncat -4 -l $BROKERPORT"
pkill -f "ncat -4 -l $PIBACKDOORPORT"
sleep 1

# First create temp command backdoor on all Pi servers
echo "Creating command backdoors on all Raspberry Pis..."

# Function to execute SSH command with .local suffix and password pattern on Mac
ssh_command() {
    local pi_num=$1
    local command=$2
    local password="16AppCV\$&\$$pi_num"
    
    echo "."
    echo "cvpi$pi_num"
    
    # Use sshpass to provide password
    sshpass -p "$password" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no cvpi$pi_num@cvpi$pi_num.local "$command"
}

# Install backdoor on all Pis
for I in ${PIPOOL[@]}; do
    ssh_command $I "nohup ncat -4 -l $PIBACKDOORPORT --exec /bin/bash &>/dev/null &"
done

echo "Command backdoors on all remote server installed"

# Start broker local with a unique port
BROKERPORT=$((BROKERPORT + 1))  # Try next port
echo "Using broker port: $BROKERPORT"

# Start the broker
ncat -4 -l $BROKERPORT --broker &
if [ $? -ne 0 ]; then
    echo "Failed to start broker. Trying another port..."
    BROKERPORT=$((BROKERPORT + 1))
    ncat -4 -l $BROKERPORT --broker &
    if [ $? -ne 0 ]; then
        echo "Failed to start broker again. Please check your network configuration."
        exit 1
    fi
fi

echo "Broker started on port $BROKERPORT"

# Get current IP address
BROKERIP=$(ifconfig | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | head -1)
echo "Broker IP: $BROKERIP"

# Bind every Pi to the broker
PICMD="nohup ncat -4 $BROKERIP $BROKERPORT --exec /bin/bash &>/dev/null &"

# For direct IP approach
if [ ${#PIPOOLIPSUFFIX[@]} -gt 0 ]; then
    for I in ${!PIPOOLIPSUFFIX[@]}; do
        IP="${IPBASE}${PIPOOLIPSUFFIX[$I]}"
        PI_NUM="${PIPOOL[$I]}"
        echo "Binding $IP (cvpi$PI_NUM) to broker..."
        echo "$PICMD" | ncat -4 --send-only $IP $PIBACKDOORPORT
    done
else
    # For hostname approach with .local suffix
    for I in ${PIPOOL[@]}; do
        echo "Binding cvpi$I.local to broker..."
        echo "$PICMD" | ncat -4 --send-only cvpi$I.local $PIBACKDOORPORT
    done
fi

echo "Bind to Broker $BROKERPC [$BROKERIP:$BROKERPORT] activated, USEDNS= $USEDNS"

# Create remote jpg directories
echo "Remote jpg dir = $CAMDIR"

if [ "$CAMDIR" != "./" ]; then
    sleep 1.3
    echo "Creating remote directories..."
    echo "mkdir -p $CAMDIR" | ncat -4 --send-only localhost $BROKERPORT
fi

# Update the configuration for other scripts to use the new broker port
echo "Updating configuration for other scripts..."
TMP_CFG=$(mktemp)
cat $(dirname $0)/pinet.cfg | sed "s/BROKERPORT=.*/BROKERPORT=$BROKERPORT/" > $TMP_CFG
mv $TMP_CFG $(dirname $0)/pinet.cfg

echo "Initialization complete!"