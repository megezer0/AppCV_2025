#!/bin/bash
# create-server-list-simple.sh - Direct nmap scan processing

# Set the directory for the script
SCRIPT_DIR=$(dirname "$0")
CONFIG_DIR=$(dirname "$SCRIPT_DIR")

# Load configuration
if [ -f "$CONFIG_DIR/pinet.cfg" ]; then 
    source "$CONFIG_DIR/pinet.cfg"
else
    echo "$CONFIG_DIR/pinet.cfg not found"
    exit 1
fi

echo "Creating server list using direct nmap scan..."

# Clean up IPBASE if needed
CLEAN_IPBASE=$(echo "$IPBASE" | sed 's/\.$//g')
echo "Scanning network: $CLEAN_IPBASE.0/24"

# Run the nmap scan command (using the one that worked in your previous test)
echo "Running nmap scan..."
SCAN_RESULT=$(sudo nmap -sn "$CLEAN_IPBASE.0/24")

# Extract Raspberry Pi IP addresses
PI_IPS=$(echo "$SCAN_RESULT" | grep -A 1 "Raspberry Pi" | grep "Nmap scan report" | awk '{print $5}')

if [ -z "$PI_IPS" ]; then
    echo "No Raspberry Pi devices found on the network!"
    exit 1
fi

# Convert to array and extract last octets
declare -a IP_ARRAY
declare -a LAST_OCTET_ARRAY

while read -r ip; do
    if [ -n "$ip" ]; then
        IP_ARRAY+=("$ip")
        last_octet=$(echo "$ip" | cut -d. -f4)
        LAST_OCTET_ARRAY+=("$last_octet")
        echo "Found Raspberry Pi at $ip (last octet: $last_octet)"
    fi
done <<< "$PI_IPS"

echo "Found ${#IP_ARRAY[@]} Raspberry Pi devices."

# Use PIPOOL numbers
NUM_PIS=${#IP_ARRAY[@]}
NUM_POOL=${#PIPOOL[@]}
USED_POOL=("${PIPOOL[@]:0:NUM_PIS}")

# If we have fewer PIPOOL numbers than found Pis, add some
if [ "$NUM_PIS" -gt "$NUM_POOL" ]; then
    for ((i = NUM_POOL; i < NUM_PIS; i++)); do
        USED_POOL+=("${LAST_OCTET_ARRAY[$i]}")
    done
fi

# Create the server config file
echo "#!/bin/bash" > "$CONFIG_DIR/pinet-server.cfg"
echo "#PINAME_=$PINAME_" >> "$CONFIG_DIR/pinet-server.cfg"
echo "#IPBASE=\"$CLEAN_IPBASE\"" >> "$CONFIG_DIR/pinet-server.cfg"
echo "PIPOOL=( ${USED_POOL[@]} )" >> "$CONFIG_DIR/pinet-server.cfg"
echo "PIPOOLIPSUFFIX=( ${LAST_OCTET_ARRAY[@]} )" >> "$CONFIG_DIR/pinet-server.cfg"
echo "#Found Raspberry Pi devices: ${#IP_ARRAY[@]}" >> "$CONFIG_DIR/pinet-server.cfg"
echo "#Matched Raspberry Pis: ${#USED_POOL[@]}" >> "$CONFIG_DIR/pinet-server.cfg"

echo
echo "Server configuration created successfully:"
echo "IPBASE: $CLEAN_IPBASE"
echo "PIPOOL: ${USED_POOL[@]}"
echo "PIPOOLIPSUFFIX: ${LAST_OCTET_ARRAY[@]}"
echo
echo "Configuration saved to $CONFIG_DIR/pinet-server.cfg"