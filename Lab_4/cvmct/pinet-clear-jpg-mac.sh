#!/bin/bash
#
# Wolfgang Stinner, Berlin  19.01.2018 / … / 19.05.2025
# pinet-clear-jpg-mac.sh

# ensure HOSTNAME is set
HOSTNAME=${HOSTNAME:-$(hostname)}

#####################
# read local config
if [ -f "$(dirname "$0")/pinet-mac.cfg" ]; then
    . "$(dirname "$0")/pinet-mac.cfg"
else
    echo "Error: $(dirname "$0")/pinet-mac.cfg not found"
    exit 1
fi
#######################

# send the delete command over netcat
# -n skip DNS lookups
# -w 1 quit after 1 second of inactivity (closes the connection)
# echo "rm $CAMDIR\$HOSTNAME*.jpg" | nc -N -w 1 localhost "$BROKERPORT"
echo "rm $CAMDIR\$HOSTNAME*.jpg" | ncat -4 --send-only localhost "$BROKERPORT"

# DONE