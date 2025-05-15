#!/bin/bash
#
# Wolfgang Stinner, Berlin  19.01.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# execute on client (= brokerpc = localhost)
# ./pinet-trigger-camera.sh [cameraconfigfile]
#
# #read configfile, overwrite variables
if [ -f $(dirname $0)/pinet.cfg ];then 
	. $(dirname $0)/pinet.cfg
else
exit
fi

# commandline camerafile
if [ -n "$1" ];then
PICAMERAFILE=$1
fi


# Output remote trigger cmd:
CAMSET3="pkill -SIGUSR1 raspistill"


#
# capture pictures 
echo "$CAMSET3" | ncat -v -4 --send-only localhost $BROKERPORT

echo "Cameras triggered"
## END
