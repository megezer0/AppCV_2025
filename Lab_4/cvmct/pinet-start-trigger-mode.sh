#!/bin/bash
#
# Wolfgang Stinner, Berlin  14.03.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# execute on client (= brokerpc = localhost)
# ./pinet-start-trigger-mode.sh [cameraconfigfile] # in background, triggered by USR1
#
# #read configfile, overwrite variables
if [ -f $(dirname $0)/pinet.cfg ];then 
	. $(dirname $0)/pinet.cfg
else
exit
fi

# default *background* cameraconfigfile, dirty overwrite for background:
PICAMERAFILE="pinet-camera-trigger.cfg"

# read optinal commandline camerafile
if [ -n "$1" ];then
PICAMERAFILE=$1
fi

#read CAMERA configfile, overwrite variables
if [ -f $(dirname $0)/$PICAMERAFILE ];then 
	. $(dirname $0)/$PICAMERAFILE
else
exit
fi

# Output filename:
#CAMSET2="-o $CAMDIR\$HOSTNAME-%d.jpg &"
CAMSET2="--timeout $TIMEOUT -o $CAMDIR\$HOSTNAME-$(date +%s%N)-Nr%d.jpg &"
# show CAMSET1:
echo Start Camera Trigger Mode with:
echo "raspistill" "$CAMSET1"
#
# capture pictures 
echo "raspistill ""$CAMSET1" "$CAMSET2" | ncat -v -4 --send-only localhost $BROKERPORT
## END
