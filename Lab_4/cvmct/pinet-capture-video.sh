#!/bin/bash
#
# Wolfgang Stinner, Berlin  19.01.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# execute on client (= brokerpc = localhost)
# ./pinet-capture.sh [cameraconfigfile]
#
# #read configfile, overwrite variables
if [ -f $(dirname $0)/pinet.cfg ];then 
	. $(dirname $0)/pinet.cfg
else
exit
fi

# default *background* cameraconfigfile, dirty overwrite for background:
PICAMERAFILE="video-camera.cfg"


# commandline camerafile
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
#CAMSET2="-o $CAMDIR\$HOSTNAME-%d.h264 --timeout $TIMEOUT"
MYDATE=$(date +"%s%N")
CAMSET2="-o $CAMDIR\$HOSTNAME-$MYDATE.h264"

# show CAMSET1:
echo "raspivid " "$CAMSET1" "$CAMSET2"

#
# capture pictures 
echo "raspivid ""$CAMSET1" "$CAMSET2" | ncat -v -4 --send-only localhost $BROKERPORT
## END
