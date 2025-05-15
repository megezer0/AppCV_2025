#!/bin/bash
#
# Wolfgang Stinner, Berlin  26.02.2018/ 29.06.2016 / 20.02.2017 16.06.2017
#
# ncat-pool.sh renamed to pinet-sub-nc-pool.sh
# create local temporary ncat receive-port for each cvpi in PIPOOL
# Called by pinet-retrieve.sh on client
# Optional accept only 1 pi
# pinet-sub-nc-pool.sh [PiNr]

#######################
#local config file
# read file , overwrite variables
if [ -f $(dirname $0)/pinet.cfg ];then 
	. $(dirname $0)/pinet.cfg
else
echo $(dirname $0)/pinet.cfg not found
exit
fi
#######################


## if cmdline $1 not empty, create only one port
if [ -n "$1" ] 
then
PIPOOL=( $1 )
fi


# create local directory with given path:
if [ ! -d $OUTPUTDIR ]; then
  mkdir -p $OUTPUTDIR
fi

# create ncat|tar receive ports
# use "> /dev/null &" instead of "&" only, otherwise it will not return to cmdline!
#  Important: decouple ncat from keyboard input (</dev/null), otherwise any pressed key will be add 
#  to the data stream, crashing ncat/tar.
# 
# without zip
# ncat -4 -l $AP </dev/null | tar xvf - --warning=no-timestamp -C $OUTPUTDIR > /dev/null &

#with zip
# ncat -4 -l $AP </dev/null | tar xzvf - --warning=no-timestamp -C $OUTPUTDIR > /dev/null &
IJ=0
 for I in ${PIPOOL[@]}; do
  AP=$[$CLIENTPORTBASE+$I]
ncat -4 -l $AP </dev/null | tar xvf - --warning=no-timestamp -C $OUTPUTDIR > /dev/null &
(( ++IJ ))
done

echo "$IJ Client Receive ports created"
