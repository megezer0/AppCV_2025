#!/bin/bash
#
# Wolfgang Stinner, Berlin  28.02.2018/ 1.11.2017 / 20.02.2017 16.06.2017
# trial to reduce dns usage
#
# runs on client: clientpc = brokerpc, so I can send to localhost
#

# pinet init?
PIDTEST=($(pidof ncat))
if [ "${#PIDTEST[@]}" -eq 0 ]; then echo pinet is not initialized; exit ;fi;
if [ "${#PIDTEST[@]}" -gt 1 ]; then echo transfer is not ready, check and try again; exit ;fi;
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


# Call pinet-sub-nc-pool.sh 
# defines the common local retrieve folder and client listening ports for each pi:
$(dirname $0)/pinet-sub-nc-pool.sh


# file-wildcards will not work with pathoption "-C". trial without trailing path
CAMDIRPATH=$(dirname $CAMDIR)
FILEDIR=$(basename $CAMDIR)"/"
#
#
RETRIEVECMD="tar vcH posix -C $CAMDIRPATH $FILEDIR | ncat -4 --send-only $CLIENTIP \$((\${HOSTNAME:$ALPHALENGTH}+$CLIENTPORTBASE))"
#
# Trigger retrieve
echo $RETRIEVECMD | ncat -4 --send-only localhost $BROKERPORT

# tar --remove-files OR remove remote pictures with "pinet-clear.sh" 
# zip before transfer:# tar vzc .... //change also in ncatpool.sh
# 
echo "Retrieve triggered."
echo "Transfer from remote $CAMDIR to local $OUTPUTDIR"

# Check if all images are retrieved with:
# pidof ncat
# All pictures received gives "1" as result, because one instance is used for "Broker"-ncat
