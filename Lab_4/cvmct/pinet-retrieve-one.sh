#!/bin/bash
#
# Wolfgang Stinner, Berlin  14.03.2018/ 1.11.2017 / 20.02.2017 16.06.2017
#
# pinet-retrieve-one.sh [PiNr] 
# with reduced dns usage
#
# runs on client: clientpc = brokerpc, so I can send to localhost
#
# pinet init? , transfer ready?
PIDTEST=($(pidof ncat))
if [ "${#PIDTEST[@]}" -eq 0 ]; then echo pinet is not initialized; exit ;fi;
if [ "${#PIDTEST[@]}" -gt 1 ]; then echo transfer in progress, check and try again; exit ;fi;


#
if [ $# -ne 1 ];then 
echo "syntax: pinet-retrieve-one.sh [PiNr]"
fi

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

# default remote pi server nr., first element in pipool  
R_PINR=$PIPOOL
## $1 not empty, set retrieve server  
if [ -n "$1" ] 
then
R_PINR=$1
fi

PISERVER_=""
for I in ${PIPOOL[@]}; do
   if [ $I -eq $R_PINR ]; then
   # piserver is active
    PISERVER_=$PINAME_$R_PINR
    break
    fi
done

if ! [ -n "$PISERVER_" ] ; then
echo $PINAME_$R_PINR is not part of PIPOOL 
exit
fi

#echo $PISERVER_, $R_PINR; exit #bugtest
##################################################### 
 
# pinet-sub-nc-pool.sh defines the common local retrieve folder and client listening ports for each pi:
# accept only one pi by nr.
$(dirname $0)/pinet-sub-nc-pool.sh $R_PINR


# file-wildcards will not work with pathoption "-C". trial without trailing path
CAMDIRPATH=$(dirname $CAMDIR)
FILEDIR=$(basename $CAMDIR)"/"

RETRIEVECMD="if [ \$HOSTNAME == $PISERVER_ ] ; then tar vcH posix -C $CAMDIRPATH $FILEDIR | ncat -4 --send-only $CLIENTIP \$((\${HOSTNAME:$ALPHALENGTH}+$CLIENTPORTBASE)) ; fi ;"
# trigger retrieve
echo "$RETRIEVECMD" | ncat -4 --send-only localhost $BROKERPORT

# tar --remove-files OR remove remote pictures with "pinet-clear.sh" 
# zip before transfer:# tar vzc .... //change also in ncatpool.sh
# 
echo "Retrieve triggered."
echo "Transfer from remote $PISERVER_ $CAMDIR to local $OUTPUTDIR"

# Check if all images are retrieved with:
# pidof ncat
# All pictures received resultes in "1",  because one instance is used for "Broker"-ncat
