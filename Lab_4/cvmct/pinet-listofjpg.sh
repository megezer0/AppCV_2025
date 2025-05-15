 #!/bin/bash
#
# Wolfgang Stinner, Berlin  21.02.2018/ 1.11.2017 / 20.02.2017 16.06.2017
#
# **** Get list of current remote picture-folder for given cvpi # ***
# default is pi server Nr. 1
# syntax: pinet-list-pic.sh "[#]"
# I'm the broker therefore send to localhost instead of 'BROKERPC'
#
###
# pinet init?
if ! [ -n "$(pidof ncat)" ]; then echo pinet is not initialized; exit ;fi;

if [ $# -ne 1 ];then 
echo "syntax: pinet-list-jpg.sh [PiNr]"
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
#
#temp_listenport=30001
temp_listenport=$(( $BROKERPORT + 1 )) # for jpg-list-receive

# tempor. local Listfile prefix
LISTFILE="/dev/shm/listfile-"

## $1 not empty?  
if [ -n "$1" ] 
then
R_PINR=$1
fi

#remote listfile
remlistfile="/dev/shm/mylistfile"

PISERVER_=$PINAME_$R_PINR

for I in ${PIPOOL[@]}; do
 if [ $I -eq $R_PINR ]; then
# execute only if pinr is in list

# mylistfile is a temp. file on remote pi server
#
# cmd executes remote on selectet pi server and send result by remote ncat to local temporary ncat listener port
#PICOMMAND2_="if [ \$HOSTNAME == $PISERVER_ ] ; then ls $CAMDIR > $remlistfile && ncat -w 1 $BROKERIP $temp_listenport < $remlistfile ; fi ;"
PICOMMAND2_="if [ \$HOSTNAME == $PISERVER_ ] ; then ls $CAMDIR > $remlistfile && ncat -w 1 $BROKERIP $temp_listenport < $remlistfile ; fi ;"

#
# start local listener (receives result file)
ncat -l -p $temp_listenport > $LISTFILE$PISERVER_ &

# send executable cmd to remote pi server by brokerpc
echo "$PICOMMAND2_" | ncat -4 --send-only localhost $BROKERPORT

# wait 0.5 second
sleep 0.5	

 if [ -s "$LISTFILE$PISERVER_" ] ; then
  echo "jpg files in remote folder $CAMDIR on "$PISERVER_ " :"
  cat $LISTFILE$PISERVER_
 else echo "No jpg files in remote folder $CAMDIR on "$PISERVER_ 
 fi
rm $LISTFILE$PISERVER_
# 
exit
fi
done
echo $PISERVER_ is not part of PIPOOL 
