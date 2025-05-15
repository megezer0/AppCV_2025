#!/bin/bash
#
# Wolfgang Stinner, Berlin  7.02.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# pinet-start.sh
# runs on client
# Creates all basic settings for capture retrieve etc
# client side: depends on nmap (includes ncat), sysvinit-utils(includes pidof), sshpass
# server side depends on ncat and raspistill 
# pinet usage: On client copy all the pinet-files files into  a folder "~/pinet/" and cd into the folder OR
# add the folder path: #PATH=$PATH:~/pinet/
#
#######################
#
#local config file
# read file , overwrite variables
if [ -f $(dirname $0)/pinet.cfg ];then 
	. $(dirname $0)/pinet.cfg
else
echo $(dirname $0)/pinet.cfg not found
exit
fi
#######################

echo $BROKERPC
#first create temp. command backdoor on all pi-server for faster broker-pc binding:
# Command could be run on each raspberry server at boot, after Broker is running:
# 

# only as precaution:
if [ $USEDNS -ne "0" ]; then
USEDNS=1
fi

if [ $USEDNS -eq "1" ]; then
$(dirname $0)/pinet-sub-cmdu.sh "nohup ncat -4 -l $PIBACKDOORPORT --exec /bin/bash &>/dev/null &"
else
$(dirname $0)/pinet-sub-cmdu-ip.sh "nohup ncat -4 -l $PIBACKDOORPORT --exec /bin/bash &>/dev/null &"
fi


echo "Command backdoors on all remote server installed"

# start broker local
ncat -4 -l $BROKERPORT --broker &
echo " Broker started"

# check if a Pi is accessible. Put in bind loop, otherwise could hang if a pi is down.
#PINGGOOD="$(ping -c1 $PINAME_$I | grep '1 rec')"
#if [ "$PINGGOOD" \> " " ]
#then
# 
#fi

# bind every pi to IP of brokerpc by sending ncat-command to each pi-server backdoor port:
# trial without dns calls

PICMD="nohup ncat -4 $BROKERIP $BROKERPORT --exec /bin/bash &>/dev/null &"
if [ $USEDNS -eq "1" ]; then
for I in ${PIPOOL[@]}; do
 echo "$PICMD" | ncat -4 --send-only $PINAME_$I $PIBACKDOORPORT
 done
else
# use IPs instead of names, PIPOOLIPSUFFIX must be correct set in pinet.cfg!
for I in ${PIPOOLIPSUFFIX[@]}; do
 echo "$PICMD" | ncat -4 --send-only $IPBASE$I $PIBACKDOORPORT
 done
fi

echo "Bind to Broker $BROKERPC [$BROKERIP:$BROKERPORT] activated, USEDNS= $USEDNS"
#
# Now use the backdoor ports
# to create the remote-jpg-directories, if necessary.
echo "Remote jpg dir = $CAMDIR"
#
# forbidden directories
# " . / /. .." d/ ./d/

if [ "$CAMDIR" \!\= "./" ]
then
# if ["${#CAMDIR}" -ge 2 ]
sleep 1.3s
#either works: 
#$(dirname $0)/pinet-ncat-cmd.sh "mkdir -p $CAMDIR"
echo "mkdir -p $CAMDIR" | ncat -4 --send-only localhost $BROKERPORT
# fi
fi
#############################
