#!/bin/bash
#
# Wolfgang Stinner, Berlin  7.02.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# pinet-firewall-ports.sh
# sudo iptables -I INPUT -p tcp --dport 31101:31122 --syn -j ACCEPT
# sudo iptables -I INPUT -p tcp -m multiport --dports 30001,31001 --syn -j ACCEPT
#
# !!IMPORTANT pinet.cfg MUST be set!!

#######################
#local config file
# read file , overwrite variables
if [ -f $(dirname $0)/../pinet.cfg ];then 
	. $(dirname $0)/../pinet.cfg
else
echo $(dirname $0)/../pinet.cfg not found
exit
fi
###########################
# find max. pi-nr:
HPINR=0
for N in "${PIPOOL[@]}" ; do
    (( N > HPINR )) && HPINR=$N
done
##############

TEMPLISTENPORT=$(( $BROKERPORT + 1 )) # for jpg-list-receive

IFIRSTPORT=$(( $CLIENTPORTBASE + 1 ))
ILASTPORT=$(( $CLIENTPORTBASE + $HPINR ))

iptables -I INPUT -p tcp --dport $IFIRSTPORT:$ILASTPORT --syn -j ACCEPT
iptables -I INPUT -p tcp -m multiport --dports $TEMPLISTENPORT,$BROKERPORT --syn -j ACCEPT

echo iptables: Local tcp listen ports temporary opened:
echo $BROKERPORT, $TEMPLISTENPORT, $IFIRSTPORT to $ILASTPORT
