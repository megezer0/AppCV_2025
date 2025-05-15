#!/bin/bash
#
# Wolfgang Stinner, Berlin  26.01.2018/ 29.06.2016 / 20.02.2017 /
#
# 
# changed to pinet-sub-cmdu.sh
# is called by pinet-init.sh
# ***** works only on raspberries set in pinet.cfg *****
# **** "pinet-cmdu" hier für **UBUNTU** ( keine sudo Befehle, USER-Variante) für pi-Farm***
## using IP-addresses, NO dns calls
#
# Folgender  Hinweis ist OK, tritt nur während dem ersten Durchlauf auf.
# Warning: Permanently added the ECDSA host key for IP address 
#
# Raspberries host-names: cvpi1,cvpi2...cvpi10..
# Raspberries user-names: cvpi01,cpi02,..cvpi10..  könnte man auch umbenennen zu cvpi1,cvpi2... 

# ***** trial without dns calls, IPOOLIPSUFFIX array must be set in pinet.cfg ****
 ###############################################################
#
# leading zero for usernames 1..9
NUMBERPREFIX_=0

#default dummy command:
PICOMMAND_="date"

# all commands allowed , be carefull!!!
#
# commandline option, $1 not empty?
if [ -n "$1" ] 
then
PICOMMAND_=$1
fi

#########################
#local config file
# read file , overwrite variables
if [ -f $(dirname $0)/pinet.cfg ];then 
	. $(dirname $0)/pinet.cfg
else
echo $(dirname $0)/pinet.cfg not found
exit
fi
#######################
# 
#
# password for piuser is 16AppCV$&$"PINR"
# "PINR" is the trailing number of the pi-server hostname

AI=-1;for A in ${PIPOOL[@]}; do
(( ++AI ))
PIUSER_=$PINAME_$A
PIWORD_=16AppCV\$\&\$$A

PIIP=$IPBASE"${PIPOOLIPSUFFIX[$AI]}"


echo .
echo $PIUSER_


PINGGOOD="$(ping -c1 $PIIP | grep '1 rec')"
if [ "$PINGGOOD" \> " " ]
then
# ** For Ubuntu Only:
sshpass -p $PIWORD_ ssh -oStrictHostKeyChecking=no $PIUSER_@$PIIP "$PICOMMAND_"
echo "."
fi

done
##########################
