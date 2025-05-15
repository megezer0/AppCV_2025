#!/bin/bash
# Wolfgang Stinner, Berlin  15.01.2018 /02.11.2017/ 29.06.2016 / 20.02.2017 
# Version: 2
# 
# **** "pi-cmdu" , remote command line tool for one pi ***
# DO NOT execute as sudo or root.
# 
#
# pi-cmdu.sh "command [arguments]" PiNr
#  Executes bash-commands on one remote pi. 
#  Use remote Commandline with command-arguments in "quotes"
# pi-cmdu.sh "ls -al" PiNr
#
# pi-cmd.sh "mkdir  ./dumdir" 22
#  create folder on cvpi25
#

# At first time usage you may receive:
# Warning: Permanently added the ECDSA host key for IP address 

###############################################################
PIUSER_B_=cvpi
PINAME_B_=cvpi

PIPOOL=( {1..6} 7 {8..20} 22 25 {31..52} )

#default command:
#PICOMMAND_="poweroff"
# PICOMMAND_=reboot
PICOMMAND_="date"

# pi-cmd.sh "cmd" nr
# all commands allowed , be carefull!!!
if [ $# -ne 2 ] 
then
echo "syntax: pi-cmdu.sh \"cmd [arguments]\" PiNr"
exit
fi

## A=PiNr, argument-2
A=$2

## PICOMMAND= argument-1
PICOMMAND_=$1

##### 
# for PIUSER # <10
NUMBERPREFIX_=0


for I in ${PIPOOL[@]}; do
 if [ $I -eq $A ]; then

PINAME_=$PINAME_B_$A
PIUSER_=$PIUSER_B_$A
PIWORD_=16AppCV\$\&\$$A

echo .
echo $PINAME_

PINGGOOD="$(ping -c1 $PINAME_ | grep '1 rec')"
if [ "$PINGGOOD" \> " " ]
then
sshpass -p $PIWORD_ ssh -oStrictHostKeyChecking=no $PIUSER_@$PINAME_ "$PICOMMAND_"
echo "."
fi
# ;;
#
exit
fi
#
done

#############################
