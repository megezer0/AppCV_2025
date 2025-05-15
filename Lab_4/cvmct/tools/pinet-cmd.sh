#!/bin/bash
#
# Wolfgang Stinner, Berlin  30.01.2018/  20.02.2017 /29.06.2016 
# Version: 2
# 
# **** "pinet-cmd" für pi-Farm***
# NICHT als sudo or root starten
# 
#
# "pinet-cmd.sh" 
# bash commands für alle, commandline commands with options in "quotes"
# default cmd: "date"
# "pinet-cmd.sh "ls -al" "
#
# "pinet-cmd.sh poweroff"
#  schaltet alle aus
#
# "apt-get -y autoremove"
# "apt-get -y dist-upgrade"
#
# Folgender  Hinweis ist OK, tritt nur während des ersten Durchlauf auf.
# Warning: Permanently added the ECDSA host key for IP address 

###############################################################

NUMBERPREFIX_=0

#default command:
#PICOMMAND_="poweroff"
# PICOMMAND_=reboot
PIUSER_B_=cvpi
PINAME_B_=cvpi
PIPOOL=( {1..6} {8..20} 22 25 )
#PIPOOL=( {31..50} 52 )
PICOMMAND_="date"



# allowed commands
#if [ "$1" == "poweroff" ] 
#then
## PICOMMAND_="ls -al"
#PICOMMAND_=poweroff
#fi
# all commands allowed , be carefull!!!
#
# $1 not empty?
if [ -n "$1" ] 
then
PICOMMAND_=$1
fi

#######

# 
for A in ${PIPOOL[@]}
do
PINAME_=$PINAME_B_$A
PIUSER_=$PIUSER_B_$A
PIWORD_=16AppCV\$\&\$$A

echo .
echo $PINAME_

PINGGOOD="$(ping -c1 $PINAME_ | grep '1 rec')"
if [ "$PINGGOOD" \> " " ]
then
sshpass -p $PIWORD_ ssh  -oStrictHostKeyChecking=no $PIUSER_@$PINAME_ "echo '$PIWORD_' | sudo -S $PICOMMAND_"
echo "."
fi

done
##########################
