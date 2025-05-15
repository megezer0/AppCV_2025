#!/bin/bash
#
# Wolfgang Stinner, Berlin   15.01.2018/ 29.06.2016 / 20.02.2017 /
#Version: 2
# 

# **** "pinet-cmdu" hier für **UBUNTU** ( keine sudo Befehle, USER-Variante) für pi-Farm***
# NICHT als sudo or root starten
# 
#
# "pinet-cmdu.sh" 
# keine sudo commands!!
#
# bash commands für alle, commandline commands with arguments in "quotes"
# default cmd: "date"
# pinet-cmdu.sh "ls -al" 
#
# pinet-cmdu.sh date
#  listet von allen pi das Datum
#
#
# Folgender  Hinweis ist OK, tritt nur während dem ersten Durchlauf auf.
# Warning: Permanently added the ECDSA host key for IP address 

###############################################################
PIUSER_B_=cvpi
PINAME_B_=cvpi

PIPOOL=( {1..6} {8..20} 22 )
#PIPOOL=( {31..50} 52 )

NUMBERPREFIX_=0
#default command:
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
echo $PIUSER_


PINGGOOD="$(ping -c1 $PINAME_ | grep '1 rec')"
if [ "$PINGGOOD" \> " " ]
then
# ** For Ubuntu Only:
sshpass -p $PIWORD_ ssh -oStrictHostKeyChecking=no $PIUSER_@$PINAME_ "$PICOMMAND_"
echo "."
fi

done
##########################
