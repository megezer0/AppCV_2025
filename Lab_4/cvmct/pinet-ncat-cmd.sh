 #!/bin/bash
#
# Wolfgang Stinner, Berlin  24.01.2018/ 1.11.2017 / 20.02.2017 16.06.2017
#
# **** execute user or sudo commands on remote pinet ***
# syntax: pinet-ncat-cmd.sh "cmd [arguments]" [-S]
#  option [-S] executes remote cmd as sudo"
#
# BROKERPC=xxxx
# I'm the broker therefore send to localhost instead of 'BROKERPC'
#
# all commands allowed , be carefull!!!
# examples: 
#PICOMMAND_="rm /home/\$HOSTNAME/\$HOSTNAME*.jpg"
#PICOMMAND_="rm \$HOSTNAME*.jpg"
#PICOMMAND_="poweroff" -S
#PICOMMAND_="rm du*.jpg"
#PICOMMAND_="apt-get -y update" -S
#PICOMMAND_="apt-get -y upgrade" -S
#PICOMMAND_="touch dudel2.jpg"
#PICOMMAND_="service cpid stop" -S
#
#PICOMMAND_="SKIP_WARNING=1 rpi-update" -S


###################################
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
# password, only for sudo-mode, not used for user-commands
PIWORD_='16AppCV$&$'\"\${HOSTNAME:$ALPHALENGTH}\"


# $1 not empty?
if [ -n "$1" ] 
then
PICOMMAND_=$1
else
echo "syntax: pinet-ncat-cmd.sh \"cmd [arguments]\" [-S]"
echo " option [-S] executes remote cmd as sudo"
exit
fi

# 2. argument exists but is not "-S" means Syntax error
if [ $# -eq 2 ] && [ "$2" != "-S" ] 
then
echo "syntax: pinet-ncat-cmd.sh \"cmd [arguments]\" [-S]"
echo " option [-S] executes remote cmd as sudo"
exit
fi


if [ "$2" == "-S" ] 
then
echo "Send with sudomode"
# execute remote cmd as sudo:
PICOMMAND2_="echo \"$PIWORD_\" | sudo -S $PICOMMAND_"
else
# execute remote user command:
PICOMMAND2_="$PICOMMAND_"
fi

##echo "$PICOMMAND2_"

# execute remote
echo "$PICOMMAND2_" | ncat -4 --send-only localhost $BROKERPORT
#done
