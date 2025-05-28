 #!/bin/bash
#
# Wolfgang Stinner, Berlin  19.01.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# BROKERPC=xxxx
# I'm the broker therefore 'BROKERPC=localhost'

#####################

HOSTNAME=${HOSTNAME:-$(hostname)}


#local config file
# read file , overwrite variables
if [ -f $(dirname $0)/pinet-mac.cfg ];then 
	. $(dirname $0)/pinet-mac.cfg
else
echo $(dirname $0)/pinet-mac.cfg not found
exit
fi
#######################

# rm "\$HOSTNAME\*.jpg" ### the *double quotes* and "\$" are important, 
#     otherwise the local hostname will be used, not the remote one!
#
echo "rm $CAMDIR\$HOSTNAME*.h264" | ncat -4 --send-only localhost $BROKERPORT

# DONE

