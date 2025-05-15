#!/bin/bash
#
#
# Wolfgang Stinner, Berlin  14.02.2018/ 29.06.2016 / 20.02.2017 16.06.2017
#
# pinet-check-retrieve.sh
#
# Check jpg transfer tasks
# currently a bit quick and dirty because I don't check the pids of ncat or tar but the number of pids.
# check endof transfer with "#pgrep -fl pinet-check-retrieve.sh" 
# "#pgrep -fl pinet-check-ret" # use only 15 characters??
#
# pinet init?
if ! [ -n "$(pidof ncat)" ]; then echo pinet is not initialized; exit ;fi;

PROC="tar"
RES=0
# or
#PROC="ncat"
#RES=1

#"timeout 10  command"

APIDS_=($(pidof $PROC))
# Nr. of tasks:
AL=${#APIDS_[@]}
ATASKS=$AL
#
AL1=0
echo List of retrieve tasks, please wait:
while [ $AL -gt $RES ]; do
  if [ $AL1 -ne $AL ];then
   echo -ne $[AL-RES]     \\r
  fi

AL1=$AL
sleep 1
APIDS_=($(pidof $PROC))
AL=${#APIDS_[@]}

done

#image transfer successfull: 
echo Transfer done, $ATASKS servers used
# check exit code in calling routine
exit 199

