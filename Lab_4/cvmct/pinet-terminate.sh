#!/bin/bash
#
##
# Wolfgang Stinner, Berlin  23.01.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# execute on client (= brokerpc = localhost)
#
# terminates "ncat broker" and all bound server
# !!and  **all other tar and ncat tasks!!
# 
#
KIP="ncat"
AB=$(pidof $KIP);if [ -n "$AB" ];  then  kill $(pidof $KIP);  fi;
AB=$(pidof $KIP);if [ -n "$AB" ];  then  kill $(pidof $KIP);  fi;


KIP="tar"
AB=$(pidof $KIP);if [ -n "$AB" ];  then  kill $(pidof $KIP);  fi;
AB=$(pidof $KIP);if [ -n "$AB" ];  then  kill $(pidof $KIP);  fi;

# 
KIP="ncat"
AB=$(pidof $KIP);if [ -n "$AB" ];  then  kill $(pidof $KIP);  fi;
AB=$(pidof $KIP);if [ -n "$AB" ];  then  kill $(pidof $KIP);  fi;

#######

#APIDS_=($(pidof $KIP))
#AL=${#APIDS_[@]}
