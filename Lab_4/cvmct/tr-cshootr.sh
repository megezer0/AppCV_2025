#!/bin/bash
#
# Wolfgang Stinner, Berlin  8.02.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# without path

# read file , overwrite variables
IPATH=$(dirname $0)/
#
# sleep time is in sec., adjust between .2 and .3
# it's faster and better to clear/delete the files before this script
"$IPATH"pinet-clear-jpg.sh && "$IPATH"pinet-trigger-camera.sh && sleep 0.3 && "$IPATH"pinet-retrieve.sh && "$IPATH"pinet-check-retrieve.sh
#######################

