#!/bin/bash
#
# Wolfgang Stinner, Berlin  8.02.2018/ 29.06.2016 / 20.02.2017 16.06.2017
# without path

# read file , overwrite variables
IPATH=$(dirname $0)/
# with clear ( it's faster to delete files before this script):
#
#default cameraxx.cfg
#"$IPATH"pinet-clear-jpg.sh && "$IPATH"pinet-capture.sh && "$IPATH"pinet-retrieve.sh && "$IPATH"pinet-check-retrieve.sh
#
# camera-auto.cfg
"$IPATH"pinet-retrieve.sh && "$IPATH"pinet-check-retrieve.sh
#######################

