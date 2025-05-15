#!/bin/bash
#
# Wolfgang Stinner, Berlin  16.03.2018, 29.06.2016
# 
# **** "pinet-server-table-pipool.sh" # tool für pinet
# creates server tables for "/24" network
# needs Nmap
# # creates 2 arrays from all detected servers:
# PIPOOL and PIPOOLIPSUFFIX
# Saved in "./pinet-server.cfg"

#
# read file , overwrite variables
if [ -f $(dirname $0)/../pinet.cfg ];then 
	. $(dirname $0)/../pinet.cfg
else
echo $(dirname $0)/../pinet.cfg not found
exit
fi
#######################
# overwrite pinet.cfg
#PINAME_=cvpi
# 2016:
#PIPOOL=( {1..6} {8..20} 22 25 )
#
# 2018:
#PIPOOL=( {31..50} )
#### 

echo " Please wait"
echo 

PCNAME_B_="cvpi"
IPBASE="192.168.93"

NETIP=$IPBASE".0/24"
#####################
# Create by pcnameNr sorted server list: "Name IP"
iplist=$(nmap -sP $NETIP | grep -o "$PCNAME_B_.\{1,\}" | sort -n --key=1.5 )
iplist=$(echo $iplist | tr "( )" "  " )

# convert to array
#iplista=($(echo $iplist | tr " " "\n"))
# or
iplista=(${iplist})

SERVN=${#iplista[@]}
echo **Found $PCNAME_B_ server IP-Addresses: $((SERVN / 2))

# split into two arrays
PIPOOLOCT=();PIPOOLNAMENR=()
I=0;while ((I < SERVN)); do
#name-nr
PINA=$( echo ${iplista[ ((I)) ]} | cut -d" " -f1 | grep -o -E '[0-9]+')
#ip-l.octett
PIIP=$( echo ${iplista[ ((I+1)) ]} | cut -d" " -f1 |cut -d"." -f4,4 )

PIPOOLNAMENR=( ${PIPOOLNAMENR[@]} "$PINA" )
PIPOOLOCT=( ${PIPOOLOCT[@]} "$PIIP" )
# loop by 2
((I += 2))
done
# while end

######################

# filter by PIPOOL nrs. and their ips
# PIPOOL=( 4 8 11 ) # testdummy

PIPOOLOCT1=(); PIPOOLNAMENR1=()
for i in ${PIPOOL[@]}; do
    for (( j=0;j<=${#PIPOOLNAMENR[@]};j++ )); do
      if [[ "$i" = "${PIPOOLNAMENR[$j]}" ]]; then
      PIPOOLNAMENR1=( ${PIPOOLNAMENR1[@]} "${PIPOOLNAMENR[$j]}" )
      PIPOOLOCT1=( ${PIPOOLOCT1[@]} "${PIPOOLOCT[$j]}" ) && break
      fi
    done
done

##################

echo IPBASE: $IPBASE
echo Servers to find ${#PIPOOL[@]}

echo Servers found $((SERVN / 2)) ":"
echo PIPOOL-Name-Nrs: ${PIPOOLNAMENR[@]}
echo PIPOOL-least-IP-Octet: ${PIPOOLOCT[@]}
echo
echo Servers used ${#PIPOOLNAMENR1[@]} ":"
echo PIPOOL-Name-Nrs: ${PIPOOLNAMENR1[@]}
echo PIPOOL-least-IP-Octet: ${PIPOOLOCT1[@]}


# create ip-config-file:
echo \#!/bin/bash > ./pinet-server.cfg
echo \#PINAME_=$PCNAME_B_ >> ./pinet-server.cfg

#in one line
#echo $'#!/bin/bash\n#\$PCNAME_B_' > ./pinet-server.cfg

##echo \#IPBASE=\"192.168.93\" >> ./pinet-server.cfg
echo "#IPBASE=\"$IPBASE\"" >> ./pinet-server.cfg

echo PIPOOL=\( ${PIPOOLNAMENR1[@]} \) >> ./pinet-server.cfg
echo PIPOOLIPSUFFIX=\( ${PIPOOLOCT1[@]} \) >> ./pinet-server.cfg

echo \#Found $PCNAME_B_: $((SERVN / 2)) >> ./pinet-server.cfg
echo \#Used $PCNAME_B_ : ${#PIPOOLNAMENR1[@]} >> ./pinet-server.cfg
####

