#!/bin/bash
#
# Wolfgang Stinner, Berlin  22.01.2018, 29.06.2016
# 
# **** "pinet-list.sh" für pi-Farm***
# version: 2
# needs Nmap
# "pinet-list.sh [all]" 
# Listet standardmäßig alle cvpi** mit Namen,IP,(MAC)
# mit option "all" werden ALLE Namen gelistet
#
echo " syntax:pinet-list.sh [all]" 
echo " Please wait"
echo 

# nmap -sP --system-dns 192.168.93.0/24 > null && arp -a | sort -n --key=1.5
NETIP=192.168.93.0/24
PCNAME_B_=cvpi

if [ -n "$1" ] 
then
# list all
nmap -sP --system-dns $NETIP | grep 'Nmap scan'| cut -d" " -f5,6  | sort -n --key=1.5 | nl
else
#list only PCNAME_B_
nmap -sP --system-dns $NETIP |  grep -o "$PCNAME_B_.\{1,\}" | sort -n --key=1.5 | nl
fi
