@echo off
REM create-server-list-simple.bat - Windows script for CVMCT server list

setlocal enabledelayedexpansion

echo Creating server list for CVMCT on Windows...

REM Set default values
set PINAME_=cvpi
set IPBASE=192.168.93
set PIPOOL=3 4 6 36 40 41 42 43

REM Check if pinet.cfg exists and load it
if exist pinet.cfg (
    echo Loading configuration from pinet.cfg
    for /f "tokens=1,* delims==" %%a in (pinet.cfg) do (
        set line=%%a
        set line=!line: =!
        if "!line!"=="PINAME_" set PINAME_=%%b
        if "!line!"=="IPBASE" set IPBASE=%%b
        if "!line!"=="PIPOOL" set PIPOOL=%%b
    )
)

REM Clean up IPBASE (remove quotes and trailing dot)
set IPBASE=!IPBASE:"=!
set IPBASE=!IPBASE:.=!

echo Using network: %IPBASE%.0/24
echo Scanning network for Raspberry Pis...

REM Save scan results to a file
nmap -sn %IPBASE%.0/24 > pi_scan.txt

REM Extract IP addresses and MAC addresses
echo Processing scan results...
findstr /c:"Raspberry Pi" pi_scan.txt > pi_mac.txt
findstr /b /c:"Nmap scan report" pi_scan.txt > pi_ip.txt

REM Create a combined file with IP and MAC info
type nul > pi_combined.txt
for /f "tokens=5" %%i in (pi_ip.txt) do (
    echo %%i >> pi_combined.txt
)

REM Count found Pis
set count=0
for /f %%a in (pi_combined.txt) do (
    set /a count+=1
)

if %count%==0 (
    echo No Raspberry Pi devices found on the network!
    goto :cleanup
)

echo Found %count% Raspberry Pi devices.

REM Create arrays for IP addresses and last octets
set "pi_ips="
set "last_octets="

for /f %%i in (pi_combined.txt) do (
    echo Found Raspberry Pi at %%i
    set "pi_ips=!pi_ips! %%i"
    
    REM Extract the last octet
    for /f "tokens=4 delims=." %%o in ("%%i") do (
        set "last_octets=!last_octets! %%o"
        echo   Last octet: %%o
    )
)

REM Remove leading space
set "pi_ips=!pi_ips:~1!"
set "last_octets=!last_octets:~1!"

REM Use PIPOOL numbers in order
set "used_pool="
set used_count=0

REM Convert PIPOOL from array notation to space-separated list
set pipool_clean=!PIPOOL!
set pipool_clean=!pipool_clean:{=!
set pipool_clean=!pipool_clean:}=!
set pipool_clean=!pipool_clean:..= !

REM Use the first N numbers from PIPOOL to match the number of discovered Pis
set counter=0
for %%p in (!pipool_clean!) do (
    if !counter! LSS %count% (
        set "used_pool=!used_pool! %%p"
        set /a counter+=1
    )
)

REM Remove leading space
set "used_pool=!used_pool:~1!"

REM Create the server config file
echo Creating server configuration file...
echo #!/bin/bash > pinet-server.cfg
echo #PINAME_=%PINAME_% >> pinet-server.cfg
echo #IPBASE="%IPBASE%" >> pinet-server.cfg
echo PIPOOL=^( %used_pool% ^) >> pinet-server.cfg
echo PIPOOLIPSUFFIX=^( %last_octets% ^) >> pinet-server.cfg
echo #Found Raspberry Pi devices: %count% >> pinet-server.cfg
echo #Matched Raspberry Pis: %counter% >> pinet-server.cfg

echo.
echo Server configuration created successfully:
echo IPBASE: %IPBASE%
echo PIPOOL: %used_pool%
echo PIPOOLIPSUFFIX: %last_octets%
echo.
echo Configuration saved to pinet-server.cfg

:cleanup
REM Clean up temp files
del pi_scan.txt
del pi_mac.txt
del pi_ip.txt
del pi_combined.txt

endlocal