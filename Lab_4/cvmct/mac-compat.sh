#!/usr/bin/env bash
# mac-compat.sh – fill Linux gaps on macOS

# --- GNU pidof ---
if ! command -v pidof >/dev/null 2>&1 ; then
  pidof() { gpidof "$@"; }
  export -f pidof
fi

# --- ping success pattern ---
export CVMCT_PING_OK_REGEXP='\(1 rec\|1 packets received\)'

# --- pick a non-loopback IPv4 if BROKERIP is empty ---
if [[ -z "$BROKERIP" ]]; then
  BROKERIP=$(ipconfig getifaddr en0 2>/dev/null \
        || ipconfig getifaddr "$(route get default | awk '/interface:/{print $2}')" )
fi
