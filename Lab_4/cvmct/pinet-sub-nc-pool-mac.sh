# #!/bin/bash
# # pinet-sub-nc-pool-mac.sh  (mac-friendly receive-port factory)

# CFG="$(dirname "$0")/pinet-mac.cfg"
# [ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
# . "$CFG"

# # optional: one Pi only
# [ -n "$1" ] && PIPOOL=( "$1" )

# mkdir -p "$OUTPUTDIR"

# J=0
# for I in "${PIPOOL[@]}"; do
#     # skip non-numeric host indices (“43b” etc.)
#     [[ "$I" =~ ^[0-9]+$ ]] || { echo "skip non-numeric Pi id: $I"; continue; }

#     PORT=$((CLIENTPORTBASE + I))
#     # launch listener only if the port is free
#     if ! lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
#         ncat -4 -l "$PORT" </dev/null | tar xv -C "$OUTPUTDIR" >/dev/null &
#         ((++J))
#     fi
# done
# echo "$J client receive port(s) created"

#!/bin/bash
# -----------------------------------------------------------
#  pinet-sub-nc-pool-mac.sh
#  macOS‑friendly variant of Wolfgang Stinner’s ncat‑pool helper.
#  • Starts one  ncat | tar  receive pair per alive Raspberry‑Pi.
#  • "Alive" == answers a 1‑second ping probe (or explicitly given on CLI)
#  • Skips non‑numeric Pi IDs (e.g. 43b) so arithmetic never breaks.
#  • Each listener self‑destroys after 30 s (-w 30) if no Pi connects.
# -----------------------------------------------------------
#!/bin/bash
# pinet-sub-nc-pool-mac.sh  –  robust, self-timing version for macOS
set -euo pipefail

CFG="$(dirname "$0")/pinet-mac.cfg"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 1; }
. "$CFG"

[[ -n "${1:-}" ]] && PIPOOL=( "$1" )   # optional: single-Pi mode

mkdir -p "$OUTPUTDIR"

PING_TMO=1      # seconds for ping probe
LISTEN_TMO=10   # seconds to wait for first connection
alive=()

for id in "${PIPOOL[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "skip non-numeric Pi id: $id"; continue; }
  host="${PINAME_}${id}${DOMAIN_SUFFIX}"
  if ping -c1 -W$PING_TMO "$host" >/dev/null 2>&1; then
      alive+=( "$id" )
  else
      echo "Pi $host offline – no listener"
  fi
done

started=0
for id in "${alive[@]}"; do
  port=$((CLIENTPORTBASE + id))
  if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "listener on $port already exists – skipping"
      continue
  fi

  (
      nc -4 -l "$port" </dev/null | tar xv -C "$OUTPUTDIR" >/dev/null &
      PID_NC=$!
      PID_TAR=$!
      SECS=$LISTEN_TMO
      while (( SECS )); do
          kill -0 "$PID_NC" 2>/dev/null || exit        # nc exited => done
          sleep 1; (( SECS-- ))
      done
      echo "No connection after ${LISTEN_TMO}s on $port – killing nc/tar"
      kill "$PID_NC" "$PID_TAR" 2>/dev/null || true
  ) &
  echo "listener for Pi #$id on port $port (watchdog ${LISTEN_TMO}s)"
  ((++started))
done

echo "$started client receive port(s) created"