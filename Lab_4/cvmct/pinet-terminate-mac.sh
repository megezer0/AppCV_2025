#!/usr/bin/env bash
# ------------------------------------------------------------
#  pinet-terminate-mac.sh   (v2 – local + remote cleanup)
#  • Kills the Mac-side Pi-Net broker (port 31004) and 311xx listeners.
#  • SSHes every Pi in PIPOOL and kills:
#        – ncat --broker             (31004)
#        – back-door listener        (31010)
#        – any 311xx data listener
# ------------------------------------------------------------
set -euo pipefail

CFG="$(dirname "$0")/pinet-mac.cfg"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 1; }
. "$CFG"                  # brings in BROKERPORT, CLIENTPORTBASE, PIPOOL, PINAME_, PIPOWORD pattern

PORT=$BROKERPORT          # 31004
LISTEN_RE="ncat .* -l[[:space:]]*31[0-1][0-9][0-9]"   # 31100-31199

echo "──────────────────────────────────────────────"
echo "■ 1/3  Killing Mac broker on $PORT …"
lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null | xargs -r kill || true

echo "■ 2/3  Killing Mac 311xx listeners + tar children …"
for pid in $(pgrep -f "$LISTEN_RE" || true); do
  echo "  • listener PID $pid"
  pkill -P "$pid" -f tar || true
  kill "$pid"
done

echo "Remaining ncat listeners on the Mac:"
lsof -nP -iTCP -sTCP:LISTEN | grep -E 'ncat|COMMAND' || echo "  none"

echo "──────────────────────────────────────────────"
echo "■ 3/3  Cleaning brokers on each Raspberry Pi …"
for id in "${PIPOOL[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "  • skip non-numeric id $id"; continue; }
  host="${PINAME_}${id}.local"
  user="${PINAME_}${id}"
  pass="16AppCV\$&\$${id}"

  ping -c1 -W1 "$host" >/dev/null 2>&1 \
      || { echo "  • $host offline – skipped"; continue; }

  echo "  • $host"
  sshpass -p "$pass" \
    ssh -o ConnectTimeout=4 -o StrictHostKeyChecking=no \
        "$user@$host" \
        "pkill -f 'ncat .* --broker'            2>/dev/null || true ; \
         pkill -f 'ncat .* -l .* $PIBACKDOORPORT' 2>/dev/null || true ; \
         pkill -f 'ncat .* -l 31[0-9][0-9][0-9]' 2>/dev/null || true" \
    || true
done

echo "──────────────────────────────────────────────"
echo "Cleanup complete."