#!/bin/bash
# macOS version of pinet-check-retrieve.sh

PROC=tar             # what we count (same as the original script)
SLEEP=1              # seconds between polls

# Wait until all ‘tar | ncat’ retrieve tasks are gone
while :; do
    RUNNING=$(pgrep -f "$PROC" | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        break
    fi
    printf "\rWaiting for %s retrieve task(s) to finish..." "$RUNNING"
    sleep "$SLEEP"
done
echo
echo "Transfer done."
exit 199