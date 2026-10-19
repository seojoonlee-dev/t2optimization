#!/bin/bash
# Body of the t2opt-night unit: workload -> stop logger -> report -> power off
# once the logger's CSV is saved. A manual `systemctl stop t2opt-night` kills
# this script before the power-off.
set -u
HOURS=${1:-6}
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA=$(dirname "$TOOLS")/data

/usr/bin/systemd-inhibit --what=sleep --mode=block --who=t2opt --why="overnight data run" \
  /usr/bin/python3 -I "$TOOLS/overnight.py" "$HOURS"
# stopping the logger closes its CSV; then flush everything to disk
systemctl stop t2opt-logger-night
sync
# report only, for the morning; it does not decide the power-off
/usr/bin/python3 -I "$TOOLS/verify_night.py" "$HOURS" || true
sync
csv=$(ls -t "$DATA"/passive-*.csv 2>/dev/null | head -1)
if [[ -n $csv && -s $csv ]]; then
  sleep 10
  systemctl poweroff
fi
