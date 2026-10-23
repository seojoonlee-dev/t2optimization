#!/bin/bash
# Start the overnight data run as root: suspend off, 1 Hz logger, quiet workload
# schedule, then power off once the log is saved. Returns immediately; the run
# continues in the background as systemd units.
#
#   sudo tools/overnight-start.sh [hours]   (default 6)
#   sudo systemctl stop t2opt-night t2opt-logger-night   abort, no power-off
set -euo pipefail
HOURS=${1:-6}
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# other power-limit services to pause for the run (local.env)
[[ -f $TOOLS/../local.env ]] && . "$TOOLS/../local.env"
ENV=(-E "EXTRA_UNITS=${EXTRA_UNITS:-}")
SECS=$(python3 -c "print(int(float('$HOURS') * 3600))")

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

# runtime-only mask: gone after any reboot, and the run unmasks it at the end
systemctl mask --runtime sleep.target suspend.target hibernate.target \
  hybrid-sleep.target suspend-then-hibernate.target
systemd-run --unit=t2opt-logger-night -p Nice=10 -p RuntimeMaxSec=$((SECS + 3600)) \
  /usr/bin/python3 -I "$TOOLS/logger.py"
systemd-run --unit=t2opt-night "${ENV[@]}" \
  -p "ExecStopPost=/usr/bin/python3 -I $TOOLS/overnight.py --cleanup" \
  /bin/bash "$TOOLS/overnight-run.sh" "$HOURS"
echo "started: ${HOURS} h; power-off after the log is saved."
echo "watch:   journalctl -fu t2opt-night"
