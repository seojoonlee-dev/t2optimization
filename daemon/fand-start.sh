#!/bin/bash
# Start fand as a transient systemd unit (gone after a reboot, when t2fanrd
# takes over again as before).
#
#   sudo daemon/fand-start.sh            control the fans
#   sudo daemon/fand-start.sh --shadow   plan and log only
#   sudo systemctl stop t2opt-fand              stop
#   journalctl -fu t2opt-fand                   watch decisions
#   cat /run/t2opt-fand.json                    current state
#
# In control mode t2fanrd (and EXTRA_UNITS from local.env) are stopped first:
# fand owns the fans, the CPU power limits and the dGPU cap. When fand stops
# for any reason (stop, crash, watchdog), the fans go back to the SMC and
# those services restart.
# Switch modes with daemon/power-mode (eco | balanced | perf | toggle).
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN=$D/target/release/t2opt-fand
# t2fanrd plus this machine's other fan/power services (local.env)
[[ -f $D/../local.env ]] && . "$D/../local.env"
UNITS="t2fanrd.service ${EXTRA_UNITS:-}"

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi
if [[ ! -x $BIN ]]; then
  echo "build it first (as your user): cargo build --release --manifest-path $D/Cargo.toml" >&2
  exit 1
fi
if systemctl is-active -q t2opt-fand; then
  echo "t2opt-fand is already running (sudo systemctl stop t2opt-fand)" >&2
  exit 1
fi
systemctl reset-failed t2opt-fand 2>/dev/null || true

props=(-p Type=notify -p WatchdogSec=15 -p Nice=-5)
if [[ ${1:-} == --shadow ]]; then
  args=(--shadow)
else
  args=()
  props+=(-p "ExecStopPost=$BIN --release"
          -p "ExecStopPost=-/usr/bin/systemctl start --no-block $UNITS")
  systemctl stop $UNITS 2>/dev/null || true
fi
systemd-run --unit=t2opt-fand "${props[@]}" "$BIN" "${args[@]}"
echo "watch: journalctl -fu t2opt-fand"
