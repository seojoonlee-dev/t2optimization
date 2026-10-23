#!/bin/bash
# Put t2fanrd (and EXTRA_UNITS from local.env) back in charge, in place of
# t2opt-fand. With TOUCHBAR_CONFIG and FALLBACK_CMD set in local.env, the
# Touch Bar's power-mode button goes over to FALLBACK_CMD too. Run as your
# user; asks for sudo.
#
#   ./revert-to-t2fanrd.sh               keep the scx_lavd scheduler
#   ./revert-to-t2fanrd.sh --scheduler   also go back to the kernel's default
#                                        scheduler (EEVDF)
#
# Re-install fand later with daemon/install-service.sh.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $HERE/local.env ]] && . "$HERE/local.env"
UNITS="t2fanrd.service ${EXTRA_UNITS:-}"
NEW_CMD=$HOME/.local/bin/power-mode

if [[ $EUID -eq 0 ]]; then
  echo "run as your user (it edits your Touch Bar config); it asks for sudo itself" >&2
  exit 1
fi
scheduler=0
case ${1:-} in
  "") ;;
  --scheduler) scheduler=1 ;;
  *) echo "usage: $0 [--scheduler]" >&2; exit 2 ;;
esac

sudo bash -euo pipefail -s "$scheduler" $UNITS <<'EOF2'
scheduler=$1
shift
# stopping runs fand's ExecStopPost: fans back to the SMC, dGPU cap restored
systemctl disable --now t2opt-fand.service 2>/dev/null || true
rm -f /etc/systemd/system/t2opt-fand.service
systemctl daemon-reload
systemctl reset-failed t2opt-fand.service 2>/dev/null || true
# each re-applies its own state on start (fan curve, power limits)
systemctl enable --now "$@"
if (( scheduler )); then
  systemctl disable --now scx_loader.service
fi
EOF2

if [[ -n ${TOUCHBAR_CONFIG:-} && -n ${FALLBACK_CMD:-} && -f $TOUCHBAR_CONFIG ]] && grep -q "$NEW_CMD" "$TOUCHBAR_CONFIG"; then
  sed -i "s|$NEW_CMD --json|$FALLBACK_CMD --json|; s|$NEW_CMD toggle|$FALLBACK_CMD toggle|" "$TOUCHBAR_CONFIG"
  echo "Touch Bar button: back to $FALLBACK_CMD"
fi

for u in $UNITS; do
  echo "$u: $(systemctl is-active "$u")"
done
echo "scheduler: $(cat /sys/kernel/sched_ext/state 2>/dev/null | sed 's/disabled/kernel default (EEVDF)/; s/enabled/scx_lavd/')"
