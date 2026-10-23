#!/bin/bash
# Install the scx_lavd CPU scheduler and run it at boot through scx_loader.
#
#   sudo daemon/setup-scx.sh
#
# scx_lavd (sched_ext) prefers latency-critical threads (compositor, input,
# audio) over batch work, so the desktop stays responsive under load.
# scx_loader starts it in Auto mode (--autopilot); t2opt-fand then switches
# its profile with the power mode: eco -> PowerSave (--powersave),
# balanced -> Auto, perf -> Gaming (--performance).
#
# Undo: systemctl disable --now scx_loader   (the kernel's default EEVDF scheduler
#       takes over at once; no reboot), then pacman -R scx-tools scx-scheds
set -euo pipefail

CONF_DIR=/etc/scx_loader
CONF=$CONF_DIR/config.toml

if [[ $EUID -ne 0 ]]; then
  echo "run as root" >&2
  exit 1
fi

echo "== installing scx-scheds scx-tools"
pacman -S --needed --noconfirm scx-scheds scx-tools

install -d "$CONF_DIR"
if [[ -e $CONF ]]; then
  cp -a "$CONF" "$CONF.bak-$(date +%Y%m%d-%H%M%S)"
fi
cat > "$CONF" <<'EOF'
# scx_lavd at boot in Auto mode (--autopilot). t2opt-fand switches the
# profile with its power mode (eco PowerSave, balanced Auto, perf Gaming).
default_sched = "scx_lavd"
default_mode = "Auto"
EOF
echo "== $CONF written"

systemctl enable scx_loader.service
# restart, so an already running loader reads the new config
systemctl restart scx_loader.service

for _ in {1..20}; do
  [[ $(</sys/kernel/sched_ext/state) == enabled ]] && break
  sleep 0.5
done
state=$(</sys/kernel/sched_ext/state)
ops=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || echo none)
echo "== sched_ext state: $state, scheduler: $ops"
if [[ $state != enabled ]]; then
  echo "scx_lavd did not start; see: journalctl -u scx_loader" >&2
  exit 1
fi
