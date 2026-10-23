#!/bin/bash
# Make t2opt-fand the boot default in place of t2fanrd (and EXTRA_UNITS from
# local.env).
#
#   sudo daemon/install-service.sh
#
# Build first (as your user): cargo build --release --manifest-path daemon/Cargo.toml
# Undo: revert-to-t2fanrd.sh in the project root.
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIT=/etc/systemd/system/t2opt-fand.service
# t2fanrd plus this machine's other fan/power services (local.env)
[[ -f $D/../local.env ]] && . "$D/../local.env"
UNITS="t2fanrd.service ${EXTRA_UNITS:-}"

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi
if [[ ! -x $D/target/release/t2opt-fand ]]; then
  echo "build it first: cargo build --release --manifest-path $D/Cargo.toml" >&2
  exit 1
fi

# the unit runs the binary from this checkout
sed "s|@REPO@|$(dirname "$D")|g; s|@UNITS@|${UNITS% }|g" "$D/t2opt-fand.service" > "$UNIT"
chmod 0644 "$UNIT"
systemctl daemon-reload
# a unit that isn't installed is fine
systemctl disable --now $UNITS 2>/dev/null || true
systemctl enable --now t2opt-fand.service
sleep 2
systemctl --no-pager --lines=5 status t2opt-fand.service
