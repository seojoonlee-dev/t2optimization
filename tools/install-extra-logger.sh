#!/bin/bash
# Install the extra logger as a service that starts at boot. It runs from this
# checkout and writes data/extra-*.csv.
#
#   sudo tools/install-extra-logger.sh
#   sudo systemctl disable --now t2opt-extralog     undo
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

mkdir -p "$(dirname "$D")/data"
chown --reference="$(dirname "$D")" "$(dirname "$D")/data"
sed "s|@REPO@|$(dirname "$D")|g" "$D/t2opt-extralog.service" > /etc/systemd/system/t2opt-extralog.service
chmod 0644 /etc/systemd/system/t2opt-extralog.service
systemctl daemon-reload
systemctl enable --now t2opt-extralog
sleep 3
systemctl --no-pager --lines=3 status t2opt-extralog
