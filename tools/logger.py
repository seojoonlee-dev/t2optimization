#!/usr/bin/python3 -I
"""Passive 1 Hz logger for model training data (run as root: SMC keys need it).

Read-only: never touches fans, RAPL limits or services, and never reads the
amdgpu hwmon (that keeps the dGPU awake). Writes one CSV per run to the repo's
data/ dir, same key columns as the 2026-10-19 stress log, owned by the user.

    sudo systemd-run --unit=t2opt-logger -p Nice=10 \
        /usr/bin/python3 -I "$PWD/tools/logger.py"
    sudo systemctl stop t2opt-logger
"""

import glob
import os
import signal
import struct
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the CSVs belong to whoever owns the checkout
USER_UID = os.stat(REPO).st_uid
OUT_DIR = os.path.join(REPO, "data")
SMC = "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:102/APP0001:00"
DGPU = "/sys/bus/pci/devices/0000:01:00.0"
# temperatures, power rails, fans (current, voltage keys are not needed)
WANT = {"T": ("sp78",), "P": ("flt", "sp87"), "F": ("flt", "ui8 ", "ui16")}


def rd(p):
    with open(p) as f:
        return f.read().strip()


def gt_dir():
    for c in glob.glob("/sys/class/drm/card*"):
        try:
            if os.path.basename(os.readlink(c + "/device/driver")) == "i915":
                return c + "/gt/gt0"
        except OSError:
            pass


def decode(typ, b):
    if typ == "flt":
        return struct.unpack("<f", b[:4])[0]
    if typ == "sp78":
        return struct.unpack(">h", b[:2])[0] / 256
    if typ == "sp87":
        return struct.unpack(">h", b[:2])[0] / 128
    if typ == "ui8 ":
        return b[0]
    if typ == "ui16":
        return struct.unpack(">H", b[:2])[0]
    raise ValueError(typ)


def energy(zone):
    return int(rd(f"/sys/class/powercap/{zone}/energy_uj"))


def main():
    gt = gt_dir()
    idx_fd = os.open(SMC + "/key_at_index", os.O_WRONLY)
    data_fd = os.open(SMC + "/key_at_index_data", os.O_RDONLY)
    keys = []
    for i in range(int(rd(SMC + "/key_count"))):
        os.pwrite(idx_fd, str(i).encode(), 0)
        name, typ = rd(SMC + "/key_at_index_name"), rd(SMC + "/key_at_index_type")
        if typ in WANT.get(name[:1], ()):
            keys.append((i, name, typ))

    def smc_row():
        out = []
        for i, _, typ in keys:
            try:
                os.pwrite(idx_fd, str(i).encode(), 0)
                out.append(f"{decode(typ, os.pread(data_fd, 8, 0)):.4g}")
            except (OSError, ValueError, struct.error):
                out.append("")
        return out

    path = os.path.join(OUT_DIR, time.strftime("passive-%Y%m%d-%H%M%S.csv"))
    f = open(path, "w", buffering=1)
    os.chown(path, USER_UID, USER_UID)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    f.write(
        ",".join(
            ["timestamp", "unix_time", "fan_smc_controlled", "rapl_pkg_w", "rapl_gfx_w",
             "igpu_act_mhz", "igpu_cur_mhz", "dgpu_active"]
            + [k[1] for k in keys]
        )
        + "\n"
    )
    prev = (energy("intel-rapl:0"), energy("intel-rapl:0:1"), time.monotonic(), time.time())
    nxt = time.monotonic()
    while True:
        nxt += 1.0
        row = smc_row()
        now = (energy("intel-rapl:0"), energy("intel-rapl:0:1"), time.monotonic(), time.time())
        dt = now[2] - prev[2]
        # monotonic stops in suspend, wall time doesn't: a gap means the energy
        # delta spans the suspend, so leave power blank for that sample
        if now[3] - prev[3] > dt + 2 or dt <= 0:
            pkg = gfx = ""
        else:
            pkg = f"{((now[0] - prev[0]) % (1 << 32)) / dt / 1e6:.2f}"
            gfx = f"{((now[1] - prev[1]) % (1 << 32)) / dt / 1e6:.2f}"
        prev = now
        manual = rd(SMC + "/fan1_manual")
        f.write(
            ",".join(
                [time.strftime("%F %T"), f"{now[3]:.1f}", "0" if manual == "1" else "1", pkg, gfx,
                 rd(gt + "/rps_act_freq_mhz"), rd(gt + "/rps_cur_freq_mhz"),
                 "1" if rd(DGPU + "/power/runtime_status") == "active" else "0"]
                + row
            )
            + "\n"
        )
        delay = nxt - time.monotonic()
        if delay < -5:  # woke from suspend: restart the 1 s schedule
            nxt = time.monotonic()
        time.sleep(max(0.0, delay))


if __name__ == "__main__":
    main()
