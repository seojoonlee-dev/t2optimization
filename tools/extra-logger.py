#!/usr/bin/python3 -I
"""1 Hz logger for model training data (run as root: SMC keys need it).

The only logger: fand writes no CSV. Each row has the SMC keys the model and
fand use (the five Ts1S sensors, Ts1S, the SMC budget and throttle, CPU, dGPU
and memory power, fans, die temperatures), the other power rails and
temperatures (Wi-Fi/SSD path power POLR + PORR + PSLC, charging power
PDTR - PSTR, battery and chassis sensors), RAPL power, CPU load, Wi-Fi
traffic, iGPU clocks, SSD, display, keyboard light and charging state, and
fand's own state from its status file while it runs. sim/data.py trains on
these CSVs and t2opt-fand --replay / --sim run on them. Writes
data/extra-*.csv, owned by whoever owns the checkout.

Core temperatures come from the thermal MSRs at 10 Hz between the 1 s SMC
reads (coretemp's sysfs only refreshes once a second, so short spikes fall
between samples). The hottest-core model needs the hottest core and how many
cores share the power; the kernel's throttle counters are logged too.

Read-only: never touches fans, RAPL limits or services, and never reads the
amdgpu hwmon (that keeps the dGPU awake). Every SMC key read holds the SMC
lock that fand takes around its own reads, so both can run at once.

    sudo tools/install-extra-logger.sh        install and start at boot
    systemctl start|stop t2opt-extralog
"""

import fcntl
import glob
import json
import os
import signal
import struct
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the CSVs belong to whoever owns the checkout
USER_UID = os.stat(REPO).st_uid
OUT_DIR = os.path.join(REPO, "data")
# applesmc binds to the T2's ACPI SMC device
SMC = "/sys/bus/acpi/devices/APP0001:00"
# shared with fand (daemon/src/hw.rs SMC_LOCK)
SMC_LOCK = "/run/t2opt-smc.lock"

KEYS = [
    # used by the model and fand
    "TW0P", "TM0P", "Ts0P", "TG0P", "Ts1P", "Ts1S", "PZ0E", "PZ0F", "PZ0T", "PC0R", "PG0R", "F0Ac", "F1Ac", "TCMX", "TGDD",
    "PM0C",
    # fan limits (min / max rpm), for --replay away from this machine
    "F0Mn", "F1Mn", "F0Mx", "F1Mx",
    # other rails and sensors
    "PCPC", "PCPG", "PCPT", "PDTR", "PH0C", "PH0R", "PH1C", "PH1R", "PHPC", "PO5R", "POLR", "PORR", "PPBR", "PSLC", "PSTR",
    "TA0V", "TB0T", "TB1T", "TB2T", "TC0E", "TC0F", "TC0P", "TCGC", "TCSA", "TCXC", "TF0S", "TGDF", "TGVP", "TH0F", "TH0X",
    "TH0a", "TH0b", "TPCD", "TTLD", "TTRD", "TaLC", "TaRC", "Th1H", "Th2H", "Tm0P", "Ts0S", "Ts2S",
]

# per 1 s window: hottest core / package (max of 10 Hz samples), mean of the
# core means, busy logical CPUs (sum of busy fractions) and those over 50 %,
# and the kernel's thermal throttle counters (cumulative; cores summed)
CORE = ["core_max_c", "pkg_max_c", "core_mean_c", "busy_cpus", "cpus_busy50", "throttle_core", "throttle_pkg"]
FAST_S = 0.1
IA32_THERM_STATUS = 0x19C
IA32_PACKAGE_THERM_STATUS = 0x1B1
MSR_TEMPERATURE_TARGET = 0x1A2

# pl1_w / pl2_w: the RAPL package limits actually in force (fand's own log has
# the plan's PL1, before the hottest-core limit cuts PL1/PL2)
SYS = ["nvme_c", "disk_mb_s", "backlight_pct", "kbd_backlight_pct", "ac_online", "charging", "ext_displays", "pl1_w", "pl2_w"]
RAPL = "/sys/class/powercap/intel-rapl:0"
RAPL_DOMAINS = [RAPL, RAPL + "/intel-rapl:0:0", RAPL + "/intel-rapl:0:1"]

# RAPL package / core / iGPU power (W), Wi-Fi traffic received + sent (MB/s)
# on the first wireless interface, iGPU actual / requested clock (MHz)
METER = ["rapl_pkg_w", "rapl_core_w", "rapl_gfx_w", "wifi_mb_s", "igpu_act_mhz", "igpu_req_mhz"]

# fand's status file (written every second while it runs, removed on exit)
FAND_STATUS = "/run/t2opt-fand.json"
# column -> status key; names match the CSV fand used to write
FAND = {
    "mode": "mode",
    "power_mode": "power_mode",
    "cmd_rpm": "cmd_rpm",
    "reason": "reason",
    "pred_ts1s_max": "pred_ts1s_max",
    "kf_ambient": "room_offset_k",
    "fan_ceiling": "fan_ceiling_rpm",
    "cpu_pl1_w": "cpu_pl1_w",
    "dgpu_cap_w": "dgpu_cap_w",
    "dgpu_active": "dgpu_active",
}


def rd(p):
    with open(p) as f:
        return f.read().strip()


def decode(typ, b):
    if typ == "flt":
        return struct.unpack("<f", b[:4])[0]
    if typ == "sp78":
        return struct.unpack(">h", b[:2])[0] / 256
    if typ == "sp87":
        return struct.unpack(">h", b[:2])[0] / 128
    if typ == "ui8":
        return b[0]
    if typ == "ui16":
        return struct.unpack(">H", b[:2])[0]
    raise ValueError(typ)


def first(pattern, test=lambda p: True):
    def ok(p):
        try:
            return test(p)
        except OSError:  # e.g. an hwmon dir without a name file
            return False

    return next((p for p in sorted(glob.glob(pattern)) if ok(p)), None)


def level(d):
    """brightness / max_brightness in dir d, %"""
    try:
        m = float(rd(d + "/max_brightness"))
        return f"{100 * float(rd(d + '/brightness')) / m:.1f}" if m > 0 else ""
    except (OSError, ValueError):
        return ""


def disk_bytes():
    """bytes read + written on nvme0n1 so far"""
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            # major minor name reads merged sectors_read ms writes merged sectors_written
            if p[2] == "nvme0n1":
                return (int(p[5]) + int(p[9])) * 512
    return None


class Sys:
    def __init__(self):
        self.nvme = self.backlight = self.kbd = self.ac = self.bat = None
        self.disk = None

    def find(self):
        # devices can register after we start at boot (the keyboard LED does), so look again while any is missing
        kind = lambda t: lambda p: os.path.exists(p + "/type") and rd(p + "/type") == t
        self.nvme = self.nvme or first("/sys/class/hwmon/hwmon*", lambda p: rd(p + "/name") == "nvme")
        self.backlight = self.backlight or first("/sys/class/backlight/*", lambda p: "appletb" not in p)
        self.kbd = self.kbd or first("/sys/class/leds/*kbd_backlight")
        self.ac = self.ac or first("/sys/class/power_supply/*", kind("Mains"))
        self.bat = self.bat or first("/sys/class/power_supply/*", kind("Battery"))

    def row(self):
        if not all((self.nvme, self.backlight, self.kbd, self.ac, self.bat)):
            self.find()
        out = {}
        try:
            out["nvme_c"] = f"{int(rd(self.nvme + '/temp1_input')) / 1000:.1f}" if self.nvme else ""
        except (OSError, ValueError):
            out["nvme_c"] = ""
        b, t = disk_bytes(), time.monotonic()
        prev, self.disk = self.disk, (b, t)
        ok = prev and b is not None and prev[0] is not None and b >= prev[0] and 0.2 < t - prev[1] < 5
        out["disk_mb_s"] = f"{(b - prev[0]) / (t - prev[1]) / 1e6:.2f}" if ok else ""
        out["backlight_pct"] = level(self.backlight) if self.backlight else ""
        out["kbd_backlight_pct"] = level(self.kbd) if self.kbd else ""
        try:
            out["ac_online"] = rd(self.ac + "/online") if self.ac else ""
            out["charging"] = ("1" if rd(self.bat + "/status") == "Charging" else "0") if self.bat else ""
        except OSError:
            out["ac_online"] = out["charging"] = ""
        # connected displays other than the built-in panel and the Touch Bar
        ext = 0
        for s in glob.glob("/sys/class/drm/card*-*/status"):
            name = os.path.basename(os.path.dirname(s))
            if "-eDP-" not in name and "-USB-" not in name and rd(s) == "connected":
                ext += 1
        out["ext_displays"] = str(ext)
        for k, c in (("pl1_w", 0), ("pl2_w", 1)):
            try:
                out[k] = f"{int(rd(f'{RAPL}/constraint_{c}_power_limit_uw')) / 1e6:.2f}"
            except (OSError, ValueError):
                out[k] = ""
        return [out[k] for k in SYS]


class Meters:
    """RAPL power, Wi-Fi traffic and iGPU clocks (rates from two reads)."""

    def __init__(self):
        self.last_e = self.last_w = None
        self.wifi = first("/sys/class/net/*", lambda p: os.path.exists(p + "/wireless"))
        self.gt = first("/sys/class/drm/card*/gt/gt0", lambda p: os.path.exists(p + "/rps_act_freq_mhz"))

    def row(self):
        t = time.monotonic()
        out = []
        try:
            e = [int(rd(d + "/energy_uj")) for d in RAPL_DOMAINS]
        except (OSError, ValueError):
            e = None
        prev, self.last_e = self.last_e, (e, t)
        if e and prev and prev[0] and 0.2 < t - prev[1] < 5:
            for d, a, b in zip(RAPL_DOMAINS, prev[0], e):
                if b < a:  # the counter wrapped
                    try:
                        b += int(rd(d + "/max_energy_range_uj"))
                    except (OSError, ValueError):
                        out.append("")
                        continue
                out.append(f"{(b - a) / (t - prev[1]) / 1e6:.2f}")
        else:
            out += [""] * 3
        try:
            w = sum(int(rd(f"{self.wifi}/statistics/{f}")) for f in ("rx_bytes", "tx_bytes")) if self.wifi else None
        except (OSError, ValueError):
            w = None
        prev, self.last_w = self.last_w, (w, t)
        ok = w is not None and prev and prev[0] is not None and w >= prev[0] and t - prev[1] > 0.2
        out.append(f"{(w - prev[0]) / (t - prev[1]) / 1e6:.3f}" if ok else "")
        for f in ("rps_act_freq_mhz", "rps_cur_freq_mhz"):
            try:
                out.append(rd(f"{self.gt}/{f}") if self.gt else "")
            except OSError:
                out.append("")
        return out


def fand_row():
    """fand's state, if it is running (empty columns otherwise)."""
    try:
        with open(FAND_STATUS) as f:
            st = json.load(f)
        if time.time() - st["time"] > 3:  # stale: fand died without removing it
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        return [""] * len(FAND)
    out = []
    for k in FAND.values():
        v = st.get(k)
        out.append("" if v is None else str(int(v)) if isinstance(v, bool) else str(v))
    return out


def msr(fd, reg):
    return struct.unpack("<Q", os.pread(fd, 8, reg))[0]


def cpu_times():
    """per logical CPU: (busy, total) jiffies"""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            p = line.split()
            if p[0].startswith("cpu") and p[0] != "cpu":
                v = [int(x) for x in p[1:9]]
                # idle + iowait count as idle
                out[p[0]] = (sum(v) - v[3] - v[4], sum(v))
    return out


class Cores:
    def __init__(self):
        # one logical CPU per physical core
        first_cpu = {}
        for c in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"), key=lambda c: int(c.rsplit("cpu", 1)[1])):
            first_cpu.setdefault(rd(c + "/topology/core_id"), int(c.rsplit("cpu", 1)[1]))
        self.cpus = list(first_cpu.values())
        self.fds = [os.open(f"/dev/cpu/{n}/msr", os.O_RDONLY) for n in self.cpus]
        self.tjmax = (msr(self.fds[0], MSR_TEMPERATURE_TARGET) >> 16) & 0xFF
        self.prev = cpu_times()
        self.reset()

    def reset(self):
        self.core_max = self.pkg_max = None
        self.mean_sum, self.n = 0.0, 0

    def sample(self):
        # digital readout (bits 22:16) is degrees below TjMax; skip it when not valid (bit 31)
        temps = []
        for fd in self.fds:
            v = msr(fd, IA32_THERM_STATUS)
            if v >> 31 & 1:
                temps.append(self.tjmax - (v >> 16 & 0x7F))
        if not temps:
            return
        pkg = self.tjmax - (msr(self.fds[0], IA32_PACKAGE_THERM_STATUS) >> 16 & 0x7F)
        self.core_max = max(self.core_max or 0, max(temps))
        self.pkg_max = max(self.pkg_max or 0, pkg)
        self.mean_sum += sum(temps) / len(temps)
        self.n += 1

    def row(self):
        now = cpu_times()
        busy = []
        for k, (b, t) in now.items():
            if k in self.prev and t > self.prev[k][1]:
                busy.append((b - self.prev[k][0]) / (t - self.prev[k][1]))
        self.prev = now
        # sibling threads share a core's counter and every CPU repeats the package's,
        # so count each core once and the package once
        tt = "/sys/devices/system/cpu/cpu{}/thermal_throttle/{}_throttle_count"
        thr = {}
        for kind, cpus in (("core", self.cpus), ("package", self.cpus[:1])):
            try:
                thr[kind] = sum(int(rd(tt.format(n, kind))) for n in cpus)
            except (OSError, ValueError):
                thr[kind] = ""
        out = [
            "" if self.core_max is None else str(self.core_max),
            "" if self.pkg_max is None else str(self.pkg_max),
            f"{self.mean_sum / self.n:.2f}" if self.n else "",
            f"{sum(busy):.2f}" if busy else "",
            str(sum(1 for x in busy if x > 0.5)) if busy else "",
            str(thr["core"]),
            str(thr["package"]),
        ]
        self.reset()
        return out


def main():
    lock = open(SMC_LOCK, "a")
    idx_fd = os.open(SMC + "/key_at_index", os.O_WRONLY)
    data_fd = os.open(SMC + "/key_at_index_data", os.O_RDONLY)

    found = {}
    for i in range(int(rd(SMC + "/key_count"))):
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            os.pwrite(idx_fd, str(i).encode(), 0)
            name = rd(SMC + "/key_at_index_name")
            if name in KEYS:
                # trimmed: "ui8 " reads as "ui8"
                found[name] = (i, rd(SMC + "/key_at_index_type"))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    keys = [(k, *found[k]) for k in KEYS if k in found]
    missing = [k for k in KEYS if k not in found]
    if missing:
        print("SMC keys not found:", " ".join(missing), flush=True)

    def smc_row():
        out = []
        for _, i, typ in keys:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                os.pwrite(idx_fd, str(i).encode(), 0)
                out.append(f"{decode(typ, os.pread(data_fd, 8, 0)):.4g}")
            except (OSError, ValueError, struct.error, IndexError):
                out.append("")
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return out

    sysm = Sys()
    cores = Cores()
    meters = Meters()
    path = os.path.join(OUT_DIR, time.strftime("extra-%Y%m%d-%H%M%S.csv"))
    f = open(path, "w", buffering=1)
    os.chown(path, USER_UID, USER_UID)
    print("log:", path, flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    f.write(",".join(["timestamp", "unix_time", "smc_read_ms"] + CORE + SYS + METER + list(FAND) + [k[0] for k in keys]) + "\n")
    nxt = time.monotonic()
    while True:
        nxt += 1.0
        t0 = time.monotonic()
        row = smc_row()
        ms = (time.monotonic() - t0) * 1e3
        f.write(",".join([time.strftime("%F %T"), f"{time.time():.1f}", f"{ms:.1f}"] + cores.row() + sysm.row() + meters.row() + fand_row() + row) + "\n")
        if nxt - time.monotonic() < -5:  # woke from suspend: restart the 1 s schedule
            nxt = time.monotonic()
        # sample the cores at 10 Hz until the next row
        while True:
            try:
                cores.sample()
            except OSError:
                pass
            delay = nxt - time.monotonic()
            if delay <= 0:
                break
            time.sleep(min(FAST_S, delay))


if __name__ == "__main__":
    main()
