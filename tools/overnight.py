#!/usr/bin/python3 -I
"""Quiet overnight excitation run for model training data (root).

Steps through a seeded random schedule of light loads (CPU threads, iGPU or
dGPU glmark2 off-screen), flat CPU power caps and fixed low fan speeds, so the
logger (tools/logger.py, run separately) sees varied power and fan inputs
without waking anyone.

Noise ceiling = what t2fanrd runs at 75 °C (~3100 rpm, measured 2026-10-19).
To avoid PL2 bursts at load starts, other power-limit services (EXTRA_UNITS
in local.env) are paused for the run and each phase gets a flat cap
(PL1 = PL2); dGPU phases get the 'low' DPM level.

The caps are chosen to stay clear of the ceiling: 14 W RAPL (~18.5 W PC0R)
settled at 2400-2600 rpm under t2fanrd on 2026-10-19, so steady phases should
never trip a guard (a trip cuts a phase short and skews the data).

Guards, checked every second: fans or CPU package above their limits averaged
over GUARD_WINDOW (t2fanrd would pass the ceiling when it takes over; brief
turbo spikes are not noise), fans above HARD_RPM at any instant, Ts1S above
TS1S_LIMIT, any SMC throttle -> all load stops, fans go back to t2fanrd, and
the machine idles for GUARD_COOLDOWN before the schedule resumes.

On exit (end, SIGTERM, crash via the launcher's ExecStopPost --cleanup):
no loads, t2fanrd + EXTRA_UNITS running (they re-apply fans and power
limits), suspend unmasked.

    python3 tools/overnight.py --plan [hours]     print the schedule
"""

import collections
import glob
import os
import pwd
import random
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the workload runs as whoever owns the checkout
UID = os.stat(REPO).st_uid
_PW = pwd.getpwuid(UID)
USER = _PW.pw_name
# applesmc binds to the T2's ACPI SMC device
SMC = "/sys/bus/acpi/devices/APP0001:00"
RAPL = "/sys/class/powercap/intel-rapl:0"
# other power-limit services, from local.env via overnight-start.sh
EXTRA_UNITS = os.environ.get("EXTRA_UNITS", "").split()
SLEEP_TARGETS = ["sleep.target", "suspend.target", "hibernate.target",
                 "hybrid-sleep.target", "suspend-then-hibernate.target"]
UENV = {"XDG_RUNTIME_DIR": f"/run/user/{UID}", "HOME": _PW.pw_dir, "PATH": "/usr/bin"}

NOISE_RPM = 3100
HARD_RPM = 3500
CPU_TEMP_LIMIT = 76.0
GUARD_WINDOW = 20  # s
# the SMC only throttles once the budget drains below zone power; at <= 18.5 W
# that takes Ts1S > 45 for 10+ min, so 46 leaves the 44-45 region in the data
TS1S_LIMIT = 46.0
GUARD_COOLDOWN = 600

# name: (cpu threads, gpu)
LEVELS = {
    "idle": (0, None),
    "cpu1": (1, None),
    "cpu2": (2, None),
    "cpu4": (4, None),
    "cpu8": (8, None),
    "cpu16": (16, None),
    "igpu": (0, "igpu"),
    "igpu+cpu2": (2, "igpu"),
    "dgpu": (0, "dgpu"),
    "dgpu+cpu2": (2, "dgpu"),
}
WEIGHTS = {"idle": 4, "cpu1": 1, "cpu2": 2, "cpu4": 2, "cpu8": 1.5, "cpu16": 2,
           "igpu": 1.5, "igpu+cpu2": 1, "dgpu": 1.5, "dgpu+cpu2": 1}
CAPS_W = [8, 10, 12, 14]
DGPU_CAP_W = 10  # CPU cap while the dGPU works (they share the heatsink)
FAN_OVERRIDES = [2400, 2700]


def schedule(total, seed=20261019):
    """[(level, seconds, cpu cap W, fan)] summing to `total`; fan None = t2fanrd."""
    rng = random.Random(seed)
    plan = [("idle", 900, 12, None)]
    end_idle = 1800
    used = 900 + end_idle
    prev = "idle"
    while used < total:
        names = [n for n in LEVELS if n != prev]
        level = rng.choices(names, [WEIGHTS[n] for n in names])[0]
        secs = rng.randint(300, 1800) if level == "idle" else rng.randint(240, 1200)
        secs = min(secs, total - used)
        threads, gpu = LEVELS[level]
        cap = DGPU_CAP_W if gpu == "dgpu" else rng.choice(CAPS_W)
        fan = None
        if gpu != "dgpu" and threads <= 4 and rng.random() < 0.35:
            fan = rng.choice(FAN_OVERRIDES)
        plan.append((level, secs, cap, fan))
        used += secs
        prev = level
    plan.append(("idle", end_idle, 12, None))
    return plan


def rd(p):
    with open(p) as f:
        return f.read().strip()


def wr(p, v):
    with open(p, "w") as f:
        f.write(str(v))


def pkg_temp_path():
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        if rd(z + "/type") == "x86_pkg_temp":
            return z + "/temp"


class LoggerTail:
    """Ts1S/PZ0T from the last row the logger wrote.

    Never read SMC keys here: key_at_index is one shared register, and a
    second reader interleaves with the logger and corrupts both (2026-10-19).
    """

    def __init__(self, since):
        self.since = since
        self.path = None
        self.cols = None

    def read(self):
        """{name: float} or None if no fresh row is available."""
        if self.path is None:
            files = [f for f in glob.glob(os.path.join(REPO, "data", "passive-*.csv"))
                     if os.path.getmtime(f) >= self.since]
            if not files:
                return None
            self.path = max(files, key=os.path.getmtime)
            with open(self.path) as f:
                self.cols = f.readline().strip().split(",")
        if time.time() - os.path.getmtime(self.path) > 10:
            return None
        with open(self.path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            lines = f.read().decode(errors="replace").splitlines()
        # the last line may be half-written
        for line in reversed(lines):
            vals = line.split(",")
            if len(vals) == len(self.cols):
                row = dict(zip(self.cols, vals))
                try:
                    return {k: float(row[k]) for k in ("Ts1S", "PZ0T")}
                except ValueError:
                    return None
        return None


def amdgpu_dev():
    """The dGPU's PCI device dir, or None."""
    for c in glob.glob("/sys/class/drm/card*"):
        try:
            if os.path.basename(os.readlink(c + "/device/driver")) == "amdgpu":
                return os.path.realpath(c + "/device")
        except OSError:
            pass


def wayland_display():
    """The user's compositor socket, e.g. wayland-1."""
    socks = sorted(p for p in glob.glob(f"/run/user/{UID}/wayland-*") if not p.endswith(".lock"))
    return os.path.basename(socks[0]) if socks else "wayland-0"


def dgpu_low(dev):
    """Stock overdrive table and the 'low' DPM level; only while it is awake."""
    try:
        wr(dev + "/pp_od_clk_voltage", "r")
        wr(dev + "/pp_od_clk_voltage", "c")
    except OSError:
        pass  # overdrive off: the table is stock anyway
    wr(dev + "/power_dpm_force_performance_level", "low")


def fan_rpm():
    return max(int(rd(f"{SMC}/fan{n}_input")) for n in (1, 2))


FANS = [(n, int(rd(f"{SMC}/fan{n}_min")), int(rd(f"{SMC}/fan{n}_max"))) for n in (1, 2)]
procs = []
fan_override = False


def set_loads(threads, gpu):
    global procs
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    procs = []
    # runuser may not forward the signal; make sure nothing is left behind
    subprocess.run(["pkill", "-u", USER, "-x", "yes"])
    subprocess.run(["pkill", "-u", USER, "-f", "glmark2-wayland --off-screen"])
    if gpu:
        env = dict(UENV, WAYLAND_DISPLAY=wayland_display())
        if gpu == "dgpu":
            env["DRI_PRIME"] = "1"
        procs.append(subprocess.Popen(
            ["runuser", "-u", USER, "--", "env", *[f"{k}={v}" for k, v in env.items()],
             "glmark2-wayland", "--off-screen", "--size", "1440x900", "--run-forever",
             "-b", "terrain"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for _ in range(threads):
        procs.append(subprocess.Popen(["runuser", "-u", USER, "--", "nice", "-n", "5", "yes"],
                                      stdout=subprocess.DEVNULL))


def set_cap(watts):
    uw = int(watts * 1e6)
    wr(RAPL + "/constraint_0_time_window_us", 2000000)
    wr(RAPL + "/constraint_0_power_limit_uw", uw)
    wr(RAPL + "/constraint_1_power_limit_uw", uw)


def set_fans(rpm):
    """rpm None = hand fans back to t2fanrd."""
    global fan_override
    if rpm is None:
        if fan_override:
            subprocess.run(["systemctl", "start", "t2fanrd"])
            fan_override = False
        return
    if not fan_override:
        subprocess.run(["systemctl", "stop", "t2fanrd"])
        fan_override = True
    for n, fmin, fmax in FANS:
        wr(f"{SMC}/fan{n}_manual", 1)
        wr(f"{SMC}/fan{n}_output", max(fmin, min(fmax, rpm)))


done = False


def cleanup(*_):
    global done
    if done:
        return
    done = True
    set_loads(0, None)
    # each re-applies its own state on start: fan curve, power limits
    subprocess.run(["systemctl", "start", "--no-block", "t2fanrd", *EXTRA_UNITS])
    subprocess.run(["systemctl", "unmask", "--runtime", *SLEEP_TARGETS])


def main():
    total = int(float(sys.argv[1]) * 3600) if len(sys.argv) > 1 else 6 * 3600
    plan = schedule(total)
    smc = LoggerTail(time.time() - 120)
    pkg_temp = pkg_temp_path()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if EXTRA_UNITS:
        subprocess.run(["systemctl", "stop", *EXTRA_UNITS])
    dgpu = amdgpu_dev()

    path = os.path.join(REPO, "data", time.strftime("workload-%Y%m%d-%H%M%S.csv"))
    log = open(path, "w", buffering=1)
    os.chown(path, UID, UID)
    log.write("timestamp,unix_time,event,level,cap_w,fan_rpm,planned_s\n")

    def note(event, level="", cap="", fan="", secs=""):
        log.write(f"{time.strftime('%F %T')},{time.time():.1f},{event},{level},{cap},{fan},{secs}\n")
        print(event, level, cap, fan, secs, flush=True)

    note("start", secs=total)
    try:
        for level, secs, cap, fan in plan:
            threads, gpu = LEVELS[level]
            set_cap(cap)
            set_fans(fan)
            set_loads(threads, gpu)
            note("phase", level, cap, fan or "t2fanrd", secs)
            dpm_done = gpu != "dgpu" or dgpu is None
            awake_since = None
            recent_rpm = collections.deque(maxlen=GUARD_WINDOW)
            recent_temp = collections.deque(maxlen=GUARD_WINDOW)
            end = time.monotonic() + secs
            while time.monotonic() < end:
                time.sleep(1)
                if not dpm_done:
                    if rd(dgpu + "/power/runtime_status") == "active":
                        awake_since = awake_since or time.monotonic()
                        if time.monotonic() - awake_since >= 3:
                            try:
                                dgpu_low(dgpu)
                            except OSError as e:
                                print("dgpu-low:", e, flush=True)
                            note("dgpu-low", level)
                            dpm_done = True
                    else:
                        awake_since = None
                try:
                    rpm = fan_rpm()
                    temp = int(rd(pkg_temp)) / 1000
                    tail = smc.read() or {}
                except (OSError, ValueError):
                    continue
                ts1s, pz0t = tail.get("Ts1S", 0.0), tail.get("PZ0T", 0.0)
                recent_rpm.append(rpm)
                recent_temp.append(temp)
                full = len(recent_rpm) == GUARD_WINDOW
                avg_rpm = sum(recent_rpm) / len(recent_rpm)
                avg_temp = sum(recent_temp) / len(recent_temp)
                reason = None
                if rpm > HARD_RPM:
                    reason = f"fans {rpm:.0f}"
                elif full and avg_rpm > NOISE_RPM:
                    reason = f"fans avg {avg_rpm:.0f}"
                elif full and avg_temp > CPU_TEMP_LIMIT:
                    reason = f"pkg avg {avg_temp:.0f}C"
                elif ts1s > TS1S_LIMIT:
                    reason = f"Ts1S {ts1s:.1f}"
                elif pz0t > 0:
                    reason = f"PZ0T {pz0t:.0f}"
                if reason:
                    set_loads(0, None)
                    set_fans(None)
                    note("guard", reason, fan="t2fanrd", secs=GUARD_COOLDOWN)
                    time.sleep(GUARD_COOLDOWN)
                    break
        note("end")
    finally:
        cleanup()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plan":
        hours = float(sys.argv[2]) if len(sys.argv) > 2 else 6
        for p in schedule(int(hours * 3600)):
            print(*p)
    elif len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup()
    else:
        main()
