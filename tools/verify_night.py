#!/usr/bin/python3 -I
"""Check the newest overnight logger + workload CSVs; exit 0 only if usable.

    verify_night.py HOURS

Writes data/night-report-<time>.txt. Stdlib only (runs as root, system python).
"""

import csv
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
UID = os.stat(REPO).st_uid
KEYS = ["TW0P", "TM0P", "Ts0P", "TG0P", "Ts1P", "Ts1S", "PZ0E", "PZ0F", "PZ0T",
        "PC0R", "PG0R", "F0Ac", "F1Ac"]
MAX_GAP_S = 5
MIN_COVERAGE = 0.95
MIN_KEY_FILL = 0.99


def newest(pattern):
    files = sorted(glob.glob(os.path.join(DATA, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    lines, ok = [], True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        lines.append(("PASS " if cond else "FAIL ") + msg)

    log = newest("passive-*.csv")
    work = newest("workload-*.csv")
    check(log is not None, f"logger csv: {log}")
    check(work is not None, f"workload csv: {work}")
    if log:
        with open(log) as f:
            rows = list(csv.DictReader(f))
        check(len(rows) > 0, f"{len(rows)} rows")
        if rows:
            t = [float(r["unix_time"]) for r in rows]
            span = t[-1] - t[0]
            gaps = [b - a for a, b in zip(t, t[1:])]
            check(span >= MIN_COVERAGE * hours * 3600, f"span {span / 3600:.2f} h (need {MIN_COVERAGE * hours:.2f})")
            check(max(gaps, default=0) <= MAX_GAP_S, f"largest gap {max(gaps, default=0):.1f} s (max {MAX_GAP_S})")
            check(len(rows) >= 0.97 * span, f"rate {len(rows) / max(span, 1):.3f} rows/s")
            for k in KEYS:
                if k not in rows[0]:
                    check(False, f"key {k} missing")
                    continue
                fill = sum(1 for r in rows if r[k] not in ("", None)) / len(rows)
                check(fill >= MIN_KEY_FILL, f"key {k} filled {fill:.3%}")
            pz0t = [float(r["PZ0T"]) for r in rows if r.get("PZ0T")]
            rpm = [float(r["F0Ac"]) for r in rows if r.get("F0Ac")]
            pc = [float(r["PC0R"]) for r in rows if r.get("PC0R")]
            lines.append(f"INFO PC0R {min(pc):.1f}..{max(pc):.1f} W, fan1 {min(rpm):.0f}..{max(rpm):.0f} rpm, "
                         f"PZ0T max {max(pz0t):.0f}")
    if work:
        with open(work) as f:
            events = list(csv.DictReader(f))
        check(any(e["event"] == "end" for e in events), "workload finished ('end' event)")
        phases = sum(e["event"] == "phase" for e in events)
        guards = [e["level"] for e in events if e["event"] == "guard"]
        lines.append(f"INFO {phases} phases, {len(guards)} guard trips {guards}")

    lines.append("RESULT " + ("OK -> power off" if ok else "NOT OK -> staying on"))
    path = os.path.join(DATA, time.strftime("night-report-%Y%m%d-%H%M%S.txt"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chown(path, UID, UID)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
