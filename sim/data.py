"""Load the logged CSVs onto a common 1 s grid.

A Segment is one continuous stretch of the machine being awake: inputs are
interpolated to every second, measured outputs are kept only at the seconds
they were actually sampled (NaN elsewhere).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import smc

DATA = Path(__file__).resolve().parent.parent / "data"
STRESS_CSV = DATA / "ts1s-stress-2026-10-19.csv"
SESSION_CSV = DATA / "igpu-clamp-session-2026-10-19.csv"

# PC0R (CPU rail at VR input) from RAPL package watts, fitted on the 1 Hz
# stress log (rms 1.3 W). The session log has RAPL every 2 s but PC0R only
# every 10 s, so the finer RAPL series makes the better input there.
PC0R_FROM_PKG = (1.2648, 0.8040)

OUTPUTS = smc.SENSORS + ["Ts1S", "PZ0E", "PZ0F", "PZ0T"]

# held-out validation: everything in the session log from here on
VALIDATION_START = pd.Timestamp("2026-10-19 19:45:00")
# long passive logs (the overnight run) train on all but their last part
LONG_LOG_S = 3 * 3600
LONG_LOG_HOLDOUT_S = 5400
# fand logs join the fit, all but their last part (fand runs near the throttle
# at mid-to-high fan speeds, where the other logs have little data)
FAND_HOLDOUT_FRAC = 0.3
# log sources kept out of the fit entirely (pure test), e.g. to score a model
# on a run it never saw: T2OPT_TEST_ONLY=fand-control-20261020-214635
TEST_ONLY = set(filter(None, os.environ.get("T2OPT_TEST_ONLY", "").split(",")))


@dataclass
class Segment:
    name: str
    t0: pd.Timestamp
    pc: np.ndarray  # CPU package rail PC0R, W
    pg: np.ndarray  # dGPU rail PG0R, W
    rpm: np.ndarray  # mean of the two fans
    y: dict  # output name -> array (NaN = not sampled)
    smc_resets: list = field(default_factory=list)  # indices (excluding 0)
    val_from: int | None = None  # index where validation starts, None = all train
    source: str = ""  # log file it came from (each has its own room temperature)
    pm: np.ndarray | None = None  # memory power PM0C, W; None if the log lacks it
    step: float = 1.0  # seconds between SMC samples

    def __len__(self):
        return len(self.pc)

    @property
    def zone_power(self):
        return self.pc + self.pg

    def times(self):
        return self.t0 + pd.to_timedelta(np.arange(len(self)), unit="s")


def _grid(ts, t0, n):
    """Seconds since t0 for a timestamp series, and the integer grid."""
    sec = (ts - t0).dt.total_seconds().to_numpy()
    return sec, np.arange(n, dtype=float)


def _interp(sec, values, grid):
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    return np.interp(grid, sec[ok], v[ok])


def _place(sec, values, n):
    out = np.full(n, np.nan)
    v = np.asarray(values, dtype=float)
    idx = np.rint(sec).astype(int)
    ok = ~np.isnan(v) & (idx >= 0) & (idx < n)
    out[idx[ok]] = v[ok]
    return out


def load_stress():
    df = pd.read_csv(STRESS_CSV)
    # local wall clock of the first row + sub-second-accurate unix offsets
    t0 = pd.Timestamp(df["timestamp"].iloc[0])
    ts = t0 + pd.to_timedelta(df["unix_time"] - df["unix_time"].iloc[0], unit="s")
    n = int((ts.iloc[-1] - t0).total_seconds()) + 1
    sec, grid = _grid(ts, t0, n)
    seg = Segment(
        name="stress",
        source="stress",
        t0=t0,
        pc=_interp(sec, df["PC0R"], grid),
        pg=_interp(sec, df["PG0R"], grid),
        rpm=_interp(sec, (df["F0Ac"] + df["F1Ac"]) / 2, grid),
        y={k: _place(sec, df[k], n) for k in OUTPUTS},
        pm=_interp(sec, df["PM0C"], grid),
    )
    return [seg]


def load_session():
    df = pd.read_csv(SESSION_CSV, parse_dates=["timestamp"])
    gap = df["timestamp"].diff().dt.total_seconds().fillna(0)
    # >5 min gap = long suspend: new segment, unknown thermal state.
    # Short gaps (the 10 s suspend at 19:42) keep the thermal state but reset the SMC.
    seg_id = (gap > 300).cumsum()
    segs = []
    for k, part in df.groupby(seg_id):
        ts = part["timestamp"]
        t0 = ts.iloc[0]
        n = int((ts.iloc[-1] - t0).total_seconds()) + 1
        sec, grid = _grid(ts, t0, n)
        a, b = PC0R_FROM_PKG
        # RAPL watts are energy deltas: the first sample after a gap spans the
        # suspend and reads hundreds of W
        pkg = part["pkg_w"].where(gap.loc[part.index] <= 3)
        y = {}
        for name in OUTPUTS:
            y[name] = _place(sec, part["smc_" + name], n)
        resets = [int(round(s)) for s, g in zip(sec, gap.loc[part.index]) if 8 <= g <= 300]
        seg = Segment(
            name=f"session{k}",
            source="session",
            step=10.0,
            t0=t0,
            pc=_interp(sec, a * pkg + b, grid),
            pg=_interp(sec, part["smc_PG0R"], grid),
            rpm=_interp(sec, (part["fan1_rpm"] + part["fan2_rpm"]) / 2, grid),
            pm=_interp(sec, part["smc_PM0C"], grid),
            y=y,
            smc_resets=resets,
        )
        cut = int((VALIDATION_START - t0).total_seconds())
        if 0 < cut < n:
            seg.val_from = cut
        elif cut <= 0:
            seg.val_from = 0
        segs.append(seg)
    return segs


def load_passive(path):
    """A tools/logger.py or fand run: 1 Hz, all held out as fresh test data."""
    # a running fand may be mid-way through its last line
    df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["unix_time"])
    t0 = pd.Timestamp(df["timestamp"].iloc[0])
    ts = t0 + pd.to_timedelta(df["unix_time"] - df["unix_time"].iloc[0], unit="s")
    gap = ts.diff().dt.total_seconds().fillna(0)
    seg_id = (gap > 300).cumsum()
    segs = []
    for k, part in df.groupby(seg_id):
        pts = ts.loc[part.index]
        start = pts.iloc[0].floor("s")
        n = int((pts.iloc[-1] - start).total_seconds()) + 1
        sec, grid = _grid(pts, start, n)
        resets = [int(round(s)) for s, g in zip(sec, gap.loc[part.index]) if 8 <= g <= 300]
        segs.append(
            Segment(
                name=f"{Path(path).stem}-{k}",
                source=Path(path).stem,
                t0=start,
                pc=_interp(sec, part["PC0R"], grid),
                pg=_interp(sec, part["PG0R"], grid),
                rpm=_interp(sec, (part["F0Ac"] + part["F1Ac"]) / 2, grid),
                pm=_interp(sec, part["PM0C"], grid) if "PM0C" in part else None,
                y={name: _place(sec, part[name], n) for name in OUTPUTS},
                smc_resets=resets,
                val_from=0,
            )
        )
    return segs


def load_all(test=True):
    """Fit segments (+ the short passive logs as pure test segments if `test`).

    Long passive logs join the fit, with their last LONG_LOG_HOLDOUT_S held out
    (that part is scored as test, never fitted).
    """
    segs = load_stress() + load_session()
    # fand logs (daemon/) have the same columns as the passive logger's
    for path in sorted(DATA.glob("passive-*.csv")) + sorted(DATA.glob("fand-*.csv")):
        for seg in load_passive(path):
            if seg.source in TEST_ONLY:
                if test:
                    segs.append(seg)
            elif seg.source.startswith("fand-"):
                seg.val_from = int(len(seg) * (1 - FAND_HOLDOUT_FRAC))
                segs.append(seg)
            elif len(seg) >= LONG_LOG_S:
                seg.val_from = len(seg) - LONG_LOG_HOLDOUT_S
                segs.append(seg)
            elif test:
                segs.append(seg)
    return segs
