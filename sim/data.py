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
OUTPUTS = smc.SENSORS + ["Ts1S", "PZ0E", "PZ0F", "PZ0T"]

# long passive logs (the overnight run) train on all but their last part
LONG_LOG_S = 3 * 3600
LONG_LOG_HOLDOUT_S = 5400
# extra-logger (and older fand) logs join the fit, all but their last part
# (they cover the throttle region at mid-to-high fan speeds, where the other
# logs have little data)
FAND_HOLDOUT_FRAC = 0.3
# I/O power (POLR + PORR + PSLC: Wi-Fi + SSD path rails) where a log lacks
# those keys: from wifi_mb_s where fand logged it (fit on the extra-logger
# logs 2026-10-28, rms 0.41 W, R2 0.73), else the idle level. Charge power
# (PDTR - PSTR) unknown = 0 (not charging).
IO_FROM_WIFI = (6.2011, 0.0463, 0.0859)
IO_IDLE_W = IO_FROM_WIFI[0]
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
    io: np.ndarray | None = None  # I/O power POLR + PORR + PSLC, W
    chg: np.ndarray | None = None  # battery charge power PDTR - PSTR (>= 0), W
    io_measured: bool = False  # io and chg come from the rails, not estimated

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


def _io_chg(part, sec, grid, prefix=""):
    """I/O and charge power on the grid, from the rails or estimated (see IO_FROM_WIFI)."""
    k = lambda c: prefix + c
    if all(k(c) in part and part[k(c)].notna().mean() > 0.9 for c in ("POLR", "PORR", "PSLC", "PDTR", "PSTR")):
        io = part[k("POLR")] + part[k("PORR")] + part[k("PSLC")]
        chg = (part[k("PDTR")] - part[k("PSTR")]).clip(lower=0)
        return _interp(sec, io, grid), _interp(sec, chg, grid), True
    if "wifi_mb_s" in part and part["wifi_mb_s"].notna().mean() > 0.9:
        w = part["wifi_mb_s"].clip(lower=0)
        a, b, c = IO_FROM_WIFI
        io = _interp(sec, a + b * w + c * np.sqrt(w), grid)
    else:
        io = np.full(len(grid), IO_IDLE_W)
    return io, np.zeros(len(grid)), False


_EXTRA = None


def _full_extra(path):
    """An extra-logger CSV with every key the fit needs (fand no longer logs).
    Older extra logs only had the other rails and are joined to fand's logs."""
    with open(path) as f:
        return "Ts1S" in f.readline().strip().split(",")


def _extra():
    """Older extra-logger CSVs (tools/extra-logger.py), joined to fand's old logs on unix_time."""
    global _EXTRA
    if _EXTRA is None:
        files = [p for p in sorted(DATA.glob("extra-*.csv")) if not _full_extra(p)]
        if files:
            ex = pd.concat([pd.read_csv(p, on_bad_lines="skip") for p in files])
            _EXTRA = ex.dropna(subset=["unix_time"]).drop(columns=["timestamp"]).sort_values("unix_time")
        else:
            _EXTRA = pd.DataFrame(columns=["unix_time"])
    return _EXTRA


def load_passive(path):
    """A 1 Hz log (extra logger, or an older fand / passive logger run)."""
    # a running fand may be mid-way through its last line
    df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["unix_time"])
    ex = _extra()
    if "POLR" not in df and len(ex):
        df = pd.merge_asof(df.sort_values("unix_time"), ex[["unix_time", "POLR", "PORR", "PSLC", "PDTR", "PSTR"]],
                           on="unix_time", direction="nearest", tolerance=0.7)
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
        io, chg, measured = _io_chg(part, sec, grid)
        segs.append(
            Segment(
                io=io,
                chg=chg,
                io_measured=measured,
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
    """Fit segments from every log in data/ (+ the short passive logs as pure
    test segments if `test`).

    Long passive logs join the fit, with their last LONG_LOG_HOLDOUT_S held out
    (that part is scored as test, never fitted).
    """
    segs = []
    # extra-logger logs, plus older fand and passive-logger logs (same key columns)
    full = [p for p in sorted(DATA.glob("extra-*.csv")) if _full_extra(p)]
    for path in sorted(DATA.glob("passive-*.csv")) + sorted(DATA.glob("fand-*.csv")) + full:
        for seg in load_passive(path):
            if seg.source in TEST_ONLY:
                if test:
                    segs.append(seg)
            elif seg.source.startswith(("fand-", "extra-")):
                seg.val_from = int(len(seg) * (1 - FAND_HOLDOUT_FRAC))
                segs.append(seg)
            elif len(seg) >= LONG_LOG_S:
                seg.val_from = len(seg) - LONG_LOG_HOLDOUT_S
                segs.append(seg)
            elif test:
                segs.append(seg)
    return segs
