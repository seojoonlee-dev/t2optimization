"""Fit the thermal model on the logged data and score it on the held-out part.

    python -m sim.fit rc3 rc4 rc5
    python -m sim.fit rc3 --warm rc3-v1 --free a1_C,fan_exp --out rc3-v4

The loss is free-run simulation error over whole segments (not one-step
prediction), since the controller relies on ~90 s look-ahead. The session log
after data.VALIDATION_START is never seen by the fit.

The free runs are spread over one worker process per CPU: every (trial
parameter vector, segment) run of a finite-difference Jacobian is a separate
task, so the long overnight segments don't leave the other CPUs idle.
"""

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# the parallelism is the worker processes; library threads on top of them
# only fight over the CPUs (must be set before numpy loads)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.optimize import least_squares

from . import data, smc
from .thermal import TOPOLOGIES, ThermalModel, solve_outputs

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
# per-log room temperature vs the stress log, K; wider bounds let the shift
# soak up model error (it fit +4..+10 K for logs from the same day)
MAX_AMBIENT_SHIFT = 3.0
# residuals are scored every RES_STEP s (weighted by it): with 53 h of logs a
# 1 s step made the least-squares Jacobian ~15 GB and ran out of memory
# (2026-10-28); the sensors' filters are 16 s and slower
RES_STEP = 5
WORKERS = os.cpu_count() or 1

# the Problem being fitted; the fork workers inherit it, so a task only
# carries its segment index and parameters
_PROB = None


def _simulate_segment(task):
    i, params, x0, full = task
    s = _PROB.segs[i]
    states = ThermalModel(_PROB.topo, params, {}, {}).simulate(x0, s.pc, s.pg, s.rpm, s.pm, s.io, s.chg)
    return states if full else states[_PROB.rows[i]]


def _masks(seg):
    n = len(seg)
    idx = np.arange(n)
    measured = ~np.isnan(seg.y["TW0P"])
    val_from = n if seg.val_from is None else seg.val_from
    train = measured & (idx < val_from)
    val = measured & (idx >= val_from)
    return train, val


class Problem:
    def __init__(self, topo, segs):
        self.topo = topo
        # a memory-power model can only fit logs that recorded PM0C
        self.segs = segs = [s for s in segs if s.pm is not None or not topo.pm_node]
        self.specs = topo.param_specs()
        self.n_dyn = len(self.specs)
        self.n_nodes = len(topo.nodes)
        self.masks = [_masks(s) for s in segs]
        # each sample weighs the seconds it stands for (session SMC data is 10 s;
        # elsewhere every RES_STEP-th second stands for RES_STEP)
        def sampled(s, train):
            if s.step >= RES_STEP:
                return train * s.step
            keep = np.zeros(len(train), bool)
            keep[::RES_STEP] = True
            return (train & keep) * float(RES_STEP)

        self.weights = [sampled(s, train) for s, (train, _) in zip(segs, self.masks)]
        # then each log file gets an equal share: 4.5 h of quiet overnight data
        # otherwise outweighs the short heavy-load logs and the fit gives up
        # accuracy where the controller matters most
        totals = {}
        for s, w in zip(segs, self.weights):
            totals[s.source] = totals.get(s.source, 0.0) + float(w.sum())
        share = float(np.median(list(totals.values())))
        self.weights = [w * share / totals[s.source] for s, w in zip(segs, self.weights)]
        # room temperature differs per log file; the stress log is the reference
        self.sources = sorted({s.source for s in segs} - {"stress"})
        # the residual rows of each segment
        self.rows = [np.flatnonzero(w > 0) for w in self.weights]
        # longest first, so the pool doesn't end on one long segment
        self.order = sorted(range(len(segs)), key=lambda i: -len(segs[i]))
        self.pool = None
        global _PROB
        _PROB = self

    def start_workers(self):
        self.pool = mp.get_context("fork").Pool(WORKERS)

    def stop_workers(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    # parameter vector: log(dynamic params) | ambient shift per source | x0 per segment
    def initial(self):
        v = [np.log(init) for _, init, _, _ in self.specs]
        lo = [np.log(low) for _, _, low, _ in self.specs]
        hi = [np.log(high) for _, _, _, high in self.specs]
        for _ in self.sources:
            v.append(0.0)
            lo.append(-MAX_AMBIENT_SHIFT)
            hi.append(MAX_AMBIENT_SHIFT)
        for s in self.segs:
            first = {k: s.y[k][~np.isnan(s.y[k])][0] for k in ("TW0P", "Ts1P")}
            hot, cool = first["TW0P"] - 22.0, first["Ts1P"] - 22.0
            for node in self.topo.nodes:
                guess = cool if node == "S" else (hot * 0.7 if node == "C" else hot)
                v.append(max(guess, 0.5))
                lo.append(-5.0)
                hi.append(80.0)
        return np.array(v), (np.array(lo), np.array(hi))

    def unpack(self, v):
        params = {name: float(np.exp(v[i])) for i, (name, *_) in enumerate(self.specs)}
        k = self.n_dyn + len(self.sources)
        shift = dict(zip(self.sources, map(float, v[self.n_dyn : k])))
        x0s = v[k:].reshape(len(self.segs), self.n_nodes)
        return params, shift, x0s

    def ambient(self, seg, shift):
        return shift.get(seg.source, 0.0)

    def simulate_many(self, vs, full=False):
        """(model, states, shift, x0s) for each parameter vector in vs. States
        are the residual rows only unless `full`."""
        unpacked = [self.unpack(v) for v in vs]
        tasks = [(j, i) for i in self.order for j in range(len(vs))]
        args = [(i, unpacked[j][0], unpacked[j][2][i], full) for j, i in tasks]
        runs = self.pool.imap(_simulate_segment, args) if self.pool else map(_simulate_segment, args)
        states = [[None] * len(self.segs) for _ in vs]
        for (j, i), st in zip(tasks, runs):
            states[j][i] = st
        out = []
        for (params, shift, x0s), sts in zip(unpacked, states):
            model = ThermalModel(self.topo, params, {}, {})
            rows = [slice(None) if full else r for r in self.rows]
            ys = [{k: s.y[k][r] - self.ambient(s, shift) for k in smc.SENSORS} for s, r in zip(self.segs, rows)]
            ws = [w[r] for w, r in zip(self.weights, rows)]
            model.out_w, model.out_d = solve_outputs(sts, ys, ws)
            out.append((model, sts, shift, x0s))
        return out

    def simulate(self, v):
        return self.simulate_many([v], full=True)[0]

    def sparsity(self):
        """Which parameters move which residuals: dynamic params all of them,
        a log's room shift its own segments, a segment's start state its own
        rows. (The output map couples everything a little; ignored here, so
        least_squares nudges all segments' start states in one pass.)"""
        from scipy.sparse import lil_matrix

        k = self.n_dyn + len(self.sources)
        rows, r0 = [], 0
        for s, w in zip(self.segs, self.weights):
            n = int((w > 0).sum()) * len(smc.SENSORS)
            rows.append((r0, r0 + n))
            r0 += n
        m = lil_matrix((r0, k + self.n_nodes * len(self.segs)), dtype=bool)
        m[:, : self.n_dyn] = True
        for i, (s, (a, b)) in enumerate(zip(self.segs, rows)):
            if s.source in self.sources:
                m[a:b, self.n_dyn + self.sources.index(s.source)] = True
            m[a:b, k + i * self.n_nodes : k + (i + 1) * self.n_nodes] = True
        return m

    def residuals_many(self, vs):
        out = []
        for model, states, shift, _ in self.simulate_many(vs):
            res = []
            for s, st, w, r in zip(self.segs, states, self.weights, self.rows):
                pred = model.sensors(st)
                for k in smc.SENSORS:
                    res.append((pred[k] - (s.y[k][r] - self.ambient(s, shift))) * np.sqrt(w[r]))
            out.append(np.concatenate(res))
        return out

    def residuals(self, v):
        return self.residuals_many([v])[0]


def evaluate(problem, v):
    """Per segment and split: sensor RMS, Ts1S/PZ0E/PZ0T from predicted sensors."""
    model, states, shift, _ = problem.simulate(v)
    rows = []
    for s, st, (train, val) in zip(problem.segs, states, problem.masks):
        model.ambient = problem.ambient(s, shift)
        pred = model.sensors(st)
        ts = smc.ts1s(pred, s.smc_resets)
        pz0e, _ = smc.budget(ts, s.smc_resets)
        _, pz0t = smc.throttle(s.zone_power, pz0e, s.smc_resets)
        for split, m in (("train", train), ("val", val)):
            if not m.any():
                continue
            r = {"segment": s.name, "split": split, "n": int(m.sum())}
            for k in smc.SENSORS:
                r[k] = float(np.sqrt(np.mean((pred[k][m] - s.y[k][m]) ** 2)))
            mt = m & ~np.isnan(s.y["Ts1S"])
            r["Ts1S"] = float(np.sqrt(np.mean((ts[mt] - s.y["Ts1S"][mt]) ** 2)))
            r["Ts1S_bias"] = float(np.mean(ts[mt] - s.y["Ts1S"][mt]))
            r["PZ0E"] = float(np.sqrt(np.mean((pz0e[mt] - s.y["PZ0E"][mt]) ** 2)))
            pinned_true = s.y["PZ0T"][mt] >= smc.PIN_LEVEL
            pinned_pred = pz0t[mt] >= smc.PIN_LEVEL
            r["pin_true_s"] = int(pinned_true.sum())
            r["pin_pred_s"] = int(pinned_pred.sum())
            r["pin_agree"] = float(np.mean(pinned_true == pinned_pred))
            rows.append(r)
        model.ambient = 0.0
    return model, rows


def fit(name, segs, warm=None, free=None):
    """free: names of the dynamic params to fit; the rest stay at the warm model's values."""
    topo = TOPOLOGIES[name]
    prob = Problem(topo, segs)
    v0, bounds = prob.initial()
    if warm:
        # start the dynamic params from an earlier fit (models/<warm>.json)
        prev = json.loads((MODELS / f"{warm}.json").read_text())["params"]
        for i, (pname, *_) in enumerate(prob.specs):
            if pname in prev:
                v0[i] = np.clip(np.log(prev[pname]), bounds[0][i], bounds[1][i])
    if free is not None:
        if not warm:
            raise SystemExit("--free needs --warm")
        unknown = set(free) - {p for p, *_ in prob.specs}
        if unknown:
            raise SystemExit(f"unknown params: {sorted(unknown)}")
    # frozen params stay at the warm model's values and out of the solve
    # (pinning them with near-equal bounds made each step's solve crawl)
    keep = np.ones(len(v0), bool)
    if free is not None:
        for i, (pname, *_) in enumerate(prob.specs):
            keep[i] = pname in free

    def full(z):
        v = v0.copy()
        v[keep] = z
        return v

    t = time.time()
    prob.start_workers()
    try:
        # verbose=2: one line per step (cost, step size, optimality), for long fits.
        # workers: the Jacobian's trial vectors all go to the pool at once
        sol = least_squares(lambda z: prob.residuals(full(z)), v0[keep], bounds=(bounds[0][keep], bounds[1][keep]),
                            x_scale="jac", max_nfev=400, verbose=2, jac_sparsity=prob.sparsity()[:, keep],
                            workers=lambda _, zs: prob.residuals_many([full(z) for z in zs]))
        sol.x = full(sol.x)
        model, rows = evaluate(prob, sol.x)
    finally:
        prob.stop_workers()
    return prob, sol, model, rows, time.time() - t


def print_rows(rows):
    cols = smc.SENSORS + ["Ts1S", "Ts1S_bias", "PZ0E"]
    print(f"  {'segment':9s} {'split':5s} {'n':>5s} " + " ".join(f"{c:>9s}" for c in cols) + "  pin s true/pred agree")
    for r in rows:
        vals = " ".join(f"{r[c]:9.3f}" for c in cols)
        print(f"  {r['segment']:9s} {r['split']:5s} {r['n']:5d} {vals}  {r['pin_true_s']:5d}/{r['pin_pred_s']:<5d} {r['pin_agree']:.3f}")


def main(names, warm=None, free=None, out=None):
    segs = data.load_all(test=False)
    MODELS.mkdir(exist_ok=True)
    for name in names:
        print(f"fitting {name} on {len(segs)} segments ({time.strftime('%H:%M:%S')})", flush=True)
        prob, sol, model, rows, dt = fit(name, segs, warm, free)
        n_par = len(sol.x)
        print(f"\n== {name}: cost {sol.cost:.1f}, {sol.nfev} evals, {dt:.0f} s, {n_par} params ({prob.n_dyn} dynamic), status {sol.status}")
        params, shift, x0s = prob.unpack(sol.x)
        print("  " + "  ".join(f"{k}={v:.4g}" for k, v in params.items()))
        print("  ambient shift vs stress log: " + ", ".join(f"{k} {v:+.2f} K" for k, v in shift.items()))
        for k in smc.SENSORS:
            print(f"  {k}: w={np.round(model.out_w[k], 3)} d={model.out_d[k]:.2f}")
        print_rows(rows)
        out_name = out or name
        (MODELS / f"{out_name}.json").write_text(model.to_json())
        np.save(MODELS / f"{out_name}-solution.npy", sol.x)


if __name__ == "__main__":
    args = sys.argv[1:]

    def opt(flag):
        if flag not in args:
            return None
        i = args.index(flag)
        val = args[i + 1]
        del args[i : i + 2]
        return val

    warm, free, out = opt("--warm"), opt("--free"), opt("--out")
    main(args or ["rc3"], warm, free.split(",") if free else None, out)
