"""Fit the thermal model on the logged data and score it on the held-out part.

    python -m sim.fit rc3 rc4 rc5
    python -m sim.fit rc3 --warm rc3-v1 --free a1_C,fan_exp --out rc3-v4

The loss is free-run simulation error over whole segments (not one-step
prediction), since the controller relies on ~90 s look-ahead. The session log
after data.VALIDATION_START is never seen by the fit.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import data, smc
from .thermal import TOPOLOGIES, ThermalModel, solve_outputs

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
# per-log room temperature vs the stress log, K; wider bounds let the shift
# soak up model error (it fit +4..+10 K for logs from the same day)
MAX_AMBIENT_SHIFT = 3.0


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
        # each sample weighs the seconds it stands for (session SMC data is 10 s)
        self.weights = [train * s.step for s, (train, _) in zip(segs, self.masks)]
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

    def simulate(self, v):
        params, shift, x0s = self.unpack(v)
        model = ThermalModel(self.topo, params, {}, {})
        states = [model.simulate(x0, s.pc, s.pg, s.rpm, s.pm) for s, x0 in zip(self.segs, x0s)]
        ys = [{k: s.y[k] - self.ambient(s, shift) for k in smc.SENSORS} for s in self.segs]
        out_w, out_d = solve_outputs(states, ys, self.weights)
        model.out_w, model.out_d = out_w, out_d
        return model, states, shift, x0s

    def residuals(self, v):
        model, states, shift, _ = self.simulate(v)
        res = []
        for s, st, w in zip(self.segs, states, self.weights):
            pred = model.sensors(st)
            m = w > 0
            for k in smc.SENSORS:
                y = s.y[k] - self.ambient(s, shift)
                res.append((pred[k][m] - y[m]) * np.sqrt(w[m]))
        return np.concatenate(res)


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
        # least_squares needs lo < hi, so pin frozen params to a sliver around their value
        for i, (pname, *_) in enumerate(prob.specs):
            if pname not in free:
                bounds[0][i], bounds[1][i] = v0[i] - 1e-9, v0[i] + 1e-9
    t = time.time()
    sol = least_squares(prob.residuals, v0, bounds=bounds, x_scale="jac", max_nfev=400, verbose=0)
    model, rows = evaluate(prob, sol.x)
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
