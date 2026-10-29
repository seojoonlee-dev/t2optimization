"""Score the model the way the daemon will use it: correct the state from the
sensors (Kalman filter), then predict H seconds ahead with the known inputs.

    python -m sim.forecast rc3

The inputs over the horizon are the logged ones (in the daemon they are the
candidate plan being evaluated). Baseline "hold" = assume nothing changes.
"""

import sys

import numpy as np
from scipy.linalg import expm

from . import data, smc
from .fit import MODELS, _masks
from .thermal import ThermalModel, fan_feature

HORIZONS = (30, 60, 90, 180)
SENSOR_NOISE = 0.3  # K, rms
# process noise per second, per node; slow nodes drift less
PROCESS_NOISE = {"H": 0.05, "Hc": 0.05, "Hg": 0.05, "F": 0.05, "C": 0.02, "S": 0.005, "M": 0.05, "W": 0.05, "B": 0.05}
# room temperature is an extra random-walk state: unknown on a new day, drifts slowly
AMBIENT_NOISE = 0.003
TEST_WARMUP = 300  # s of filtering from a cold start before scoring


class Kalman:
    """State = node temperatures + ambient offset (shared by all sensors)."""

    def __init__(self, model):
        self.model = model
        # a model's "kalman" block (tuned for it) overrides the values above
        t = model.kalman or {}
        noise = {**PROCESS_NOISE, **t.get("process_noise", {})}
        sensor, ambient = t.get("sensor_noise", SENSOR_NOISE), t.get("ambient_noise", AMBIENT_NOISE)
        self.p0, self.p0_ambient = t.get("p0", 25.0), t.get("p0_ambient", 9.0)
        cinv, gmat, ga0, ga1, bmat = model.topo.build(model.params)
        self.cinv, self.gmat, self.ga0, self.ga1, self.bmat = cinv, gmat, ga0, ga1, bmat
        self.n = len(cinv)
        w = np.array([model.out_w[k] for k in smc.SENSORS])
        self.h = np.hstack([w, np.ones((len(smc.SENSORS), 1))])
        self.d = np.array([model.out_d[k] for k in smc.SENSORS])
        self.q = np.diag([noise[k] ** 2 for k in model.topo.nodes] + [ambient**2])
        self.r = np.eye(len(smc.SENSORS)) * sensor**2

    def step_matrices(self, pc, pg, rpm, pm=0.0, io=0.0, chg=0.0):
        """Exact 1 s discretisation with inputs held constant."""
        f = fan_feature(np.array(rpm), self.model.params["fan_exp"])
        m = self.cinv[:, None] * (self.gmat - np.diag(self.ga0 + self.ga1 * f + self.gmat.sum(axis=1)))
        b = self.cinv * (self.bmat @ np.array([pc, pg, pm, io, chg], float))
        aug = np.zeros((self.n + 1, self.n + 1))
        aug[: self.n, : self.n] = m
        aug[: self.n, self.n] = b
        e = expm(aug)
        return e[: self.n, : self.n], e[: self.n, self.n]

    def run(self, seg, x0, ambient, p0=4.0, p0_ambient=0.25):
        """Filtered node states and ambient at every second."""
        n_t = len(seg)
        m = self.n + 1
        xs = np.empty((n_t, m))
        x = np.append(np.asarray(x0, float), ambient)
        p = np.diag([p0] * self.n + [p0_ambient])
        ys = np.array([seg.y[k] for k in smc.SENSORS]).T - self.d
        a_full = np.eye(m)
        b_full = np.zeros(m)
        for t in range(n_t):
            if t > 0:
                pm = 0.0 if seg.pm is None else seg.pm[t - 1]
                io = 0.0 if seg.io is None else seg.io[t - 1]
                chg = 0.0 if seg.chg is None else seg.chg[t - 1]
                a, b = self.step_matrices(seg.pc[t - 1], seg.pg[t - 1], seg.rpm[t - 1], pm, io, chg)
                a_full[: self.n, : self.n] = a
                b_full[: self.n] = b
                x = a_full @ x + b_full
                p = a_full @ p @ a_full.T + self.q
            y = ys[t]
            if not np.isnan(y).any():
                s = self.h @ p @ self.h.T + self.r
                k = p @ self.h.T @ np.linalg.inv(s)
                x = x + k @ (y - self.h @ x)
                p = (np.eye(m) - k @ self.h) @ p
            xs[t] = x
        return xs[:, : self.n], xs[:, self.n]


def _ema_states(sensors, resets, upto):
    """The SMC's Ts1S filter states after sample `upto` (from measured sensors)."""
    out = {}
    for name, (_, tau) in smc.TS1S_TERMS.items():
        a = smc.ema_alpha(tau)
        x = sensors[name]
        f = x[0]
        for i in range(1, upto + 1):
            f = x[i] if i in resets else f + a * (x[i] - f)
        out[name] = f
    return out


def main(name):
    """Score a saved model (models/<name>.json) on every log.

    Every segment starts cold, as the daemon would: steady state at the first
    inputs, room temperature unknown, and the first TEST_WARMUP s of filtering
    are not scored. So models fitted on different data compare fairly.
    """
    model = ThermalModel.from_json((MODELS / f"{name}.json").read_text())
    kf = Kalman(model)
    fit_names = {s.name for s in data.load_all(test=False)}
    splits = ("train", "val", "night-test", "eve-test", "fand-val", "fand-test")
    errs = {(split, h): {"model": [], "hold": [], "pz0e": []} for split in splits for h in HORIZONS}
    for seg in data.load_all():
        if model.topo.pm_node and seg.pm is None:
            print(f"  ({seg.name}: no PM0C in this log, skipped)")
            continue
        val = _masks(seg)[1] if seg.name in fit_names else None
        x0 = model.steady_state(seg.pc[0], seg.pg[0], seg.rpm[0], 0.0 if seg.pm is None else seg.pm[0],
                                seg.io[0] if seg.io is not None else 0.0, seg.chg[0] if seg.chg is not None else 0.0)
        xs, ambs = kf.run(seg, x0, 0.0, kf.p0, kf.p0_ambient)
        idx = np.arange(len(seg))
        meas = {
            k: np.interp(idx, np.flatnonzero(~np.isnan(seg.y[k])), seg.y[k][~np.isnan(seg.y[k])])
            for k in smc.SENSORS
        }
        ts_meas = smc.ts1s(meas, seg.smc_resets)
        _, integ = smc.budget(ts_meas, seg.smc_resets)
        resets = set(seg.smc_resets)
        sampled = np.flatnonzero(~np.isnan(seg.y["Ts1S"]))
        sampled_set = set(sampled.tolist())
        # a forecast origin every ~30 s
        for t in sampled[:: max(1, round(30 / seg.step))]:
            if t < TEST_WARMUP:
                continue
            if val is None:
                split = "fand-test" if seg.source.startswith("fand-") else "eve-test"
            elif not val[t]:
                split = "train"
            elif seg.source.startswith("fand-"):
                split = "fand-val"
            else:
                split = "val" if seg.source == "session" else "night-test"
            for h in HORIZONS:
                end = t + h
                if end >= len(seg) or end not in sampled_set or any(t < r <= end for r in resets):
                    continue
                model.ambient = ambs[t]
                pm = None if seg.pm is None else seg.pm[t:end + 1]
                io = None if seg.io is None else seg.io[t:end + 1]
                chg = None if seg.chg is None else seg.chg[t:end + 1]
                st = model.simulate(xs[t], seg.pc[t:end + 1], seg.pg[t:end + 1], seg.rpm[t:end + 1], pm, io, chg)
                pred = model.sensors(st)
                # SMC filters and integrator start from their replayed state at t
                fk = _ema_states(meas, resets, t)
                acc = integ[t]
                for i in range(1, h + 1):
                    ts_pred = smc.TS1S_OFFSET
                    for k, (gain, tau) in smc.TS1S_TERMS.items():
                        fk[k] += smc.ema_alpha(tau) * (pred[k][i] - fk[k])
                        ts_pred += gain * fk[k]
                    acc = min(max(acc + smc.KI_E * (ts_pred - smc.SETPOINT), 0.0), smc.I_MAX)
                pz0e = min(max(smc.E_MAX - smc.KP_E * (ts_pred - smc.SETPOINT) - acc, 0.0), smc.E_MAX)
                truth = seg.y["Ts1S"][end]
                bucket = errs[(split, h)]
                bucket["model"].append(ts_pred - truth)
                bucket["hold"].append(seg.y["Ts1S"][t] - truth)
                bucket["pz0e"].append(pz0e - seg.y["PZ0E"][end])
        model.ambient = 0.0
    print(f"{name}: Ts1S error h seconds ahead, Kalman-corrected state (rms / bias °C), vs 'hold' baseline")
    print(f"  {'split':10s} {'h':>4s} {'n':>5s}  {'model rms':>9s} {'bias':>6s}  {'hold rms':>8s}  {'PZ0E rms W':>10s}")
    for (split, h), b in errs.items():
        if not b["model"]:
            continue
        m, hd, pe = np.array(b["model"]), np.array(b["hold"]), np.array(b["pz0e"])
        print(f"  {split:10s} {h:4d} {len(m):5d}  {np.sqrt(np.mean(m**2)):9.3f} {m.mean():+6.2f}  {np.sqrt(np.mean(hd**2)):8.3f}  {np.sqrt(np.mean(pe**2)):10.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rc3")
