"""Grey-box lumped thermal model: power + fan speed -> the five Ts1S sensors.

Each node i has a heat capacity C_i and a temperature x_i measured relative to
ambient. Heat flows between linked nodes (fixed conductance), into nodes from
CPU (PC0R), dGPU (PG0R) and, per topology, memory (PM0C), I/O (Wi-Fi + SSD
path rails POLR + PORR + PSLC) and battery charge (PDTR - PSTR) power, and
out to ambient through
g_i(rpm) = a0_i + a1_i * (rpm/1000)^k (a1 = 0 for nodes the fans don't reach;
the exponent k is fitted, ~0.8 for textbook forced convection):

    C_i dx_i/dt = P_i - g_i(rpm) x_i + sum_j G_ij (x_j - x_i)

Each sensor reads a non-negative mix of node temperatures plus an offset and
the ambient temperature; those output weights are linear, so the fit solves
them exactly (NNLS) for every trial of the dynamic parameters.
"""

import json
import math
from dataclasses import dataclass

import numba
import numpy as np
from scipy.optimize import nnls

from . import smc


def fan_feature(rpm, exp):
    return (np.maximum(rpm, 0.0) / 1000.0) ** exp


@numba.njit(cache=True)
def _simulate(x0, u, fan, cinv, gmat, ga0, ga1, bmat, nsub):
    """u: (n_t, k) power inputs; bmat: (n, k) where each input's heat goes."""
    n_t = u.shape[0]
    n = x0.shape[0]
    k_in = u.shape[1]
    out = np.empty((n_t, n))
    x = x0.copy()
    q = np.empty(n)
    h = 1.0 / nsub
    out[0] = x
    for t in range(1, n_t):
        f = fan[t - 1]
        for _ in range(nsub):
            for i in range(n):
                acc = -(ga0[i] + ga1[i] * f) * x[i]
                for j in range(k_in):
                    acc += bmat[i, j] * u[t - 1, j]
                for j in range(n):
                    acc += gmat[i, j] * (x[j] - x[i])
                q[i] = acc
            for i in range(n):
                x[i] += h * q[i] * cinv[i]
        out[t] = x
    return out


@dataclass
class Topology:
    name: str
    nodes: list  # node names
    edges: list  # (a, b) pairs with a conductance each
    fan_nodes: list  # nodes whose loss to ambient scales with fan speed
    pc_node: str  # where CPU power goes (a fitted fraction goes to the board node)
    pg_node: str  # where dGPU power goes
    board_node: str = "C"
    pm_node: str | None = None  # where memory power (PM0C) goes; None = not an input
    io_node: str | None = None  # where I/O power (POLR + PORR + PSLC) goes
    chg_node: str | None = None  # where battery charge power (PDTR - PSTR) goes

    def param_specs(self):
        """(name, init, low, high); all positive, fitted in log space."""
        caps = {"H": 80.0, "Hc": 60.0, "Hg": 60.0, "F": 60.0, "C": 600.0, "S": 1500.0, "M": 20.0, "W": 20.0, "B": 20.0}
        specs = [(f"cap_{n}", caps[n], 0.5, 2e5) for n in self.nodes]
        specs += [(f"g_{a}_{b}", 1.0, 1e-3, 50.0) for a, b in self.edges]
        for n in self.nodes:
            specs.append((f"a0_{n}", 0.2 if n in self.fan_nodes else 0.1, 1e-4, 20.0))
            if n in self.fan_nodes:
                specs.append((f"a1_{n}", 0.2, 1e-4, 20.0))
        specs.append(("pc_board_frac", 0.1, 1e-3, 0.9))
        specs.append(("fan_exp", 0.8, 0.3, 3.0))
        return specs

    def build(self, p):
        """Parameter dict -> simulation arrays."""
        idx = {n: i for i, n in enumerate(self.nodes)}
        n = len(self.nodes)
        cinv = np.array([1.0 / p[f"cap_{k}"] for k in self.nodes])
        gmat = np.zeros((n, n))
        for a, b in self.edges:
            g = p[f"g_{a}_{b}"]
            gmat[idx[a], idx[b]] = gmat[idx[b], idx[a]] = g
        ga0 = np.array([p[f"a0_{k}"] for k in self.nodes])
        ga1 = np.array([p.get(f"a1_{k}", 0.0) for k in self.nodes])
        bpc = np.zeros(n)
        frac = p["pc_board_frac"]
        bpc[idx[self.pc_node]] += 1.0 - frac
        bpc[idx[self.board_node]] += frac
        bpg = np.zeros(n)
        bpg[idx[self.pg_node]] = 1.0
        bpm = np.zeros(n)
        if self.pm_node:
            bpm[idx[self.pm_node]] = 1.0
        bio = np.zeros(n)
        if self.io_node:
            bio[idx[self.io_node]] = 1.0
        bch = np.zeros(n)
        if self.chg_node:
            bch[idx[self.chg_node]] = 1.0
        # columns: CPU, dGPU, memory, I/O, charge
        bmat = np.stack([bpc, bpg, bpm, bio, bch], 1)
        return cinv, gmat, ga0, ga1, bmat

    def substeps(self, p, max_fan):
        """Explicit-Euler substeps so h stays well under the fastest time constant."""
        cinv, gmat, ga0, ga1, _ = self.build(p)
        g_tot = ga0 + ga1 * max_fan + gmat.sum(axis=1)
        tau_min = float(np.min(1.0 / (cinv * g_tot)))
        return max(1, math.ceil(3.0 / tau_min))


TOPOLOGIES = {
    # one heatsink node shared by CPU and dGPU
    "rc3": Topology("rc3", ["H", "C", "S"], [("H", "C"), ("C", "S")], ["H", "C"], "H", "H"),
    # rc3 + a small memory node M fed by PM0C (memory heat that PC0R does not
    # include; TM0P read ~2 K above rc3's prediction in a compile)
    "rc3m": Topology(
        "rc3m", ["H", "C", "S", "M"], [("H", "C"), ("C", "S"), ("M", "C")], ["H", "C", "M"], "H", "H", pm_node="M"
    ),
    # rc3m + I/O node W (Wi-Fi/SSD path rails) and charge node B: heat the
    # 2026-10-28 relevance pass found under TW0P/TM0P with a ~2 min lag
    "rc3mwb": Topology(
        "rc3mwb",
        ["H", "C", "S", "M", "W", "B"],
        [("H", "C"), ("C", "S"), ("M", "C"), ("W", "C"), ("B", "C")],
        ["H", "C", "M", "W", "B"],
        "H",
        "H",
        pm_node="M",
        io_node="W",
        chg_node="B",
    ),
    # CPU and dGPU sides of the heatsink as separate nodes
    "rc4": Topology(
        "rc4",
        ["Hc", "Hg", "C", "S"],
        [("Hc", "Hg"), ("Hc", "C"), ("Hg", "C"), ("C", "S")],
        ["Hc", "Hg", "C"],
        "Hc",
        "Hg",
    ),
    # both die sides feed a shared fin stack F, which the fans cool
    "rc5": Topology(
        "rc5",
        ["Hc", "Hg", "F", "C", "S"],
        [("Hc", "F"), ("Hg", "F"), ("Hc", "C"), ("Hg", "C"), ("C", "S")],
        ["F", "C"],
        "Hc",
        "Hg",
    ),
}


class ThermalModel:
    """A fitted model: topology + dynamic params + per-sensor output map."""

    def __init__(self, topo, params, out_w, out_d, ambient=0.0):
        self.topo = topo
        self.params = params  # dict
        self.out_w = out_w  # sensor -> node weights (array, >= 0)
        self.out_d = out_d  # sensor -> offset (absorbs the ambient of the fit data)
        self.ambient = ambient  # shift for a different ambient than the fit data
        self.kalman = None  # the model's own Kalman settings (sim/forecast.py), if tuned

    def simulate(self, x0, pc, pg, rpm, pm=None, io=None, chg=None):
        fan = fan_feature(rpm, self.params["fan_exp"])
        cinv, gmat, ga0, ga1, bmat = self.topo.build(self.params)
        nsub = self.topo.substeps(self.params, float(fan.max()) if len(fan) else 0.0)
        z = np.zeros(len(pc))
        u = np.stack([pc, pg, z if pm is None else pm, z if io is None else io, z if chg is None else chg], 1)
        return _simulate(np.asarray(x0, float), np.ascontiguousarray(u, float), fan, cinv, gmat, ga0, ga1, bmat, nsub)

    def sensors(self, states):
        return {k: states @ self.out_w[k] + self.out_d[k] + self.ambient for k in smc.SENSORS}

    def steady_state(self, pc, pg, rpm, pm=0.0, io=0.0, chg=0.0):
        """Node temperatures for constant inputs held forever."""
        cinv, gmat, ga0, ga1, bmat = self.topo.build(self.params)
        f = fan_feature(np.array(rpm), self.params["fan_exp"])
        a = np.diag(ga0 + ga1 * f + gmat.sum(axis=1)) - gmat
        return np.linalg.solve(a, bmat @ np.array([pc, pg, pm, io, chg], float))

    def to_json(self):
        return json.dumps(
            {
                "topology": self.topo.name,
                "params": self.params,
                "out_w": {k: list(map(float, v)) for k, v in self.out_w.items()},
                "out_d": {k: float(v) for k, v in self.out_d.items()},
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        m = cls(
            TOPOLOGIES[d["topology"]],
            d["params"],
            {k: np.array(v) for k, v in d["out_w"].items()},
            d["out_d"],
        )
        m.kalman = d.get("kalman")
        return m


def solve_outputs(states_list, y_list, w_list):
    """Output map per sensor: y ~ states @ w + d with w >= 0 and sum(w) <= 1.

    A sensor sits on the heat path between the nodes and ambient, so it reads
    a convex mix of node temperatures (the remainder weight goes to ambient).
    Without the cap, node scale and output weights trade off freely. The cap is
    an equality with a non-negative ambient slack column, enforced by a heavily
    weighted extra row.

    states_list: per segment (N, n) arrays; y_list: per segment dict sensor ->
    (N,) with NaN where unused; w_list: per segment sample weights (N,).
    """
    out_w, out_d = {}, {}
    for k in smc.SENSORS:
        rows, ys = [], []
        for st, y, w in zip(states_list, y_list, w_list):
            m = ~np.isnan(y[k]) & (w > 0)
            sw = np.sqrt(w[m])[:, None]
            ones = np.ones((m.sum(), 1))
            # columns: node weights | +offset | -offset | ambient slack
            a = np.hstack([st[m], ones, -ones, 0 * ones])
            rows.append(a * sw)
            ys.append(y[k][m] * sw[:, 0])
        n = states_list[0].shape[1]
        big = 1e4 * np.sqrt(sum(float(w.sum()) for w in w_list))
        cap = np.concatenate([np.ones(n), [0.0, 0.0, 1.0]]) * big
        a = np.vstack(rows + [cap[None, :]])
        b = np.concatenate(ys + [[big]])
        coef, _ = nnls(a, b)
        out_w[k] = coef[:n]
        out_d[k] = coef[n] - coef[n + 1]
    return out_w, out_d
