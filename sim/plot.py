"""Plot a fitted model's free-run prediction against the logs.

    python -m sim.plot rc4
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data, smc
from .fit import MODELS, ROOT, Problem
from .thermal import TOPOLOGIES


def main(name):
    segs = data.load_all(test=False)
    prob = Problem(TOPOLOGIES[name], segs)
    v = np.load(MODELS / f"{name}-solution.npy")
    model, states, shift, _ = prob.simulate(v)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    for s, st in zip(segs, states):
        model.ambient = prob.ambient(s, shift)
        pred = model.sensors(st)
        ts = smc.ts1s(pred, s.smc_resets)
        pz0e, _ = smc.budget(ts, s.smc_resets)
        _, pz0t = smc.throttle(s.zone_power, pz0e, s.smc_resets)
        t = s.times()
        fig, ax = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
        for k, c in zip(smc.SENSORS, ["C0", "C1", "C2", "C3", "C4"]):
            m = ~np.isnan(s.y[k])
            ax[0].plot(t[m], s.y[k][m], ".", ms=2, color=c, label=k)
            ax[0].plot(t, pred[k], "-", lw=0.8, color=c)
        ax[0].legend(ncol=5, fontsize=8)
        ax[0].set_ylabel("sensors °C (dots = measured)")
        m = ~np.isnan(s.y["Ts1S"])
        ax[1].plot(t[m], s.y["Ts1S"][m], ".", ms=2, color="k", label="Ts1S measured")
        ax[1].plot(t, ts, "-", lw=0.8, color="C3", label="Ts1S model")
        ax[1].axhline(smc.SETPOINT, color="gray", lw=0.5)
        ax[1].legend(fontsize=8)
        ax[2].plot(t[m], s.y["PZ0E"][m], ".", ms=2, color="k", label="PZ0E measured")
        ax[2].plot(t, pz0e, "-", lw=0.8, color="C3", label="PZ0E model")
        ax[2].plot(t[m], s.y["PZ0T"][m], ".", ms=2, color="C1", label="PZ0T measured")
        ax[2].plot(t, pz0t, "-", lw=0.8, color="C1", alpha=0.6, label="PZ0T model")
        ax[2].legend(fontsize=8)
        ax[3].plot(t, s.pc, lw=0.6, label="PC0R W")
        ax[3].plot(t, s.pg, lw=0.6, label="PG0R W")
        ax[3].legend(fontsize=8)
        ax[4].plot(t, s.rpm, lw=0.6, color="C7")
        ax[4].set_ylabel("fan rpm")
        if s.val_from is not None:
            for a in ax:
                a.axvline(t[s.val_from], color="r", ls="--", lw=1)
        fig.suptitle(f"{name} — {s.name} (red dashed = start of held-out validation)")
        fig.tight_layout()
        path = out / f"{name}-{s.name}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)
        model.ambient = 0.0


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rc4")
