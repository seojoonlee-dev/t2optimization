"""The SMC's own throttle chain, as reverse-engineered on 2026-10-19 (MacBookPro15,1).

    sensors --EMA mix--> Ts1S --PI--> PZ0E (zone-0 power budget, W)
    PC0R + PG0R = PZ0G --30 s EMA--> PZ0F
    PZ0F - PZ0E --PI--> PZ0T (0..100); iGPU pinned at 150 MHz from PZ0T ~83

Everything here runs at a 1 s step. A suspend resets the SMC: the Ts1S filters
re-initialise to the current sensor values and both integrators drop to 0.
"""

import math

import numpy as np

# Ts1S = offset + sum(gain * EMA(sensor, tau))
TS1S_OFFSET = -1.386
TS1S_TERMS = {  # sensor: (gain, tau_s)
    "TW0P": (1.0373, 22.5),
    "TM0P": (-0.6553, 25.5),
    "Ts0P": (0.3332, 16.0),
    "TG0P": (0.1835, 69.3),
    "Ts1P": (0.1442, 1915.0),
}
SENSORS = list(TS1S_TERMS)

# stage 1: Ts1S -> budget
SETPOINT = 44.989
KP_E = 12.25
KI_E = 0.0595
I_MAX = 65.45
E_MAX = 70.0

# stage 2: budget vs filtered zone power -> throttle
TAU_F = 30.0
KP_T = 4.07
KI_T = 0.148

PIN_LEVEL = 83.0  # iGPU pinned at 150 MHz above this


def ema_alpha(tau, dt=1.0):
    return 1.0 - math.exp(-dt / tau)


def ts1s(sensors, resets=()):
    """Ts1S from the five sensor series (dict name -> array, 1 s step).

    `resets` are sample indices where the SMC restarted (suspend); the filters
    re-initialise to the sensor value there, as they do at index 0.
    """
    n = len(sensors[SENSORS[0]])
    resets = set(resets) | {0}
    out = np.full(n, TS1S_OFFSET)
    for name, (gain, tau) in TS1S_TERMS.items():
        x = sensors[name]
        a = ema_alpha(tau)
        f = np.empty(n)
        for i in range(n):
            f[i] = x[i] if i in resets else f[i - 1] + a * (x[i] - f[i - 1])
        out += gain * f
    return out


def budget(ts1s_series, resets=(), i0=0.0):
    """PZ0E (W) and the integrator state from a Ts1S series."""
    n = len(ts1s_series)
    resets = set(resets)
    pz0e = np.empty(n)
    integ = np.empty(n)
    acc = i0
    for i in range(n):
        if i in resets:
            acc = 0.0
        e = ts1s_series[i] - SETPOINT
        acc = min(max(acc + KI_E * e, 0.0), I_MAX)
        integ[i] = acc
        pz0e[i] = min(max(E_MAX - KP_E * e - acc, 0.0), E_MAX)
    return pz0e, integ


def throttle(zone_power, pz0e, resets=(), f0=None, j0=0.0):
    """PZ0F (30 s EMA of PC0R+PG0R) and PZ0T (0..100)."""
    n = len(zone_power)
    resets = set(resets)
    a = ema_alpha(TAU_F)
    pz0f = np.empty(n)
    pz0t = np.empty(n)
    f = zone_power[0] if f0 is None else f0
    j = j0
    for i in range(n):
        if i in resets:
            f, j = zone_power[i], 0.0
        else:
            f += a * (zone_power[i] - f)
        d = f - pz0e[i]
        j = min(max(j + KI_T * d, 0.0), 100.0)
        pz0f[i] = f
        pz0t[i] = min(max(KP_T * d + j, 0.0), 100.0)
    return pz0f, pz0t
