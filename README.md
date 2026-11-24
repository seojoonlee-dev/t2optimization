# t2optimization

A fan and power controller for Intel MacBooks with the T2 chip, running Linux,
used in place of `t2fanrd`. It uses model-predictive control, a long-established
technique: a thermal model of the laptop predicts where the temperature is
heading, and fans and power limits are set from that prediction. The goal is to
avoid the SMC's own throttling while keeping the fans as quiet as possible.

> **Status: proof of concept.** It was developed and tuned on one machine, a
> 2019 15" MacBook Pro (MacBookPro15,1, i9 + Radeon dGPU). The thermal model
> that ships in `models/` describes *that* laptop. Other T2 Macs have different
> heatsinks, fans and sensors, so on them you should treat the shipped model as
> a starting point and train your own (see [Training your own model](#training-your-own-model)).

## Why

Inside every T2 Mac, the SMC chip keeps track of a "skin" temperature called
`Ts1S`, roughly how hot the case feels to your hands. Once `Ts1S` goes above
about 45 °C, the SMC lowers the power budget for the CPU and GPU on its own
(the `PZ0T` throttle value climbs above 0). If it pushes hard enough, the iGPU
gets pinned at 150 MHz and the desktop starts to stutter.

`t2fanrd` sets the fan speed from a curve. It takes the higher of the CPU
package temperature and the GPU temperature (when a GPU sensor is found),
averages it over the last few seconds, and maps it between `low_temp` and
`high_temp` using one of three curve shapes: `linear`, `exponential` or
`logarithmic`. It doesn't look at `Ts1S` or the SMC's power budget, and it
doesn't limit power. Die temperatures change within seconds, while `Ts1S`
changes over minutes, so a curve tuned on die temperature can let the SMC
throttle under sustained load, or run the fans faster than needed.

This project controls `Ts1S` directly instead, using fans and power limits
together, ensuring the highest performance while keeping noise low.

## How it works

`t2opt-fand` is a small Rust daemon. Every second it:

1. **Reads the SMC**: the five temperature sensors that make up `Ts1S`, CPU
   and dGPU power, fan speeds and the throttle state.
2. **Updates its estimate** of how much heat is stored in the laptop, using a
   simple thermal model and correcting it against the real sensors (a Kalman
   filter). It also tracks room temperature, since that changes from day to day.
3. **Looks ahead.** Every few seconds it simulates the next ~90 seconds and
   picks the **slowest fan speed** that keeps `Ts1S` just under the SMC's limit
   (44.6 °C by default).
4. **Caps power when fans aren't enough.** If even the highest allowed fan
   speed won't hold the temperature, it lowers the CPU power limits (Intel
   RAPL PL1/PL2) and the dGPU power cap (amdgpu `power1_cap`) just enough that
   the SMC never has to throttle. Short bursts still get full power. Caps are
   tightened only when the long-term heat really calls for it.
5. **Has safety floors.** High CPU or GPU die temperatures, a `Ts1S` reading
   above the limit (meaning the model was wrong) or any SMC throttle all push
   the fans up, whatever the plan says. The hottest CPU core is held under a
   set limit (98 °C by default) through PL2.

Fans speed up quickly and slow down gently. If the daemon stops, crashes or
misses its systemd watchdog, the fans go back to the SMC's own control.

### Power modes

There are three modes: **eco**, **balanced** and **perf**. Each mode sets a
fan-speed ceiling, and the daemon sets power caps to fit under it. Eco also
gives the CPU a low average power budget that still allows short bursts, plus
a hard cap on the dGPU.

```sh
daemon/power-mode                    # show the current mode
daemon/power-mode eco|balanced|perf  # switch
daemon/power-mode toggle             # eco -> balanced -> perf -> eco
```

Switching modes doesn't need root. The mode file is writable by the `wheel`
group.

## The thermal model

The model is a lumped thermal model (`sim/thermal.py` for training,
`daemon/src/model.rs` at runtime). It describes the laptop as a few
connected blocks that each store heat.

### The blocks

The current model (`rc3mwb`) has six blocks:

| Block | What it stands for | Heated by |
| --- | --- | --- |
| H | the heatsink, shared by CPU and dGPU | CPU power (`PC0R`), dGPU power (`PG0R`) |
| C | the chassis and logic board | part of the CPU power, plus heat flowing from the other blocks |
| S | a slow mass that warms and cools over hours | heat flowing from C |
| M | the memory | memory power (`PM0C`) |
| W | the Wi-Fi and SSD area | Wi-Fi + SSD power (`POLR + PORR + PSLC`) |
| B | the battery and charging circuitry | charging power (`PDTR − PSTR`) |

Every block's temperature is tracked as *degrees above room air*. Each second,
for each block:

- **heat in** = the power fed to it
- **heat between blocks** = a fixed conductance × the temperature difference
  to each block it's joined to
- **heat out to the air** = (a + b × fan speed^k) × how far it is above room
  temperature. `a` is passive cooling, `b` is how strongly the fans reach that
  block, and `k` is fitted. Blocks the fans don't reach get `b = 0`.

Each block also has a heat capacity, which sets how fast it warms up. All of
these numbers (capacities, conductances, `a`, `b`, `k`) are what training
finds.

### From blocks to sensors

The SMC has five temperature sensors that it uses to work out `Ts1S`: `TW0P`,
`TM0P`, `Ts0P`, `TG0P` and `Ts1P`. The model predicts each one as

> sensor = room temperature + fixed offset + a weighted mix of block temperatures

The weights are fitted, and none of them can be negative.

### From sensors to `Ts1S` and throttling

`sim/smc.py` (copied into `model.rs`) is a reverse-engineered copy of the SMC's
own logic. `Ts1S` isn't a physical sensor: the SMC computes it by smoothing each
of the five sensors (each with its own time constant) and adding them up with fixed weights. The SMC then turns `Ts1S`
into a power budget and a throttle value. With this copy, the daemon can predict
whether the SMC *would* throttle, not just how hot the sensors will get.

### The hottest CPU core

A separate small formula predicts the hottest core from the heatsink and
chassis temperatures, the CPU cores' power, and how many cores share that
power. One busy core at 30 W gets much hotter than sixteen at 2 W each. This
lets the daemon keep core temperatures under `die_limit_c` by adjusting PL2.

### Estimating room air temperature

No sensor measures the air the laptop sits in, but every prediction depends on
it. So the daemon estimates it. A Kalman filter runs every second: it compares
the predicted sensor readings with the real ones and nudges the block
temperatures toward what the sensors say. Room temperature is one more unknown
in that filter, assumed to drift slowly. When all five sensors read warmer (or
cooler) than the model expects for a long stretch, the filter puts it down to
the room being warmer (or cooler) and adjusts its estimate. You can see the
current estimate as `room_offset_k` in `/run/t2opt-fand.json`. It's measured
relative to the room during training, not in absolute °C.

Because it's inferred, this estimate also soaks up part of whatever the model
gets wrong. A better-trained model makes it more trustworthy.

### Predicting ahead

Every few seconds the daemon takes the filter's current picture (block
temperatures plus room temperature) and runs the model forward about 90
seconds. It assumes power stays at its recent level and tries different fan
speeds. It picks the slowest one whose predicted `Ts1S` stays under
`ceiling_c`. For power caps, it looks further ahead (minutes) at the power
level the laptop could sustain.

### Models in this repo

`rc3mwb-a` in `models/` is the default model. Simpler layouts (`rc3` with only H, C and S;
`rc3m` adding memory; `rc4` and `rc5` with separate CPU and dGPU heatsink
blocks) are defined in `sim/thermal.py`.

## Requirements

- An Intel T2 Mac with **two fans**, running a T2 Linux kernel (see
  [t2linux.org](https://t2linux.org)) with `applesmc` exposing the SMC keys and
  fan control under `/sys/bus/acpi/devices/APP0001:00`
- Intel RAPL (`/sys/class/powercap/intel-rapl:0`) and the `msr` module
- Rust (cargo) to build the daemon
- Python 3 with the packages in `requirements.txt`, only if you want to train
  a model (**highly recommended**: the shipped model was trained on one
  MacBookPro15,1, so your own machine will be predicted better by a model
  trained on its own data)
- systemd

The daemon needs these SMC keys: `TW0P TM0P Ts0P TG0P Ts1P Ts1S PZ0T PC0R PG0R
F0Ac F1Ac`. Optional ones: `TGDD` (dGPU die), `PM0C` (memory power; required
if the model uses it), and `POLR PORR PSLC PDTR PSTR` (Wi-Fi/SSD and charging
power). Check that your machine has them before you trust the controller:

```sh
ls /sys/bus/acpi/devices/APP0001:00/ | head
```

## Install

```sh
# build (as your user)
cargo build --release --manifest-path daemon/Cargo.toml

# 1. try it without touching the fans: plans only
sudo daemon/fand-start.sh --shadow
journalctl -fu t2opt-fand          # watch what it would do
sudo systemctl stop t2opt-fand

# 2. try it for real, until the next reboot (t2fanrd comes back after)
sudo daemon/fand-start.sh
cat /run/t2opt-fand.json            # current state

# 3. make it the boot default in place of t2fanrd
sudo daemon/install-service.sh
```

The service runs the binary straight from this checkout. After pulling changes,
rebuild and run `sudo systemctl restart t2opt-fand`.

If you run other services that also control fans or power limits, list them in
`local.env` (copy `local.env.example`) as `EXTRA_UNITS`. They're stopped while
`t2opt-fand` runs and started again when it stops.

### Going back to t2fanrd

```sh
./revert-to-t2fanrd.sh
```

### Optional: scx_lavd scheduler

`daemon/setup-scx.sh` installs the `scx_lavd` CPU scheduler (Arch packages,
via `pacman`). It favours latency-sensitive threads like the compositor, input
and audio, so the desktop stays smooth while the CPU is capped. When it's
installed, the daemon switches its profile to match the power mode. You can
undo it with `systemctl disable --now scx_loader`.

## Configuration

Settings live in `daemon/fand.toml`. Every key has a comment explaining what
it does. Any key you leave out falls back to the default in
`daemon/src/control.rs`. To override settings for your machine without
touching the tracked file, put them in `daemon/fand.local.toml` (copy
`daemon/fand.local.toml.example`). Restart the daemon to apply changes.

The ones you're most likely to change:

| Key | What it does |
| --- | --- |
| `model` | which thermal model to load |
| `ceiling_c` | the `Ts1S` the plan stays under (the SMC's limit is ~44.99 °C) |
| `die_limit_c` | the hottest CPU core is kept under this |
| `[fan_ceiling_rpm]` | the fan-speed ceiling for each mode |
| `eco_sustained_w`, `eco_dgpu_cap_w` | eco's CPU budget and dGPU cap |
| `cpu_floor_c`, `gpu_floor_c` | die temperatures where the fans start rising, and where they reach full speed |

## Testing changes offline

The daemon can run against a log recorded by the logger (`tools/extra-logger.py`,
see [Training your own model](#training-your-own-model)), without root and
without touching the fans:

```sh
t2opt-fand --replay data/extra-<time>.csv [--trace]          # what it would have decided
t2opt-fand --sim data/extra-<time>.csv [--from T] [--to T]   # closed loop against the model
```

`--sim` feeds the daemon's own fan and cap decisions back through the thermal
model. That lets you try a new setting or model on a recorded workload (a
compile, a game, a long download) before running it on the real machine.

## Training

### How it was trained

- **Data:** about two weeks of 1-second logs from one MacBookPro15,1, covering
  idle, desktop use, overnight runs, CPU and GPU stress tests, compiles, games,
  long Wi-Fi downloads, and time both on battery and charging.
- **What the fit scores:** each candidate model is run freely over whole
  recorded stretches, minutes to hours, and compared with what the sensors
  actually read. It isn't scored on one-second-ahead guesses, because the
  daemon relies on 90-second look-ahead, so the model has to stay accurate over
  long runs.
- **Fairness:** every log file counts equally, so hours of quiet overnight data
  don't drown out the short heavy-load runs where the controller matters most.
  Each log also gets its own small room-temperature offset.
- **Held-out data:** the last part of each log is never used for fitting. It's
  used only to check the model's predictions on data it hasn't seen.
  `sim/forecast.py` scores the model the way the daemon uses it (correct from
  the sensors, then predict 30/60/90/180 s ahead).
- **Speed:** the fit is spread across every CPU, one worker process per core.

### Training your own model

Because the model describes one particular laptop, a different T2 Mac will
predict better with a model trained on its own data.

1. **Collect data.** Install the logger and leave it running:
   ```sh
   sudo tools/install-extra-logger.sh
   ```
   It writes `data/extra-*.csv` once a second, and it's the only source of
   training data (the daemon writes no logs). Each row has the SMC sensors,
   `Ts1S`, the SMC's power budget and throttle value, CPU, dGPU, memory,
   Wi-Fi/SSD and charging power, fan speeds, the hottest CPU core (sampled 10
   times a second), busy CPU count, RAPL power, Wi-Fi traffic, SSD, screen and
   charging state, and the daemon's own decisions while it runs. It only
   reads; it never changes fans or power limits, so it can run alongside
   `t2fanrd` or this daemon.

   Use the machine normally, and also cover the cases you care about: idle,
   sustained CPU load, sustained GPU load, a long download, charging and on
   battery, at a range of fan speeds.
2. **Fit:**
   ```sh
   python -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   python -m sim.fit rc3mwb --out my-model     # writes models/my-model.json
   ```
   To refine an existing model instead of starting from scratch, use
   `--warm <model> --free <param,...>`.
3. **Check it:**
   ```sh
   python -m sim.forecast my-model   # prediction error at 30-180 s ahead
   python -m sim.plot my-model       # prediction vs. logs, in reports/
   ```
4. **Try it offline** with `t2opt-fand --sim data/extra-<time>.csv`. Then set
   `model = "../models/my-model.json"` in `fand.local.toml` and restart.

Simpler layouts like `rc3` need fewer SMC keys and less data.

## Project layout

```
daemon/            the controller (Rust) and its service/install scripts
  src/main.rs      main loop, logging, --replay / --sim / --release
  src/control.rs   fan planning, power caps, safety floors, settings
  src/model.rs     thermal model + Kalman filter at runtime
  src/hw.rs        SMC, fans, RAPL, amdgpu, MSR access
  fand.toml        settings
  power-mode       switch eco / balanced / perf
sim/               model training and evaluation (Python)
  thermal.py       the lumped thermal model and its layouts
  smc.py           the SMC's throttle logic, reverse-engineered
  data.py          loading logs onto a 1 s grid
  fit.py           fitting
  forecast.py      scoring look-ahead predictions
  plot.py          plots
tools/
  extra-logger.py  1 Hz logger: training data and the daemon's decisions (service: t2opt-extralog)
models/            fitted models
revert-to-t2fanrd.sh
```

## Safety notes

- The daemon writes fan speeds, CPU power-limit MSRs and the dGPU power cap.
  It restores the fans to the SMC on exit, and restores the PL2 clamp bit
  that it sets.
- SMC reads go through one shared register, so two programs reading SMC keys
  at once can corrupt each other's reads. The daemon and the extra logger
  share a lock file (`/run/t2opt-smc.lock`) so they can run together; don't
  run other SMC-polling tools alongside them.
- If the service fails repeatedly, systemd starts `t2fanrd` again.
