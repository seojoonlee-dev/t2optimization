//! Planner and controller: which fan speed and power caps to command.
//!
//! The power mode sets a fan ceiling (and eco an energy budget). Under it, the
//! controller picks the slowest fan speed that keeps predicted Ts1S under the
//! ceiling. CPU + dGPU power is capped at what the mode's fan ceiling can
//! sustain (the model's steady state), with bursts above that only while
//! there is room to drop back to it. Light loads never reach the cap; heavy
//! ones get the fans first, then glide down to the sustainable power.
//!
//! Heat the model doesn't know about (Wi-Fi under a long Steam download ran
//! TW0P 2+ K over the model, 2026-10-21) is learned by integral action on
//! Ts1S while power-limited and added to every prediction.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use serde::Deserialize;

use crate::model::{self, Kalman, Model, Power, Replica, FAN_TAU_S, SETPOINT};

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// model JSON, relative to the config file's directory
    pub model: String,
    /// where the 1 Hz CSV logs go, relative to the config file's directory
    pub log_dir: String,
    pub log_csv: bool,
    /// Ts1S the plan must stay under; the SMC budget's set point is 44.989.
    /// Fans plan for it, and the power caps settle Ts1S there
    pub ceiling_c: f64,
    /// a burst is planned as held this long, then cut to the sustainable power
    pub cap_hold_s: u32,
    /// CPU + dGPU power (PC0R + PG0R) never allowed over this: the SMC's
    /// budget (PZ0E) is 70 W even when cool, and its 30 s mean of the zone
    /// power going over it throttles (CPU 45 W + dGPU 30 W would)
    pub zone_max_w: f64,
    /// how far ahead the burst plan looks (the chassis takes minutes)
    pub sustain_horizon_s: u32,
    /// integral action on Ts1S over the ceiling while power-limited, W of
    /// extra heat per K·s; limited to ± heat_max_w, fades with heat_fade_s
    /// once the load is gone
    pub heat_gain: f64,
    pub heat_max_w: f64,
    pub heat_fade_s: f64,
    /// under the ceiling, extra heat isn't lowered while Ts1S has risen more
    /// than heat_rising_c over the last heat_rising_s: the plan only brings
    /// Ts1S up to the ceiling over minutes, so it is still on its way
    pub heat_rising_s: u32,
    pub heat_rising_c: f64,
    /// extra margin a slower fan speed needs before the fans come down
    pub down_hysteresis_c: f64,
    pub horizon_s: u32,
    pub plan_every_s: u32,
    /// after a start or resume the filter needs a moment (room estimate);
    /// hold the fans where they are until then
    pub warmup_s: u32,
    pub rpm_step: u32,
    /// fans come down at most this fast; up is immediate
    pub down_rpm_per_s: f64,
    /// user hard limits on the mean fan speed (clamped to the SMC's fan range)
    pub min_rpm: u32,
    pub max_rpm: u32,
    /// die-temperature floors: min fans at the first value, max fans at the second
    pub cpu_floor_c: [f64; 2],
    /// the CPU floor follows the mean of the cores over this many seconds, so
    /// a one-second spike on a core or two doesn't set it
    pub cpu_floor_avg_s: u32,
    pub gpu_floor_c: [f64; 2],
    /// measured Ts1S this far above the set point -> fans past the mode's
    /// ceiling (full speed at cut_over_setpoint_c) until back under the ceiling
    pub panic_over_setpoint_c: f64,
    /// measured Ts1S this far above the set point -> both caps to their floors
    pub cut_over_setpoint_c: f64,
    /// mode at start when no mode file says otherwise: eco, balanced or perf
    pub default_mode: String,
    pub fan_ceiling_rpm: FanCeilings,
    /// eco holds the CPU package (RAPL W) to this on average, and lets it
    /// burst to cpu_pl1_max_w for about eco_burst_s once it has been under
    /// it for a while (an energy budget; the cap slides down as it drains)
    pub eco_sustained_w: f64,
    pub eco_burst_s: f64,
    /// eco's hard dGPU cap (power1_cap W, no burst); without it eco's dGPU
    /// got the full thermal cap (Minecraft held ~20-25 W, 2026-10-24)
    pub eco_dgpu_cap_w: f64,
    /// while eco's energy budget holds the package, the cores alone are
    /// limited (RAPL PP0) to what they got under it, the package minus
    /// eco_uncore_w, and the package limit rises by eco_igpu_w, so the extra
    /// is the iGPU's. Without it the cores took the whole package and the
    /// iGPU sat at 100-250 MHz asking for 1250 (desktop lag, 2026-10-24)
    pub eco_igpu_w: f64,
    /// package power outside the cores at eco's cap with the iGPU idle (ring,
    /// LLC, system agent; 3.6-3.9 W in the 2026-10-2x fand logs)
    pub eco_uncore_w: f64,
    /// CPU package limit range (RAPL PL1, W); the low end keeps the iGPU,
    /// which shares the package budget, from starving (9 W stuttered video)
    pub cpu_pl1_min_w: f64,
    pub cpu_pl1_max_w: f64,
    /// PL2 = PL1 + this, for short bursts; PL1 averages over pl1_window_s
    pub pl2_extra_w: f64,
    pub pl1_window_s: f64,
    /// dGPU power cap range (amdgpu power1_cap, W) while it is awake
    pub dgpu_cap_min_w: f64,
    pub dgpu_cap_max_w: f64,
    /// caps drop at once but rise at most this much per plan
    pub cap_raise_w: f64,
    /// CPU PL1 until the first plan (warm-up after start): at boot the
    /// firmware default would otherwise run unchecked
    pub warmup_pl1_w: f64,
    /// follow the power mode with scx_lavd's profile through scx_loader
    /// (only while scx_lavd is the running scheduler)
    pub lavd_profiles: bool,
    /// command that puts the dGPU back on its stock DPM level and overdrive
    /// table, run with the argument "auto" when the card wakes with a custom
    /// one (e.g. a gaming undervolt); unset leaves the table alone
    pub dgpu_dpm_cmd: Option<String>,
    /// dGPU cap (W) restored on exit; unset uses the driver's power1_cap_default
    pub dgpu_release_cap_w: Option<f64>,
}

/// Fan ceiling per power mode, rpm (mean of the two fans).
#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct FanCeilings {
    pub eco: u32,
    pub balanced: u32,
    pub perf: u32,
}

impl Default for FanCeilings {
    fn default() -> Self {
        Self { eco: 2500, balanced: 3800, perf: 99999 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Eco,
    Balanced,
    Perf,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "eco" => Some(Self::Eco),
            "balanced" => Some(Self::Balanced),
            "perf" => Some(Self::Perf),
            _ => None,
        }
    }

    /// scx_loader's mode for scx_lavd: PowerSave (2) runs --powersave, Auto (0)
    /// --autopilot, Gaming (1) --performance (scx_loader 1.1.3 defaults)
    pub fn scx_mode(self) -> u32 {
        match self {
            Self::Eco => 2,
            Self::Balanced => 0,
            Self::Perf => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Eco => "eco",
            Self::Balanced => "balanced",
            Self::Perf => "perf",
        }
    }
}

// PC0R (CPU rail at the VR input) = PC0R_GAIN * RAPL package W + PC0R_OFFSET,
// fitted on the 2026-10-19 stress log (sim/data.py PC0R_FROM_PKG)
const PC0R_GAIN: f64 = 1.2648;
const PC0R_OFFSET: f64 = 0.804;

pub fn pc0r_from_pkg(w: f64) -> f64 {
    PC0R_GAIN * w + PC0R_OFFSET
}

pub fn pkg_from_pc0r(w: f64) -> f64 {
    (w - PC0R_OFFSET) / PC0R_GAIN
}

/// Power caps, in the SMC's units (PC0R and PG0R watts).
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub cpu: f64,
    /// None while the dGPU sleeps (it draws nothing to cap)
    pub gpu: Option<f64>,
    /// cores-only limit (RAPL PP0, W), set while eco's energy budget holds
    /// the package so the iGPU keeps its share; None = off
    pub cores: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "../models/rc3-v1.json".into(),
            log_dir: "../data".into(),
            log_csv: true,
            ceiling_c: 44.6,
            cap_hold_s: 30,
            zone_max_w: 64.0,
            sustain_horizon_s: 600,
            heat_gain: 0.1,
            heat_max_w: 20.0,
            heat_fade_s: 1800.0,
            heat_rising_s: 30,
            heat_rising_c: 0.05,
            down_hysteresis_c: 0.15,
            horizon_s: 90,
            plan_every_s: 5,
            warmup_s: 30,
            rpm_step: 100,
            down_rpm_per_s: 30.0,
            min_rpm: 0,
            max_rpm: 99999,
            cpu_floor_c: [90.0, 100.0],
            cpu_floor_avg_s: 1,
            gpu_floor_c: [90.0, 100.0],
            panic_over_setpoint_c: 0.4,
            cut_over_setpoint_c: 1.5,
            default_mode: "balanced".into(),
            fan_ceiling_rpm: FanCeilings::default(),
            eco_sustained_w: 8.0,
            eco_burst_s: 30.0,
            eco_dgpu_cap_w: 10.0,
            eco_igpu_w: 6.0,
            eco_uncore_w: 4.0,
            cpu_pl1_min_w: 12.0,
            cpu_pl1_max_w: 45.0,
            pl2_extra_w: 5.0,
            pl1_window_s: 20.0,
            dgpu_cap_min_w: 10.0,
            dgpu_cap_max_w: 30.0,
            cap_raise_w: 2.0,
            warmup_pl1_w: 25.0,
            lavd_profiles: true,
            dgpu_dpm_cmd: None,
            dgpu_release_cap_w: None,
        }
    }
}

impl Config {
    /// `path`, then `<stem>.local.toml` next to it (this machine's settings,
    /// kept out of git) on top; either may be missing.
    pub fn load(path: &Path) -> Result<Self, String> {
        let read = |p: &Path| -> Result<toml::Table, String> {
            match std::fs::read_to_string(p) {
                Ok(text) => text.parse().map_err(|e| format!("{}: {e}", p.display())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(toml::Table::new()),
                Err(e) => Err(format!("{}: {e}", p.display())),
            }
        };
        let mut table = read(path)?;
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        // tables (fan_ceiling_rpm) merge key by key, so a local file can set one mode
        for (k, v) in read(&path.with_file_name(format!("{stem}.local.toml")))? {
            match (table.get_mut(&k), v) {
                (Some(toml::Value::Table(base)), toml::Value::Table(over)) => base.extend(over),
                (_, v) => {
                    table.insert(k, v);
                }
            }
        }
        let cfg: Self = toml::Value::Table(table).try_into().map_err(|e| format!("{} (or {stem}.local.toml): {e}", path.display()))?;
        Mode::parse(&cfg.default_mode).ok_or_else(|| format!("default_mode {:?}: eco, balanced or perf", cfg.default_mode))?;
        Ok(cfg)
    }

    pub fn fan_ceiling(&self, mode: Mode) -> f64 {
        (match mode {
            Mode::Eco => self.fan_ceiling_rpm.eco,
            Mode::Balanced => self.fan_ceiling_rpm.balanced,
            Mode::Perf => self.fan_ceiling_rpm.perf,
        }) as f64
    }

    fn cpu_range(&self) -> (f64, f64) {
        (pc0r_from_pkg(self.cpu_pl1_min_w), pc0r_from_pkg(self.cpu_pl1_max_w))
    }
}

/// One SMC sample, the values the controller needs.
pub struct Reading {
    pub y: [f64; 5],
    pub ts1s: f64,
    pub pz0t: f64,
    pub p: Power,
    pub rpm: f64,
    pub tcmx: Option<f64>,
    /// mean of the CPU cores (coretemp), the CPU floor's input; TCMX where it
    /// is missing (logs from before it was logged)
    pub cores_c: Option<f64>,
    pub tgdd: Option<f64>,
    /// dGPU awake (runtime PM active); its power cap only applies then
    pub dgpu_active: bool,
}

/// Split a CPU + dGPU budget by what each draws: headroom shared equally,
/// a shortfall cut in proportion; then each kept inside its own range.
///
/// This is a plain rule, not something the model learned: thermally a watt
/// costs about the same on either side (one shared heatsink), and nothing
/// measured yet says which split gives more fps or a faster compile.
fn split(cfg: &Config, total: f64, demand: Power, dgpu_active: bool) -> Caps {
    let (cmin, cmax) = cfg.cpu_range();
    if !dgpu_active {
        return Caps { cpu: total.clamp(cmin, cmax), gpu: None, cores: None };
    }
    let (dc, dg) = (demand.pc.max(0.0), demand.pg.max(0.0));
    let (c, g) = if total >= dc + dg {
        let h = (total - dc - dg) / 2.0;
        (dc + h, dg + h)
    } else {
        let k = total / (dc + dg).max(1e-9);
        (dc * k, dg * k)
    };
    Caps { cpu: c.clamp(cmin, cmax), gpu: Some(g.clamp(cfg.dgpu_cap_min_w, cfg.dgpu_cap_max_w)), cores: None }
}

struct Planner {
    lo: f64,
    candidates: Vec<f64>,
    alpha: [f64; 5],
    /// heat the model misses (W, CPU-rail equivalent), added to every prediction
    heat: f64,
}

impl Planner {
    fn new(cfg: &Config, lo: f64, hi: f64) -> Self {
        let step = cfg.rpm_step as f64;
        let mut candidates = vec![lo, hi];
        let mut r = (lo / step).ceil() * step;
        while r < hi.trunc() {
            candidates.push(r);
            r += step;
        }
        candidates.sort_by(f64::total_cmp);
        candidates.dedup();
        Self { lo, candidates, alpha: model::ema_alpha(), heat: 0.0 }
    }

    /// CPU + dGPU budget (each drawing its whole share), from two questions:
    ///
    /// - sustainable: the most that, held forever at the fan speed `fan`,
    ///   settles Ts1S at `ceiling_c` (the model's steady state, with the
    ///   filter's room estimate, so heat the model misses lowers it);
    /// - now: the most that can be held for `cap_hold_s` and then cut to the
    ///   sustainable level with Ts1S staying under `ceiling_c` over
    ///   `sustain_horizon_s`.
    ///
    /// The second is never below the first unless the machine is already
    /// too hot, so a long load glides down to the sustainable level instead
    /// of being cut to the floor and let back up.
    #[allow(clippy::too_many_arguments)]
    fn power_cap(&self, cfg: &Config, m: &mut Model, kf: &Kalman, rep: &Replica, demand: Power, dgpu_active: bool, fan: f64, rpm_now: f64) -> (Caps, f64) {
        let (cmin, cmax) = cfg.cpu_range();
        let (lo, hi) = if dgpu_active { (cmin + cfg.dgpu_cap_min_w, cmax + cfg.dgpu_cap_max_w) } else { (cmin, cmax) };
        // the SMC budget is 70 W at most: more than that throttles at any temperature
        let hi = hi.min(cfg.zone_max_w).max(lo);
        let power = |total: f64| {
            let c = split(cfg, total, demand, dgpu_active);
            Power { pc: c.cpu, pg: c.gpu.unwrap_or(demand.pg), pm: demand.pm }
        };
        let sus = bisect(lo, hi, |t| self.steady(m, kf, rep, power(t), fan) <= cfg.ceiling_c);
        let ps = power(sus);
        let mut ok = |t: f64| {
            let pb = power(t);
            self.predict_seq(m, kf, rep, |i| if i < cfg.cap_hold_s { pb } else { ps }, fan, cfg.sustain_horizon_s, rpm_now) <= cfg.ceiling_c
        };
        // already past the burst ceiling: hold the sustainable power, which
        // brings Ts1S back to the ceiling; cutting deeper made a sawtooth
        let now = if ok(sus) { bisect(sus, hi, ok) } else { sus };
        (split(cfg, now, demand, dgpu_active), sus)
    }

    /// Ts1S once power `p` has been held at fan speed `rpm` for good.
    fn steady(&self, m: &Model, kf: &Kalman, rep: &Replica, p: Power, rpm: f64) -> f64 {
        let p = Power { pc: p.pc + self.heat, ..p };
        model::ts1s(&m.sensors(&m.steady(p, rpm), kf.ambient())) + rep.bias
    }

    /// Max predicted Ts1S over the horizon with power held at `p`.
    #[allow(clippy::too_many_arguments)]
    fn predict(&self, m: &mut Model, kf: &Kalman, rep: &Replica, p: Power, rpm: f64, horizon: u32, rpm_now: f64) -> f64 {
        self.predict_seq(m, kf, rep, |_| p, rpm, horizon, rpm_now)
    }

    /// Max predicted Ts1S over the horizon, power `p(second)`; the fans move
    /// from rpm_now toward the target with FAN_TAU_S lag.
    #[allow(clippy::too_many_arguments)]
    fn predict_seq(&self, m: &mut Model, kf: &Kalman, rep: &Replica, p: impl Fn(u32) -> Power, rpm: f64, horizon: u32, rpm_now: f64) -> f64 {
        let mut x = kf.nodes();
        let mut ema = rep.ema;
        let (amb, heat) = (kf.ambient(), self.heat);
        let mut ts_max = f64::NEG_INFINITY;
        for i in 0..horizon {
            // mid-step fan speed; after ~10 s it is the target
            let fan = if i < 10 { rpm + (rpm_now - rpm) * (-(i as f64 + 0.5) / FAN_TAU_S).exp() } else { rpm };
            let pi = p(i);
            x = model::advance(&m.step(fan), &x, Power { pc: pi.pc + heat, ..pi });
            let y = m.sensors(&x, amb);
            for k in 0..5 {
                ema[k] += self.alpha[k] * (y[k] - ema[k]);
            }
            ts_max = ts_max.max(model::ts1s(&ema) + rep.bias);
        }
        ts_max
    }

    /// Slowest fan speed whose predicted Ts1S stays under the ceiling.
    ///
    /// `current` = the speed last commanded, `rpm_now` = what the fans run at.
    /// Faster fans always predict a cooler Ts1S, and the ceiling is only
    /// stricter below `current` (hysteresis), so feasibility is monotone in
    /// rpm and a bisection over the candidates finds the slowest one.
    #[allow(clippy::too_many_arguments)]
    fn plan(&self, cfg: &Config, m: &mut Model, kf: &Kalman, rep: &Replica, p: Power, current: f64, rpm_now: f64, fan_max: f64) -> (f64, f64, &'static str) {
        let mut cache: HashMap<usize, f64> = HashMap::new();
        let mut ok = |i: usize, m: &mut Model| {
            let rpm = self.candidates[i];
            let ts = *cache.entry(i).or_insert_with(|| self.predict(m, kf, rep, p, rpm, cfg.horizon_s, rpm_now));
            let ceiling = cfg.ceiling_c - if rpm < current - 1.0 { cfg.down_hysteresis_c } else { 0.0 };
            (ts <= ceiling, ts)
        };
        // the mode's fan ceiling: the fastest candidate at or under it
        let top = self.candidates.iter().rposition(|&c| c <= fan_max + 0.5).unwrap_or(0);
        let (mut lo, mut hi) = (0, top);
        let (pass, ts) = ok(hi, m);
        if !pass {
            return (self.candidates[top], ts, "plan-max");
        }
        let (pass, ts) = ok(lo, m);
        if pass {
            return (self.lo, ts, "plan");
        }
        // invariant: lo fails, hi passes
        let mut ts_hi = ok(hi, m).1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            let (pass, ts) = ok(mid, m);
            if pass {
                hi = mid;
                ts_hi = ts;
            } else {
                lo = mid;
            }
        }
        (self.candidates[hi], ts_hi, "plan")
    }
}

/// Largest value in [lo, hi] that passes `ok` (to 0.25 W), assuming passing
/// is monotone; lo if even lo fails.
fn bisect(lo: f64, hi: f64, ok: impl FnMut(f64) -> bool) -> f64 {
    let mut ok = ok;
    if ok(hi) {
        return hi;
    }
    if !ok(lo) {
        return lo;
    }
    let (mut lo, mut hi) = (lo, hi);
    while hi - lo > 0.25 {
        let mid = (lo + hi) / 2.0;
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// rpm_lo below lo, rpm_hi above hi, linear between.
fn ramp(v: Option<f64>, [lo, hi]: [f64; 2], rpm_lo: f64, rpm_hi: f64) -> f64 {
    match v {
        Some(v) if v > lo => rpm_lo + ((v - lo) / (hi - lo)).min(1.0) * (rpm_hi - rpm_lo),
        _ => rpm_lo,
    }
}

pub struct Controller {
    pub cfg: Config,
    pub model: Model,
    pub kf: Kalman,
    pub rep: Replica,
    planner: Planner,
    pub lo: f64,
    pub hi: f64,
    pub cmd: f64,
    plan_rpm: f64,
    pub pred: f64,
    plan_reason: &'static str,
    pub reason: &'static str,
    panic: bool,
    zone_hist: VecDeque<Power>,
    /// the CPU floor's input over the last cpu_floor_avg_s
    cpu_hist: VecDeque<f64>,
    ts1s_hist: VecDeque<f64>,
    started: bool,
    t: u32,
    pub mode: Mode,
    replan: bool,
    /// caps to apply; None until the first plan (the hardware keeps what it has)
    pub caps: Option<Caps>,
    /// the thermal plan's caps, before eco's energy budget and safety cuts
    plan_caps: Option<Caps>,
    /// eco's energy budget, J of package energy above eco_sustained_w
    energy_j: f64,
    /// sustainable CPU + dGPU power at the mode's fan ceiling (PC0R + PG0R W), last plan
    pub sustainable: f64,
}

impl Controller {
    pub fn new(cfg: Config, model: Model, rpm_lo: f64, rpm_hi: f64) -> Self {
        let energy_j = (cfg.cpu_pl1_max_w - cfg.eco_sustained_w) * cfg.eco_burst_s;
        let lo = rpm_lo.max(cfg.min_rpm as f64);
        let hi = rpm_hi.min(cfg.max_rpm as f64);
        Self {
            planner: Planner::new(&cfg, lo, hi),
            kf: Kalman::new(&model),
            rep: Replica::new(),
            cfg,
            model,
            lo,
            hi,
            cmd: lo,
            plan_rpm: lo,
            pred: f64::NAN,
            plan_reason: "start",
            reason: "start",
            panic: false,
            zone_hist: VecDeque::with_capacity(61),
            cpu_hist: VecDeque::new(),
            ts1s_hist: VecDeque::new(),
            started: false,
            t: 0,
            mode: Mode::Balanced,
            replan: false,
            caps: None,
            plan_caps: None,
            energy_j,
            sustainable: f64::NAN,
        }
    }

    /// Switch power mode; the next tick plans for it at once.
    /// eco's CPU cap from its energy budget (PC0R W): the full PL1 range with
    /// a full budget, sliding to eco_sustained_w as it drains; None outside eco.
    fn energy_cap(&self) -> Option<f64> {
        let cfg = &self.cfg;
        (self.mode == Mode::Eco).then(|| {
            let room = cfg.cpu_pl1_max_w - cfg.eco_sustained_w;
            pc0r_from_pkg(cfg.eco_sustained_w + room * self.energy_j / (room * cfg.eco_burst_s))
        })
    }

    /// eco's hard dGPU cap (W); None outside eco.
    fn eco_gpu_cap(&self) -> Option<f64> {
        (self.mode == Mode::Eco).then_some(self.cfg.eco_dgpu_cap_w)
    }

    /// The package cap (PC0R W) and cores limit (RAPL W) while eco's energy
    /// cap `e` (PC0R W) holds: the cores keep what they got under `e`, the
    /// iGPU gets eco_igpu_w on top.
    fn igpu_fence(&self, e: f64) -> (f64, f64) {
        let pkg = pkg_from_pc0r(e);
        (pc0r_from_pkg(pkg + self.cfg.eco_igpu_w), (pkg - self.cfg.eco_uncore_w).max(1.0))
    }

    /// Extra heat the integral action has found (W).
    pub fn heat(&self) -> f64 {
        self.planner.heat
    }

    pub fn set_mode(&mut self, mode: Mode) {
        if mode != self.mode {
            self.mode = mode;
            self.replan = true;
        }
    }

    /// Caps drop at once and rise by at most cap_raise_w per plan.
    fn limit_rise(&self, want: Caps) -> Caps {
        let Some(prev) = self.plan_caps else { return want };
        let rise = |p: f64, w: f64| if w > p { w.min(p + self.cfg.cap_raise_w) } else { w };
        Caps {
            cpu: rise(prev.cpu, want.cpu),
            gpu: match (prev.gpu, want.gpu) {
                (Some(p), Some(w)) => Some(rise(p, w)),
                (_, w) => w,
            },
            cores: None,
        }
    }

    /// Start over: new model state, SMC filters re-read, warm-up again.
    pub fn reset(&mut self, r: &Reading) {
        self.kf.reset(&self.model, r.p, r.rpm);
        self.planner.heat = 0.0;
        self.rep.reset(&r.y);
        self.zone_hist.clear();
        self.cpu_hist.clear();
        self.ts1s_hist.clear();
        self.t = 0;
        self.cmd = r.rpm.clamp(self.lo, self.hi);
        self.plan_rpm = self.cmd;
        self.plan_reason = "warmup";
        self.pred = f64::NAN;
        self.started = true;
    }

    /// Integral action: with power capped (or Ts1S over the ceiling) and the
    /// fans where the cap plan put them, Ts1S should sit at the ceiling. What
    /// it is off by is heat the model doesn't know about (Wi-Fi under a long
    /// download, model error at this load), so it becomes extra heat in every
    /// prediction. Time over the ceiling always counts; time under it only
    /// with the fans no faster than the plan (not while they come down from
    /// a safety floor) and Ts1S no longer rising, and it fades once the load
    /// is gone.
    fn integrate_heat(&mut self, r: &Reading, fan_max: f64) {
        let cfg = &self.cfg;
        let limited = self.plan_reason == "power-cap" || r.ts1s > cfg.ceiling_c;
        self.ts1s_hist.push_back(r.ts1s);
        if self.ts1s_hist.len() > cfg.heat_rising_s as usize + 1 {
            self.ts1s_hist.pop_front();
        }
        // still climbing towards the ceiling: no evidence of less heat yet
        // (counting it ran extra heat to -5 W and overshot to 45.5 °C, 2026-10-27)
        let rising = r.ts1s - self.ts1s_hist[0] > cfg.heat_rising_c;
        let h = &mut self.planner.heat;
        let over = r.ts1s > cfg.ceiling_c;
        if limited && (over || (!rising && self.cmd <= fan_max + cfg.rpm_step as f64)) {
            *h = (*h + cfg.heat_gain * (r.ts1s - cfg.ceiling_c)).clamp(-cfg.heat_max_w, cfg.heat_max_w);
        } else if !limited {
            *h *= (-1.0 / cfg.heat_fade_s).exp();
        }
    }

    /// One second. Returns the commanded mean fan speed.
    pub fn tick(&mut self, r: &Reading) -> f64 {
        if !self.started {
            self.reset(r);
        } else {
            self.kf.update(&mut self.model, r.p, r.rpm, &r.y);
            self.rep.update(&r.y, r.ts1s);
        }
        let cfg = &self.cfg;
        self.zone_hist.push_back(r.p);
        if self.zone_hist.len() > 60 {
            self.zone_hist.pop_front();
        }
        if let Some(t) = r.cores_c.or(r.tcmx) {
            self.cpu_hist.push_back(t);
            while self.cpu_hist.len() > cfg.cpu_floor_avg_s.max(1) as usize {
                self.cpu_hist.pop_front();
            }
        }

        let fan_max = cfg.fan_ceiling(self.mode).min(self.hi);
        // eco's energy budget: refills below the sustained power, drains above it
        let (sus, room) = (cfg.eco_sustained_w, cfg.cpu_pl1_max_w - cfg.eco_sustained_w);
        self.energy_j = (self.energy_j + sus - pkg_from_pc0r(r.p.pc)).clamp(0.0, room * cfg.eco_burst_s);
        self.integrate_heat(r, fan_max);
        let cfg = &self.cfg;
        if self.t >= cfg.warmup_s && (self.replan || self.t.is_multiple_of(cfg.plan_every_s)) {
            self.replan = false;
            // power over the horizon: the higher of the last 10 s and 60 s means,
            // so a rising load counts at once and a falling one only slowly
            let mean = |n: usize| {
                let it = self.zone_hist.iter().skip(self.zone_hist.len().saturating_sub(n));
                let k = n.min(self.zone_hist.len()) as f64;
                let s = it.fold(Power::default(), |s, p| Power { pc: s.pc + p.pc, pg: s.pg + p.pg, pm: s.pm + p.pm });
                Power { pc: s.pc / k, pg: s.pg / k, pm: s.pm / k }
            };
            let (m10, m60) = (mean(10), mean(60));
            // memory power follows the window CPU + dGPU power picked
            let demand = if m10.pc + m10.pg > m60.pc + m60.pg { m10 } else { m60 };
            let (want, sus) = self.planner.power_cap(cfg, &mut self.model, &self.kf, &self.rep, demand, r.dgpu_active, fan_max, r.rpm);
            self.sustainable = sus;
            let caps = self.limit_rise(want);
            // the fans plan for the power the caps (and eco's energy budget) let through
            let eco_pc = self.energy_cap().map_or(f64::INFINITY, |e| self.igpu_fence(e).0);
            let eco_pg = self.eco_gpu_cap().unwrap_or(f64::INFINITY);
            let p = Power { pc: demand.pc.min(caps.cpu).min(eco_pc), pg: caps.gpu.map_or(demand.pg, |g| demand.pg.min(g).min(eco_pg)), pm: demand.pm };
            (self.plan_rpm, self.pred, self.plan_reason) =
                self.planner.plan(cfg, &mut self.model, &self.kf, &self.rep, p, self.cmd, r.rpm, fan_max);
            // a load pressing against a cap that is below its maximum
            let cpu_capped = caps.cpu < pc0r_from_pkg(cfg.cpu_pl1_max_w) - 0.01 && demand.pc >= caps.cpu - 1.0;
            let gpu_capped = caps.gpu.is_some_and(|g| g < cfg.dgpu_cap_max_w - 0.01 && demand.pg >= g - 1.0);
            if cpu_capped || gpu_capped {
                // fans before power: a cap that binds means the fans are at the mode's ceiling
                self.plan_rpm = self.plan_rpm.max(fan_max);
                self.plan_reason = "power-cap";
            }
            self.plan_caps = Some(caps);
        }
        self.t += 1;
        let energy = self.energy_cap();
        let eco_gpu = self.eco_gpu_cap();
        self.caps = self.plan_caps.map(|c| {
            let c = match energy {
                Some(e) if e < c.cpu => {
                    let (cpu, cores) = self.igpu_fence(e);
                    Caps { cpu: cpu.min(c.cpu), cores: Some(cores), ..c }
                }
                _ => c,
            };
            Caps { gpu: c.gpu.map(|g| eco_gpu.map_or(g, |e| g.min(e))), ..c }
        });
        let cpu_held = energy.is_some_and(|e| self.caps.is_some_and(|c| c.cores.is_some()) && r.p.pc >= e - 1.0);
        let gpu_held = eco_gpu.is_some_and(|e| self.caps.and_then(|c| c.gpu).is_some_and(|g| g >= e - 0.01) && r.p.pg >= e - 1.0);
        if cpu_held || gpu_held {
            self.plan_reason = "energy-cap";
        }
        let cfg = &self.cfg;

        let fan_max = cfg.fan_ceiling(self.mode).min(self.hi);
        let (mut target, mut reason) = (self.plan_rpm, self.plan_reason);
        let cpu_c = (!self.cpu_hist.is_empty()).then(|| self.cpu_hist.iter().sum::<f64>() / self.cpu_hist.len() as f64);
        let mut floors = vec![
            (ramp(cpu_c, cfg.cpu_floor_c, self.lo, self.hi), "cpu-temp"),
            (ramp(r.tgdd, cfg.gpu_floor_c, self.lo, self.hi), "gpu-temp"),
        ];
        if r.ts1s > SETPOINT + cfg.panic_over_setpoint_c {
            self.panic = true;
        } else if r.ts1s < cfg.ceiling_c {
            self.panic = false;
        }
        // safety: fans past the mode's ceiling; both caps to their floors only
        // if the SMC throttles or Ts1S is far over (a few tenths over leave
        // the SMC budget near 65 W, and cutting there made a sawtooth)
        if self.panic {
            // graded: from the mode's ceiling to full speed over the band
            // where cutting power would start
            let band = [SETPOINT + cfg.panic_over_setpoint_c, SETPOINT + cfg.cut_over_setpoint_c];
            floors.push((ramp(Some(r.ts1s), band, fan_max, self.hi).max(fan_max), "ts1s-over"));
        }
        if r.pz0t > 0.0 {
            floors.push((self.hi, "smc-throttle"));
        }
        if r.pz0t > 0.0 || r.ts1s > SETPOINT + cfg.cut_over_setpoint_c {
            let (cmin, _) = cfg.cpu_range();
            self.caps = Some(Caps { cpu: cmin, gpu: r.dgpu_active.then_some(cfg.dgpu_cap_min_w), cores: None });
        }
        for (f, why) in floors {
            if f > target {
                (target, reason) = (f, why);
            }
        }
        // up at once, down slowly (also after a switch to a quieter mode)
        if target < self.cmd {
            target = target.max(self.cmd - cfg.down_rpm_per_s);
        }
        self.cmd = target.clamp(self.lo, self.hi);
        self.reason = reason;
        self.cmd
    }
}
