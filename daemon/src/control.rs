//! Planner and controller: which fan speed and power caps to command.
//!
//! The power mode sets only a fan ceiling. Under it, the controller picks the
//! slowest fan speed that keeps predicted Ts1S under the ceiling, and caps
//! CPU + dGPU power at the most they can draw while the mode's fan ceiling
//! still holds Ts1S there. Light loads never reach the cap; heavy ones get
//! the fans first, then a lower cap instead of louder fans.

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
    /// Ts1S the plan must stay under; the SMC budget's set point is 44.989
    pub ceiling_c: f64,
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
    pub gpu_floor_c: [f64; 2],
    /// measured Ts1S this far above the set point -> full speed until back under the ceiling
    pub panic_over_setpoint_c: f64,
    /// mode at start when no mode file says otherwise: eco, balanced or perf
    pub default_mode: String,
    pub fan_ceiling_rpm: FanCeilings,
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
        Self { eco: 2500, balanced: 3300, perf: 99999 }
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "../models/rc3-v1.json".into(),
            log_dir: "../data".into(),
            log_csv: true,
            ceiling_c: 44.6,
            down_hysteresis_c: 0.15,
            horizon_s: 90,
            plan_every_s: 5,
            warmup_s: 30,
            rpm_step: 100,
            down_rpm_per_s: 30.0,
            min_rpm: 0,
            max_rpm: 99999,
            cpu_floor_c: [88.0, 98.0],
            gpu_floor_c: [85.0, 95.0],
            panic_over_setpoint_c: 0.4,
            default_mode: "balanced".into(),
            fan_ceiling_rpm: FanCeilings::default(),
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
        return Caps { cpu: total.clamp(cmin, cmax), gpu: None };
    }
    let (dc, dg) = (demand.pc.max(0.0), demand.pg.max(0.0));
    let (c, g) = if total >= dc + dg {
        let h = (total - dc - dg) / 2.0;
        (dc + h, dg + h)
    } else {
        let k = total / (dc + dg).max(1e-9);
        (dc * k, dg * k)
    };
    Caps { cpu: c.clamp(cmin, cmax), gpu: Some(g.clamp(cfg.dgpu_cap_min_w, cfg.dgpu_cap_max_w)) }
}

struct Planner {
    lo: f64,
    candidates: Vec<f64>,
    alpha: [f64; 5],
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
        Self { lo, candidates, alpha: model::ema_alpha() }
    }

    /// Largest CPU + dGPU budget (each drawing its whole share) that keeps
    /// predicted Ts1S under the ceiling with the fans at `fan`.
    #[allow(clippy::too_many_arguments)]
    fn power_cap(&self, cfg: &Config, m: &mut Model, kf: &Kalman, rep: &Replica, demand: Power, dgpu_active: bool, fan: f64, rpm_now: f64) -> Caps {
        let (cmin, cmax) = cfg.cpu_range();
        let (lo, hi) = if dgpu_active { (cmin + cfg.dgpu_cap_min_w, cmax + cfg.dgpu_cap_max_w) } else { (cmin, cmax) };
        let ok = |total: f64, m: &mut Model| {
            let c = split(cfg, total, demand, dgpu_active);
            let p = Power { pc: c.cpu, pg: c.gpu.unwrap_or(demand.pg), pm: demand.pm };
            self.predict(m, kf, rep, p, fan, cfg.horizon_s, rpm_now) <= cfg.ceiling_c
        };
        if ok(hi, m) {
            return split(cfg, hi, demand, dgpu_active);
        }
        if !ok(lo, m) {
            return split(cfg, lo, demand, dgpu_active);
        }
        // invariant: lo passes, hi fails; 0.25 W is finer than RAPL needs
        let (mut lo, mut hi) = (lo, hi);
        while hi - lo > 0.25 {
            let mid = (lo + hi) / 2.0;
            if ok(mid, m) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        split(cfg, lo, demand, dgpu_active)
    }

    /// Max predicted Ts1S over the horizon; the fans move from rpm_now
    /// toward the target with FAN_TAU_S lag.
    #[allow(clippy::too_many_arguments)]
    fn predict(&self, m: &mut Model, kf: &Kalman, rep: &Replica, p: Power, rpm: f64, horizon: u32, rpm_now: f64) -> f64 {
        let mut x = kf.nodes();
        let mut ema = rep.ema;
        let amb = kf.ambient();
        let mut ts_max = f64::NEG_INFINITY;
        for i in 0..horizon {
            // mid-step fan speed; after ~10 s it is the target
            let fan = if i < 10 { rpm + (rpm_now - rpm) * (-(i as f64 + 0.5) / FAN_TAU_S).exp() } else { rpm };
            x = model::advance(&m.step(fan), &x, p);
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
    started: bool,
    t: u32,
    pub mode: Mode,
    replan: bool,
    /// caps to apply; None until the first plan (the hardware keeps what it has)
    pub caps: Option<Caps>,
}

impl Controller {
    pub fn new(cfg: Config, model: Model, rpm_lo: f64, rpm_hi: f64) -> Self {
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
            started: false,
            t: 0,
            mode: Mode::Balanced,
            replan: false,
            caps: None,
        }
    }

    /// Switch power mode; the next tick plans for it at once.
    pub fn set_mode(&mut self, mode: Mode) {
        if mode != self.mode {
            self.mode = mode;
            self.replan = true;
        }
    }

    /// Caps drop at once and rise by at most cap_raise_w per plan.
    fn limit_rise(&self, want: Caps) -> Caps {
        let Some(prev) = self.caps else { return want };
        let rise = |p: f64, w: f64| if w > p { w.min(p + self.cfg.cap_raise_w) } else { w };
        Caps {
            cpu: rise(prev.cpu, want.cpu),
            gpu: match (prev.gpu, want.gpu) {
                (Some(p), Some(w)) => Some(rise(p, w)),
                (_, w) => w,
            },
        }
    }

    /// Start over: new model state, SMC filters re-read, warm-up again.
    pub fn reset(&mut self, r: &Reading) {
        self.kf.reset(&self.model, r.p, r.rpm);
        self.rep.reset(&r.y);
        self.zone_hist.clear();
        self.t = 0;
        self.cmd = r.rpm.clamp(self.lo, self.hi);
        self.plan_rpm = self.cmd;
        self.plan_reason = "warmup";
        self.pred = f64::NAN;
        self.started = true;
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

        let fan_max = cfg.fan_ceiling(self.mode).min(self.hi);
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
            let want = self.planner.power_cap(cfg, &mut self.model, &self.kf, &self.rep, demand, r.dgpu_active, fan_max, r.rpm);
            let caps = self.limit_rise(want);
            // the fans plan for the power the caps let through
            let p = Power { pc: demand.pc.min(caps.cpu), pg: caps.gpu.map_or(demand.pg, |g| demand.pg.min(g)), pm: demand.pm };
            (self.plan_rpm, self.pred, self.plan_reason) =
                self.planner.plan(cfg, &mut self.model, &self.kf, &self.rep, p, self.cmd, r.rpm, fan_max);
            // a load pressing against a cap that is below its maximum
            let cpu_capped = caps.cpu < pc0r_from_pkg(cfg.cpu_pl1_max_w) - 0.01 && demand.pc >= caps.cpu - 1.0;
            let gpu_capped = caps.gpu.is_some_and(|g| g < cfg.dgpu_cap_max_w - 0.01 && demand.pg >= g - 1.0);
            if cpu_capped || gpu_capped {
                self.plan_reason = "power-cap";
            }
            self.caps = Some(caps);
        }
        self.t += 1;

        let (mut target, mut reason) = (self.plan_rpm, self.plan_reason);
        let mut floors = vec![
            (ramp(r.tcmx, cfg.cpu_floor_c, self.lo, self.hi), "cpu-temp"),
            (ramp(r.tgdd, cfg.gpu_floor_c, self.lo, self.hi), "gpu-temp"),
        ];
        if r.ts1s > SETPOINT + cfg.panic_over_setpoint_c {
            self.panic = true;
        } else if r.ts1s < cfg.ceiling_c {
            self.panic = false;
        }
        // safety: past the mode's ceiling, and both caps to their floors
        if self.panic {
            floors.push((self.hi, "ts1s-over"));
        }
        if r.pz0t > 0.0 {
            floors.push((self.hi, "smc-throttle"));
        }
        if self.panic || r.pz0t > 0.0 {
            let (cmin, _) = cfg.cpu_range();
            self.caps = Some(Caps { cpu: cmin, gpu: r.dgpu_active.then_some(cfg.dgpu_cap_min_w) });
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
