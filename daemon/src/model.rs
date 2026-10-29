//! The thermal model (rc3 / rc3m / rc3mwb), its Kalman filter, the hottest-core
//! model, and the SMC's Ts1S filters.

use std::collections::HashMap;
use std::path::Path;

use crate::linalg::{self, Mat};

// --- the SMC's Ts1S estimate (sim/smc.py, reverse-engineered 2026-10-19) ---
pub const SENSORS: [&str; 5] = ["TW0P", "TM0P", "Ts0P", "TG0P", "Ts1P"];
pub const TS1S_OFFSET: f64 = -1.386;
const TS1S_GAIN: [f64; 5] = [1.0373, -0.6553, 0.3332, 0.1835, 0.1442];
const TS1S_TAU_S: [f64; 5] = [22.5, 25.5, 16.0, 69.3, 1915.0];
pub const SETPOINT: f64 = 44.989;
// fans reach a new target with a ~2 s lag (first order, measured on the
// 2026-10-19 overnight log and confirmed live 2026-10-20: tau 1.5 s, no dead time)
pub const FAN_TAU_S: f64 = 1.5;

/// Nodes: H (heatsink), C (chassis), S (slow mass), M (memory, rc3m and
/// rc3mwb), W (I/O: Wi-Fi + SSD path rails) and B (battery charging, rc3mwb).
/// A node a topology lacks is inert: no capacity, no links, no sensor weight.
pub const N: usize = 6;
/// I/O power (POLR + PORR + PSLC) when it isn't known: the idle level
/// (sim/data.py IO_IDLE_W, fit on the extra-logger logs 2026-10-28)
pub const IO_IDLE_W: f64 = 6.2;

/// Kalman filter settings: noise per second (K rms) and the uncertainty at a
/// cold start (K²). A model's "kalman" block overrides these defaults, the
/// hand-picked values of sim/forecast.py.
#[derive(Clone, Copy)]
pub struct KalmanTune {
    pub sensor: f64,
    /// H, C, S, M, W, B
    pub process: [f64; N],
    pub ambient: f64,
    pub p0: f64,
    pub p0_ambient: f64,
}

const NODE_NAMES: [&str; N] = ["H", "C", "S", "M", "W", "B"];

impl Default for KalmanTune {
    fn default() -> Self {
        Self { sensor: 0.3, process: [0.05, 0.02, 0.005, 0.05, 0.05, 0.05], ambient: 0.003, p0: 25.0, p0_ambient: 9.0 }
    }
}

pub fn ema_alpha() -> [f64; 5] {
    TS1S_TAU_S.map(|tau| 1.0 - (-1.0 / tau).exp())
}

pub fn ts1s(ema: &[f64; 5]) -> f64 {
    TS1S_OFFSET + TS1S_GAIN.iter().zip(ema).map(|(g, e)| g * e).sum::<f64>()
}

/// Power inputs, W: CPU (PC0R), dGPU (PG0R), memory (PM0C), I/O (POLR +
/// PORR + PSLC) and battery charging (PDTR - PSTR, >= 0).
#[derive(Clone, Copy, Default)]
pub struct Power {
    pub pc: f64,
    pub pg: f64,
    pub pm: f64,
    pub io: f64,
    pub chg: f64,
}

/// 1 s update for one fan speed: x' = a x + gc*pc + gg*pg + gm*pm + gi*io + gb*chg.
#[derive(Clone, Copy)]
pub struct Step {
    pub a: Mat<N, N>,
    pub gc: [f64; N],
    pub gg: [f64; N],
    pub gm: [f64; N],
    pub gi: [f64; N],
    pub gb: [f64; N],
}

/// Hottest CPU core, °C: c0 + k_h*H + k_c*C + k_core*core W
/// + k_per_busy*min(core W / busy, per_core_max_w), with H and C the model's
/// node temperatures plus the room offset, and a correction that follows the
/// measured hottest core with feedback_tau_s (fit on the 10 Hz core
/// temperatures, 2026-10-28: 2.8 K rms at >= 92 °C). The per-core bound (32 W,
/// the 99th percentile at ~1 busy CPU) keeps "this power on these few busy
/// CPUs" sane when the busy count is near idle.
#[derive(Clone, Copy, Debug)]
pub struct Die {
    pub c0: f64,
    pub k_h: f64,
    pub k_c: f64,
    pub k_core: f64,
    pub k_per_busy: f64,
    pub per_core_max_w: f64,
    pub feedback_tau_s: f64,
}

impl Die {
    /// Busy CPUs below this count as this (a core at its own limit is one busy CPU).
    pub const MIN_BUSY: f64 = 0.5;

    /// The part that doesn't depend on this second's core power.
    pub fn base(&self, h: f64, c: f64) -> f64 {
        self.c0 + self.k_h * h + self.k_c * c
    }

    /// Rise over the base from `core_w` W of core power on `busy` busy CPUs.
    pub fn rise(&self, core_w: f64, busy: f64) -> f64 {
        self.k_core * core_w + self.k_per_busy * (core_w / busy.max(Self::MIN_BUSY)).min(self.per_core_max_w)
    }

    /// Most core power whose rise stays within `room` K at `busy` busy CPUs
    /// (the rise is piecewise linear and increasing in power).
    pub fn max_core_w(&self, room: f64, busy: f64) -> f64 {
        let b = busy.max(Self::MIN_BUSY);
        let w = room / (self.k_core + self.k_per_busy / b);
        if w / b <= self.per_core_max_w {
            w
        } else {
            (room - self.k_per_busy * self.per_core_max_w) / self.k_core
        }
    }
}

pub struct Model {
    pub topology: String,
    /// true if memory power is an input (rc3m, rc3mwb)
    pub uses_pm: bool,
    pub die: Option<Die>,
    pub kalman: KalmanTune,
    pub w: Mat<5, N>,
    pub d: [f64; 5],
    cinv: [f64; N],
    g: Mat<N, N>,
    ga0: [f64; N],
    ga1: [f64; N],
    bpc: [f64; N],
    bpg: [f64; N],
    bpm: [f64; N],
    bpi: [f64; N],
    bpb: [f64; N],
    fan_exp: f64,
    cache: HashMap<i64, Step>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let topology = v["topology"].as_str().unwrap_or("").to_owned();
        let (uses_pm, uses_io) = match topology.as_str() {
            "rc3" => (false, false),
            "rc3m" => (true, false),
            "rc3mwb" => (true, true),
            t => return Err(format!("{}: unsupported topology {t:?} (rc3, rc3m, rc3mwb)", path.display())),
        };
        let n = if uses_io { 6 } else if uses_pm { 4 } else { 3 };
        let p = |k: &str| v["params"][k].as_f64().ok_or_else(|| format!("{}: missing param {k}", path.display()));
        let mut w = [[0.0; N]; 5];
        let mut d = [0.0; 5];
        for (i, s) in SENSORS.iter().enumerate() {
            for (j, wij) in w[i].iter_mut().enumerate().take(n) {
                *wij = v["out_w"][s][j].as_f64().ok_or_else(|| format!("{}: bad out_w.{s}", path.display()))?;
            }
            d[i] = v["out_d"][s].as_f64().ok_or_else(|| format!("{}: bad out_d.{s}", path.display()))?;
        }
        let mut g = [[0.0; N]; N];
        let mut link = |a: usize, b: usize, k: &str| -> Result<(), String> {
            g[a][b] = p(k)?;
            g[b][a] = g[a][b];
            Ok(())
        };
        link(0, 1, "g_H_C")?;
        link(1, 2, "g_C_S")?;
        let fr = p("pc_board_frac")?;
        let die = match &v["die"] {
            serde_json::Value::Null => None,
            dv => {
                let k = |n: &str| dv[n].as_f64().ok_or_else(|| format!("{}: missing die.{n}", path.display()));
                // an older die block without the bound: unbounded
                let per_core_max_w = dv["per_core_max_w"].as_f64().unwrap_or(f64::INFINITY);
                Some(Die { c0: k("c0")?, k_h: k("k_h")?, k_c: k("k_c")?, k_core: k("k_core")?, k_per_busy: k("k_per_busy")?, per_core_max_w, feedback_tau_s: k("feedback_tau_s")? })
            }
        };
        let mut kalman = KalmanTune::default();
        let kv = &v["kalman"];
        if !kv.is_null() {
            let k = |n: &str| kv[n].as_f64().ok_or_else(|| format!("{}: missing kalman.{n}", path.display()));
            kalman.sensor = k("sensor_noise")?;
            kalman.ambient = k("ambient_noise")?;
            kalman.p0 = k("p0")?;
            kalman.p0_ambient = k("p0_ambient")?;
            for (q, name) in kalman.process.iter_mut().zip(NODE_NAMES) {
                if let Some(x) = kv["process_noise"][name].as_f64() {
                    *q = x;
                }
            }
        }
        let mut m = Self {
            topology,
            uses_pm,
            die,
            kalman,
            w,
            d,
            cinv: [1.0 / p("cap_H")?, 1.0 / p("cap_C")?, 1.0 / p("cap_S")?, 0.0, 0.0, 0.0],
            g,
            ga0: [p("a0_H")?, p("a0_C")?, p("a0_S")?, 0.0, 0.0, 0.0],
            ga1: [p("a1_H")?, p("a1_C")?, 0.0, 0.0, 0.0, 0.0],
            bpc: [1.0 - fr, fr, 0.0, 0.0, 0.0, 0.0],
            bpg: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            bpm: [0.0; N],
            bpi: [0.0; N],
            bpb: [0.0; N],
            fan_exp: p("fan_exp")?,
            cache: HashMap::new(),
        };
        if uses_pm {
            m.g[3][1] = p("g_M_C")?;
            m.g[1][3] = m.g[3][1];
            m.cinv[3] = 1.0 / p("cap_M")?;
            m.ga0[3] = p("a0_M")?;
            m.ga1[3] = p("a1_M")?;
            m.bpm[3] = 1.0;
        }
        if uses_io {
            // W and B both hang off the chassis
            for (i, name) in [(4, "W"), (5, "B")] {
                m.g[i][1] = p(&format!("g_{name}_C"))?;
                m.g[1][i] = m.g[i][1];
                m.cinv[i] = 1.0 / p(&format!("cap_{name}"))?;
                m.ga0[i] = p(&format!("a0_{name}"))?;
                m.ga1[i] = p(&format!("a1_{name}"))?;
            }
            m.bpi[4] = 1.0;
            m.bpb[5] = 1.0;
        }
        Ok(m)
    }

    fn fan(&self, rpm: f64) -> f64 {
        (rpm.max(0.0) / 1000.0).powf(self.fan_exp)
    }

    /// Heat loss to the room plus to neighbours, per node, at this fan speed.
    fn loss(&self, rpm: f64) -> [f64; N] {
        let f = self.fan(rpm);
        std::array::from_fn(|i| self.ga0[i] + self.ga1[i] * f + self.g[i].iter().sum::<f64>())
    }

    /// Exact 1 s update for fan speed `rpm` (rounded to 10 rpm, cached).
    pub fn step(&mut self, rpm: f64) -> Step {
        let key = (rpm / 10.0).round_ties_even() as i64;
        if let Some(s) = self.cache.get(&key) {
            return *s;
        }
        let loss = self.loss(key as f64 * 10.0);
        // nodes | CPU, dGPU, memory, I/O, charge input columns
        let mut aug = [[0.0; N + 5]; N + 5];
        for i in 0..N {
            for j in 0..N {
                aug[i][j] = self.cinv[i] * (self.g[i][j] - if i == j { loss[i] } else { 0.0 });
            }
            aug[i][N] = self.cinv[i] * self.bpc[i];
            aug[i][N + 1] = self.cinv[i] * self.bpg[i];
            aug[i][N + 2] = self.cinv[i] * self.bpm[i];
            aug[i][N + 3] = self.cinv[i] * self.bpi[i];
            aug[i][N + 4] = self.cinv[i] * self.bpb[i];
        }
        let e = linalg::expm(&aug);
        let s = Step {
            a: std::array::from_fn(|i| std::array::from_fn(|j| e[i][j])),
            gc: std::array::from_fn(|i| e[i][N]),
            gg: std::array::from_fn(|i| e[i][N + 1]),
            gm: std::array::from_fn(|i| e[i][N + 2]),
            gi: std::array::from_fn(|i| e[i][N + 3]),
            gb: std::array::from_fn(|i| e[i][N + 4]),
        };
        if self.cache.len() > 4096 {
            self.cache.clear();
        }
        self.cache.insert(key, s);
        s
    }

    /// Node temperatures for constant inputs held forever.
    pub fn steady(&self, p: Power, rpm: f64) -> [f64; N] {
        let loss = self.loss(rpm);
        // an inert node (rc3's M, W, B) has no equation: pin it at 0
        let a: Mat<N, N> = std::array::from_fn(|i| {
            std::array::from_fn(|j| match (self.cinv[i] == 0.0, i == j) {
                (true, true) => 1.0,
                (true, false) => 0.0,
                (false, true) => loss[i],
                (false, false) => -self.g[i][j],
            })
        });
        let b: [f64; N] = std::array::from_fn(|i| self.bpc[i] * p.pc + self.bpg[i] * p.pg + self.bpm[i] * p.pm + self.bpi[i] * p.io + self.bpb[i] * p.chg);
        linalg::mul_vec(&linalg::inv(&a).expect("steady-state matrix is singular"), &b)
    }

    pub fn sensors(&self, x: &[f64; N], ambient: f64) -> [f64; 5] {
        let y = linalg::mul_vec(&self.w, x);
        std::array::from_fn(|i| y[i] + self.d[i] + ambient)
    }

    fn active(&self, i: usize) -> bool {
        self.cinv[i] != 0.0
    }
}

pub fn advance(s: &Step, x: &[f64; N], p: Power) -> [f64; N] {
    let ax = linalg::mul_vec(&s.a, x);
    std::array::from_fn(|i| ax[i] + s.gc[i] * p.pc + s.gg[i] * p.pg + s.gm[i] * p.pm + s.gi[i] * p.io + s.gb[i] * p.chg)
}

/// State = node temperatures + room offset (shared by all sensors).
pub struct Kalman {
    pub x: [f64; N + 1],
    p: Mat<{ N + 1 }, { N + 1 }>,
    h: Mat<5, { N + 1 }>,
    q: Mat<{ N + 1 }, { N + 1 }>,
    r: Mat<5, 5>,
    p0: [f64; N + 1],
}

impl Kalman {
    pub fn new(m: &Model) -> Self {
        let h = std::array::from_fn(|i| std::array::from_fn(|j| if j < N { m.w[i][j] } else { 1.0 }));
        let mut q = [[0.0; N + 1]; N + 1];
        let mut p0 = [0.0; N + 1];
        let t = m.kalman;
        for i in 0..N {
            if m.active(i) {
                q[i][i] = t.process[i].powi(2);
                p0[i] = t.p0;
            }
        }
        q[N][N] = t.ambient.powi(2);
        p0[N] = t.p0_ambient;
        let r = linalg::scale(&linalg::eye::<5>(), t.sensor.powi(2));
        Self { x: [0.0; N + 1], p: linalg::eye(), h, q, r, p0 }
    }

    /// Cold start, as scored in sim/forecast.py: steady state, room unknown.
    pub fn reset(&mut self, m: &Model, p: Power, rpm: f64) {
        let s = m.steady(p, rpm);
        self.x = std::array::from_fn(|i| if i < N { s[i] } else { 0.0 });
        self.p = [[0.0; N + 1]; N + 1];
        for i in 0..=N {
            self.p[i][i] = self.p0[i];
        }
    }

    pub fn update(&mut self, m: &mut Model, p: Power, rpm: f64, y: &[f64; 5]) {
        let s = m.step(rpm);
        let mut af = linalg::eye::<{ N + 1 }>();
        for i in 0..N {
            af[i][..N].copy_from_slice(&s.a[i]);
        }
        let nodes = advance(&s, &self.nodes(), p);
        self.x[..N].copy_from_slice(&nodes);
        self.p = linalg::add(&linalg::mul(&linalg::mul(&af, &self.p), &linalg::transpose(&af)), &self.q);
        let ht = linalg::transpose(&self.h);
        let sm = linalg::add(&linalg::mul(&linalg::mul(&self.h, &self.p), &ht), &self.r);
        let Some(sinv) = linalg::inv(&sm) else { return };
        let k = linalg::mul(&linalg::mul(&self.p, &ht), &sinv);
        let hx = linalg::mul_vec(&self.h, &self.x);
        let innov: [f64; 5] = std::array::from_fn(|i| y[i] - m.d[i] - hx[i]);
        let dx = linalg::mul_vec(&k, &innov);
        for (x, d) in self.x.iter_mut().zip(dx) {
            *x += d;
        }
        self.p = linalg::mul(&linalg::sub(&linalg::eye::<{ N + 1 }>(), &linalg::mul(&k, &self.h)), &self.p);
    }

    pub fn nodes(&self) -> [f64; N] {
        std::array::from_fn(|i| self.x[i])
    }

    pub fn ambient(&self) -> f64 {
        self.x[N]
    }
}

/// The SMC's internal Ts1S filter states, replayed from the measured sensors.
pub struct Replica {
    pub ema: [f64; 5],
    pub bias: f64,
    alpha: [f64; 5],
}

impl Replica {
    pub fn new() -> Self {
        Self { ema: [0.0; 5], bias: 0.0, alpha: ema_alpha() }
    }

    pub fn reset(&mut self, y: &[f64; 5]) {
        self.ema = *y;
        self.bias = 0.0;
    }

    pub fn update(&mut self, y: &[f64; 5], ts1s_meas: f64) {
        for i in 0..5 {
            self.ema[i] += self.alpha[i] * (y[i] - self.ema[i]);
        }
        // the replica is exact up to SMC rounding; follow the measured Ts1S slowly
        self.bias += 0.05 * ((ts1s_meas - ts1s(&self.ema)) - self.bias);
    }
}
