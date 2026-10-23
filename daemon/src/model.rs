//! The thermal model (rc3 / rc3m), its Kalman filter, and the SMC's Ts1S filters.

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

/// Nodes: H (heatsink), C (chassis), S (slow mass), M (memory, rc3m only).
/// In rc3 the M node is inert: no capacity, no links, no sensor weight.
pub const N: usize = 4;

// Kalman noise, as in sim/forecast.py
const SENSOR_NOISE: f64 = 0.3;
const PROCESS_NOISE: [f64; N] = [0.05, 0.02, 0.005, 0.05];
const AMBIENT_NOISE: f64 = 0.003;

pub fn ema_alpha() -> [f64; 5] {
    TS1S_TAU_S.map(|tau| 1.0 - (-1.0 / tau).exp())
}

pub fn ts1s(ema: &[f64; 5]) -> f64 {
    TS1S_OFFSET + TS1S_GAIN.iter().zip(ema).map(|(g, e)| g * e).sum::<f64>()
}

/// Power inputs, W: CPU (PC0R), dGPU (PG0R), memory (PM0C).
#[derive(Clone, Copy, Default)]
pub struct Power {
    pub pc: f64,
    pub pg: f64,
    pub pm: f64,
}

/// 1 s update for one fan speed: x' = a x + gc*pc + gg*pg + gm*pm.
#[derive(Clone, Copy)]
pub struct Step {
    pub a: Mat<N, N>,
    pub gc: [f64; N],
    pub gg: [f64; N],
    pub gm: [f64; N],
}

pub struct Model {
    pub topology: String,
    /// true if memory power is an input (rc3m)
    pub uses_pm: bool,
    pub w: Mat<5, N>,
    pub d: [f64; 5],
    cinv: [f64; N],
    g: Mat<N, N>,
    ga0: [f64; N],
    ga1: [f64; N],
    bpc: [f64; N],
    bpg: [f64; N],
    bpm: [f64; N],
    fan_exp: f64,
    cache: HashMap<i64, Step>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let topology = v["topology"].as_str().unwrap_or("").to_owned();
        let uses_pm = match topology.as_str() {
            "rc3" => false,
            "rc3m" => true,
            t => return Err(format!("{}: unsupported topology {t:?} (rc3, rc3m)", path.display())),
        };
        let n = if uses_pm { 4 } else { 3 };
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
        let mut m = Self {
            topology,
            uses_pm,
            w,
            d,
            cinv: [1.0 / p("cap_H")?, 1.0 / p("cap_C")?, 1.0 / p("cap_S")?, 0.0],
            g,
            ga0: [p("a0_H")?, p("a0_C")?, p("a0_S")?, 0.0],
            ga1: [p("a1_H")?, p("a1_C")?, 0.0, 0.0],
            bpc: [1.0 - fr, fr, 0.0, 0.0],
            bpg: [1.0, 0.0, 0.0, 0.0],
            bpm: [0.0; N],
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
        // nodes | CPU, dGPU, memory input columns
        let mut aug = [[0.0; N + 3]; N + 3];
        for i in 0..N {
            for j in 0..N {
                aug[i][j] = self.cinv[i] * (self.g[i][j] - if i == j { loss[i] } else { 0.0 });
            }
            aug[i][N] = self.cinv[i] * self.bpc[i];
            aug[i][N + 1] = self.cinv[i] * self.bpg[i];
            aug[i][N + 2] = self.cinv[i] * self.bpm[i];
        }
        let e = linalg::expm(&aug);
        let s = Step {
            a: std::array::from_fn(|i| std::array::from_fn(|j| e[i][j])),
            gc: std::array::from_fn(|i| e[i][N]),
            gg: std::array::from_fn(|i| e[i][N + 1]),
            gm: std::array::from_fn(|i| e[i][N + 2]),
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
        // an inert node (rc3's M) has no equation: pin it at 0
        let a: Mat<N, N> = std::array::from_fn(|i| {
            std::array::from_fn(|j| match (self.cinv[i] == 0.0, i == j) {
                (true, true) => 1.0,
                (true, false) => 0.0,
                (false, true) => loss[i],
                (false, false) => -self.g[i][j],
            })
        });
        let b: [f64; N] = std::array::from_fn(|i| self.bpc[i] * p.pc + self.bpg[i] * p.pg + self.bpm[i] * p.pm);
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
    std::array::from_fn(|i| ax[i] + s.gc[i] * p.pc + s.gg[i] * p.pg + s.gm[i] * p.pm)
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
        for i in 0..N {
            if m.active(i) {
                q[i][i] = PROCESS_NOISE[i].powi(2);
                p0[i] = 25.0;
            }
        }
        q[N][N] = AMBIENT_NOISE.powi(2);
        p0[N] = 9.0;
        let r = linalg::scale(&linalg::eye::<5>(), SENSOR_NOISE.powi(2));
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
