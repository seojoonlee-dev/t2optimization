//! Closed-loop simulation: the controller drives a thermal model that stands
//! in for the machine, fed by the power a logged session asked for.
//!
//! Before `from`, the controller and the plant's Kalman filter follow the log
//! (as in --replay), so the closed loop starts from the real thermal state.
//! From then on the plant model makes the sensor readings, the SMC's Ts1S
//! filters and limiter run on them, and the CPU and dGPU draw what the log
//! drew, cut to the caps the controller sets. Where the log itself was held
//! by a cap (power within 1.5 W of it), the load is taken to want everything
//! (PL1 max, dGPU cap max): a capped log can't say how much more it wanted.
//!
//! Die temperatures aren't modelled: the floors see the logged CPU die
//! temperature (TCMX, or the core mean where logged), moved by what the sim
//! changed: the model's heatsink (H) difference under --inject, plus DIE_K_PER_W
//! per W of CPU power. The dGPU die (TGDD) is replayed as logged.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::control::{pc0r_from_pkg, pkg_from_pc0r, Config, Controller, Mode, Reading};
use crate::model::{self, Kalman, Model, Power, FAN_TAU_S, SETPOINT};
use crate::{rapl_limits, reading, KEYS};

/// Die above heatsink per W of CPU power over 2-5 s: 0.25-0.45 K/W in the
/// 10-19 stress test and two 10-20 fand logs; the top end, so capped runs
/// read on the warm side.
const DIE_K_PER_W: f64 = 0.4;

/// The SMC's skin limiter (sim/smc.py, reverse-engineered 2026-10-19).
pub struct Limiter {
    pub i: f64,
    pub j: f64,
    pub pz0f: f64,
}

impl Limiter {
    /// One second at skin estimate `ts1s` and zone power `zone` (PC0R + PG0R); returns (PZ0E, PZ0T).
    pub fn step(&mut self, ts1s: f64, zone: f64) -> (f64, f64) {
        let e = ts1s - 44.985;
        self.i = (self.i + 0.05925 * e).max(0.0);
        let pz0e = (70.0 - 12.19 * e - self.i).clamp(0.0, 70.0);
        self.pz0f += (zone - self.pz0f) * (1.0 - (-1.0f64 / 30.0).exp());
        let d = self.pz0f - pz0e;
        self.j = (self.j + 0.148 * d).clamp(0.0, 100.0);
        (pz0e, (4.07 * d + self.j).clamp(0.0, 100.0))
    }
}

/// The simulated machine.
struct Plant {
    x: [f64; model::N],
    amb: f64,
    /// the SMC's Ts1S filters
    ema: [f64; 5],
    rpm: f64,
    lim: Limiter,
    /// power drawn during the last second
    last_p: Power,
}

/// One closed-loop second, for the report.
struct Sec {
    ts1s: f64,
    pz0e: f64,
    pz0t: f64,
    rpm: f64,
    pl1: f64,
    /// package W the load asked for / got
    want: f64,
    drawn: f64,
}

pub struct SimArgs<'a> {
    pub plant: Option<&'a Path>,
    /// scales the heat the plant gets (1.1: the machine runs 10 % hotter than the model)
    pub plant_heat: f64,
    /// plant = the logged sensors + the model's response to the power and fan
    /// differences, so heat the model doesn't know about (Wi-Fi) stays in
    pub inject: bool,
    pub from: Option<&'a str>,
    pub to: Option<&'a str>,
    pub mode: Option<Mode>,
    pub out: Option<&'a Path>,
}

pub fn run_sim(cfg: Config, cfg_dir: &Path, path: &Path, a: SimArgs) -> Result<(), String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = BufReader::new(f).lines().map_while(Result::ok);
    let head: Vec<String> = lines.next().ok_or("empty CSV")?.split(',').map(str::to_owned).collect();
    let col = |n: &str| head.iter().position(|h| h == n);
    let cols: Vec<Option<usize>> = KEYS.iter().map(|k| col(k)).collect();
    let ts_col = col("timestamp").ok_or("no timestamp column")?;
    let rows: Vec<Vec<String>> = lines.map(|l| l.split(',').map(str::to_owned).collect()).collect();
    let num = |row: &Vec<String>, n: &str| col(n).and_then(|i| row.get(i)?.parse::<f64>().ok());

    let model = Model::load(&cfg_dir.join(&cfg.model))?;
    let uses_pm = model.uses_pm;
    let mut plant = match a.plant {
        Some(p) => Model::load(p)?,
        None => Model::load(&cfg_dir.join(&cfg.model))?,
    };
    let mut pkf = Kalman::new(&plant);
    let (lo, hi) = crate::hw::Fans::open().map(|f| (f.lo, f.hi)).unwrap_or((2080.0, 5708.0));
    let start_mode = a.mode.or(Mode::parse(&cfg.default_mode)).unwrap_or(Mode::Balanced);
    let pl1_max = cfg.cpu_pl1_max_w;
    let gpu_max = cfg.dgpu_cap_max_w;
    let mut ctl = Controller::new(cfg, model, lo, hi);
    ctl.set_mode(start_mode);

    let mut out = match a.out {
        Some(p) => {
            let mut w = File::create(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let _ = writeln!(w, "timestamp,ts1s,pz0e,pz0t,cmd_rpm,rpm,reason,pred,pl1_w,dgpu_cap_w,pkg_want_w,pkg_w,pg_want_w,pg_w,sus_w,amb,heat,{}", model::SENSORS.map(|k| format!("sim_{k}")).join(","));
            Some(w)
        }
        None => None,
    };

    // closed-loop state, set when the window starts
    let mut sim: Option<Plant> = None;
    // --inject: the plant run with the logged power and fans, to take off the logged sensors
    let mut shadow = [0.0; model::N];
    let alpha = model::ema_alpha();
    let (mut started, mut pkf_started) = (false, false);
    let mut log: Vec<Sec> = Vec::new();
    let mut first_ts = String::new();
    for row in &rows {
        let ts = row.get(ts_col).map(String::as_str).unwrap_or("");
        if a.to.is_some_and(|to| ts > to) {
            break;
        }
        let vals: Option<Vec<Option<f64>>> = cols
            .iter()
            .map(|c| match c.and_then(|i| row.get(i)).map(String::as_str) {
                None | Some("") => Some(None),
                Some(s) => s.parse().ok().map(Some),
            })
            .collect();
        let Some(mut r) = vals.as_deref().and_then(|v| reading(v, uses_pm).ok()) else { continue };
        r.dgpu_active = r.p.pg > 1.0;
        let in_window = a.from.is_none_or(|from| ts >= from);
        if !in_window || !started {
            // follow the log: plant filter and controller both see the real machine
            if pkf_started {
                pkf.update(&mut plant, r.p, r.rpm, &r.y);
            } else {
                pkf.reset(&plant, r.p, r.rpm);
                pkf_started = true;
            }
            if !in_window {
                ctl.tick(&r);
                continue;
            }
        }
        // what the logged load asked for: everything, where a cap held it
        let capped = |drawn: Option<f64>, cap: Option<f64>, max: f64| matches!((drawn, cap), (Some(d), Some(c)) if c < max - 0.5 && d >= c - 1.5);
        let pkg_logged = num(row, "rapl_pkg_w").unwrap_or(pkg_from_pc0r(r.p.pc));
        // eco's iGPU fence: the cores at their own limit held the package too
        let cores_capped = matches!((num(row, "rapl_core_w"), num(row, "cpu_cores_w")), (Some(d), Some(c)) if d >= c - 1.0);
        let pc_want = if cores_capped || capped(Some(pkg_logged), num(row, "cpu_pl1_w"), pl1_max) { pc0r_from_pkg(pl1_max) } else { r.p.pc };
        let pg_want = if capped(Some(r.p.pg), num(row, "dgpu_cap_w"), gpu_max) { gpu_max } else { r.p.pg };
        let want = Power { pc: pc_want, pg: pg_want, ..r.p };

        if !started {
            started = true;
            first_ts = ts.to_owned();
            let x = pkf.nodes();
            shadow = x;
            let lim = Limiter { i: 0.0, j: 0.0, pz0f: r.p.pc + r.p.pg };
            sim = Some(Plant { x, amb: pkf.ambient(), ema: ctl.rep.ema, rpm: r.rpm, lim, last_p: r.p });
        }
        let Plant { x, amb, ema, rpm, lim, last_p } = sim.as_mut().unwrap();
        let y = if a.inject {
            // the machine as logged, plus the model's response to what the controller changed
            let (ys, yx) = (plant.sensors(&shadow, 0.0), plant.sensors(x, 0.0));
            shadow = model::advance(&plant.step(r.rpm), &shadow, r.p);
            std::array::from_fn(|k| r.y[k] + yx[k] - ys[k])
        } else {
            plant.sensors(x, *amb)
        };
        for k in 0..5 {
            ema[k] += alpha[k] * (y[k] - ema[k]);
        }
        let ts1s = model::ts1s(ema);
        let (pz0e, pz0t) = lim.step(ts1s, last_p.pc + last_p.pg);
        let dh = if a.inject { x[0] - shadow[0] } else { 0.0 };
        let die = |t: Option<f64>| t.map(|t| t + dh + DIE_K_PER_W * (last_p.pc - r.p.pc));
        let sr = Reading { y, ts1s, pz0t, p: *last_p, rpm: *rpm, tcmx: die(r.tcmx), cores_c: die(num(row, "cpu_core_c")), tgdd: r.tgdd, dgpu_active: want.pg > 1.0, core_max: None, pkg_w: None, core_w: None, busy: None };
        let cmd = ctl.tick(&sr);
        let pl1 = ctl.caps.map_or(ctl.cfg.warmup_pl1_w, |c| rapl_limits(&ctl.cfg, c.cpu).0);
        let gcap = ctl.caps.and_then(|c| c.gpu);
        // under the fence the cores get their limit and the rest of the
        // package what it drew in the log (the iGPU as logged, else idle)
        let fence = ctl.caps.and_then(|c| c.cores).map_or(f64::INFINITY, |w| w + ctl.cfg.eco_uncore_w + num(row, "rapl_gfx_w").unwrap_or(0.0));
        let p = Power { pc: want.pc.min(pc0r_from_pkg(pl1.min(fence))), pg: gcap.map_or(want.pg, |g| want.pg.min(g)), ..want };
        let k = a.plant_heat;
        *x = model::advance(&plant.step(*rpm), x, Power { pc: k * p.pc, pg: k * p.pg, pm: k * p.pm, ..p });
        *rpm += (cmd - *rpm) * (1.0 - (-1.0 / FAN_TAU_S).exp());
        *last_p = p;
        log.push(Sec { ts1s, pz0e, pz0t, rpm: *rpm, pl1, want: pkg_from_pc0r(want.pc), drawn: pkg_from_pc0r(p.pc) });
        if let Some(w) = out.as_mut() {
            let o = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.1}"));
            let _ = writeln!(
                w,
                "{ts},{ts1s:.3},{pz0e:.2},{pz0t:.2},{cmd:.0},{rpm:.0},{},{:.3},{pl1:.2},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{}",
                ctl.reason,
                ctl.pred,
                o(gcap),
                pkg_from_pc0r(want.pc),
                pkg_from_pc0r(p.pc),
                want.pg,
                p.pg,
                ctl.sustainable,
                ctl.kf.ambient(),
                ctl.heat(),
                y.map(|v| format!("{v:.2}")).join(",")
            );
        }
    }
    if log.is_empty() {
        return Err("no rows in the window".into());
    }
    report(&log, path, &first_ts, start_mode, a.plant_heat, a.inject);
    Ok(())
}

fn report(log: &[Sec], path: &Path, from: &str, mode: Mode, heat: f64, inject: bool) {
    let n = log.len() as f64;
    let col = |f: fn(&Sec) -> f64| log.iter().map(f).collect::<Vec<_>>();
    let (ts, pz0e, pz0t, rpm, pl1, want, drawn) =
        (col(|r| r.ts1s), col(|r| r.pz0e), col(|r| r.pz0t), col(|r| r.rpm), col(|r| r.pl1), col(|r| r.want), col(|r| r.drawn));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let q = |v: &[f64], p: f64| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[((s.len() - 1) as f64 * p).round() as usize]
    };
    // what the load got, averaged the way it feels: 30 s means
    let m30: Vec<f64> = drawn.windows(30.min(drawn.len())).map(mean).collect();
    // cap cuts of 5 W or more within 10 s
    let mut cuts = 0;
    let mut i = 10;
    while i < pl1.len() {
        if pl1[i - 10] - pl1[i] >= 5.0 {
            cuts += 1;
            i += 10;
        } else {
            i += 1;
        }
    }
    let swing: f64 = pl1.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (n / 60.0);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    println!("{name} from {from}, {n} s closed loop, mode {}, plant heat x{heat}{}", mode.name(), if inject { ", logged residual injected" } else { "" });
    println!(
        "  Ts1S max {:.2} °C, {} s over the set point {SETPOINT}; PZ0E min {:.1} W, PZ0T max {:.1}",
        max(&ts),
        ts.iter().filter(|&&t| t > SETPOINT).count(),
        min(&pz0e),
        max(&pz0t)
    );
    println!("  fans mean {:.0}, p90 {:.0}, max {:.0} rpm", mean(&rpm), q(&rpm, 0.9), max(&rpm));
    println!(
        "  PL1 mean {:.1}, min {:.1} W; {} s at <= 15 W; {cuts} cuts of >= 5 W in 10 s; swing {swing:.1} W/min",
        mean(&pl1),
        min(&pl1),
        pl1.iter().filter(|&&p| p <= 15.0).count()
    );
    println!(
        "  CPU package: wanted {:.1} W, got {:.1} W ({:.0} %); 30 s means p10 {:.1} / p50 {:.1} W",
        mean(&want),
        mean(&drawn),
        100.0 * drawn.iter().sum::<f64>() / want.iter().sum::<f64>(),
        q(&m30, 0.1),
        q(&m30, 0.5)
    );
}
