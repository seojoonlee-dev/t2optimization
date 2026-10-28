//! Closed-loop simulation on synthetic workloads (no log): a thermal model
//! stands in for the machine and the controller drives it, as in --sim, but
//! the load is a scripted scenario: threads and the core power they ask for,
//! the dGPU, I/O (Wi-Fi + SSD path) and charge power.
//!
//! The plant's hottest core comes from its model's "die" block (none: 0 K
//! rise), plus --die-bias, so a controller can be tried against a machine
//! that runs hotter than its own model. RAPL is modelled as PL2 cutting at
//! once and PL1 holding the 20 s mean. Workload numbers are rough, from the
//! 2026-10-27/28 logs: uncapped, 1 thread drew ~14 W in the cores, 2 threads
//! ~27 W, 4+ threads the whole PL1 (~44 W); the uncore ~3 W; memory power
//! ~0.6 W + 6 % of the package (1.4 W idle, 3.3 W in a compile).

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::control::{pc0r_from_pkg, pkg_from_pc0r, rapl_with_die, Config, Controller, Mode, Reading};
use crate::model::{self, Model, Power, FAN_TAU_S, IO_IDLE_W, SETPOINT};
use crate::rapl_limits;
use crate::sim::Limiter;

const UNCORE_W: f64 = 3.0;
const CORE_W_PER_THREAD: f64 = 14.0;
const CORE_W_MAX: f64 = 44.0;
const IDLE_CORE_W: f64 = 1.3;
/// I/O power while downloading at ~55 MB/s (sim/data.py IO_FROM_WIFI)
const IO_DOWNLOAD_W: f64 = 9.4;
/// I/O power on top while the SSD writes at full speed (2026-10-28 bursts: PORR + PSLC + PH0R)
const IO_SSD_FULL_W: f64 = 5.2;

/// What the load asks for in one second.
#[derive(Clone, Copy, Default)]
struct Want {
    /// busy CPUs
    threads: f64,
    /// core power (RAPL PP0, W) the threads would draw uncapped
    core_w: f64,
    /// iGPU power, W
    gfx_w: f64,
    /// dGPU power, W (PG0R)
    pg: f64,
    io: f64,
    chg: f64,
}

fn threads(n: f64) -> Want {
    let core_w = if n <= 0.0 { IDLE_CORE_W } else { (CORE_W_PER_THREAD * n).min(CORE_W_MAX) };
    Want { threads: n.max(0.2), core_w, io: IO_IDLE_W, ..Default::default() }
}

/// Scenario by name: (description, want per second).
fn scenario(name: &str) -> Option<(&'static str, Vec<Want>)> {
    let idle = |s: usize| vec![threads(0.0); s];
    let mut v = idle(300);
    let desc = match name {
        "compile" => {
            // 16 threads, then the link steps: fewer threads, same budget
            v.extend(vec![threads(16.0); 480]);
            v.extend(vec![threads(4.0); 5]);
            v.extend(vec![threads(2.0); 10]);
            v.extend(vec![threads(1.0); 20]);
            v.extend(idle(300));
            "16 threads 8 min, then 4 -> 2 -> 1 thread (the failsafe trips of 10-27)"
        }
        "few-threads" => {
            v.extend(vec![threads(2.0); 600]);
            v.extend(vec![threads(1.0); 300]);
            v.extend(idle(300));
            "2 threads 10 min, then 1 thread 5 min (the overnight 100 °C trips)"
        }
        "steam" => {
            for t in 0..1500 {
                let mut w = threads(6.0);
                w.core_w = 20.0;
                // full-speed SSD writes 3 s of every 30
                w.io = IO_DOWNLOAD_W + if t % 30 < 3 { IO_SSD_FULL_W } else { 0.0 };
                v.push(w);
            }
            v.extend(idle(300));
            "25 min download: Wi-Fi ~55 MB/s, 6 threads at 20 W, SSD bursts"
        }
        "charge-compile" => {
            for t in 0..1800 {
                let mut w = if t < 900 { threads(16.0) } else { threads(0.0) };
                // charging slows as the battery fills
                w.chg = 40.0 - 30.0 * t as f64 / 1800.0;
                v.push(w);
            }
            "15 min compile while charging at 40 -> 10 W, then idle while charging"
        }
        "game" => {
            for _ in 0..1500 {
                let mut w = threads(4.0);
                w.core_w = 18.0;
                w.pg = 30.0;
                v.push(w);
            }
            v.extend(idle(300));
            "25 min dGPU game: dGPU 30 W, 4 threads at 18 W"
        }
        "bursty" => {
            for t in 0..900 {
                v.push(if t % 25 < 5 { threads(16.0) } else { threads(0.5) });
            }
            v.extend(idle(300));
            "15 min of 5 s 16-thread bursts every 25 s"
        }
        _ => return None,
    };
    Some((desc, v))
}

pub const SCENARIOS: [&str; 6] = ["compile", "few-threads", "steam", "charge-compile", "game", "bursty"];

pub struct SynthArgs<'a> {
    pub scenario: &'a str,
    pub plant: Option<&'a Path>,
    /// the plant's hottest core runs this much hotter than its die model, K
    pub die_bias: f64,
    pub mode: Option<Mode>,
    pub out: Option<&'a Path>,
}

pub fn run_synth(cfg: Config, cfg_dir: &Path, a: SynthArgs) -> Result<(), String> {
    let names: Vec<&str> = if a.scenario == "all" { SCENARIOS.to_vec() } else { vec![a.scenario] };
    for name in names {
        let (desc, want) = scenario(name).ok_or_else(|| format!("unknown scenario {name:?} ({}, all)", SCENARIOS.join(", ")))?;
        run_one(cfg.clone(), cfg_dir, name, desc, &want, &a)?;
    }
    Ok(())
}

fn run_one(cfg: Config, cfg_dir: &Path, name: &str, desc: &str, want: &[Want], a: &SynthArgs) -> Result<(), String> {
    let model = Model::load(&cfg_dir.join(&cfg.model))?;
    let ctl_topology = model.topology.clone();
    let mut plant = match a.plant {
        Some(p) => Model::load(p)?,
        None => Model::load(&cfg_dir.join(&cfg.model))?,
    };
    let (lo, hi) = (2080.0, 5708.0);
    let mode = a.mode.or(Mode::parse(&cfg.default_mode)).unwrap_or(Mode::Balanced);
    let pl2_extra = cfg.pl2_extra_w;
    let mut ctl = Controller::new(cfg, model, lo, hi);
    ctl.set_mode(mode);

    let mut out = match a.out {
        Some(p) => {
            let path = p.with_file_name(format!("{}-{name}.csv", p.file_stem().unwrap_or_default().to_string_lossy()));
            let mut w = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let _ = writeln!(w, "t,ts1s,pz0t,rpm,reason,pl1_w,pl2_w,die_cap_w,pkg_want_w,pkg_w,pg_w,core_max_c,busy,io_w,chg_w");
            Some(w)
        }
        None => None,
    };

    // start: idle, settled, room at the plant model's reference
    let p0 = Power { pc: pc0r_from_pkg(IDLE_CORE_W + UNCORE_W), pg: 0.0, pm: 0.8, io: IO_IDLE_W, chg: want[0].chg };
    let mut x = plant.steady(p0, lo);
    let mut ema = plant.sensors(&x, 0.0);
    let mut rpm = lo;
    let mut lim = Limiter { i: 0.0, j: 0.0, pz0f: p0.pc };
    let mut last = p0;
    let (mut last_core, mut last_pkg, mut last_busy) = (IDLE_CORE_W, IDLE_CORE_W + UNCORE_W, 0.2);
    let mut pl1_avg = last_pkg;
    let alpha = model::ema_alpha();
    let die = plant.die;

    let (mut ts, mut pz, mut rpms, mut core_max, mut wanted, mut got, mut pg_got) = (vec![], vec![], vec![], vec![], 0.0, 0.0, 0.0);
    for (t, w) in want.iter().enumerate() {
        // the machine now: sensors, SMC filters and limiter, hottest core
        let y = plant.sensors(&x, 0.0);
        for k in 0..5 {
            ema[k] += alpha[k] * (y[k] - ema[k]);
        }
        let ts1s = model::ts1s(&ema);
        let (_pz0e, pz0t) = lim.step(ts1s, last.pc + last.pg);
        let hottest = die.map(|d| d.base(x[0], x[1]) + d.rise(last_core, last_busy) + a.die_bias);
        // the CPU fan floor's input: the same rise spread over all cores
        let cores_c = die.map(|d| d.base(x[0], x[1]) + d.rise(last_core, 16.0));
        let r = Reading {
            y,
            ts1s,
            pz0t,
            p: last,
            rpm,
            tcmx: hottest,
            cores_c,
            tgdd: None,
            dgpu_active: w.pg > 1.0,
            core_max: hottest,
            pkg_w: Some(last_pkg),
            core_w: Some(last_core),
            busy: Some(last_busy),
        };
        let cmd = ctl.tick(&r);
        // what the caps let through this second
        let (pl1, pl2) = match ctl.caps {
            Some(c) => rapl_limits(&ctl.cfg, c.cpu),
            None => (ctl.cfg.warmup_pl1_w, ctl.cfg.warmup_pl1_w + pl2_extra),
        };
        let (pl1, pl2) = rapl_with_die(pl1, pl2, ctl.die_cap);
        let pkg_want = w.core_w + UNCORE_W + w.gfx_w;
        let mut pkg = pkg_want.min(pl2);
        if pl1_avg > pl1 {
            pkg = pkg.min(pl1);
        }
        pl1_avg += (pkg - pl1_avg) / ctl.cfg.pl1_window_s;
        let core = (pkg - UNCORE_W - w.gfx_w).max(0.0);
        let pg = ctl.caps.and_then(|c| c.gpu).map_or(w.pg, |g| w.pg.min(g));
        let p = Power { pc: pc0r_from_pkg(pkg), pg, pm: 0.6 + 0.06 * pkg, io: w.io, chg: w.chg };
        x = model::advance(&plant.step(rpm), &x, p);
        rpm += (cmd - rpm) * (1.0 - (-1.0 / FAN_TAU_S).exp());
        (last, last_core, last_pkg, last_busy) = (p, core, pkg, w.threads);

        ts.push(ts1s);
        pz.push(pz0t);
        rpms.push(rpm);
        if let Some(h) = hottest {
            core_max.push(h);
        }
        wanted += pkg_want;
        got += pkg;
        pg_got += pg;
        if let Some(o) = out.as_mut() {
            let _ = writeln!(
                o,
                "{t},{ts1s:.3},{pz0t:.1},{rpm:.0},{},{pl1:.1},{pl2:.1},{},{pkg_want:.1},{pkg:.1},{pg:.1},{},{:.1},{:.1},{:.1}",
                ctl.reason,
                ctl.die_cap.map_or(String::new(), |d| format!("{d:.1}")),
                hottest.map_or(String::new(), |h| format!("{h:.1}")),
                w.threads,
                w.io,
                w.chg
            );
        }
    }
    let n = ts.len() as f64;
    let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let over = |v: &[f64], lim: f64| v.iter().filter(|&&x| x >= lim).count();
    let pg_want: f64 = want.iter().map(|w| w.pg).sum();
    println!("{name} ({desc}): {n} s, mode {}, controller {ctl_topology}, plant {}{}", mode.name(), plant.topology, if a.die_bias != 0.0 { format!(", plant die +{:.1} K", a.die_bias) } else { String::new() });
    println!("  Ts1S max {:.2} °C, {} s over the set point {SETPOINT}; PZ0T max {:.1}", max(&ts), ts.iter().filter(|&&t| t > SETPOINT).count(), max(&pz));
    println!("  fans mean {:.0}, max {:.0} rpm; CPU package got {:.0} % of what it asked ({:.1} of {:.1} W mean){}", rpms.iter().sum::<f64>() / n, max(&rpms), 100.0 * got / wanted, got / n, wanted / n,
        if pg_want > 0.0 { format!("; dGPU got {:.0} %", 100.0 * pg_got / pg_want) } else { String::new() });
    if core_max.is_empty() {
        println!("  hottest core: not modelled (the plant has no die block)");
    } else {
        println!("  hottest core max {:.1} °C; {} s >= {:.0} °C, {} s >= 100 °C (the CPU's own throttle)", max(&core_max), over(&core_max, ctl.cfg.die_limit_c), ctl.cfg.die_limit_c, over(&core_max, 100.0));
    }
    let _ = pkg_from_pc0r;
    Ok(())
}
