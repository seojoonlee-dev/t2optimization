//! Model-predictive fan daemon for the T2 MacBook (PoC, this machine only).
//!
//! Every second: read the SMC, correct the rc3 thermal model's state from the
//! five Ts1S sensors (Kalman filter, room temperature as an extra state).
//! Every `plan_every_s`: run the model and the SMC's Ts1S filters `horizon_s`
//! ahead with power held at its recent level, and bisect for the slowest fan
//! speed whose predicted Ts1S stays <= `ceiling_c` (just under the SMC's 45 °C
//! set point; below it the SMC budget stays at its 70 W maximum, so PZ0T stays
//! 0). Even full speed fails -> full speed. Fans go up at once, down slowly.
//!
//! Floors on top of the plan: CPU (TCMX) or dGPU (TGDD) die temperature near
//! their limits, a measured Ts1S above the set point (the model was wrong), and
//! any SMC throttle (PZ0T > 0) all raise the fans regardless of the plan.
//!
//! This is the only process that may read SMC keys while it runs, so
//! tools/logger.py must not run at the same time; fand logs the same kind of
//! 1 Hz CSV itself.
//!
//!     sudo daemon/fand-start.sh [--shadow]              (see that script)
//!     t2opt-fand --replay data/x.csv [--trace]          offline, no root, no fans
//!     t2opt-fand --sim data/x.csv [--from T] [--to T]   closed loop against the model (src/sim.rs)
//!     t2opt-fand --release                              fans back to the SMC

mod control;
mod hw;
mod linalg;
mod model;
mod sim;

use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use control::{pkg_from_pc0r, Config, Controller, Mode, Reading};
use model::{Model, Power, SENSORS, SETPOINT};

const STATUS: &str = "/run/t2opt-fand.json";
const LOCK: &str = "/run/t2opt-fand.lock";
/// eco | balanced | perf, writable by the wheel group (power-mode, Touch Bar)
const MODE_DIR: &str = "/run/t2opt-fand";
const MODE_FILE: &str = "/run/t2opt-fand/mode";
/// the dGPU must stay awake this long before its cap and DPM files are touched
const DGPU_SETTLE_S: u64 = 15;

/// Logged SMC keys, in CSV column order. TGDD (dGPU die) and PM0C (memory
/// power, logged for model training only) may be missing.
const KEYS: [&str; 16] = [
    "TW0P", "TM0P", "Ts0P", "TG0P", "Ts1P", "Ts1S", "PZ0E", "PZ0F", "PZ0T", "PC0R", "PG0R", "F0Ac", "F1Ac", "TCMX", "TGDD",
    "PM0C",
];
const OPTIONAL: [&str; 2] = ["TGDD", "PM0C"];

fn key(vals: &[Option<f64>], name: &str) -> Option<f64> {
    vals[KEYS.iter().position(|k| *k == name).expect("unknown key")]
}

/// The controller's inputs from one row of key values, or the keys that are missing.
/// `uses_pm`: the model takes memory power, so PM0C is required too.
fn reading(vals: &[Option<f64>], uses_pm: bool) -> Result<Reading, Vec<&'static str>> {
    let need: &[&'static str] = if uses_pm {
        &["Ts1S", "PZ0T", "PC0R", "PG0R", "F0Ac", "F1Ac", "PM0C"]
    } else {
        &["Ts1S", "PZ0T", "PC0R", "PG0R", "F0Ac", "F1Ac"]
    };
    let missing: Vec<_> = SENSORS.iter().chain(need).copied().filter(|k| key(vals, k).is_none()).collect();
    if !missing.is_empty() {
        return Err(missing);
    }
    let k = |n| key(vals, n).unwrap();
    Ok(Reading {
        y: SENSORS.map(k),
        ts1s: k("Ts1S"),
        pz0t: k("PZ0T"),
        p: Power { pc: k("PC0R"), pg: k("PG0R"), pm: if uses_pm { k("PM0C") } else { 0.0 } },
        rpm: (k("F0Ac") + k("F1Ac")) / 2.0,
        tcmx: key(vals, "TCMX"),
        tgdd: key(vals, "TGDD"),
        dgpu_active: false,
    })
}

/// Create the mode file if needed (root:wheel 0664) and return the mode to start in.
fn setup_mode_file(cfg: &Config) -> Result<Mode, String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(MODE_DIR).map_err(|e| format!("{MODE_DIR}: {e}"))?;
    if let Some(m) = read_mode() {
        return Ok(m);
    }
    let start = Mode::parse(&cfg.default_mode).unwrap_or(Mode::Balanced);
    std::fs::write(MODE_FILE, format!("{}\n", start.name())).map_err(|e| format!("{MODE_FILE}: {e}"))?;
    let wheel = std::ffi::CString::new("wheel").unwrap();
    // SAFETY: getgrnam returns a pointer to static storage or null
    let gid = unsafe { libc::getgrnam(wheel.as_ptr()).as_ref().map(|g| g.gr_gid) };
    let _ = std::os::unix::fs::chown(MODE_FILE, Some(0), gid);
    let _ = std::fs::set_permissions(MODE_FILE, std::fs::Permissions::from_mode(0o664));
    Ok(start)
}

fn read_mode() -> Option<Mode> {
    Mode::parse(&std::fs::read_to_string(MODE_FILE).ok()?)
}

/// What the CPU cap means in RAPL terms: (PL1, PL2) in W.
fn rapl_limits(cfg: &Config, cpu_cap_pc0r: f64) -> (f64, f64) {
    // eco's energy budget may go under the thermal plan's floor
    let pl1 = pkg_from_pc0r(cpu_cap_pc0r).clamp(cfg.cpu_pl1_min_w.min(cfg.eco_sustained_w), cfg.cpu_pl1_max_w);
    (pl1, pl1 + cfg.pl2_extra_w)
}

/// Like printf's %.4g, for the CSV.
fn g4(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let exp = v.abs().log10().floor() as i32;
    if !(-4..4).contains(&exp) {
        return format!("{v:.3e}");
    }
    let s = format!("{v:.*}", (3 - exp).max(0) as usize);
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_owned() } else { s }
}

fn local_time(unix: i64, fmt: &str) -> String {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let mut buf = [0u8; 64];
    let cfmt = std::ffi::CString::new(fmt).unwrap();
    // SAFETY: tm and buf are valid for the calls; strftime bounds by buf.len()
    let n = unsafe {
        libc::localtime_r(&unix, &mut tm);
        libc::strftime(buf.as_mut_ptr().cast(), buf.len(), cfmt.as_ptr(), &tm)
    };
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

fn sd_notify(msg: &str) {
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixDatagram};
    let Ok(path) = std::env::var("NOTIFY_SOCKET") else { return };
    let addr = match path.strip_prefix('@') {
        Some(name) => SocketAddr::from_abstract_name(name.as_bytes()),
        None => SocketAddr::from_pathname(&path),
    };
    if let (Ok(sock), Ok(addr)) = (UnixDatagram::unbound(), addr) {
        let _ = sock.send_to_addr(msg.as_bytes(), &addr);
    }
}

fn write_status(mode: &str, ctl: &Controller, r: &Reading) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
    let st = serde_json::json!({
        "time": now, "mode": mode, "cmd_rpm": ctl.cmd.round(), "reason": ctl.reason,
        "ts1s": r.ts1s, "pred_ts1s_max": (ctl.pred * 100.0).round() / 100.0, "ceiling_c": ctl.cfg.ceiling_c,
        "room_offset_k": (ctl.kf.ambient() * 100.0).round() / 100.0, "pz0t": r.pz0t, "fan_rpm": r.rpm,
        "power_mode": ctl.mode.name(), "fan_ceiling_rpm": ctl.cfg.fan_ceiling(ctl.mode).min(ctl.hi),
        "cpu_pl1_w": ctl.caps.map(|c| (rapl_limits(&ctl.cfg, c.cpu).0 * 10.0).round() / 10.0),
        "dgpu_cap_w": ctl.caps.and_then(|c| c.gpu).map(|g| (g * 10.0).round() / 10.0),
        "dgpu_active": r.dgpu_active,
    });
    let tmp = format!("{STATUS}.tmp");
    if std::fs::write(&tmp, st.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, STATUS);
    }
}

const CSV_HEAD: &str = "timestamp,unix_time,mode,cmd_rpm,reason,pred_ts1s_max,kf_ambient,kf_H,kf_C,kf_S,replica_bias,power_mode,fan_ceiling,\
cpu_pl1_w,cpu_cores_w,dgpu_cap_w,dgpu_active,heat_w,sustainable_w,rapl_pkg_w,rapl_core_w,rapl_gfx_w,igpu_act_mhz,igpu_req_mhz,cpu_busy_pct,cpu_iowait_pct,wifi_mb_s";

/// Extra per-second measurements for the CSV (live only).
#[derive(Default)]
struct Extra {
    rapl: Option<[f64; 3]>,
    igpu: Option<(f64, f64)>,
    cpu: Option<(f64, f64)>,
    wifi: Option<f64>,
}

fn csv_row(mode: &str, ctl: &Controller, vals: &[Option<f64>], unix: f64, dgpu_active: bool, x: &Extra) -> String {
    let n = ctl.kf.nodes();
    let mut row = format!(
        "{},{unix:.1},{mode},{:.0},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        local_time(unix as i64, "%F %T"),
        ctl.cmd,
        ctl.reason,
        ctl.pred,
        ctl.kf.ambient(),
        n[0],
        n[1],
        n[2],
        ctl.rep.bias
    );
    let opt = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.1}"));
    row.push_str(&format!(
        ",{},{:.0},{},{},{},{},{:.2},{:.2}",
        ctl.mode.name(),
        ctl.cfg.fan_ceiling(ctl.mode).min(ctl.hi),
        opt(ctl.caps.map(|c| rapl_limits(&ctl.cfg, c.cpu).0)),
        opt(ctl.caps.and_then(|c| c.cores)),
        opt(ctl.caps.and_then(|c| c.gpu)),
        u8::from(dgpu_active),
        ctl.heat(),
        ctl.sustainable
    ));
    let pair = |v: Option<(f64, f64)>| v.map_or(",".to_owned(), |(a, b)| format!("{a:.2},{b:.2}"));
    let wifi = x.wifi.map_or(String::new(), |w| format!("{w:.3}"));
    let rapl = x.rapl.map_or(",,".to_owned(), |[p, c, g]| format!("{p:.2},{c:.2},{g:.2}"));
    row.push_str(&format!(",{rapl},{},{},{wifi}", pair(x.igpu), pair(x.cpu)));
    for v in vals {
        row.push(',');
        if let Some(v) = v {
            row.push_str(&g4(*v));
        }
    }
    row.push('\n');
    row
}

fn logger_running() -> bool {
    std::fs::read_dir("/proc").into_iter().flatten().flatten().any(|e| {
        e.file_name().to_string_lossy().bytes().all(|b| b.is_ascii_digit())
            && std::fs::read(e.path().join("cmdline")).is_ok_and(|c| c.windows(15).any(|w| w == b"tools/logger.py"))
    })
}

fn run_live(cfg: Config, cfg_dir: &Path, shadow: bool) -> Result<(), String> {
    if unsafe { libc::geteuid() } != 0 {
        return Err("run as root (SMC keys and fan control need it)".into());
    }
    let lock = File::create(LOCK).map_err(|e| format!("{LOCK}: {e}"))?;
    // SAFETY: valid fd for the lifetime of `lock`
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err("another fand is running".into());
    }
    if logger_running() {
        return Err("tools/logger.py is running; two SMC readers corrupt each other's reads".into());
    }

    let fans = hw::Fans::open()?;
    let model = Model::load(&cfg_dir.join(&cfg.model))?;
    let uses_pm = model.uses_pm;
    let mut ctl = Controller::new(cfg.clone(), model, fans.lo, fans.hi);
    // PM0C is only optional when the model doesn't take it
    let optional: Vec<&str> = OPTIONAL.iter().copied().filter(|k| !(uses_pm && *k == "PM0C")).collect();
    let smc = hw::SmcReader::open(&KEYS, &optional)?;
    // amdgpu can register its card after fand starts at boot, so a miss is retried each tick
    let mut dgpu = hw::Dgpu::find();
    // an earlier build could leave the cores-only power limit on
    if !shadow {
        hw::set_core_limit(None, 0.0)?;
    }
    ctl.set_mode(setup_mode_file(&cfg)?);
    let mode = if shadow { "shadow" } else { "control" };
    println!(
        "fand {mode}: fans {:.0}-{:.0} rpm (limits {:.0}-{:.0}), ceiling {} °C, model {} ({})",
        fans.lo, fans.hi, ctl.lo, ctl.hi, cfg.ceiling_c, cfg.model, ctl.model.topology
    );

    let mut out = None;
    if cfg.log_csv {
        let path = cfg_dir.join(&cfg.log_dir).join(local_time(now_unix() as i64, &format!("fand-{mode}-%Y%m%d-%H%M%S.csv")));
        let f = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        // logs belong to whoever owns the config (the checkout's user)
        if let Ok(m) = std::fs::metadata(cfg_dir) {
            use std::os::unix::fs::MetadataExt;
            let _ = std::os::unix::fs::chown(&path, Some(m.uid()), Some(m.gid()));
        }
        let mut w = LineWriter::new(f);
        let _ = writeln!(w, "{CSV_HEAD},{}", KEYS.join(","));
        println!("log: {}", path.display());
        out = Some(w);
    }

    let stop = Arc::new(AtomicBool::new(false));
    for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(sig, stop.clone()).map_err(|e| e.to_string())?;
    }
    sd_notify("READY=1");

    let res = (|| -> Result<(), String> {
        let mut next = Instant::now();
        let mut prev = (Instant::now(), now_unix());
        let mut last = ("", ctl.mode);
        // dGPU bookkeeping: awake since, DPM table checked this wake, cap last written, reset job
        let mut gpu_since: Option<Instant> = None;
        let mut dpm_checked = false;
        let mut gpu_written: Option<f64> = None;
        let mut dpm_job: Option<std::process::Child> = None;
        let mut meter = hw::RaplMeter::new();
        let (mut cpu_meter, mut wifi_meter) = (hw::CpuMeter::new(), hw::WifiMeter::new());
        let mut lavd = hw::Lavd::new();
        while !stop.load(Ordering::Relaxed) {
            next += Duration::from_secs(1);
            let vals = smc.read();
            let now = (Instant::now(), now_unix());
            // Instant (CLOCK_MONOTONIC) stops in suspend, wall time doesn't: the
            // SMC was reset and the machine cooled, so start the model over
            let resumed = now.1 - prev.1 > now.0.duration_since(prev.0).as_secs_f64() + 5.0;
            prev = now;
            if let Some(m) = read_mode() {
                ctl.set_mode(m);
            }
            match reading(&vals, uses_pm) {
                Err(missing) => println!("incomplete SMC read, skipped: {missing:?}"),
                Ok(mut r) => {
                    if dgpu.is_none() {
                        dgpu = hw::Dgpu::find();
                        if dgpu.is_some() {
                            println!("dGPU found");
                        }
                    }
                    r.dgpu_active = dgpu.as_ref().is_some_and(|d| d.active());
                    if resumed {
                        println!("resume detected: model state reset");
                        ctl.reset(&r);
                    }
                    let cmd = ctl.tick(&r);
                    let extra = Extra { rapl: meter.read(), igpu: hw::igpu_mhz(), cpu: cpu_meter.read(), wifi: wifi_meter.read() };
                    if !shadow {
                        fans.set(cmd)?;
                        match ctl.caps {
                            Some(c) => {
                                let (pl1, pl2) = rapl_limits(&cfg, c.cpu);
                                hw::set_cpu_limits(pl1, pl2, cfg.pl1_window_s)?;
                                hw::set_core_limit(c.cores, cfg.pl1_window_s)?;
                            }
                            None => {
                                hw::set_cpu_limits(cfg.warmup_pl1_w, cfg.warmup_pl1_w + cfg.pl2_extra_w, cfg.pl1_window_s)?;
                                hw::set_core_limit(None, 0.0)?;
                            }
                        }
                        if cfg.lavd_profiles {
                            lavd.follow(ctl.mode);
                        }
                        if let Some(d) = dgpu.as_ref() {
                            if !r.dgpu_active {
                                (gpu_since, dpm_checked, gpu_written) = (None, false, None);
                            } else if gpu_since.get_or_insert(now.0).elapsed().as_secs() >= DGPU_SETTLE_S {
                                if dpm_job.as_mut().is_some_and(|j| j.try_wait().ok().flatten().is_none()) {
                                    // a DPM reset ends by writing its own cap: rewrite ours after it
                                    gpu_written = None;
                                } else {
                                    dpm_job = None;
                                    if !dpm_checked {
                                        dpm_checked = true;
                                        if let Some(cmd) = cfg.dgpu_dpm_cmd.as_deref().filter(|_| !d.stock_dpm()) {
                                            println!("dGPU: overdrive table not stock, resetting to auto");
                                            dpm_job = d.reset_dpm(cmd);
                                        }
                                    }
                                    let want = ctl.caps.and_then(|c| c.gpu);
                                    let stale = |g: &f64| gpu_written.is_none_or(|w| (w - g).abs() >= 0.5);
                                    if let Some(g) = want.filter(stale).filter(|_| dpm_job.is_none()) {
                                        d.set_cap(g)?;
                                        gpu_written = Some(g);
                                    }
                                }
                            }
                        }
                    }
                    if ctl.reason != last.0 || ctl.mode != last.1 {
                        let caps = ctl.caps.map_or("caps not set yet".to_owned(), |c| {
                            let (pl1, _) = rapl_limits(&cfg, c.cpu);
                            let cores = c.cores.map_or(String::new(), |w| format!(" (cores {w:.1} W)"));
                            match c.gpu {
                                Some(g) => format!("PL1 {pl1:.1} W{cores}, dGPU {g:.1} W"),
                                None => format!("PL1 {pl1:.1} W{cores}"),
                            }
                        });
                        println!(
                            "{} [{}] {}: {cmd:.0} rpm, {caps}, Ts1S {:.2}, pred max {:.2}, room {:+.2} K, extra heat {:+.1} W",
                            local_time(now.1 as i64, "%T"),
                            ctl.mode.name(),
                            ctl.reason,
                            r.ts1s,
                            ctl.pred,
                            ctl.kf.ambient(),
                            ctl.heat()
                        );
                        last = (ctl.reason, ctl.mode);
                    }
                    write_status(mode, &ctl, &r);
                    if let Some(w) = out.as_mut() {
                        let _ = w.write_all(csv_row(mode, &ctl, &vals, now.1, r.dgpu_active, &extra).as_bytes());
                    }
                }
            }
            sd_notify("WATCHDOG=1");
            match next.checked_duration_since(Instant::now()) {
                Some(d) => std::thread::sleep(d),
                // fell behind by more than a tick: restart the 1 s schedule
                None if Instant::now() - next > Duration::from_secs(5) => next = Instant::now(),
                None => {}
            }
        }
        Ok(())
    })();
    if !shadow {
        fans.release();
        if let Err(e) = hw::set_core_limit(None, 0.0) {
            eprintln!("core limit: {e}");
        }
        println!("fans released to the SMC, core power limit off");
    }
    let _ = std::fs::remove_file(STATUS);
    res
}

fn now_unix() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

/// Feed a logged 1 Hz CSV through the controller; report what it would do.
/// The dGPU's runtime state isn't logged by the passive logger; it draws
/// ~0.1 W asleep, so above 1 W counts as awake.
fn run_replay(cfg: Config, cfg_dir: &Path, path: &Path, trace: bool, mode: Option<Mode>) -> Result<(), String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = BufReader::new(f).lines().map_while(Result::ok);
    let head: Vec<String> = lines.next().ok_or("empty CSV")?.split(',').map(str::to_owned).collect();
    let col = |n: &str| head.iter().position(|h| h == n);
    let cols: Vec<Option<usize>> = KEYS.iter().map(|k| col(k)).collect();
    let t_col = col("unix_time").ok_or("no unix_time column")?;
    let rows: Vec<Vec<String>> = lines.map(|l| l.split(',').map(str::to_owned).collect()).collect();
    let first = rows.first().ok_or("no rows")?;
    let num = |row: &Vec<String>, n: &str| col(n).and_then(|i| row.get(i)?.parse::<f64>().ok());
    let lim = |a: &str, b: &str| -> Result<f64, String> {
        Ok((num(first, a).ok_or(format!("no {a}"))? + num(first, b).ok_or(format!("no {b}"))?) / 2.0)
    };
    let model = Model::load(&cfg_dir.join(&cfg.model))?;
    let uses_pm = model.uses_pm;
    // logger CSVs carry the fan limits; fand's own logs don't, so read this machine's
    let (lo, hi) = match (lim("F0Mn", "F1Mn"), lim("F0Mx", "F1Mx")) {
        (Ok(lo), Ok(hi)) => (lo, hi),
        _ => hw::Fans::open().map(|f| (f.lo, f.hi))?,
    };
    let start_mode = mode.or(Mode::parse(&cfg.default_mode)).unwrap_or(Mode::Balanced);
    let mut ctl = Controller::new(cfg, model, lo, hi);
    ctl.set_mode(start_mode);

    let t0 = Instant::now();
    let (mut pl1s, mut capped) = (Vec::new(), 0u32);
    let (mut cmds, mut acts, mut ts_max) = (Vec::new(), Vec::new(), f64::NEG_INFINITY);
    let mut reasons: Vec<(&str, u32)> = Vec::new();
    let mut prev_t: Option<f64> = None;
    for row in &rows {
        // a value that is present but not a number drops the whole row
        let vals: Option<Vec<Option<f64>>> = cols
            .iter()
            .map(|c| match c.and_then(|i| row.get(i)).map(String::as_str) {
                None | Some("") => Some(None),
                Some(s) => s.parse().ok().map(Some),
            })
            .collect();
        let Some(mut r) = vals.as_deref().and_then(|v| reading(v, uses_pm).ok()) else { continue };
        r.dgpu_active = r.p.pg > 1.0;
        let Some(t) = row.get(t_col).and_then(|s| s.parse::<f64>().ok()) else { continue };
        if prev_t.is_some_and(|p| t - p > 8.0) {
            ctl.reset(&r);
        }
        prev_t = Some(t);
        let cmd = ctl.tick(&r);
        if trace {
            println!("{} {cmd:.3} {} {:.4}", cmds.len(), ctl.reason, ctl.pred);
        }
        cmds.push(cmd);
        acts.push(r.rpm);
        if let Some(c) = ctl.caps {
            pl1s.push(rapl_limits(&ctl.cfg, c.cpu).0);
        }
        capped += u32::from(ctl.reason == "power-cap");
        ts_max = ts_max.max(r.ts1s);
        match reasons.iter_mut().find(|(k, _)| *k == ctl.reason) {
            Some((_, n)) => *n += 1,
            None => reasons.push((ctl.reason, 1)),
        }
    }
    if trace {
        return Ok(());
    }
    let dt = t0.elapsed().as_secs_f64();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    reasons.sort_by_key(|r| std::cmp::Reverse(r.1));
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    println!("{name}: {} s replayed in {dt:.2} s ({:.3} ms/s)", cmds.len(), 1e3 * dt / cmds.len() as f64);
    println!("  logged fans: mean {:.0}, max {:.0} rpm", mean(&acts), max(&acts));
    println!("  fand would:  mean {:.0}, max {:.0} rpm", mean(&cmds), max(&cmds));
    println!("  reasons (s): {}", reasons.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    println!("  measured Ts1S max {ts_max:.2} °C (set point {SETPOINT})");
    if !pl1s.is_empty() {
        println!(
            "  mode {}: PL1 mean {:.1} W, min {:.1} W; load held by a power cap {capped} s",
            start_mode.name(),
            mean(&pl1s),
            pl1s.iter().copied().fold(f64::INFINITY, f64::min)
        );
    }
    Ok(())
}

fn usage() -> ! {
    eprintln!("usage: t2opt-fand [--config FILE] [--shadow | --replay CSV [--trace] [--mode eco|balanced|perf] | --release]");
    eprintln!("       t2opt-fand [--config FILE] --sim CSV [--plant MODEL] [--plant-heat K] [--inject] [--from TIME] [--to TIME] [--mode M] [--out CSV]");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut config = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/fand.toml"));
    let (mut shadow, mut trace, mut release, mut replay, mut mode) = (false, false, false, None, None);
    let (mut plant_heat, mut inject) = (1.0, false);
    let (mut sim, mut plant, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let (mut from, mut to): (Option<String>, Option<String>) = (None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config = args.next().unwrap_or_else(|| usage()).into(),
            "--shadow" => shadow = true,
            "--trace" => trace = true,
            "--release" => release = true,
            "--replay" => replay = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--sim" => sim = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--plant" => plant = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--plant-heat" => plant_heat = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--inject" => inject = true,
            "--from" => from = Some(args.next().unwrap_or_else(|| usage())),
            "--to" => to = Some(args.next().unwrap_or_else(|| usage())),
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--mode" => mode = Some(args.next().as_deref().and_then(Mode::parse).unwrap_or_else(|| usage())),
            _ => usage(),
        }
    }
    let cfg_dir = config.parent().unwrap_or(Path::new(".")).to_owned();
    let res = Config::load(&config).and_then(|cfg| {
        if release {
            // the CPU limits are left to whatever takes over; the dGPU cap is
            // only safe to touch while the card is awake
            if let Some(d) = hw::Dgpu::find().filter(|d| d.active()) {
                match cfg.dgpu_release_cap_w.or_else(|| d.default_cap()) {
                    Some(w) => {
                        if let Err(e) = d.set_cap(w) {
                            eprintln!("dGPU cap: {e}");
                        }
                    }
                    None => eprintln!("dGPU cap: no power1_cap_default"),
                }
            }
            if let Err(e) = hw::set_core_limit(None, 0.0) {
                eprintln!("core limit: {e}");
            }
            hw::Fans::open().map(|f| f.release())
        } else if let Some(csv) = sim {
            let a = sim::SimArgs { plant: plant.as_deref(), plant_heat, inject, from: from.as_deref(), to: to.as_deref(), mode, out: out.as_deref() };
            sim::run_sim(cfg, &cfg_dir, &csv, a)
        } else if let Some(csv) = replay {
            run_replay(cfg, &cfg_dir, &csv, trace, mode)
        } else {
            run_live(cfg, &cfg_dir, shadow)
        }
    });
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
