//! applesmc sysfs: SMC keys and fan control. Root only.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;

/// applesmc binds to the T2's ACPI SMC device
pub const SMC: &str = "/sys/bus/acpi/devices/APP0001:00";

fn rd(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map(|s| s.trim().to_owned()).map_err(|e| format!("{path}: {e}"))
}

fn wr(path: &str, v: impl std::fmt::Display) -> Result<(), String> {
    std::fs::write(path, v.to_string()).map_err(|e| format!("{path}: {e}"))
}

fn decode(typ: &str, b: &[u8]) -> Option<f64> {
    Some(match typ {
        "flt" => f32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
        "sp78" => i16::from_be_bytes(b.get(..2)?.try_into().ok()?) as f64 / 256.0,
        "sp87" => i16::from_be_bytes(b.get(..2)?.try_into().ok()?) as f64 / 128.0,
        "ui8" => *b.first()? as f64,
        "ui16" => u16::from_be_bytes(b.get(..2)?.try_into().ok()?) as f64,
        _ => return None,
    })
}

/// Every SMC reader holds this (flock) while it uses key_at_index, so a
/// second reader (a local logger) waits instead of interleaving.
pub const SMC_LOCK: &str = "/run/t2opt-smc.lock";
/// how long a read waits for another reader; past it, reading anyway beats
/// stalling the fans behind a reader that hangs
const SMC_LOCK_WAIT_MS: u32 = 200;

/// Holds the SMC lock until dropped (or nothing, if it couldn't be had).
struct SmcGuard<'a>(Option<&'a File>);

impl Drop for SmcGuard<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.0 {
            // SAFETY: flock on an open fd
            unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

fn smc_lock(lock: &File) -> SmcGuard<'_> {
    for _ in 0..SMC_LOCK_WAIT_MS / 5 {
        // SAFETY: flock on an open fd
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return SmcGuard(Some(lock));
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    println!("SMC lock held by another reader for {SMC_LOCK_WAIT_MS} ms; reading without it");
    SmcGuard(None)
}

/// Reads a fixed set of SMC keys through key_at_index.
///
/// key_at_index is one shared register: a second process reading keys at the
/// same time interleaves with this one and corrupts both, so readers take
/// SMC_LOCK around their reads.
pub struct SmcReader {
    idx: File,
    data: File,
    lock: File,
    keys: Vec<Option<(u32, String)>>,
}

impl SmcReader {
    /// `names` are the keys to read; `optional` ones may be missing on this machine.
    pub fn open(names: &[&str], optional: &[&str]) -> Result<Self, String> {
        let idx = OpenOptions::new().write(true).open(format!("{SMC}/key_at_index")).map_err(|e| format!("key_at_index: {e}"))?;
        let data = File::open(format!("{SMC}/key_at_index_data")).map_err(|e| format!("key_at_index_data: {e}"))?;
        let lock = OpenOptions::new().create(true).truncate(false).write(true).open(SMC_LOCK).map_err(|e| format!("{SMC_LOCK}: {e}"))?;
        let mut keys = vec![None; names.len()];
        // the whole index scan in one go: another reader just waits
        let guard = smc_lock(&lock);
        let count: u32 = rd(&format!("{SMC}/key_count"))?.parse().map_err(|e| format!("key_count: {e}"))?;
        for i in 0..count {
            idx.write_at(i.to_string().as_bytes(), 0).map_err(|e| format!("key_at_index: {e}"))?;
            let name = rd(&format!("{SMC}/key_at_index_name"))?;
            if let Some(k) = names.iter().position(|n| *n == name) {
                // trimmed: "ui8 " reads as "ui8"
                keys[k] = Some((i, rd(&format!("{SMC}/key_at_index_type"))?));
            }
        }
        drop(guard);
        let missing: Vec<_> = names.iter().zip(&keys).filter(|(n, k)| k.is_none() && !optional.contains(n)).map(|(n, _)| *n).collect();
        if !missing.is_empty() {
            return Err(format!("SMC keys not found: {missing:?}"));
        }
        Ok(Self { idx, data, lock, keys })
    }

    /// One value per requested key, None where the read failed or the key is missing.
    pub fn read(&self) -> Vec<Option<f64>> {
        let mut buf = [0u8; 8];
        let _guard = smc_lock(&self.lock);
        self.keys
            .iter()
            .map(|k| {
                let (i, typ) = k.as_ref()?;
                self.idx.write_at(i.to_string().as_bytes(), 0).ok()?;
                let n = self.data.read_at(&mut buf, 0).ok()?;
                decode(typ, &buf[..n])
            })
            .collect()
    }
}

/// Both fans get the same fraction of their own range, as t2fanrd does.
pub struct Fans {
    fans: Vec<(u32, f64, f64)>,
    pub lo: f64,
    pub hi: f64,
}

impl Fans {
    pub fn open() -> Result<Self, String> {
        let mut fans = Vec::new();
        for n in [1, 2] {
            let min: f64 = rd(&format!("{SMC}/fan{n}_min"))?.parse().map_err(|e| format!("fan{n}_min: {e}"))?;
            let max: f64 = rd(&format!("{SMC}/fan{n}_max"))?.parse().map_err(|e| format!("fan{n}_max: {e}"))?;
            fans.push((n, min, max));
        }
        let k = fans.len() as f64;
        let lo = fans.iter().map(|f| f.1).sum::<f64>() / k;
        let hi = fans.iter().map(|f| f.2).sum::<f64>() / k;
        Ok(Self { fans, lo, hi })
    }

    pub fn set(&self, mean_rpm: f64) -> Result<(), String> {
        let frac = ((mean_rpm - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0);
        for &(n, min, max) in &self.fans {
            // a suspend resets manual mode to 0; re-assert it every time
            let manual = format!("{SMC}/fan{n}_manual");
            if rd(&manual)? != "1" {
                wr(&manual, 1)?;
            }
            wr(&format!("{SMC}/fan{n}_output"), (min + frac * (max - min)).round() as u32)?;
        }
        Ok(())
    }

    /// Hand the fans back to the SMC's own control.
    pub fn release(&self) {
        for &(n, _, _) in &self.fans {
            if let Err(e) = wr(&format!("{SMC}/fan{n}_manual"), 0) {
                eprintln!("release fan {n}: {e}");
            }
        }
    }
}

const RAPL: &str = "/sys/class/powercap/intel-rapl:0";

/// The dGPU's power state and cap. Reading most amdgpu files restarts its
/// autosuspend timer (a 2 s poll kept it awake), so only runtime_status is
/// polled; the cap and DPM files are touched once the card stays awake.
pub struct Dgpu {
    dev: String,
    status: String,
}

impl Dgpu {
    pub fn find() -> Option<Self> {
        let cards = std::fs::read_dir("/sys/class/drm").ok()?;
        for c in cards.flatten() {
            let dev = c.path().join("device");
            let is_amdgpu = std::fs::read_link(dev.join("driver")).is_ok_and(|d| d.ends_with("amdgpu"));
            if is_amdgpu {
                let dev = std::fs::canonicalize(&dev).ok()?.to_string_lossy().into_owned();
                return Some(Self { status: format!("{dev}/power/runtime_status"), dev });
            }
        }
        None
    }

    pub fn active(&self) -> bool {
        rd(&self.status).is_ok_and(|s| s == "active")
    }

    fn hwmon(&self) -> Option<String> {
        let hw = std::fs::read_dir(format!("{}/hwmon", self.dev)).ok()?.flatten().next()?;
        Some(hw.path().display().to_string())
    }

    /// Set the power cap (W). Only call while active: writing wakes a sleeping card.
    pub fn set_cap(&self, w: f64) -> Result<(), String> {
        let hw = self.hwmon().ok_or("no amdgpu hwmon")?;
        wr(&format!("{hw}/power1_cap"), (w * 1e6).round() as u64)
    }

    /// The driver's default cap (W). Only call while active, like set_cap.
    pub fn default_cap(&self) -> Option<f64> {
        let uw: f64 = rd(&format!("{}/power1_cap_default", self.hwmon()?)).ok()?.parse().ok()?;
        Some(uw / 1e6)
    }

    /// The overdrive table survives runtime suspend, so a card left with a
    /// custom table keeps it. True if the level is stock auto and the sclk
    /// table isn't flattened (an undervolt that runs states 2-7 at one clock).
    pub fn stock_dpm(&self) -> bool {
        let level = rd(&format!("{}/power_dpm_force_performance_level", self.dev)).unwrap_or_default();
        let od = std::fs::read_to_string(format!("{}/pp_od_clk_voltage", self.dev)).unwrap_or_default();
        // mid and game flatten sclk states 2-7 to one clock; the stock table never does
        let mut sclk = od.lines().skip_while(|l| !l.starts_with("OD_SCLK")).skip(1).take_while(|l| !l.starts_with("OD_"));
        let s2 = sclk.clone().find(|l| l.trim_start().starts_with("2:")).map(|l| l.split_whitespace().nth(1).unwrap_or("").to_owned());
        let s7 = sclk.find(|l| l.trim_start().starts_with("7:")).map(|l| l.split_whitespace().nth(1).unwrap_or("").to_owned());
        level == "auto" && (s2.is_none() || s2 != s7)
    }

    /// Back to the stock table and level through `cmd` (fand.toml's
    /// dgpu_dpm_cmd), in the background.
    pub fn reset_dpm(&self, cmd: &str) -> Option<std::process::Child> {
        std::process::Command::new(cmd)
            .arg("auto")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()
    }
}

/// CPU package limits (RAPL PL1 / PL2 / PL1 window). Rewritten only when one
/// differs from what is set by `step_w` or more, so a firmware reset (e.g.
/// after resume) is corrected next time and small moves aren't applied.
pub fn set_cpu_limits(pl1_w: f64, pl2_w: f64, window_s: f64, step_w: f64) -> Result<bool, String> {
    let uw = |w: f64| (w * 1e6).round() as u64;
    let cur = |f: &str| rd(&format!("{RAPL}/{f}")).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let near = |a: u64, b: u64| a.abs_diff(b) < uw(step_w);
    if near(cur("constraint_0_power_limit_uw"), uw(pl1_w)) && near(cur("constraint_1_power_limit_uw"), uw(pl2_w)) {
        return Ok(false);
    }
    wr(&format!("{RAPL}/constraint_0_power_limit_uw"), uw(pl1_w))?;
    wr(&format!("{RAPL}/constraint_1_power_limit_uw"), uw(pl2_w))?;
    // the hardware rounds the window, so it is only rewritten with the limits
    let _ = wr(&format!("{RAPL}/constraint_0_time_window_us"), uw(window_s));
    Ok(true)
}

const RAPL_CORE: &str = "/sys/class/powercap/intel-rapl:0:0";
/// the "uncore" domain (PP1) is the iGPU
const RAPL_GFX: &str = "/sys/class/powercap/intel-rapl:0:1";

/// Cores-only limit (RAPL PP0): eco's iGPU fence while its energy budget
/// holds the package. None switches it off (also at start and exit).
pub fn set_core_limit(w: Option<f64>, window_s: f64) -> Result<(), String> {
    let enabled = rd(&format!("{RAPL_CORE}/enabled")).is_ok_and(|v| v == "1");
    let Some(w) = w else {
        if enabled {
            wr(&format!("{RAPL_CORE}/enabled"), 0)?;
        }
        return Ok(());
    };
    let uw = (w * 1e6).round() as u64;
    let cur = rd(&format!("{RAPL_CORE}/constraint_0_power_limit_uw")).ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    if cur.abs_diff(uw) >= 250_000 {
        wr(&format!("{RAPL_CORE}/constraint_0_power_limit_uw"), uw)?;
        let _ = wr(&format!("{RAPL_CORE}/constraint_0_time_window_us"), (window_s * 1e6).round() as u64);
    }
    if !enabled {
        wr(&format!("{RAPL_CORE}/enabled"), 1)?;
    }
    Ok(())
}

/// Package, core and iGPU power from the RAPL energy counters (root only), W.
pub struct RaplMeter {
    last: Option<([u64; 3], std::time::Instant)>,
}

impl RaplMeter {
    pub fn new() -> Self {
        Self { last: None }
    }

    pub fn read(&mut self) -> Option<[f64; 3]> {
        const DOMAINS: [&str; 3] = [RAPL, RAPL_CORE, RAPL_GFX];
        let e = |d: &str| rd(&format!("{d}/energy_uj")).ok()?.parse::<u64>().ok();
        let range = |d: &str| rd(&format!("{d}/max_energy_range_uj")).ok()?.parse::<u64>().ok();
        let now = ([e(RAPL)?, e(RAPL_CORE)?, e(RAPL_GFX)?], std::time::Instant::now());
        let prev = self.last.replace(now)?;
        let dt = now.1.duration_since(prev.1).as_secs_f64();
        // the counters wrap at max_energy_range_uj
        let delta = |a: u64, b: u64, r: Option<u64>| if b >= a { b - a } else { b + r.unwrap_or(u64::MAX) - a };
        (dt > 0.2 && dt < 5.0).then(|| std::array::from_fn(|k| delta(prev.0[k], now.0[k], range(DOMAINS[k])) as f64 / dt / 1e6))
    }
}

/// Mean and hottest temperature of the CPU cores (coretemp's "Core N" inputs), °C.
pub fn core_temps() -> Option<(f64, f64)> {
    let hw = std::fs::read_dir("/sys/class/hwmon").ok()?.flatten().map(|e| e.path()).find(|p| rd(&p.join("name").to_string_lossy()).is_ok_and(|n| n == "coretemp"))?;
    let (mut sum, mut n, mut max) = (0.0, 0u32, f64::NEG_INFINITY);
    for e in std::fs::read_dir(&hw).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(base) = name.strip_suffix("_label") else { continue };
        if !rd(&e.path().to_string_lossy()).is_ok_and(|l| l.starts_with("Core ")) {
            continue;
        }
        if let Some(mc) = rd(&hw.join(format!("{base}_input")).to_string_lossy()).ok().and_then(|v| v.parse::<f64>().ok()) {
            (sum, n, max) = (sum + mc / 1000.0, n + 1, max.max(mc / 1000.0));
        }
    }
    (n > 0).then(|| (sum / n as f64, max))
}

/// The iGPU's actual and requested clock (MHz), for logging.
pub fn igpu_mhz() -> Option<(f64, f64)> {
    let cards = std::fs::read_dir("/sys/class/drm").ok()?;
    for c in cards.flatten() {
        let gt = c.path().join("gt/gt0");
        if gt.join("rps_act_freq_mhz").exists() {
            let f = |n: &str| std::fs::read_to_string(gt.join(n)).ok()?.trim().parse::<f64>().ok();
            return Some((f("rps_act_freq_mhz")?, f("rps_cur_freq_mhz")?));
        }
    }
    None
}

/// Keeps scx_lavd's profile in step with the power mode through scx_loader's
/// D-Bus API (busctl, root may call it). Only touches a running scx_lavd: no
/// scx_loader, another scheduler or none at all means nothing to do. Checked
/// on a mode change and every RECHECK_S (scx_loader may have restarted).
pub struct Lavd {
    applied: Option<crate::control::Mode>,
    checked: Option<std::time::Instant>,
    job: Option<std::process::Child>,
}

const BUSCTL: &str = "/usr/bin/busctl";
const LAVD_RECHECK_S: u64 = 30;

impl Lavd {
    pub fn new() -> Self {
        Self { applied: None, checked: None, job: None }
    }

    fn get(prop: &str) -> Option<String> {
        let out = std::process::Command::new(BUSCTL)
            .args(["get-property", "org.scx.Loader", "/org/scx/Loader", "org.scx.Loader", prop])
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    }

    pub fn follow(&mut self, mode: crate::control::Mode) {
        if self.job.as_mut().is_some_and(|j| j.try_wait().ok().flatten().is_none()) {
            return;
        }
        self.job = None;
        let due = self.checked.is_none_or(|t| t.elapsed().as_secs() >= LAVD_RECHECK_S);
        if self.applied == Some(mode) && !due {
            return;
        }
        self.checked = Some(std::time::Instant::now());
        // e.g. `s "scx_lavd"` and `u 2`
        if Self::get("CurrentScheduler").as_deref() != Some("s \"scx_lavd\"") {
            self.applied = None;
            return;
        }
        let want = mode.scx_mode();
        if Self::get("SchedulerMode").as_deref() == Some(&format!("u {want}")) {
            self.applied = Some(mode);
            return;
        }
        println!("scx_lavd -> {} (scx_loader mode {want})", mode.name());
        self.job = std::process::Command::new(BUSCTL)
            .args(["call", "org.scx.Loader", "/org/scx/Loader", "org.scx.Loader", "SwitchScheduler", "su", "scx_lavd", &want.to_string()])
            .stdout(std::process::Stdio::null())
            .spawn()
            .ok();
        self.applied = Some(mode);
    }
}

/// CPU busy and iowait shares (%) from /proc/stat, between two reads.
pub struct CpuMeter {
    last: Option<(u64, u64, u64)>,
}

impl CpuMeter {
    pub fn new() -> Self {
        Self { last: None }
    }

    pub fn read(&mut self) -> Option<(f64, f64)> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        // cpu user nice system idle iowait irq softirq steal ...
        let v: Vec<u64> = stat.lines().next()?.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
        let (total, idle, iowait) = (v.iter().take(8).sum::<u64>(), *v.get(3)?, *v.get(4)?);
        let prev = self.last.replace((total, idle, iowait))?;
        let dt = total.checked_sub(prev.0).filter(|&d| d > 0)? as f64;
        let busy = dt - (idle - prev.1.min(idle)) as f64 - (iowait - prev.2.min(iowait)) as f64;
        Some((100.0 * busy / dt, 100.0 * (iowait - prev.2.min(iowait)) as f64 / dt))
    }
}

/// Wi-Fi traffic (MB/s, received + sent) on the first wireless interface:
/// a long download heats the Wi-Fi module, which the model has no input for.
pub struct WifiMeter {
    dev: Option<std::path::PathBuf>,
    last: Option<(u64, std::time::Instant)>,
}

impl WifiMeter {
    pub fn new() -> Self {
        let dev = std::fs::read_dir("/sys/class/net").ok().and_then(|d| d.flatten().map(|e| e.path()).find(|p| p.join("wireless").exists()));
        Self { dev, last: None }
    }

    pub fn read(&mut self) -> Option<f64> {
        let dev = self.dev.as_ref()?;
        let n = |f: &str| std::fs::read_to_string(dev.join("statistics").join(f)).ok()?.trim().parse::<u64>().ok();
        let now = (n("rx_bytes")? + n("tx_bytes")?, std::time::Instant::now());
        let prev = self.last.replace(now)?;
        let dt = now.1.duration_since(prev.1).as_secs_f64();
        (dt > 0.2 && now.0 >= prev.0).then(|| (now.0 - prev.0) as f64 / dt / 1e6)
    }
}
