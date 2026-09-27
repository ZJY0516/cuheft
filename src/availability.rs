//! Which kernels a given GPU can load from a cubin, only JIT from PTX, or
//! not run at all.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use regex::Regex;

use crate::cubin::CubinInfo;
use crate::ptx::PtxInfo;

/// An SM architecture target such as `sm_90a`, `sm_100f` or `compute_80`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Target {
    major: u32,
    minor: u32,
    /// `a` (arch-specific), `f` (family) or none.
    suffix: Option<char>,
}

impl Target {
    fn parse(arch: &str) -> Option<Self> {
        let digits = arch
            .strip_prefix("sm_")
            .or_else(|| arch.strip_prefix("compute_"))?;
        let (number, suffix) = match digits.strip_suffix(['a', 'f']) {
            Some(number) => (number, digits.chars().last()),
            None => (digits, None),
        };
        let number: u32 = number.parse().ok()?;
        Some(Self {
            major: number / 10,
            minor: number % 10,
            suffix,
        })
    }

    /// Whether a cubin built for this target loads on `device`.
    ///
    /// Plain and family cubins run on the same major version with an equal
    /// or newer minor (sm_100 and sm_100f on sm_103, sm_120f on sm_121);
    /// arch-specific ones only on exactly that architecture.
    fn cubin_runs_on(self, device: Device) -> bool {
        match self.suffix {
            Some('a') => (self.major, self.minor) == (device.major, device.minor),
            _ => self.major == device.major && self.minor <= device.minor,
        }
    }

    /// Whether PTX for this target can be JIT-compiled on `device`.
    fn ptx_runs_on(self, device: Device) -> bool {
        match self.suffix {
            None => (self.major, self.minor) <= (device.major, device.minor),
            _ => self.cubin_runs_on(device),
        }
    }
}

/// A physical GPU architecture, e.g. `sm_103` or `10.3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    major: u32,
    minor: u32,
}

impl FromStr for Device {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || format!("expected an architecture like sm_103 or 10.3, got '{s}'");
        let number = match s.split_once('.') {
            Some((major, minor)) if minor.len() == 1 => format!("{major}{minor}"),
            Some(_) => return Err(err()),
            None => s.strip_prefix("sm_").unwrap_or(s).to_string(),
        };
        // At least a one-digit major and a minor, e.g. 75, 103
        if number.len() < 2 || !number.bytes().all(|b| b.is_ascii_digit()) {
            return Err(err());
        }
        let number: u32 = number.parse().map_err(|_| err())?;
        Ok(Self {
            major: number / 10,
            minor: number % 10,
        })
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm_{}{}", self.major, self.minor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    /// Only PTX matches: JIT-compiled when the kernel is first loaded.
    PtxJit,
    /// Neither a cubin nor PTX can run on the device.
    Missing,
}

#[derive(Debug)]
pub struct DeviceReport {
    pub device: Device,
    /// Kernels with a loadable cubin.
    pub cubin: usize,
    /// Kernels without one, with their total code size across archs.
    pub problems: Vec<(Status, String, u64)>,
}

/// Classify every kernel, from cubins or PTX, whose name matches the filter.
pub fn check(
    devices: &[Device],
    cubins: &[CubinInfo],
    ptx: &[PtxInfo],
    name_filter: Option<&Regex>,
) -> Vec<DeviceReport> {
    #[derive(Default)]
    struct Kernel {
        cubin: BTreeSet<String>,
        ptx: BTreeSet<String>,
        size: u64,
    }
    let mut kernels: BTreeMap<&str, Kernel> = BTreeMap::new();
    for cubin in cubins {
        for kernel in &cubin.kernels {
            let entry = kernels.entry(&kernel.name).or_default();
            entry.cubin.insert(cubin.arch.clone());
            entry.size += kernel.size;
        }
    }
    for file in ptx {
        for entry in &file.entries {
            kernels
                .entry(entry)
                .or_default()
                .ptx
                .insert(file.target.clone());
        }
    }
    kernels.retain(|name, _| name_filter.is_none_or(|re| re.is_match(name)));

    let runs = |targets: &BTreeSet<String>, device, f: fn(Target, Device) -> bool| {
        targets
            .iter()
            .filter_map(|t| Target::parse(t))
            .any(|t| f(t, device))
    };
    devices
        .iter()
        .map(|&device| {
            let mut report = DeviceReport {
                device,
                cubin: 0,
                problems: Vec::new(),
            };
            for (name, kernel) in &kernels {
                if runs(&kernel.cubin, device, Target::cubin_runs_on) {
                    report.cubin += 1;
                } else {
                    let status = if runs(&kernel.ptx, device, Target::ptx_runs_on) {
                        Status::PtxJit
                    } else {
                        Status::Missing
                    };
                    report
                        .problems
                        .push((status, name.to_string(), kernel.size));
                }
            }
            // Missing first, then by size
            report
                .problems
                .sort_by(|a, b| b.0.cmp(&a.0).then(b.2.cmp(&a.2)).then(a.1.cmp(&b.1)));
            report
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubin::{KernelInfo, Usage};

    fn dev(s: &str) -> Device {
        s.parse().unwrap()
    }

    fn cubin_ok(target: &str, device: &str) -> bool {
        Target::parse(target).unwrap().cubin_runs_on(dev(device))
    }

    fn ptx_ok(target: &str, device: &str) -> bool {
        Target::parse(target).unwrap().ptx_runs_on(dev(device))
    }

    #[test]
    fn device_parsing() {
        assert_eq!(dev("sm_103"), dev("10.3"));
        assert_eq!(dev("121").to_string(), "sm_121");
        assert_eq!(dev("8.9").to_string(), "sm_89");
        assert!("sm_100a".parse::<Device>().is_err());
        assert!("10.30".parse::<Device>().is_err());
        assert!("8".parse::<Device>().is_err());
        assert!("sm_1".parse::<Device>().is_err());
        assert!("+90".parse::<Device>().is_err());
    }

    #[test]
    fn cubin_compatibility() {
        assert!(cubin_ok("sm_100", "sm_103"));
        assert!(cubin_ok("sm_100f", "sm_103"));
        assert!(!cubin_ok("sm_100a", "sm_103"));
        assert!(cubin_ok("sm_100a", "sm_100"));
        assert!(cubin_ok("sm_120f", "sm_121"));
        assert!(!cubin_ok("sm_90", "sm_100"));
        assert!(!cubin_ok("sm_103", "sm_100"));
    }

    #[test]
    fn ptx_compatibility() {
        assert!(ptx_ok("compute_80", "sm_103"));
        assert!(ptx_ok("compute_90", "sm_90"));
        assert!(!ptx_ok("compute_100", "sm_90"));
        assert!(!ptx_ok("compute_90a", "sm_100"));
        assert!(ptx_ok("compute_100f", "sm_103"));
    }

    #[test]
    fn classifies_kernels_per_device() {
        let kernel = |name: &str| KernelInfo {
            name: name.into(),
            size: 100,
            usage: Usage::default(),
        };
        let cubins = vec![
            CubinInfo {
                arch: "sm_100".into(),
                kernels: vec![kernel("generic")],
                ..Default::default()
            },
            CubinInfo {
                arch: "sm_100a".into(),
                kernels: vec![kernel("tcgen05"), kernel("jit")],
                ..Default::default()
            },
        ];
        let ptx = vec![PtxInfo {
            target: "compute_80".into(),
            entries: vec!["jit".into(), "ptx_only".into()],
        }];
        let reports = check(&[dev("sm_103")], &cubins, &ptx, None);
        let r = &reports[0];
        assert_eq!(r.cubin, 1);
        let problems: Vec<_> = r
            .problems
            .iter()
            .map(|(s, n, _)| (*s, n.as_str()))
            .collect();
        assert_eq!(
            problems,
            [
                (Status::Missing, "tcgen05"),
                (Status::PtxJit, "jit"),
                (Status::PtxJit, "ptx_only")
            ]
        );
    }
}
