//! Aggregation of per-cubin results and rendering.

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};

use bytesize::ByteSize;
use comfy_table::presets::UTF8_FULL_CONDENSED;
use comfy_table::{
    Attribute, Cell, CellAlignment, Color, ColumnConstraint, ContentArrangement, Row, Table,
};
use regex::Regex;
use serde::Serialize;

use crate::availability::{DeviceReport, Status};
use crate::cubin::{CubinInfo, KernelInfo, Sections, Usage};

/// Kernels listed in each per-architecture table.
const PER_ARCH_TOP: usize = 15;
/// Name length used when there is no terminal width to fit.
const DEFAULT_NAME_WIDTH: usize = 100;

#[derive(Debug)]
pub struct ArchSummary {
    pub arch: String,
    /// Kernels matching the name filter.
    pub kernels: usize,
    /// Code bytes of those kernels.
    pub code: u64,
    /// All section bytes for this architecture, regardless of name filter.
    pub total: u64,
}

/// A kernel's code size and resources on one architecture.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArchStats {
    pub size: u64,
    pub usage: Usage,
}

impl ArchStats {
    /// Merge another copy of the kernel, e.g. from a second cubin of the
    /// same architecture: sizes add up, resources take the larger value.
    fn add(&mut self, kernel: &KernelInfo) {
        self.size += kernel.size;
        let (u, k) = (&mut self.usage, kernel.usage);
        u.registers = u.registers.max(k.registers);
        u.max_registers = u.max_registers.max(k.max_registers);
        u.stack = u.stack.max(k.stack);
        u.shared = u.shared.max(k.shared);
    }
}

#[derive(Debug)]
pub struct Kernel {
    pub name: String,
    /// Code bytes summed over all architectures.
    pub size: u64,
    pub by_arch: BTreeMap<String, ArchStats>,
}

/// What kernel lists are ordered by, largest first.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SortKey {
    /// Code size
    #[default]
    Size,
    /// Registers per thread
    Regs,
    /// Stack (local memory) per thread
    Stack,
}

impl SortKey {
    fn of(self, stats: &ArchStats) -> u64 {
        match self {
            Self::Size => stats.size,
            Self::Regs => stats.usage.registers.unwrap_or(0).into(),
            Self::Stack => stats.usage.stack.unwrap_or(0).into(),
        }
    }

    /// Key across architectures: total size, or the largest resource value.
    fn of_kernel(self, kernel: &Kernel) -> u64 {
        match self {
            Self::Size => kernel.size,
            _ => kernel
                .by_arch
                .values()
                .map(|s| self.of(s))
                .max()
                .unwrap_or(0),
        }
    }
}

#[derive(Debug)]
pub struct Report {
    pub file: String,
    pub sections: Sections,
    pub archs: Vec<ArchSummary>,
    /// Ordered by `sort`, largest first.
    pub kernels: Vec<Kernel>,
    /// Non-kernel functions, summed over cubins; not name-filtered.
    pub device_functions: usize,
    pub device_code: u64,
    pub sort: SortKey,
    /// Availability on the GPUs requested with `--device`.
    pub devices: Vec<DeviceReport>,
}

impl Report {
    /// Build a report from cubins whose kernel names are already demangled.
    pub fn build(
        file: String,
        cubins: &[CubinInfo],
        arch: Option<&str>,
        name_filter: Option<&Regex>,
        sort: SortKey,
    ) -> Self {
        let mut sections = Sections::default();
        let mut arch_sections: BTreeMap<&str, Sections> = BTreeMap::new();
        let mut kernels: HashMap<&str, BTreeMap<String, ArchStats>> = HashMap::new();
        let (mut device_functions, mut device_code) = (0, 0);

        for cubin in cubins {
            if arch.is_some_and(|a| a != cubin.arch) {
                continue;
            }
            sections += &cubin.sections;
            device_functions += cubin.device_functions;
            device_code += cubin.device_code;
            *arch_sections.entry(&cubin.arch).or_default() += &cubin.sections;
            for kernel in &cubin.kernels {
                if name_filter.is_none_or(|re| re.is_match(&kernel.name)) {
                    kernels
                        .entry(&kernel.name)
                        .or_default()
                        .entry(cubin.arch.clone())
                        .or_default()
                        .add(kernel);
                }
            }
        }

        let archs = arch_sections
            .into_iter()
            .map(|(arch, secs)| {
                let sizes = kernels
                    .values()
                    .filter_map(|by_arch| Some(by_arch.get(arch)?.size));
                ArchSummary {
                    arch: arch.to_string(),
                    kernels: sizes.clone().count(),
                    code: sizes.sum(),
                    total: secs.total(),
                }
            })
            .collect();

        let mut kernels: Vec<Kernel> = kernels
            .into_iter()
            .map(|(name, by_arch)| Kernel {
                name: name.to_string(),
                size: by_arch.values().map(|s| s.size).sum(),
                by_arch,
            })
            .collect();
        kernels.sort_by(|a, b| {
            (sort.of_kernel(b), b.size)
                .cmp(&(sort.of_kernel(a), a.size))
                .then_with(|| a.name.cmp(&b.name))
        });

        Report {
            file,
            sections,
            archs,
            kernels,
            device_functions,
            device_code,
            sort,
            devices: Vec::new(),
        }
    }

    fn kernel_code(&self) -> u64 {
        self.kernels.iter().map(|k| k.size).sum()
    }

    pub fn write_json(&self, out: &mut impl Write) -> io::Result<()> {
        let total = self.sections.total();
        let kernel_code = self.kernel_code();
        let json = JsonReport {
            file: &self.file,
            total_size: total,
            total_size_human: human_size(total),
            kernel_code_size: kernel_code,
            kernel_code_size_human: human_size(kernel_code),
            kernel_count: self.kernels.len(),
            device_function_count: self.device_functions,
            device_function_code_size: self.device_code,
            device_function_code_size_human: human_size(self.device_code),
            architectures: self
                .archs
                .iter()
                .map(|a| JsonArch {
                    arch: &a.arch,
                    kernel_count: a.kernels,
                    code_size: a.code,
                    code_size_human: human_size(a.code),
                    total_size: a.total,
                    total_size_human: human_size(a.total),
                    percent: json_percent(a.total, total),
                })
                .collect(),
            sections: self
                .sections
                .ranked()
                .into_iter()
                .map(|(kind, size)| JsonSection {
                    name: kind.label(),
                    size,
                    size_human: human_size(size),
                    percent: json_percent(size, total),
                })
                .collect(),
            kernels: self
                .kernels
                .iter()
                .map(|k| JsonKernel {
                    name: &k.name,
                    size: k.size,
                    size_human: human_size(k.size),
                    percent: json_percent(k.size, kernel_code),
                    by_arch: k
                        .by_arch
                        .iter()
                        .map(|(arch, stats)| {
                            let json = JsonArchStats {
                                size: stats.size,
                                size_human: human_size(stats.size),
                                registers: stats.usage.registers,
                                max_registers: stats.usage.max_registers,
                                stack: stats.usage.stack,
                                likely_spill: stats.usage.likely_spills(),
                                static_shared: stats.usage.shared,
                            };
                            (arch.as_str(), json)
                        })
                        .collect(),
                })
                .collect(),
            devices: self
                .devices
                .iter()
                .map(|d| {
                    let kernels = |status: Status| {
                        d.problems
                            .iter()
                            .filter(|(s, ..)| *s == status)
                            .map(|(_, name, _)| name.as_str())
                            .collect()
                    };
                    JsonDevice {
                        device: d.device.to_string(),
                        cubin_kernel_count: d.cubin,
                        ptx_jit: kernels(Status::PtxJit),
                        missing: kernels(Status::Missing),
                    }
                })
                .collect(),
        };
        serde_json::to_writer_pretty(&mut *out, &json)?;
        writeln!(out)
    }

    pub fn write_tables(&self, out: &mut impl Write, top: usize, layout: Layout) -> io::Result<()> {
        let total = self.sections.total();
        let kernel_code = self.kernel_code();

        let mut table = new_table(
            layout,
            &[
                Label("Architecture"),
                Num("Kernels"),
                Num("Code"),
                Num("Total"),
                Num("%"),
            ],
        );
        for a in &self.archs {
            table.add_row(vec![
                label_cell(&a.arch),
                count_cell(a.kernels),
                size_cell(a.code),
                size_cell(a.total),
                percent_cell(a.total, total),
            ]);
        }
        table.add_row(
            [
                "TOTAL".to_string(),
                self.kernels.len().to_string(),
                human_size(kernel_code),
                human_size(total),
                format_percent(total, total),
            ]
            .map(|text| Cell::new(text).add_attribute(Attribute::Bold)),
        );
        let heading = format!("Architectures: {}", self.file);
        writeln!(out, "{}\n{table}", layout.title(&heading))?;
        if self.device_functions > 0 {
            writeln!(
                out,
                "Not in kernel lists: {} device functions with their own symbols, {}",
                self.device_functions,
                human_size(self.device_code)
            )?;
        }
        writeln!(out)?;

        let mut table = new_table(layout, &[Label("Section"), Num("Size"), Num("%")]);
        for (kind, size) in self.sections.ranked() {
            table.add_row(vec![
                label_cell(kind.label()),
                size_cell(size),
                percent_cell(size, total),
            ]);
        }
        writeln!(out, "{}\n{table}\n", layout.title("Sections"))?;

        // With a single architecture, resources are unambiguous and the
        // arch count is always 1, so show the former instead of the latter
        let single_arch = self.archs.len() == 1;
        let rows: Vec<KernelRow> = self
            .kernels
            .iter()
            .map(|k| KernelRow {
                name: &k.name,
                size: k.size,
                archs: (!single_arch).then_some(k.by_arch.len()),
                usage: single_arch.then(|| k.by_arch.values().next().unwrap().usage),
            })
            .collect();
        let shown = top.min(rows.len());
        let heading = format!("Top kernels ({shown} of {})", rows.len());
        let table = kernel_table(layout, &rows, shown, kernel_code);
        writeln!(out, "{}\n{table}", layout.title(&heading))?;

        if self.archs.len() > 1 {
            for arch in &self.archs {
                let mut rows: Vec<KernelRow> = self
                    .kernels
                    .iter()
                    .filter_map(|k| Some((k.name.as_str(), *k.by_arch.get(&arch.arch)?)))
                    .map(|(name, stats)| KernelRow {
                        name,
                        size: stats.size,
                        archs: None,
                        usage: Some(stats.usage),
                    })
                    .collect();
                // Kernels are ranked across architectures; re-rank for this one
                let key = |r: &KernelRow| {
                    let stats = ArchStats {
                        size: r.size,
                        usage: r.usage.unwrap_or_default(),
                    };
                    (self.sort.of(&stats), r.size)
                };
                rows.sort_by(|a, b| key(b).cmp(&key(a)).then_with(|| a.name.cmp(b.name)));
                let shown = PER_ARCH_TOP.min(rows.len());
                let heading = format!("Top kernels for {} ({shown} of {})", arch.arch, rows.len());
                let table = kernel_table(layout, &rows, shown, arch.code);
                writeln!(out, "\n{}\n{table}", layout.title(&heading))?;
            }
        }

        for device in &self.devices {
            write_device(out, device, top, layout)?;
        }
        Ok(())
    }
}

fn write_device(
    out: &mut impl Write,
    device: &DeviceReport,
    top: usize,
    layout: Layout,
) -> io::Result<()> {
    let count = |status| {
        device
            .problems
            .iter()
            .filter(|(s, ..)| *s == status)
            .count()
    };
    let heading = format!(
        "Availability on {}: {} from cubin, {} need PTX JIT, {} missing",
        device.device,
        device.cubin,
        count(Status::PtxJit),
        count(Status::Missing),
    );
    writeln!(out, "\n{}", layout.title(&heading))?;
    if device.problems.is_empty() {
        return Ok(());
    }

    let mut table = new_table(layout, &[Label("Status"), Label("Kernel"), Num("Size")]);
    layout.fit_names(&mut table, 1);
    let shown = top.min(device.problems.len());
    for (status, name, size) in &device.problems[..shown] {
        let status = match status {
            Status::Missing => Cell::new("missing")
                .fg(Color::DarkRed)
                .add_attribute(Attribute::Bold),
            Status::PtxJit => Cell::new("PTX JIT").fg(Color::DarkYellow),
        };
        let name = label_cell(layout.name(name));
        table.add_row(one_line_row(vec![status, name, size_cell(*size)]));
    }
    add_overflow_row(&mut table, device.problems.len() - shown);
    writeln!(out, "{table}")
}

/// How table output is rendered.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub color: bool,
    pub full_names: bool,
    /// Terminal width to fit, if stdout is a terminal.
    pub width: Option<u16>,
}

impl Layout {
    fn title(&self, text: &str) -> String {
        if self.color {
            format!("\x1b[1;36m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    /// Whether names are fitted to the terminal width by comfy-table.
    fn fits_terminal(&self) -> bool {
        self.width.is_some() && !self.full_names
    }

    /// In a terminal, let only the name column shrink so each row fits on
    /// one line.
    fn fit_names(&self, table: &mut Table, name_column: usize) {
        let Some(width) = self.width.filter(|_| !self.full_names) else {
            return;
        };
        table
            .set_width(width)
            .set_content_arrangement(ContentArrangement::Dynamic);
        for (i, column) in table.column_iter_mut().enumerate() {
            if i != name_column {
                column.set_constraint(ColumnConstraint::ContentWidth);
            }
        }
    }

    /// A kernel name as displayed: whole, left to the terminal fitting, or
    /// truncated to a fixed length when there is no terminal.
    fn name(&self, name: &str) -> String {
        if self.full_names || self.fits_terminal() {
            name.to_string()
        } else {
            truncate(name, DEFAULT_NAME_WIDTH)
        }
    }
}

struct KernelRow<'a> {
    name: &'a str,
    size: u64,
    /// Number of architectures, shown only in the cross-arch table.
    archs: Option<usize>,
    /// Resources, shown only in per-architecture tables.
    usage: Option<Usage>,
}

/// Table of the first `shown` rows, with sizes as a share of `whole`.
fn kernel_table(layout: Layout, rows: &[KernelRow], shown: usize, whole: u64) -> Table {
    let show_archs = rows.iter().any(|r| r.archs.is_some());
    let show_usage = rows.iter().any(|r| r.usage.is_some());
    let mut columns = vec![Num("#"), Label("Kernel")];
    if show_archs {
        columns.push(Num("Archs"));
    }
    columns.extend([Num("Size"), Num("% of code")]);
    if show_usage {
        columns.extend([Num("Regs"), Num("Stack"), Num("Shared")]);
    }
    let mut table = new_table(layout, &columns);
    layout.fit_names(&mut table, 1);

    for (rank, row) in rows[..shown].iter().enumerate() {
        let mut cells = vec![
            Cell::new(rank + 1).add_attribute(Attribute::Dim),
            label_cell(layout.name(row.name)),
        ];
        cells.extend(row.archs.map(count_cell));
        cells.extend([size_cell(row.size), percent_cell(row.size, whole)]);
        if let Some(usage) = row.usage {
            cells.extend(usage_cells(usage));
        }
        table.add_row(one_line_row(cells));
    }
    add_overflow_row(&mut table, rows.len() - shown);
    table
}

fn one_line_row(cells: Vec<Cell>) -> Row {
    let mut row = Row::from(cells);
    row.max_height(1);
    row
}

/// Note the kernels left out of a table, if any.
fn add_overflow_row(table: &mut Table, hidden: usize) {
    if hidden > 0 {
        table.add_row(vec![
            Cell::new("..."),
            Cell::new(format!("({hidden} more kernels)")).add_attribute(Attribute::Dim),
        ]);
    }
}

enum Column {
    Label(&'static str),
    Num(&'static str),
}
use Column::{Label, Num};

fn new_table(layout: Layout, columns: &[Column]) -> Table {
    let mut table = Table::new();
    table
        .load_style(UTF8_FULL_CONDENSED.with_rounded_corners())
        .set_content_arrangement(ContentArrangement::Disabled);
    if layout.color {
        table.enforce_styling();
    } else {
        table.force_no_tty();
    }
    table.set_header(columns.iter().map(|c| {
        let (Label(name) | Num(name)) = c;
        Cell::new(name).add_attribute(Attribute::Bold)
    }));
    for (i, c) in columns.iter().enumerate() {
        if matches!(c, Num(_))
            && let Some(column) = table.column_mut(i)
        {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }
    table
}

// Standard ANSI colors (not the bright variants) so terminal themes can keep
// them readable on both light and dark backgrounds.

fn label_cell(text: impl ToString) -> Cell {
    Cell::new(text).fg(Color::DarkCyan)
}

fn count_cell(count: usize) -> Cell {
    Cell::new(count).fg(Color::DarkMagenta)
}

fn size_cell(bytes: u64) -> Cell {
    Cell::new(human_size(bytes))
        .fg(Color::DarkYellow)
        .add_attribute(Attribute::Bold)
}

/// Registers, stack and static shared memory. The stack is highlighted when
/// the kernel most likely spills, i.e. its registers are at the limit.
fn usage_cells(usage: Usage) -> [Cell; 3] {
    let optional = |value: Option<String>| value.unwrap_or_else(|| "-".into());
    let stack = Cell::new(optional(usage.stack.map(|b| human_size(b.into()))));
    let stack = if usage.likely_spills() {
        stack.fg(Color::DarkRed).add_attribute(Attribute::Bold)
    } else {
        stack
    };
    [
        Cell::new(optional(usage.registers.map(|r| r.to_string()))),
        stack,
        Cell::new(human_size(usage.shared)),
    ]
}

fn percent_cell(part: u64, whole: u64) -> Cell {
    Cell::new(format_percent(part, whole)).fg(Color::DarkGreen)
}

#[derive(Serialize)]
struct JsonReport<'a> {
    file: &'a str,
    total_size: u64,
    total_size_human: String,
    kernel_code_size: u64,
    kernel_code_size_human: String,
    kernel_count: usize,
    /// Functions that are not kernels, e.g. device functions in -rdc builds.
    device_function_count: usize,
    device_function_code_size: u64,
    device_function_code_size_human: String,
    architectures: Vec<JsonArch<'a>>,
    sections: Vec<JsonSection>,
    kernels: Vec<JsonKernel<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    devices: Vec<JsonDevice<'a>>,
}

#[derive(Serialize)]
struct JsonDevice<'a> {
    device: String,
    /// Kernels with a cubin that loads on the device.
    cubin_kernel_count: usize,
    /// Kernels that only run after JIT-compiling their PTX.
    ptx_jit: Vec<&'a str>,
    /// Kernels with no code that can run on the device.
    missing: Vec<&'a str>,
}

#[derive(Serialize)]
struct JsonArch<'a> {
    arch: &'a str,
    kernel_count: usize,
    code_size: u64,
    code_size_human: String,
    total_size: u64,
    total_size_human: String,
    /// Share of the total size across all architectures.
    percent: f64,
}

#[derive(Serialize)]
struct JsonSection {
    name: &'static str,
    size: u64,
    size_human: String,
    percent: f64,
}

#[derive(Serialize)]
struct JsonKernel<'a> {
    name: &'a str,
    size: u64,
    size_human: String,
    /// Share of all kernel code.
    percent: f64,
    by_arch: BTreeMap<&'a str, JsonArchStats>,
}

#[derive(Serialize)]
struct JsonArchStats {
    size: u64,
    size_human: String,
    /// Per thread.
    registers: Option<u32>,
    /// Per-thread register limit set at compile time.
    max_registers: Option<u32>,
    /// Local memory per thread, in bytes: spills, local arrays or calls.
    stack: Option<u32>,
    /// Registers at the limit with a non-zero stack.
    likely_spill: bool,
    /// Shared memory per block allocated at compile time, in bytes,
    /// including any system reservation; excludes dynamic shared memory.
    static_shared: u64,
}

fn human_size(bytes: u64) -> String {
    ByteSize::b(bytes).display().iec().to_string()
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 * 100.0 / whole as f64
    }
}

fn format_percent(part: u64, whole: u64) -> String {
    format!("{:.1}%", percent(part, whole))
}

/// Percentage rounded to two decimals.
fn json_percent(part: u64, whole: u64) -> f64 {
    (percent(part, whole) * 100.0).round() / 100.0
}

fn truncate(name: &str, max_chars: usize) -> String {
    match name.char_indices().nth(max_chars.saturating_sub(3)) {
        Some((cut, _)) if name.chars().count() > max_chars => format!("{}...", &name[..cut]),
        _ => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubin::SectionKind;

    fn cubin(arch: &str, kernels: &[(&str, u64)], code: u64) -> CubinInfo {
        let mut sections = Sections::default();
        sections.add(SectionKind::Code, code);
        sections.add(SectionKind::Metadata, 10);
        CubinInfo {
            arch: arch.into(),
            kernels: kernels
                .iter()
                .map(|&(name, size)| KernelInfo {
                    name: name.to_string(),
                    size,
                    usage: Usage::default(),
                })
                .collect(),
            sections,
            ..Default::default()
        }
    }

    fn sample() -> Vec<CubinInfo> {
        vec![
            cubin("sm_90a", &[("gemm", 300), ("norm", 50)], 350),
            cubin("sm_100", &[("gemm", 200)], 200),
            cubin("sm_100", &[("topk", 400)], 400),
        ]
    }

    #[test]
    fn merges_kernels_across_archs() {
        let report = Report::build("lib.so".into(), &sample(), None, None, SortKey::Size);
        let names: Vec<_> = report
            .kernels
            .iter()
            .map(|k| (k.name.as_str(), k.size))
            .collect();
        assert_eq!(names, [("gemm", 500), ("topk", 400), ("norm", 50)]);
        assert_eq!(report.kernels[0].by_arch.len(), 2);
        assert_eq!(report.sections.total(), 950 + 30);
        let sm100 = &report.archs[0];
        assert_eq!(
            (sm100.arch.as_str(), sm100.kernels, sm100.code, sm100.total),
            ("sm_100", 2, 600, 620)
        );
    }

    #[test]
    fn arch_and_name_filters() {
        let re = Regex::new("(?i)GEMM").unwrap();
        let report = Report::build(
            "lib.so".into(),
            &sample(),
            Some("sm_100"),
            Some(&re),
            SortKey::Size,
        );
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.kernels[0].size, 200);
        // Section totals ignore the name filter but honor the arch filter
        assert_eq!(report.sections.total(), 620);
        assert_eq!(report.archs.len(), 1);
    }

    #[test]
    fn json_has_raw_and_human_sizes() {
        let report = Report::build("lib.so".into(), &sample(), None, None, SortKey::Size);
        let mut buf = Vec::new();
        report.write_json(&mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(json["kernel_count"], 3);
        let gemm = &json["kernels"][0];
        assert_eq!(gemm["size"], 500);
        assert_eq!(gemm["size_human"], "500 B");
        assert_eq!(gemm["percent"], 52.63);
        assert_eq!(gemm["by_arch"]["sm_90a"]["size"], 300);
    }

    #[test]
    fn tables_render_per_arch_breakdown() {
        let report = Report::build("lib.so".into(), &sample(), None, None, SortKey::Size);
        let layout = Layout {
            color: false,
            full_names: false,
            width: None,
        };
        let mut buf = Vec::new();
        report.write_tables(&mut buf, 2, layout).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("Top kernels (2 of 3)"));
        assert!(text.contains("(1 more kernels)"));
        assert!(text.contains("Top kernels for sm_100 (2 of 2)"));
        assert!(text.contains("Top kernels for sm_90a (2 of 2)"));
        assert!(!text.contains('\x1b'));
    }

    #[test]
    fn sizes_use_iec_units() {
        assert_eq!(human_size(390_568_592), "372.5 MiB");
    }

    #[test]
    fn percent_formatting() {
        assert_eq!(json_percent(1, 3), 33.33);
        assert_eq!(format_percent(1, 3), "33.3%");
        assert_eq!(format_percent(1, 0), "0.0%");
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        assert_eq!(truncate("abcdefgh", 6), "abc...");
        assert_eq!(truncate("abcdef", 6), "abcdef");
        assert_eq!(truncate("αβγδεζη", 6), "αβγ...");
    }
}
