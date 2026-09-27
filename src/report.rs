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

use crate::cubin::{CubinInfo, Sections};

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

#[derive(Debug)]
pub struct Kernel {
    pub name: String,
    /// Code bytes summed over all architectures.
    pub size: u64,
    pub by_arch: BTreeMap<String, u64>,
}

#[derive(Debug)]
pub struct Report {
    pub file: String,
    pub sections: Sections,
    pub archs: Vec<ArchSummary>,
    /// Sorted by size, largest first.
    pub kernels: Vec<Kernel>,
}

impl Report {
    /// Build a report from cubins whose kernel names are already demangled.
    pub fn build(
        file: String,
        cubins: &[CubinInfo],
        arch: Option<&str>,
        name_filter: Option<&Regex>,
    ) -> Self {
        let mut sections = Sections::default();
        let mut arch_sections: BTreeMap<&str, Sections> = BTreeMap::new();
        let mut kernels: HashMap<&str, BTreeMap<String, u64>> = HashMap::new();

        for cubin in cubins {
            if arch.is_some_and(|a| a != cubin.arch) {
                continue;
            }
            sections += &cubin.sections;
            *arch_sections.entry(&cubin.arch).or_default() += &cubin.sections;
            for (name, size) in &cubin.kernels {
                if name_filter.is_none_or(|re| re.is_match(name)) {
                    *kernels
                        .entry(name)
                        .or_default()
                        .entry(cubin.arch.clone())
                        .or_default() += size;
                }
            }
        }

        let archs = arch_sections
            .into_iter()
            .map(|(arch, secs)| {
                let sizes = kernels.values().filter_map(|by_arch| by_arch.get(arch));
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
                size: by_arch.values().sum(),
                by_arch,
            })
            .collect();
        kernels.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.name.cmp(&b.name)));

        Report {
            file,
            sections,
            archs,
            kernels,
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
                        .map(|(arch, &size)| {
                            let size_human = human_size(size);
                            (arch.as_str(), JsonSize { size, size_human })
                        })
                        .collect(),
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
        writeln!(out, "{}\n{table}\n", layout.title(&heading))?;

        let mut table = new_table(layout, &[Label("Section"), Num("Size"), Num("%")]);
        for (kind, size) in self.sections.ranked() {
            table.add_row(vec![
                label_cell(kind.label()),
                size_cell(size),
                percent_cell(size, total),
            ]);
        }
        writeln!(out, "{}\n{table}\n", layout.title("Sections"))?;

        let rows: Vec<KernelRow> = self
            .kernels
            .iter()
            .map(|k| KernelRow {
                name: &k.name,
                size: k.size,
                archs: Some(k.by_arch.len()),
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
                    .filter_map(|k| {
                        Some(KernelRow {
                            name: &k.name,
                            size: *k.by_arch.get(&arch.arch)?,
                            archs: None,
                        })
                    })
                    .collect();
                // Kernels are ranked by their total size; re-rank for this arch
                rows.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.name.cmp(b.name)));
                let shown = PER_ARCH_TOP.min(rows.len());
                let heading = format!("Top kernels for {} ({shown} of {})", arch.arch, rows.len());
                let table = kernel_table(layout, &rows, shown, arch.code);
                writeln!(out, "\n{}\n{table}", layout.title(&heading))?;
            }
        }
        Ok(())
    }
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
}

struct KernelRow<'a> {
    name: &'a str,
    size: u64,
    /// Number of architectures, shown only in the cross-arch table.
    archs: Option<usize>,
}

/// Table of the first `shown` rows, with sizes as a share of `whole`.
fn kernel_table(layout: Layout, rows: &[KernelRow], shown: usize, whole: u64) -> Table {
    let show_archs = rows.iter().any(|r| r.archs.is_some());
    let mut columns = vec![Num("#"), Label("Kernel")];
    if show_archs {
        columns.push(Num("Archs"));
    }
    columns.extend([Num("Size"), Num("% of code")]);
    let mut table = new_table(layout, &columns);

    // In a terminal, shrink only the name column so each kernel fits on one
    // line; otherwise truncate names to a fixed length
    let fit_width = layout.width.filter(|_| !layout.full_names);
    if let Some(width) = fit_width {
        table
            .set_width(width)
            .set_content_arrangement(ContentArrangement::Dynamic);
        for (i, column) in table.column_iter_mut().enumerate() {
            if i != 1 {
                column.set_constraint(ColumnConstraint::ContentWidth);
            }
        }
    }

    for (rank, row) in rows[..shown].iter().enumerate() {
        let name = if layout.full_names || fit_width.is_some() {
            row.name.to_string()
        } else {
            truncate(row.name, DEFAULT_NAME_WIDTH)
        };
        let mut cells = vec![
            Cell::new(rank + 1).add_attribute(Attribute::Dim),
            label_cell(name),
        ];
        cells.extend(row.archs.map(count_cell));
        cells.extend([size_cell(row.size), percent_cell(row.size, whole)]);
        let mut row = Row::from(cells);
        row.max_height(1);
        table.add_row(row);
    }
    if shown < rows.len() {
        table.add_row(vec![
            Cell::new("..."),
            Cell::new(format!("({} more kernels)", rows.len() - shown))
                .add_attribute(Attribute::Dim),
        ]);
    }
    table
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
    architectures: Vec<JsonArch<'a>>,
    sections: Vec<JsonSection>,
    kernels: Vec<JsonKernel<'a>>,
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
    by_arch: BTreeMap<&'a str, JsonSize>,
}

#[derive(Serialize)]
struct JsonSize {
    size: u64,
    size_human: String,
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
            kernels: kernels.iter().map(|&(n, s)| (n.to_string(), s)).collect(),
            sections,
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
        let report = Report::build("lib.so".into(), &sample(), None, None);
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
        let report = Report::build("lib.so".into(), &sample(), Some("sm_100"), Some(&re));
        assert_eq!(report.kernels.len(), 1);
        assert_eq!(report.kernels[0].size, 200);
        // Section totals ignore the name filter but honor the arch filter
        assert_eq!(report.sections.total(), 620);
        assert_eq!(report.archs.len(), 1);
    }

    #[test]
    fn json_has_raw_and_human_sizes() {
        let report = Report::build("lib.so".into(), &sample(), None, None);
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
        let report = Report::build("lib.so".into(), &sample(), None, None);
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
