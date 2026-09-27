mod cubin;
mod demangle;
mod extract;
mod report;

use std::collections::{HashMap, HashSet};
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use regex::RegexBuilder;

use crate::cubin::CubinInfo;
use crate::report::{Layout, Report};

/// Size profiler for CUDA binaries: per-kernel code size by SM architecture.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Shared library or cubin to analyze
    file: PathBuf,

    /// Number of kernels to list in table output
    #[arg(short = 'n', long, default_value_t = 30)]
    top: usize,

    /// Only include this architecture (e.g. sm_90a, sm_100f)
    #[arg(short, long)]
    arch: Option<String>,

    /// Only include kernels whose name matches this regex (case-insensitive)
    #[arg(short = 'r', long, value_name = "REGEX")]
    filter: Option<String>,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = Format::Table)]
    format: Format,

    /// Do not truncate kernel names
    #[arg(long)]
    full_names: bool,

    /// When to color table output
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto, value_name = "WHEN")]
    color: ColorChoice,

    /// Print progress information to stderr
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Table,
    Json,
}

#[derive(Clone, Copy, ValueEnum)]
enum ColorChoice {
    Auto,
    Always,
    Never,
}

impl ColorChoice {
    fn enabled(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    env_logger::Builder::new()
        .filter_level(if cli.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Warn
        })
        .format_timestamp(None)
        .format_target(false)
        .init();

    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        // Output piped into e.g. `head` that exits early
        Err(e) if is_broken_pipe(&e) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let name_filter = cli
        .filter
        .as_deref()
        .map(|p| RegexBuilder::new(p).case_insensitive(true).build())
        .transpose()
        .context("invalid --filter regex")?;

    let found = extract::collect(&cli.file)?;
    let mut cubins: Vec<CubinInfo> = found
        .paths
        .par_iter()
        .filter_map(|path| {
            cubin::analyze(path)
                .inspect_err(|e| log::warn!("skipping {}: {e:#}", path.display()))
                .ok()
        })
        .collect();
    drop(found);
    if cubins.is_empty() {
        bail!("no cubins could be analyzed in {}", cli.file.display());
    }
    log::debug!("parsed {} cubin(s)", cubins.len());

    if let Some(arch) = &cli.arch
        && !cubins.iter().any(|c| &c.arch == arch)
    {
        let mut available: Vec<&str> = cubins.iter().map(|c| c.arch.as_str()).collect();
        available.sort_unstable();
        available.dedup();
        bail!(
            "architecture '{arch}' not found (available: {})",
            available.join(", ")
        );
    }

    demangle_kernels(&mut cubins);

    let report = Report::build(
        cli.file.display().to_string(),
        &cubins,
        cli.arch.as_deref(),
        name_filter.as_ref(),
    );
    let mut out = io::stdout().lock();
    match cli.format {
        Format::Json => report.write_json(&mut out)?,
        Format::Table => {
            let layout = Layout {
                color: cli.color.enabled(),
                full_names: cli.full_names,
                width: terminal_width(),
            };
            report.write_tables(&mut out, cli.top, layout)?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Demangle kernel names once per unique symbol, in parallel.
fn demangle_kernels(cubins: &mut [CubinInfo]) {
    let unique: HashSet<&str> = cubins
        .iter()
        .flat_map(|c| c.kernels.iter().map(|(name, _)| name.as_str()))
        .collect();
    let names: HashMap<String, String> = unique
        .into_par_iter()
        .map(|name| (name.to_string(), demangle::kernel_name(name)))
        .collect();
    for (name, _) in cubins.iter_mut().flat_map(|c| &mut c.kernels) {
        if let Some(demangled) = names.get(name.as_str()) {
            name.clone_from(demangled);
        }
    }
}

/// Width of the terminal on stdout, or `None` when output is redirected.
fn terminal_width() -> Option<u16> {
    // comfy-table performs the tty check and size query
    comfy_table::Table::new().width()
}

fn is_broken_pipe(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<io::Error>()
            .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
    })
}
