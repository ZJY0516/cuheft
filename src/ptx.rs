//! Kernel entry points in PTX.

use std::path::Path;

use anyhow::{Context, Result};

/// Entry points of one PTX module, names still mangled.
#[derive(Debug, Clone)]
pub struct PtxInfo {
    /// e.g. `compute_80`, `compute_90a`.
    pub target: String,
    pub entries: Vec<String>,
}

pub fn analyze(path: &Path) -> Result<PtxInfo> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(parse(&text))
}

fn parse(text: &str) -> PtxInfo {
    let mut target = String::from("unknown");
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.split("//").next().unwrap_or_default();
        let mut tokens = line.split([' ', '\t', ',', '(']).filter(|t| !t.is_empty());
        // Directives only: `.target sm_90a[, debug]`, `[.visible|.weak] .entry name(`
        match tokens.next() {
            Some(".target") => {
                if let Some(arch) = tokens.next() {
                    target = arch.replace("sm_", "compute_");
                }
            }
            Some(first) if first.starts_with('.') => {
                // Skip linkage modifiers such as .visible or .weak
                let mut directive = first;
                while directive != ".entry" && directive.starts_with('.') {
                    match tokens.next() {
                        Some(token) => directive = token,
                        None => break,
                    }
                }
                if directive == ".entry"
                    && let Some(name) = tokens.next()
                {
                    entries.push(name.to_string());
                }
            }
            _ => {}
        }
    }
    PtxInfo { target, entries }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_and_entries() {
        let ptx = parse(
            "//\n.version 9.0\n.target sm_90a\n.address_size 64\n\n\
             .visible .entry _Z6kernelPf(\n\t.param .u64 p\n)\n{\n}\n\
             .entry plain_c_kernel (\n)\n.visible .func helper()\n\
             // .entry not_a_kernel(\n\tmov.u32 %r1, 0; // .entry x\n",
        );
        assert_eq!(ptx.target, "compute_90a");
        assert_eq!(ptx.entries, ["_Z6kernelPf", "plain_c_kernel"]);
    }
}
