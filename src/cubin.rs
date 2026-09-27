//! Parsing of a single cubin (CUDA ELF) file.

use std::collections::HashMap;
use std::ops::AddAssign;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use object::{Object, ObjectSection, ObjectSymbol, SymbolKind};
use regex::Regex;
use regex::bytes::Regex as BytesRegex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    Code,
    /// Blackwell+ Mercury sections: a second encoding of each kernel's code
    /// (`.nv.capmerc.text.*`) plus its relocations, info and debug data.
    Mercury,
    Metadata,
    Data,
    Debug,
}

impl SectionKind {
    pub const ALL: [Self; 5] = [
        Self::Code,
        Self::Mercury,
        Self::Metadata,
        Self::Data,
        Self::Debug,
    ];

    pub fn of(name: &str) -> Option<Self> {
        let starts = |prefixes: &[&str]| prefixes.iter().any(|p| name.starts_with(p));
        if name.starts_with(".text.") {
            Some(Self::Code)
        } else if starts(&[".nv.capmerc.", ".nv.merc."]) {
            Some(Self::Mercury)
        } else if starts(&[".debug_", ".nv_debug_", ".nv.debug_"]) {
            Some(Self::Debug)
        } else if starts(&[".nv.shared.", ".nv.constant", ".nv.global"]) {
            Some(Self::Data)
        } else if matches!(name, ".symtab" | ".strtab" | ".shstrtab")
            || starts(&[".nv.info", ".rela."])
        {
            Some(Self::Metadata)
        } else {
            None
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Code => "Code",
            Self::Mercury => "Mercury (capmerc)",
            Self::Metadata => "Metadata",
            Self::Data => "Data",
            Self::Debug => "Debug Info",
        }
    }
}

/// Bytes per section kind.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sections([u64; SectionKind::ALL.len()]);

impl Sections {
    pub fn get(&self, kind: SectionKind) -> u64 {
        self.0[kind as usize]
    }

    pub fn add(&mut self, kind: SectionKind, bytes: u64) {
        self.0[kind as usize] += bytes;
    }

    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }

    /// Non-empty sections, largest first.
    pub fn ranked(&self) -> Vec<(SectionKind, u64)> {
        let mut sections: Vec<_> = SectionKind::ALL
            .into_iter()
            .map(|kind| (kind, self.get(kind)))
            .filter(|&(_, bytes)| bytes > 0)
            .collect();
        sections.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
        sections
    }
}

impl AddAssign<&Sections> for Sections {
    fn add_assign(&mut self, other: &Sections) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a += b;
        }
    }
}

/// Sizes of one cubin.
#[derive(Debug, Default)]
pub struct CubinInfo {
    pub arch: String,
    /// Kernel names (mangled as parsed) with their code size.
    pub kernels: Vec<(String, u64)>,
    pub sections: Sections,
}

pub fn analyze(path: &Path) -> Result<CubinInfo> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let file =
        object::File::parse(&*data).with_context(|| format!("parsing ELF {}", path.display()))?;

    let mut sections = Sections::default();
    let mut tkinfo: &[u8] = &[];
    for section in file.sections() {
        let Ok(name) = section.name() else { continue };
        if name == ".note.nv.tkinfo" {
            tkinfo = section.data().unwrap_or_default();
        }
        // NOBITS sections such as .nv.shared.* only declare a size and take
        // no space in the file
        if let (Some(kind), Some((_, bytes))) = (SectionKind::of(name), section.file_range()) {
            sections.add(kind, bytes);
        }
    }

    let funcs = file.symbols().filter_map(|sym| {
        if sym.kind() != SymbolKind::Text {
            return None;
        }
        Some(Function {
            name: sym.name().ok()?.to_string(),
            section: sym.section_index()?.0,
            offset: sym.address(),
            size: sym.size(),
        })
    });

    Ok(CubinInfo {
        arch: detect_arch(tkinfo, path),
        kernels: fold_functions(funcs),
        sections,
    })
}

/// True SM architecture of a cubin.
///
/// cuobjdump names extracted cubins without the family suffix (an sm_100f
/// cubin appears as `*.sm_100.cubin`) and the ELF flags are identical, so
/// the authoritative source is the ptxas command line recorded in the
/// `.note.nv.tkinfo` section (`-arch sm_100f`). Falls back to the filename.
fn detect_arch(tkinfo: &[u8], path: &Path) -> String {
    static PTXAS_ARCH: LazyLock<BytesRegex> =
        LazyLock::new(|| BytesRegex::new(r"-arch\s+(sm_\d+[a-z]?)[\s\x00]").unwrap());
    static FILENAME_ARCH: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\.(sm_\d+[a-z]?)\.cubin$").unwrap());

    if let Some(caps) = PTXAS_ARCH.captures(tkinfo) {
        return String::from_utf8_lossy(&caps[1]).into_owned();
    }
    FILENAME_ARCH
        .captures(&path.to_string_lossy())
        .map_or_else(|| "unknown".to_string(), |caps| caps[1].to_string())
}

/// A function symbol; `offset` is relative to its section.
struct Function {
    name: String,
    section: usize,
    offset: u64,
    size: u64,
}

/// Attribute function symbols to kernels.
///
/// Non-inlined device functions appear as LOCAL symbols named
/// `$<kernel>$<callee>` (or `$__internal_N_$...`) placed inside the kernel's
/// own `.text` section, where the kernel symbol already spans them. They are
/// therefore not listed as kernels and add no size, except to extend a
/// kernel whose symbol ends before them.
fn fold_functions(funcs: impl IntoIterator<Item = Function>) -> Vec<(String, u64)> {
    let mut kernels: Vec<(String, u64)> = Vec::new();
    let mut index_of: HashMap<String, usize> = HashMap::new();
    let mut index = |kernels: &mut Vec<(String, u64)>, name: String| {
        *index_of.entry(name.clone()).or_insert_with(|| {
            kernels.push((name, 0));
            kernels.len() - 1
        })
    };

    // Per section: the owning kernel and the byte range its symbol covers
    let mut owners: HashMap<usize, (usize, u64, u64)> = HashMap::new();
    let mut locals = Vec::new();
    for func in funcs {
        if func.size == 0 {
            continue;
        }
        if func.name.starts_with('$') {
            locals.push(func);
        } else {
            let idx = index(&mut kernels, func.name);
            kernels[idx].1 = func.size;
            owners.insert(func.section, (idx, func.offset, func.offset + func.size));
        }
    }
    for local in locals {
        match owners.get_mut(&local.section) {
            Some((idx, start, end)) => {
                let local_end = local.offset + local.size;
                if local_end > *end {
                    *end = local_end;
                    kernels[*idx].1 = local_end - *start;
                }
            }
            None => {
                let idx = index(&mut kernels, local.name);
                kernels[idx].1 += local.size;
            }
        }
    }
    kernels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(name: &str, section: usize, offset: u64, size: u64) -> Function {
        Function {
            name: name.to_string(),
            section,
            offset,
            size,
        }
    }

    #[test]
    fn locals_inside_kernel_symbol_add_nothing() {
        // Layout of real cubins: the kernel symbol spans the whole section,
        // outlined callees sit inside it
        let kernels = fold_functions([
            f("$_Z1kv$_Z6calleev", 7, 0x190, 240),
            f("_Z1kv", 7, 0, 640),
            f("$__internal_3_$__cuda_sm3x_div_rn", 7, 0x100, 64),
            f("_Z5otherv", 8, 0, 256),
        ]);
        assert_eq!(
            kernels,
            vec![("_Z1kv".to_string(), 640), ("_Z5otherv".to_string(), 256)]
        );
    }

    #[test]
    fn local_past_kernel_symbol_extends_it() {
        let kernels = fold_functions([f("_Z1kv", 7, 0, 400), f("$_Z1kv$_Z1gv", 7, 400, 100)]);
        assert_eq!(kernels, vec![("_Z1kv".to_string(), 500)]);
    }

    #[test]
    fn orphan_local_and_zero_size_symbols() {
        let kernels = fold_functions([f("$orphan", 3, 0, 64), f("_Z3undv", 0, 0, 0)]);
        assert_eq!(kernels, vec![("$orphan".to_string(), 64)]);
    }

    #[test]
    fn arch_prefers_ptxas_command_line_over_filename() {
        let path = Path::new("lib.3.sm_100.cubin");
        assert_eq!(detect_arch(b"ptxas -arch sm_100f -m64\0", path), "sm_100f");
        assert_eq!(detect_arch(b"", path), "sm_100");
        assert_eq!(detect_arch(b"", Path::new("x.bin")), "unknown");
    }

    #[test]
    fn section_classification() {
        use SectionKind::*;
        let cases = [
            (".text._Z1kv", Some(Code)),
            (".nv.capmerc.text._Z1kv", Some(Mercury)),
            (".nv.merc.debug_info", Some(Mercury)),
            (".nv.merc.nv.info._Z1kv", Some(Mercury)),
            (".debug_info", Some(Debug)),
            (".nv.constant0._Z1kv", Some(Data)),
            (".nv.info._Z1kv", Some(Metadata)),
            (".symtab", Some(Metadata)),
            (".nv.callgraph", None),
        ];
        for (name, kind) in cases {
            assert_eq!(SectionKind::of(name), kind, "{name}");
        }
    }

    #[test]
    fn sections_rank_non_empty_largest_first() {
        let mut sections = Sections::default();
        sections.add(SectionKind::Code, 10);
        sections.add(SectionKind::Data, 30);
        sections.add(SectionKind::Debug, 20);
        sections.add(SectionKind::Debug, 0);
        let kinds: Vec<_> = sections.ranked().into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            kinds,
            [SectionKind::Data, SectionKind::Debug, SectionKind::Code]
        );
        assert_eq!(sections.total(), 60);
    }
}
