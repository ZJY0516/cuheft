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

/// Per-thread and per-block resources of a kernel, as recorded by ptxas.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub registers: Option<u32>,
    /// Register limit per thread from `__launch_bounds__` or -maxrregcount.
    pub max_registers: Option<u32>,
    /// Stack frame per thread in local memory: spills, local arrays and
    /// the call ABI of non-inlined functions.
    pub stack: Option<u32>,
    /// Shared memory per block allocated at compile time, including the
    /// 1 KiB some architectures reserve. Dynamic shared memory is only
    /// known at launch.
    pub shared: u64,
}

impl Usage {
    /// Registers at the limit together with a stack frame: most likely
    /// register spills. The binary does not record spill counts, and a stack
    /// alone may just be a local array or a function call.
    pub fn likely_spills(&self) -> bool {
        match (self.registers, self.stack) {
            (Some(regs), Some(stack)) => stack > 0 && regs >= self.max_registers.unwrap_or(255),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Mangled as parsed, demangled later.
    pub name: String,
    /// Code bytes.
    pub size: u64,
    pub usage: Usage,
}

/// Sizes of one cubin.
#[derive(Debug, Default)]
pub struct CubinInfo {
    pub arch: String,
    pub kernels: Vec<KernelInfo>,
    /// Functions that are not kernels, as in `-rdc` builds where device
    /// functions get their own global symbols.
    pub device_functions: usize,
    /// Code bytes of those functions.
    pub device_code: u64,
    pub sections: Sections,
}

pub fn analyze(path: &Path) -> Result<CubinInfo> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let file =
        object::File::parse(&*data).with_context(|| format!("parsing ELF {}", path.display()))?;

    let mut sections = Sections::default();
    let mut tkinfo: &[u8] = &[];
    let mut nv_info: &[u8] = &[];
    let mut shared: HashMap<&str, u64> = HashMap::new();
    let mut max_registers: HashMap<&str, u32> = HashMap::new();
    for section in file.sections() {
        let Ok(name) = section.name() else { continue };
        match name {
            ".note.nv.tkinfo" => tkinfo = section.data().unwrap_or_default(),
            ".nv.info" => nv_info = section.data().unwrap_or_default(),
            _ => {}
        }
        if let Some(kernel) = name.strip_prefix(".nv.shared.") {
            shared.insert(kernel, section.size());
        }
        if let Some(kernel) = name.strip_prefix(".nv.info.")
            && let Some(limit) = parse_max_registers(section.data().unwrap_or_default())
        {
            max_registers.insert(kernel, limit);
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
            entry: is_entry(sym.flags()),
        })
    });

    let mut usage: HashMap<&str, Usage> = HashMap::new();
    let by_symbol = parse_nv_info(nv_info);
    for sym in file.symbols() {
        if let (Some(u), Ok(name)) = (by_symbol.get(&sym.index().0), sym.name()) {
            usage.insert(name, *u);
        }
    }
    let folded = fold_functions(funcs);
    // Without any entry mark, treat every function as a kernel rather than
    // report none
    let has_entries = folded.iter().any(|f| f.entry);
    let (kernels, device_functions): (Vec<_>, Vec<_>) =
        folded.into_iter().partition(|f| f.entry || !has_entries);
    let kernels = kernels
        .into_iter()
        .map(|Folded { name, size, .. }| {
            let mut u = usage.get(name.as_str()).copied().unwrap_or_default();
            u.shared = shared.get(name.as_str()).copied().unwrap_or(0);
            u.max_registers = max_registers.get(name.as_str()).copied();
            KernelInfo {
                name,
                size,
                usage: u,
            }
        })
        .collect();

    Ok(CubinInfo {
        arch: detect_arch(tkinfo, path),
        kernels,
        device_functions: device_functions.len(),
        device_code: device_functions.iter().map(|f| f.size).sum(),
        sections,
    })
}

/// Whether a symbol is a `__global__` kernel (STO_CUDA_ENTRY in st_other).
fn is_entry(flags: object::SymbolFlags<object::SectionIndex, object::SymbolIndex>) -> bool {
    const STO_CUDA_ENTRY: u8 = 0x10;
    matches!(flags, object::SymbolFlags::Elf { st_other, .. } if st_other.0 & STO_CUDA_ENTRY != 0)
}

/// Registers and stack size per symbol index from the `.nv.info` section.
///
/// The section is a sequence of attributes: a format byte, an attribute
/// byte, then either a 2-byte value or, for format 4, a 2-byte length and
/// that many bytes. The attributes read here carry a (symbol index, value)
/// pair of u32s. Checked against `cuobjdump -res-usage`.
fn parse_nv_info(data: &[u8]) -> HashMap<usize, Usage> {
    const MIN_STACK_SIZE: u8 = 0x12;
    const MAX_STACK_SIZE: u8 = 0x23;
    const REGCOUNT: u8 = 0x2f;

    let mut usage: HashMap<usize, Usage> = HashMap::new();
    let mut max_stack: HashMap<usize, u32> = HashMap::new();
    for (attr, payload) in nv_info_attributes(data) {
        let Some((sym, value)) = payload
            .split_first_chunk::<4>()
            .and_then(|(sym, v)| Some((*sym, *v.first_chunk::<4>()?)))
        else {
            continue;
        };
        let sym = u32::from_le_bytes(sym) as usize;
        let value = u32::from_le_bytes(value);
        match attr {
            REGCOUNT => usage.entry(sym).or_default().registers = Some(value),
            MIN_STACK_SIZE => usage.entry(sym).or_default().stack = Some(value),
            MAX_STACK_SIZE => {
                max_stack.insert(sym, value);
            }
            _ => {}
        }
    }
    // Recursion makes the minimum a lower bound; prefer the maximum when set
    for (sym, stack) in max_stack {
        usage.entry(sym).or_default().stack = Some(stack);
    }
    usage
}

/// Register limit (MAXREG_COUNT) from a kernel's own `.nv.info.<kernel>`.
fn parse_max_registers(data: &[u8]) -> Option<u32> {
    const MAXREG_COUNT: u8 = 0x1b;
    nv_info_attributes(data)
        .find(|&(attr, _)| attr == MAXREG_COUNT)
        .and_then(|(_, value)| Some(u16::from_le_bytes(*value.first_chunk::<2>()?).into()))
}

/// `(attribute, payload)` pairs of a `.nv.info*` section. Each attribute is
/// a format byte and an attribute byte, followed either by a 2-byte value
/// or, for format 4, by a 2-byte length and that many bytes. Stops at the
/// first truncated attribute.
fn nv_info_attributes(data: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    const FMT_SIZED: u8 = 0x04;
    let mut rest = data;
    std::iter::from_fn(move || {
        let [format, attr, header @ ..] = rest else {
            return None;
        };
        let (payload, next) = if *format == FMT_SIZED {
            let (len, body) = header.split_first_chunk::<2>()?;
            let len = u16::from_le_bytes(*len) as usize;
            (len <= body.len()).then(|| body.split_at(len))?
        } else {
            (header.len() >= 2).then(|| header.split_at(2))?
        };
        let attr = *attr;
        rest = next;
        Some((attr, payload))
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
    /// A `__global__` kernel rather than a device function.
    entry: bool,
}

/// A function with its outlined callees folded in.
#[derive(Debug, PartialEq, Eq)]
struct Folded {
    name: String,
    size: u64,
    entry: bool,
}

/// Attribute local function symbols to the functions containing them.
///
/// Non-inlined device functions appear as LOCAL symbols named
/// `$<kernel>$<callee>` (or `$__internal_N_$...`) placed inside the kernel's
/// own `.text` section, where the kernel symbol already spans them. They are
/// therefore not listed on their own and add no size, except to extend a
/// kernel whose symbol ends before them.
fn fold_functions(funcs: impl IntoIterator<Item = Function>) -> Vec<Folded> {
    let mut folded: Vec<Folded> = Vec::new();
    let mut index_of: HashMap<String, usize> = HashMap::new();
    let mut index = |folded: &mut Vec<Folded>, name: String, entry: bool| {
        *index_of.entry(name.clone()).or_insert_with(|| {
            folded.push(Folded {
                name,
                size: 0,
                entry,
            });
            folded.len() - 1
        })
    };

    // Per section: the owning function and the byte range its symbol covers
    let mut owners: HashMap<usize, (usize, u64, u64)> = HashMap::new();
    let mut locals = Vec::new();
    for func in funcs {
        if func.size == 0 {
            continue;
        }
        if func.name.starts_with('$') {
            locals.push(func);
        } else {
            let idx = index(&mut folded, func.name, func.entry);
            folded[idx].size = func.size;
            owners.insert(func.section, (idx, func.offset, func.offset + func.size));
        }
    }
    for local in locals {
        match owners.get_mut(&local.section) {
            Some((idx, start, end)) => {
                let local_end = local.offset + local.size;
                if local_end > *end {
                    *end = local_end;
                    folded[*idx].size = local_end - *start;
                }
            }
            None => {
                let idx = index(&mut folded, local.name, local.entry);
                folded[idx].size += local.size;
            }
        }
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel, or a local callee when the name starts with `$`.
    fn f(name: &str, section: usize, offset: u64, size: u64) -> Function {
        Function {
            name: name.to_string(),
            section,
            offset,
            size,
            entry: !name.starts_with('$'),
        }
    }

    fn sizes(folded: &[Folded]) -> Vec<(&str, u64, bool)> {
        folded
            .iter()
            .map(|f| (f.name.as_str(), f.size, f.entry))
            .collect()
    }

    #[test]
    fn locals_inside_kernel_symbol_add_nothing() {
        // Layout of real cubins: the kernel symbol spans the whole section,
        // outlined callees sit inside it
        let folded = fold_functions([
            f("$_Z1kv$_Z6calleev", 7, 0x190, 240),
            f("_Z1kv", 7, 0, 640),
            f("$__internal_3_$__cuda_sm3x_div_rn", 7, 0x100, 64),
            f("_Z5otherv", 8, 0, 256),
        ]);
        assert_eq!(
            sizes(&folded),
            [("_Z1kv", 640, true), ("_Z5otherv", 256, true)]
        );
    }

    #[test]
    fn local_past_kernel_symbol_extends_it() {
        let folded = fold_functions([f("_Z1kv", 7, 0, 400), f("$_Z1kv$_Z1gv", 7, 400, 100)]);
        assert_eq!(sizes(&folded), [("_Z1kv", 500, true)]);
    }

    #[test]
    fn orphan_local_and_zero_size_symbols() {
        let folded = fold_functions([f("$orphan", 3, 0, 64), f("_Z3undv", 0, 0, 0)]);
        assert_eq!(sizes(&folded), [("$orphan", 64, false)]);
    }

    #[test]
    fn global_device_functions_are_not_entries() {
        // -rdc: a device function has its own global symbol and section
        let device_fn = Function {
            entry: false,
            ..f("_Z5scaleff", 9, 0, 256)
        };
        let folded = fold_functions([f("_Z1kv", 7, 0, 400), device_fn]);
        assert_eq!(
            sizes(&folded),
            [("_Z1kv", 400, true), ("_Z5scaleff", 256, false)]
        );
    }

    #[test]
    fn nv_info_registers_and_stack() {
        let mut data = Vec::new();
        let mut attr = |format: u8, attr: u8, payload: &[u8]| {
            data.extend([format, attr]);
            if format == 4 {
                data.extend((payload.len() as u16).to_le_bytes());
            }
            data.extend(payload);
        };
        let pair = |sym: u32, v: u32| [sym.to_le_bytes(), v.to_le_bytes()].concat();
        attr(4, 0x2f, &pair(3, 40)); // REGCOUNT
        attr(3, 0x1b, &[0xff, 0]); // MAXREG_COUNT, 2-byte value, ignored
        attr(4, 0x12, &pair(3, 16)); // MIN_STACK_SIZE
        attr(4, 0x23, &pair(3, 64)); // MAX_STACK_SIZE wins
        attr(4, 0x2f, &pair(5, 8));
        attr(4, 0x12, &pair(5, 0));
        let usage = parse_nv_info(&data);
        assert_eq!(usage[&3].registers, Some(40));
        assert_eq!(usage[&3].stack, Some(64));
        assert_eq!(usage[&5].registers, Some(8));
        assert_eq!(usage[&5].stack, Some(0));
        // Truncated input must not panic
        assert!(parse_nv_info(&data[..data.len() - 3]).len() <= 2);
    }

    #[test]
    fn nv_info_register_limit() {
        // MIN_STACK_SIZE (sized) then MAXREG_COUNT = 168 (2-byte value)
        let mut data = vec![4, 0x12, 8, 0];
        data.extend(3u32.to_le_bytes());
        data.extend(0u32.to_le_bytes());
        data.extend([3, 0x1b, 168, 0]);
        assert_eq!(parse_max_registers(&data), Some(168));
        assert_eq!(parse_max_registers(&data[..8]), None);
    }

    #[test]
    fn spills_need_registers_at_limit_and_a_stack() {
        let usage = |registers, max_registers, stack| Usage {
            registers: Some(registers),
            max_registers,
            stack: Some(stack),
            shared: 0,
        };
        assert!(usage(255, Some(255), 80).likely_spills());
        assert!(usage(32, Some(32), 160).likely_spills());
        assert!(usage(255, None, 16).likely_spills());
        // A local array or a call with registers to spare
        assert!(!usage(53, Some(255), 32).likely_spills());
        // At the limit but nothing in local memory
        assert!(!usage(255, Some(255), 0).likely_spills());
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
