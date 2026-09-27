//! Locating cubins in the input file.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use object::{Object, ObjectSection, ReadCache};
use tempfile::TempDir;

/// Device code to analyze. Extracted files live in a temporary directory
/// that is removed when this is dropped.
pub struct DeviceCode {
    pub cubins: Vec<PathBuf>,
    pub ptx: Vec<PathBuf>,
    _tmpdir: Option<TempDir>,
}

/// Collect cubins, and PTX if requested, from a standalone cubin, an object
/// file or library with fatbin sections, or a static archive of those.
pub fn collect(file: &Path, with_ptx: bool) -> Result<DeviceCode> {
    let format = detect(file).with_context(|| format!("reading {}", file.display()))?;
    if format == Format::Cubin {
        return Ok(DeviceCode {
            cubins: vec![file.to_path_buf()],
            ptx: Vec::new(),
            _tmpdir: None,
        });
    }
    // Members of an archive are not checked up front; cuobjdump reads them
    if format == Format::Other && !has_fatbin_sections(file)? {
        bail!("no CUDA fatbin sections found in {}", file.display());
    }
    let tmpdir = tempfile::tempdir()?;
    let cubins = extract(file, Kind::Elf, &tmpdir.path().join("cubin"))?;
    log::debug!(
        "extracted {} cubin(s) from {}",
        cubins.len(),
        file.display()
    );
    let ptx = if with_ptx {
        let ptx = extract(file, Kind::Ptx, &tmpdir.path().join("ptx"))?;
        log::debug!("extracted {} PTX file(s)", ptx.len());
        ptx
    } else {
        Vec::new()
    };
    if cubins.is_empty() && ptx.is_empty() {
        bail!("{} contains no cubins", file.display());
    }
    Ok(DeviceCode {
        cubins,
        ptx,
        _tmpdir: Some(tmpdir),
    })
}

#[derive(Clone, Copy)]
enum Kind {
    Elf,
    Ptx,
}

#[derive(Debug, PartialEq, Eq)]
enum Format {
    /// An ELF for the CUDA machine (EM_CUDA = 190).
    Cubin,
    /// A static library (`ar` archive).
    Archive,
    Other,
}

fn detect(path: &Path) -> io::Result<Format> {
    const EM_CUDA: u16 = 190;
    let mut header = [0u8; 20];
    match File::open(path)?.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(Format::Other),
        Err(e) => return Err(e),
    }
    Ok(if header.starts_with(b"!<arch>\n") {
        Format::Archive
    } else if header.starts_with(b"\x7fELF")
        && u16::from_le_bytes([header[18], header[19]]) == EM_CUDA
    {
        Format::Cubin
    } else {
        Format::Other
    })
}

/// Whether an object file contains any CUDA fatbin section.
///
/// Reads only the headers, not the (possibly hundreds of MB) whole file.
fn has_fatbin_sections(path: &Path) -> Result<bool> {
    let cache = ReadCache::new(File::open(path)?);
    let file = object::File::parse(&cache)
        .with_context(|| format!("{} is not a recognized object file", path.display()))?;
    Ok(file
        .sections()
        .any(|s| s.name().is_ok_and(|n| n.contains("nv_fatbin"))))
}

/// Extract every embedded cubin or PTX file into `out_dir` with cuobjdump.
///
/// The fatbin container format is undocumented and can be compressed, so
/// cuobjdump is used instead of a native parser.
fn extract(lib: &Path, kind: Kind, out_dir: &Path) -> Result<Vec<PathBuf>> {
    let (flag, extension) = match kind {
        Kind::Elf => ("-xelf", "cubin"),
        Kind::Ptx => ("-xptx", "ptx"),
    };
    let lib = std::fs::canonicalize(lib)?;
    std::fs::create_dir(out_dir)?;
    let output = Command::new("cuobjdump")
        .args([flag, "all"])
        .arg(&lib)
        .current_dir(out_dir)
        .output()
        .map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => anyhow!("cuobjdump not found; is the CUDA toolkit on PATH?"),
            _ => anyhow!(e).context("running cuobjdump"),
        })?;
    if !output.status.success() {
        bail!(
            "cuobjdump failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(out_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == extension))
        .collect();
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, bytes).unwrap();
        file
    }

    fn format_of(bytes: &[u8]) -> Format {
        detect(temp_file(bytes).path()).unwrap()
    }

    #[test]
    fn detects_format_from_header() {
        let mut header = [0u8; 64];
        header[..4].copy_from_slice(b"\x7fELF");
        header[18..20].copy_from_slice(&190u16.to_le_bytes());
        assert_eq!(format_of(&header), Format::Cubin);

        header[18..20].copy_from_slice(&62u16.to_le_bytes()); // x86-64
        assert_eq!(format_of(&header), Format::Other);
        assert_eq!(
            format_of(b"!<arch>\n/               0           "),
            Format::Archive
        );
        assert_eq!(format_of(b"short"), Format::Other);
    }
}
