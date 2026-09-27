//! Locating cubins in the input file.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use object::{Object, ObjectSection, ReadCache};
use tempfile::TempDir;

/// Cubins to analyze. Extracted cubins live in a temporary directory that is
/// removed when this is dropped.
pub struct Cubins {
    pub paths: Vec<PathBuf>,
    _tmpdir: Option<TempDir>,
}

/// Collect cubins from a standalone cubin or a library with fatbin sections.
pub fn collect(file: &Path) -> Result<Cubins> {
    if is_cubin(file).with_context(|| format!("reading {}", file.display()))? {
        return Ok(Cubins {
            paths: vec![file.to_path_buf()],
            _tmpdir: None,
        });
    }
    if !has_fatbin_sections(file)? {
        bail!("no CUDA fatbin sections found in {}", file.display());
    }
    let tmpdir = tempfile::tempdir()?;
    let paths = extract_cubins(file, tmpdir.path())?;
    log::debug!("extracted {} cubin(s) from {}", paths.len(), file.display());
    if paths.is_empty() {
        bail!("{} contains no cubins (PTX only?)", file.display());
    }
    Ok(Cubins {
        paths,
        _tmpdir: Some(tmpdir),
    })
}

/// Whether the file is an ELF for the CUDA machine (EM_CUDA = 190).
fn is_cubin(path: &Path) -> io::Result<bool> {
    const EM_CUDA: u16 = 190;
    let mut header = [0u8; 20];
    match File::open(path)?.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(e),
    }
    Ok(header.starts_with(b"\x7fELF") && u16::from_le_bytes([header[18], header[19]]) == EM_CUDA)
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

/// Extract every embedded cubin into `out_dir` with cuobjdump.
///
/// The fatbin container format is undocumented and can be compressed, so
/// cuobjdump is used instead of a native parser.
fn extract_cubins(lib: &Path, out_dir: &Path) -> Result<Vec<PathBuf>> {
    let lib = std::fs::canonicalize(lib)?;
    let output = Command::new("cuobjdump")
        .args(["-xelf", "all"])
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
    let mut cubins: Vec<PathBuf> = std::fs::read_dir(out_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "cubin"))
        .collect();
    cubins.sort();
    Ok(cubins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, bytes).unwrap();
        file
    }

    #[test]
    fn detects_cubin_by_elf_machine() {
        let mut header = [0u8; 64];
        header[..4].copy_from_slice(b"\x7fELF");
        header[18..20].copy_from_slice(&190u16.to_le_bytes());
        assert!(is_cubin(temp_file(&header).path()).unwrap());

        header[18..20].copy_from_slice(&62u16.to_le_bytes()); // x86-64
        assert!(!is_cubin(temp_file(&header).path()).unwrap());
        assert!(!is_cubin(temp_file(b"short").path()).unwrap());
    }
}
