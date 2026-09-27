//! End-to-end tests on binaries compiled from tests/fixtures/kernels.cu.
//!
//! They need nvcc (and cuobjdump for the shared library test). Without them
//! the tests are skipped, unless CUHEFT_REQUIRE_CUDA is set, as in CI.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::Value;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const OUT_DIR: &str = env!("CARGO_TARGET_TMPDIR");

const KERNELS: [&str; 3] = [
    "plain_c_kernel",
    "scale_kernel<4096>(float*, float const*, float)",
    "scale_kernel<8192>(float*, float const*, float)",
];

fn require_cuda() -> bool {
    std::env::var_os("CUHEFT_REQUIRE_CUDA").is_some()
}

fn tool_available(tool: &str) -> bool {
    let found = Command::new(tool).arg("--version").output().is_ok();
    if !found {
        assert!(
            !require_cuda(),
            "{tool} not found but CUHEFT_REQUIRE_CUDA is set"
        );
        eprintln!("skipping: {tool} not found");
    }
    found
}

struct Fixtures {
    sm90a_cubin: PathBuf,
    sm100f_cubin: PathBuf,
    library: PathBuf,
}

/// Compile the fixtures once per test run, or `None` if nvcc is missing.
fn fixtures() -> Option<&'static Fixtures> {
    static FIXTURES: OnceLock<Option<Fixtures>> = OnceLock::new();
    FIXTURES
        .get_or_init(|| {
            if !tool_available("nvcc") {
                return None;
            }
            let source = Path::new(FIXTURE_DIR).join("kernels.cu");
            let out = |name: &str| Path::new(OUT_DIR).join(name);
            let nvcc = |args: &[&str], output: &Path| {
                let status = Command::new("nvcc")
                    .args(["-O3", "-o"])
                    .arg(output)
                    .args(args)
                    .arg(&source)
                    .status()
                    .expect("running nvcc");
                assert!(status.success(), "nvcc {args:?} failed");
            };

            let fixtures = Fixtures {
                sm90a_cubin: out("kernels.sm_90a.cubin"),
                sm100f_cubin: out("kernels.sm_100f.cubin"),
                library: out("libkernels.so"),
            };
            nvcc(&["-cubin", "-arch=sm_90a"], &fixtures.sm90a_cubin);
            nvcc(&["-cubin", "-arch=sm_100f"], &fixtures.sm100f_cubin);
            nvcc(
                &[
                    "-shared",
                    "-Xcompiler=-fPIC",
                    "-cudart=shared",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-gencode=arch=compute_100f,code=sm_100f",
                ],
                &fixtures.library,
            );
            Some(fixtures)
        })
        .as_ref()
}

fn cuheft(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cuheft"))
        .args(args)
        .output()
        .expect("running cuheft")
}

fn cuheft_json(file: &Path, extra: &[&str]) -> Value {
    let mut args = vec![file.to_str().unwrap(), "--format", "json"];
    args.extend(extra);
    let output = cuheft(&args);
    assert!(
        output.status.success(),
        "cuheft failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("valid JSON")
}

fn strings<'a>(json: &'a Value, array: &str, field: &str) -> Vec<&'a str> {
    let mut values: Vec<&str> = json[array]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item[field].as_str().unwrap())
        .collect();
    values.sort_unstable();
    values
}

fn section_size(json: &Value, name: &str) -> Option<u64> {
    json["sections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == name)
        .map(|s| s["size"].as_u64().unwrap())
}

#[test]
fn cubin_kernels_are_demangled_and_folded() {
    let Some(fx) = fixtures() else { return };
    let json = cuheft_json(&fx.sm100f_cubin, &[]);

    // No `void ` prefix and no `$kernel$callee` locals listed as kernels
    assert_eq!(strings(&json, "kernels", "name"), KERNELS);

    // Kernel symbols already span their outlined callees, so kernel code
    // must equal the SASS sections exactly rather than exceed them
    assert_eq!(
        json["kernel_code_size"].as_u64(),
        section_size(&json, "Code")
    );
}

#[test]
fn arch_comes_from_cubin_not_filename() {
    let Some(fx) = fixtures() else { return };
    // cuobjdump would name an sm_100f cubin like this
    let renamed = Path::new(OUT_DIR).join("renamed.sm_100.cubin");
    std::fs::copy(&fx.sm100f_cubin, &renamed).unwrap();
    let json = cuheft_json(&renamed, &[]);
    assert_eq!(strings(&json, "architectures", "arch"), ["sm_100f"]);
}

#[test]
fn shared_memory_takes_no_file_space() {
    let Some(fx) = fixtures() else { return };
    let json = cuheft_json(&fx.sm90a_cubin, &[]);
    // The kernels declare 48 KiB of shared memory
    let data = section_size(&json, "Data").unwrap();
    assert!(data < 16 * 1024, "data = {data}");
    assert!(json["total_size"].as_u64().unwrap() < 48 * 1024);
}

#[test]
fn mercury_only_on_blackwell() {
    let Some(fx) = fixtures() else { return };
    let hopper = cuheft_json(&fx.sm90a_cubin, &[]);
    let blackwell = cuheft_json(&fx.sm100f_cubin, &[]);
    assert_eq!(section_size(&hopper, "Mercury (capmerc)"), None);
    assert!(section_size(&blackwell, "Mercury (capmerc)").unwrap() > 0);
}

#[test]
fn shared_library_merges_architectures() {
    let Some(fx) = fixtures() else { return };
    if !tool_available("cuobjdump") {
        return;
    }
    let json = cuheft_json(&fx.library, &[]);
    assert_eq!(
        strings(&json, "architectures", "arch"),
        ["sm_100f", "sm_90a"]
    );
    assert_eq!(strings(&json, "kernels", "name"), KERNELS);
    for kernel in json["kernels"].as_array().unwrap() {
        assert_eq!(kernel["by_arch"].as_object().unwrap().len(), 2);
    }

    let json = cuheft_json(&fx.library, &["--arch", "sm_90a", "--filter", "^SCALE"]);
    assert_eq!(json["kernel_count"], 2);
    assert_eq!(strings(&json, "architectures", "arch"), ["sm_90a"]);
}

#[test]
fn unknown_arch_lists_available_ones() {
    let Some(fx) = fixtures() else { return };
    let output = cuheft(&[fx.sm90a_cubin.to_str().unwrap(), "--arch", "sm_80"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("available: sm_90a"), "{stderr}");
}

#[test]
fn table_output_lists_kernels() {
    let Some(fx) = fixtures() else { return };
    let output = cuheft(&[fx.sm100f_cubin.to_str().unwrap(), "--color", "never"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Top kernels (3 of 3)"));
    assert!(stdout.contains("scale_kernel<8192>"));
    assert!(!stdout.contains('\x1b'));
}
