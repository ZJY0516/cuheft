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

const KERNELS: [&str; 4] = [
    "local_array_kernel(float*, int)",
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
    /// Relocatable device code: device functions get global symbols
    rdc_library: PathBuf,
    archive: PathBuf,
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
                rdc_library: out("libkernels_rdc.so"),
                archive: out("libkernels.a"),
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
                    // Portable PTX, JIT-compiled on GPUs without a cubin
                    "-gencode=arch=compute_80,code=compute_80",
                ],
                &fixtures.library,
            );
            nvcc(
                &[
                    "-shared",
                    "-rdc=true",
                    "-Xcompiler=-fPIC",
                    "-cudart=shared",
                    "-arch=sm_90a",
                ],
                &fixtures.rdc_library,
            );
            nvcc(&["-lib", "-arch=sm_100f"], &fixtures.archive);
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
fn rdc_device_functions_are_not_kernels() {
    let Some(fx) = fixtures() else { return };
    if !tool_available("cuobjdump") {
        return;
    }
    let json = cuheft_json(&fx.rdc_library, &[]);
    // scale(float, float) is a global but not an entry point
    assert_eq!(strings(&json, "kernels", "name"), KERNELS);
    assert_eq!(json["device_function_count"], 1);
    let device_code = json["device_function_code_size"].as_u64().unwrap();
    assert!(device_code > 0);
    assert_eq!(
        json["kernel_code_size"].as_u64().unwrap() + device_code,
        section_size(&json, "Code").unwrap()
    );

    let output = cuheft(&[fx.rdc_library.to_str().unwrap(), "--color", "never"]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Not in kernel lists: 1 device functions"));
}

#[test]
fn static_archive() {
    let Some(fx) = fixtures() else { return };
    if !tool_available("cuobjdump") {
        return;
    }
    let json = cuheft_json(&fx.archive, &[]);
    assert_eq!(strings(&json, "architectures", "arch"), ["sm_100f"]);
    assert_eq!(strings(&json, "kernels", "name"), KERNELS);
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
    assert!(stdout.contains("Top kernels (4 of 4)"));
    assert!(stdout.contains("scale_kernel<8192>"));
    assert!(!stdout.contains('\x1b'));
}

fn kernel<'a>(json: &'a Value, name_prefix: &str) -> &'a Value {
    json["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["name"].as_str().unwrap().starts_with(name_prefix))
        .unwrap_or_else(|| panic!("no kernel {name_prefix}"))
}

#[test]
fn resources_per_kernel() {
    let Some(fx) = fixtures() else { return };
    let json = cuheft_json(&fx.sm90a_cubin, &[]);
    let stats = |name| &kernel(&json, name)["by_arch"]["sm_90a"];

    for name in KERNELS {
        assert!(stats(name)["registers"].as_u64().unwrap() > 0, "{name}");
    }
    // float buf[64] in local memory, with registers to spare: not a spill
    assert_eq!(stats("local_array_kernel")["stack"], 256);
    assert_eq!(stats("local_array_kernel")["max_registers"], 255);
    assert_eq!(stats("local_array_kernel")["likely_spill"], false);
    assert_eq!(stats("plain_c_kernel")["stack"], 0);
    // __shared__ float tile[8192], plus up to the 1 KiB some architectures
    // reserve per block
    let shared = stats("scale_kernel<8192>")["static_shared"]
        .as_u64()
        .unwrap();
    assert!((32 * 1024..=33 * 1024).contains(&shared), "{shared}");
    assert_eq!(stats("plain_c_kernel")["static_shared"], 0);
}

#[test]
fn sort_by_stack_surfaces_local_memory() {
    let Some(fx) = fixtures() else { return };
    let json = cuheft_json(&fx.sm100f_cubin, &["--sort", "stack"]);
    assert_eq!(
        json["kernels"][0]["name"],
        "local_array_kernel(float*, int)"
    );

    // A single architecture shows resource columns in the main table
    let output = cuheft(&[fx.sm100f_cubin.to_str().unwrap(), "--color", "never"]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Regs") && stdout.contains("Stack"));
}

#[test]
fn device_availability() {
    let Some(fx) = fixtures() else { return };
    if !tool_available("cuobjdump") {
        return;
    }
    let json = cuheft_json(
        &fx.library,
        &[
            "--device", "sm_103", "--device", "12.0", "--device", "sm_75",
        ],
    );
    let devices = json["devices"].as_array().unwrap();
    let summary: Vec<_> = devices
        .iter()
        .map(|d| {
            (
                d["device"].as_str().unwrap(),
                d["cubin_kernel_count"].as_u64().unwrap(),
                d["ptx_jit"].as_array().unwrap().len(),
                d["missing"].as_array().unwrap().len(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            // sm_100f cubin runs on sm_103
            ("sm_103", 4, 0, 0),
            // No sm_12x cubin; compute_80 PTX can be JIT-compiled
            ("sm_120", 0, 4, 0),
            // Older than every target
            ("sm_75", 0, 0, 4),
        ]
    );

    let output = cuheft(&[
        fx.library.to_str().unwrap(),
        "-d",
        "sm_75",
        "--color",
        "never",
    ]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Availability on sm_75: 0 from cubin, 0 need PTX JIT, 4 missing"));
}

#[test]
fn invalid_device_is_rejected() {
    let output = cuheft(&["whatever.so", "--device", "sm_100a"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("sm_100a"));
}
