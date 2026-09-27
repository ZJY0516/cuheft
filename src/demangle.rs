//! C++ symbol demangling.

/// Demangle a kernel symbol, returning it unchanged on failure.
///
/// The leading `void ` is dropped: `__global__` functions always return
/// void, so it carries no information. Doing it here keeps the displayed
/// name, the JSON name and what `--filter` matches identical.
pub fn kernel_name(symbol: &str) -> String {
    // CUTLASS kernels nest templates far deeper than the default limit
    let parse = cpp_demangle::ParseOptions::default().recursion_limit(4096);
    let demangled = cpp_demangle::Symbol::new_with_options(symbol.as_bytes(), &parse)
        .ok()
        .and_then(|sym| sym.demangle().ok());
    match demangled {
        Some(name) => match name.strip_prefix("void ") {
            Some(stripped) => stripped.to_string(),
            None => name,
        },
        None => symbol.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demangles_and_drops_void() {
        assert_eq!(kernel_name("_Z6kernelPfi"), "kernel(float*, int)");
        assert_eq!(kernel_name("_Z3fooi"), "foo(int)");
        assert_eq!(kernel_name("plain_c_symbol"), "plain_c_symbol");
    }
}
