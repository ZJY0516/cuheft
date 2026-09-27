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

/// The template a demangled kernel name instantiates: every top-level
/// template argument list becomes `<…>` and the parameter list is dropped,
/// e.g. `ns::Foo<int>::kernel<4>(Bar<int>)` gives `ns::Foo<…>::kernel<…>`.
pub fn template_name(name: &str) -> String {
    const ANONYMOUS: &str = "(anonymous namespace)";
    let mut out = String::with_capacity(name.len());
    let mut depth = 0usize;
    let mut prev = '\0';
    let mut chars = name.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '<' => {
                if depth == 0 {
                    out.push_str("<…>");
                }
                depth += 1;
            }
            // `->` in a decltype expression is not a closing bracket
            '>' if depth > 0 && prev != '-' => depth -= 1,
            _ if depth > 0 => {}
            '(' if name[i..].starts_with(ANONYMOUS) => {
                out.push_str(ANONYMOUS);
                chars.nth(ANONYMOUS.len() - 2);
            }
            // The parameter list
            '(' => break,
            // `{lambda(int)#1}` names a closure type, parentheses included
            '{' => {
                out.push(c);
                for (_, c) in chars.by_ref() {
                    out.push(c);
                    if c == '}' {
                        break;
                    }
                }
            }
            _ if name[i..].starts_with("operator") => {
                // Keep operator names such as `operator<` or `operator()`
                let rest = &name[i + "operator".len()..];
                let symbol_len = if rest.starts_with("()") {
                    2
                } else {
                    rest.find(|c: char| !"<>=!+-*/%^&|~[],".contains(c))
                        .unwrap_or(rest.len())
                };
                let end = i + "operator".len() + symbol_len;
                out.push_str(&name[i..end]);
                while chars.next_if(|&(j, _)| j < end).is_some() {}
            }
            _ => out.push(c),
        }
        prev = c;
    }
    out
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

    #[test]
    fn folds_template_arguments_and_parameters() {
        let cases = [
            (
                "scale_kernel<4096>(float*, float const*, float)",
                "scale_kernel<…>",
            ),
            ("plain_c_kernel", "plain_c_kernel"),
            ("kernel(float*, int)", "kernel"),
            (
                "ns::Foo<Bar<int>, (unsigned int)3>::kernel<4>(Baz<int>)",
                "ns::Foo<…>::kernel<…>",
            ),
            (
                "(anonymous namespace)::topk<float, 8>(float*)",
                "(anonymous namespace)::topk<…>",
            ),
            ("f<decltype(a->b), int>(int)", "f<…>"),
            (
                "ns::{lambda(int)#1}::operator()(int)",
                "ns::{lambda(int)#1}::operator()",
            ),
            ("ns::operator<<(int)", "ns::operator<<"),
            ("ns::operator< <int>(int)", "ns::operator< <…>"),
        ];
        for (name, template) in cases {
            assert_eq!(template_name(name), template, "{name}");
        }
    }
}
