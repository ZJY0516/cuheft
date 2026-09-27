# cuheft

Shows how much space each CUDA kernel takes in a `.so`, `.o`, `.a` or
`.cubin`, per SM architecture. Useful when a wheel gets too big and you want to know which
kernels and which `TORCH_CUDA_ARCH_LIST` entries are responsible.

It also catches build problems that are easy to miss: kernels a given GPU
cannot load (a `sm_100a`-only kernel on a GB300, say), kernels that are only
JIT-compiled from PTX at startup, and kernels that likely spill registers.

Inspired by [cubloaty](https://github.com/flashinfer-ai/cubloaty).

## Install

```sh
cargo install --path .
```

Needs `cuobjdump` (CUDA toolkit) on `PATH` for libraries, object files and
archives. A standalone cubin is parsed directly.

## Usage

```sh
cuheft libfoo.so                 # top 30 kernels
cuheft libfoo.so -n 100          # top 100
cuheft libfoo.so -a sm_90a       # one architecture only
cuheft libfoo.so -r 'gemm|attn'  # regex on kernel names, case-insensitive
cuheft foo.sm_100f.cubin
cuheft libfoo.so -f json | jq '.kernels[:5]'

cuheft libfoo.so -d sm_103 -d 12.1    # what GB300 and DGX Spark can load
cuheft libfoo.so -a sm_90a -s stack   # kernels using local memory first
cuheft libfoo.so -a sm_90a -s regs    # highest register counts first
```

Output for vLLM's MoE extension (trimmed):

```
Architectures: _moe_C_stable_libtorch.abi3.so
╭──────────────┬─────────┬───────────┬───────────┬────────╮
│ Architecture ┆ Kernels ┆      Code ┆     Total ┆      % │
╞══════════════╪═════════╪═══════════╪═══════════╪════════╡
│ sm_100       ┆    1568 ┆  28.3 MiB ┆  41.3 MiB ┆   9.6% │
│ sm_120f      ┆     876 ┆  61.4 MiB ┆  80.2 MiB ┆  18.6% │
│ sm_80        ┆    2241 ┆  88.9 MiB ┆  97.2 MiB ┆  22.5% │
│ sm_90        ┆    1568 ┆  34.3 MiB ┆  40.0 MiB ┆   9.3% │
│ ...          ┆         ┆           ┆           ┆        │
│ TOTAL        ┆    2445 ┆ 348.2 MiB ┆ 432.1 MiB ┆ 100.0% │
╰──────────────┴─────────┴───────────┴───────────┴────────╯

Sections
╭───────────────────┬───────────┬───────╮
│ Section           ┆      Size ┆     % │
╞═══════════════════╪═══════════╪═══════╡
│ Code              ┆ 348.2 MiB ┆ 80.6% │
│ Metadata          ┆  36.2 MiB ┆  8.4% │
│ Mercury (capmerc) ┆  34.9 MiB ┆  8.1% │
│ Data              ┆   8.7 MiB ┆  2.0% │
│ Debug Info        ┆   4.1 MiB ┆  0.9% │
╰───────────────────┴───────────┴───────╯
```

After these come the largest kernels overall and the largest kernels for each
architecture. `Code` is the kernels' SASS; `Total` also counts metadata,
constant banks and so on. Per-architecture tables, and the main table when
only one architecture is shown, add registers per thread, stack (local
memory) per thread and compile-time shared memory per block. A stack comes
from spills, local arrays or calls to non-inlined functions. It is
highlighted only when the kernel also uses every register it is allowed
(the `__launch_bounds__` or `-maxrregcount` limit), which usually means
spills. The binary does not record actual spill counts, so check those with
`ptxas -v` or Nsight Compute. JSON output lists every kernel with its size
and resources per architecture.

With `--device`, each kernel is checked against CUDA's loading rules: a cubin
runs on the same major version with an equal or newer minor (`sm_100` and
`sm_100f` on `sm_103`), an arch-specific `a` cubin only on exactly its
architecture, and PTX is JIT-compiled on anything newer than its target.
Kernels that need JIT or cannot run at all are listed. The check looks at
every architecture in the file even with `--arch`, since that option only
narrows what is displayed; `--filter` does limit which kernels are checked.

## Notes

- Cubins are extracted with `cuobjdump -xelf all`. The fatbin container is
  undocumented and can be compressed, so there is no native parser for it.
- cuobjdump names an `sm_100f` cubin `*.sm_100.cubin`. The real target is
  read from the ptxas command line stored in each cubin.
- Device functions that did not get inlined show up as local
  `$kernel$callee` symbols inside the kernel's own code, which the kernel's
  symbol already covers. They are not listed as separate kernels.
- With `-rdc=true`, device functions get global symbols of their own. Only
  symbols marked as entry points (`__global__`) count as kernels; the rest
  are summed in one line under the architecture table.
- Blackwell and later cubins store every kernel a second time in a format
  NVIDIA calls Mercury (`.nv.capmerc.*`, `.nv.merc.*`). It is reported as its
  own section rather than folded into `Code`.
- Sections that only declare a size, like `.nv.shared.*` for static shared
  memory, take no space in the file and are not counted. cubloaty counts
  them, so its data totals are larger.
- Registers and stack come from each cubin's `.nv.info` section and match
  `cuobjdump -res-usage`. Shared memory is what ptxas allocated, which
  includes the 1 KiB some architectures reserve per block; dynamic shared
  memory is set at launch and cannot be seen in the binary.
- Names are demangled with `cpp_demangle`, so integer template arguments look
  like `(unsigned int)4` rather than `4u` as c++filt prints them.

## Tests

```sh
cargo test
```

The end-to-end tests compile `tests/fixtures/kernels.cu` with nvcc and are
skipped when nvcc or cuobjdump is missing. Set `CUHEFT_REQUIRE_CUDA=1` to
make that an error instead, as CI does.

## License

Apache-2.0, see [LICENSE](LICENSE).
