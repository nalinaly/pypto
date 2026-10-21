# CSA single-SPMD companion branch (2026-10-19)

The `dsv4-csa-spmd-repro-20261019` branch freezes the PyPTO source used for
nalinaly/vllm-ascend CSA single-SPMD validation. It starts from the debugging
branch at `d01de8523` and retains adopted local fixes. This does not imply
that these changes have landed on upstream PyPTO.

See the [CSA reproduction guide](https://github.com/nalinaly/vllm-ascend/tree/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/csa_toolchain_repro_20261019)
for clone/build commands, explicit environment binding and existing CSA tests.
Its `versions.json` pins the three repositories; `verification.json`
distinguishes source-rebuild checks from previously completed device validation.

## Companion dependencies

- Simpler fork: `https://github.com/nalinaly/simpler.git`, commit
  `57e499238fd4ee3548947ad860be456aaeeb16b5`.
- The `runtime` URL in `.gitmodules`, its gitlink and the revision in
  `python/pypto/_kernel_abi.py` identify that same source.
- Simpler's 256-Tensor plus 16-scalar ABI must be used with this PyPTO build;
  the package version `0.1.0` alone does not identify it.
- Use the matching reproduction branch of nalinaly/PTOAS. Its exact commit
  is recorded in the version manifest above.
- The measured environment uses CANN 9.2.0-beta.2, Torch 2.10.0 and
  torch_npu 2.10.0.post2, rather than the session's initial CANN 9.0 setup.

## Local patch scope

1. Resolve shared InOut SSA aliases in explicit MIX Group member arguments.
2. Preserve descriptor-derived reshape metadata and inline Tile annotations.
3. Provide `system.signal_notify/signal_wait` for ordinary local GM signals;
   signal visibility does not replace arbitrary payload synchronization.
4. Correct PTOAS codegen handles for branch views, conditional in-place
   gathers and row/column loop carries.
5. Preserve in-place/view alias chains and traverse While bodies for nested
   For/Acc rules. Correct branch-local lifetimes and allocation window ranges
   while retaining capacity, interference and pinned-buffer constraints.
6. Correct signed/INDEX FloorMod emission to match Python and IR floor
   remainder semantics.
7. Retain proven constant slice-start folding from `d46a813ca` and dead
   Tile forwarding-cycle elimination from `d01de8523`.

Some fixes cover alternative lowering paths. Keeping them preserves the
validated compiler source; it does not make each fix a runtime prerequisite
for every CSA case. Relevant regression tests and mirrored development docs
are included. Temporary import-redirection scripts are not compiler code.

Related upstream records include PyPTO #3014, #3015, #3016, #3017, #3020,
#3026, #3027, #3028 and #3037. Publishing this reproduction branch neither
closes those issues nor creates an upstream pull request.

