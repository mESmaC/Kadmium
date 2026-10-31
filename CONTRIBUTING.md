# Contributing to Kadmium

## Development setup

Install `rustup` and use the repository-pinned Rust 1.99.0 toolchain; `rust-toolchain.toml` selects it automatically with `rustfmt` and Clippy. Use locked dependency resolution for checks that match CI.

Linux native-WebView development additionally requires WebKitGTK 4.1 and GTK 3 development packages. On Debian/Ubuntu CI these are `libwebkit2gtk-4.1-dev` and `libgtk-3-dev`. The excluded CUDA device harness has separate Rust/CUDA requirements; see `crates/kadmium-cuda/README.md`.

Ordinary Intel NPU host tests run with the workspace. The opt-in hardware suite requires a supported Intel NPU/driver and is run explicitly with `cargo test --locked -p kadmium-npu --test intel_hardware -- --ignored`; see `crates/kadmium-npu/README.md` for its arithmetic and platform limitations.

## Rust checks

Run the relevant checks for your change. The full host-workspace sequence is:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

On Linux with the native-WebView dependencies installed, also run:

```bash
cargo clippy -p kadmium-desktop -p kadmium-condenser --features linux-native --all-targets --locked -- -D warnings
```

`cargo test --workspace --locked` covers host Rust unit, integration, and documentation tests. It does not cover the excluded CUDA device harness, Node workflows, browser tests, installer fixtures, or Python business-model checks.

## Other CI checks

Node-based checks use Node 24:

```bash
node --test crates/kadmium-condenser/tests/workflow.test.cjs
```

For the Control Room browser workflows:

```bash
cd crates/kadmium-gateway/tests/ui
npm ci
npx playwright install --with-deps chromium
npm test
```

See `crates/kadmium-gateway/tests/ui/README.md` for browser-test configuration and artifacts.

The separate Control Room navigation and request-recovery regressions use a router fixture. From the repository root:

```bash
npm install --prefix target/ui-test --no-package-lock --no-audit --no-fund playwright@1.64.1
node target/ui-test/node_modules/playwright/cli.js install --with-deps chromium
cargo build -p kadmium-gateway --example control_room_fixture --locked
node --test crates/kadmium-gateway/tests/control_room.test.cjs
```

The fixture test uses installed Edge on Windows and Chromium elsewhere. For the workspace suite on Windows, set `KADMIUM_TEST_BROWSER=msedge` to use installed Edge.

Python business-model checks use Python 3.12:

```bash
python scripts/test_business_model.py
python scripts/business_model.py --check
```

Installer fixture checks are platform-specific:

```bash
sh scripts/test_install.sh
```

```powershell
# Build target/debug/kadmium.exe first.
./scripts/test_installers.ps1 -Executable target/debug/kadmium.exe
```

CI also smoke-tests CLI and Condenser reference commands. The authoritative matrix is `.github/workflows/ci.yml`; run the checks affected by your change before opening a pull request.

## Documentation changes

Keep user and repository documentation synchronized with implementation:

- Distinguish implemented reference behavior from modelled/simulated output and target architecture.
- Do not turn fixture prices, synthetic throughput, estimated latency, or sample failover results into measured guarantees.
- Update command references from clap definitions and API references from route handlers.
- Use the pinned toolchain and exact command flags/defaults.
- Check repository-relative Markdown links and anchors after moving or renaming sections.
- Record performance claims in `docs/BENCHMARKS_2026-10.md` with provenance and Measured/Derived/Modelled labels.

Avoid unrelated formatting or rewrites. Keep changes focused and include validation results in the pull-request description.
