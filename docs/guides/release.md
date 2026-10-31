# Maintainer release runbook

The tag-driven workflow builds and publishes CLI archives for four platforms and includes the desktop companion on Windows and macOS. A release is ready only when its version, binaries, documentation, and checks all refer to the same commit.

## 1. Verify the release candidate

1. Set `workspace.package.version` in `Cargo.toml` to the intended `X.Y.Z` and refresh lockfile version metadata through the normal Cargo build/check process if needed.
2. The Git tag must be exactly `vX.Y.Z`. `scripts/package_release.py` rejects a tag that does not equal `v{workspace.package.version}` and a CLI whose `--version` is not `kadmium X.Y.Z`.
3. Ensure the **exact tagged commit** has passed the full CI workflow, including Rust, Node, browser, installer, and business-model checks. The tag release workflow does not rerun all of those jobs.
4. Update installation/release documentation and confirm the archive names and platform qualifications in `docs/getting-started.md` remain accurate.
5. Verify the declared MIT/Apache-2.0 license texts are present in the repository and distributed source/archive contents. `docs/OPERATING_PLAN.md` records missing license files as a release blocker; do not treat Cargo metadata alone as satisfying that gate.

## 2. Build and package

The release workflow uses Rust 1.99.0 and Python 3.12 with these target/platform pairs:

| Platform | Rust target | Archive |
| :--- | :--- | :--- |
| `windows-x86_64` | `x86_64-pc-windows-msvc` | `kadmium-windows-x86_64.zip` |
| `macos-arm64` | `aarch64-apple-darwin` | `kadmium-macos-arm64.tar.gz` |
| `linux-x86_64` | `x86_64-unknown-linux-gnu` | `kadmium-linux-x86_64.tar.gz` |
| `linux-arm64` | `aarch64-unknown-linux-gnu` | `kadmium-linux-arm64.tar.gz` |

From the repository root, the per-platform commands are:

```bash
cargo build --locked --release --bin kadmium --target <TARGET>
# Windows and macOS only:
cargo build --locked --release --bin kadmium-desktop --target <TARGET>
python scripts/package_release.py --target <TARGET> --platform <PLATFORM> --output dist
```

The packager requires the expected executables under `target/<TARGET>/release/`, checks the CLI version, writes the archive, and emits `kadmium-<PLATFORM>.sha256`. Linux archives contain only `kadmium`; Windows/macOS archives also contain `kadmium-desktop`.

## 3. Test installers offline

Use the fixtures before publishing:

```bash
sh scripts/test_install.sh
```

```powershell
./scripts/test_installers.ps1 -Executable target/<TARGET>/release/kadmium.exe
```

The release workflow runs the corresponding fixture on every build platform. Installers verify archive checksums, executable names, and CLI versions, then install into versioned per-user directories without modifying `PATH`.

SHA-256 manifests detect corrupt or mismatched downloads. They are generated alongside the archives and are **not independent signatures or provenance attestations**.

## 4. Publish through the tag workflow

Pushing a matching `v*.*.*` tag starts `.github/workflows/release.yml`. After all platform builds succeed, the `publish` job downloads the artifacts, concatenates the per-platform checksums into `SHA256SUMS`, verifies every archive, and runs `gh release create` with:

- all `.zip` and `.tar.gz` archives;
- the aggregate `SHA256SUMS`;
- `scripts/install.sh` and `scripts/install.ps1`;
- generated release notes and `--verify-tag`.

A manual workflow dispatch builds artifacts but does not publish unless the ref is a version tag. Confirm the published assets before directing users to the checked installer commands.
