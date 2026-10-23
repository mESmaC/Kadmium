"""Package only the documented release binaries, with matching installer asset names."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib
import zipfile


TARGETS = {
    "windows-x86_64": "x86_64-pc-windows-msvc",
    "macos-arm64": "aarch64-apple-darwin",
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=TARGETS, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if TARGETS[args.platform] != args.target:
        parser.error("platform does not match target")
    with open("Cargo.toml", "rb") as manifest:
        version = tomllib.load(manifest)["workspace"]["package"]["version"]
    tag = os.environ.get("GITHUB_REF", "")
    if tag.startswith("refs/tags/") and tag != f"refs/tags/v{version}":
        parser.error(f"release tag must match workspace version v{version}")
    suffix = ".exe" if args.platform.startswith("windows-") else ""
    names = [f"kadmium{suffix}"]
    if not args.platform.startswith("linux-"):
        names.append(f"kadmium-desktop{suffix}")
    binaries = [Path("target") / args.target / "release" / name for name in names]
    for binary in binaries:
        if not binary.is_file():
            parser.error(f"missing built executable: {binary}")
    actual = subprocess.check_output([str(binaries[0].resolve()), "--version"], text=True).strip()
    if actual != f"kadmium {version}":
        parser.error(f"unexpected executable version: {actual!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    extension = ".zip" if suffix else ".tar.gz"
    archive = args.output / f"kadmium-{args.platform}{extension}"
    if suffix:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for binary in binaries:
                bundle.write(binary, binary.name)
    else:
        with tarfile.open(archive, "w:gz") as bundle:
            for binary in binaries:
                info = bundle.gettarinfo(str(binary), arcname=binary.name)
                info.mode = 0o755
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with binary.open("rb") as contents:
                    bundle.addfile(info, contents)
    with archive.open("rb") as contents:
        digest = hashlib.file_digest(contents, "sha256").hexdigest()
    (args.output / f"kadmium-{args.platform}.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="ascii"
    )
    print(f"Packaged {archive.name} ({digest})")


if __name__ == "__main__":
    main()
