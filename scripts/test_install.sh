#!/bin/sh
# Offline tests: source the unmodified installer in a subprocess with network/OS fixtures.
set -eu
repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$repo_root/target"
test_root=$(mktemp -d "$repo_root/target/installer-shell.XXXXXXXX")
trap 'rm -rf -- "$test_root"' EXIT
fixture="$test_root/fixture"
fixture_platform=linux-x86_64
mkdir "$fixture"
printf '#!/bin/sh\nprintf "kadmium 0.1.0\\n"\n' > "$fixture/kadmium"
chmod 755 "$fixture/kadmium"
tar -czf "$fixture/kadmium-linux-x86_64.tar.gz" -C "$fixture" kadmium
manifest() {
    fixture_asset="kadmium-$fixture_platform.tar.gz"
    if command -v sha256sum >/dev/null 2>&1; then
        fixture_hash=$(sha256sum "$fixture/$fixture_asset" | awk '{print $1}')
    else
        fixture_hash=$(shasum -a 256 "$fixture/$fixture_asset" | awk '{print $1}')
    fi
    printf '%s  %s\n' "$fixture_hash" "$fixture_asset" > "$fixture/SHA256SUMS"
}
manifest
uname() {
    case "$fixture_platform:$1" in
        linux-x86_64:-s) printf 'Linux\n' ;; linux-x86_64:-m) printf 'x86_64\n' ;;
        macos-arm64:-s) printf 'Darwin\n' ;; macos-arm64:-m) printf 'arm64\n' ;;
    esac
}
curl() {
    download_url=''
    download_output=''
    while [ "$#" -gt 0 ]; do
        case "$1" in
            https://*) download_url=$1; shift ;;
            -o) download_output=$2; shift 2 ;;
            *) shift ;;
        esac
    done
    [ -n "$download_url" ] && [ -n "$download_output" ]
    cp "$fixture/${download_url##*/}" "$download_output"
}
install_case() (
    KADMIUM_INSTALL_DIR="$test_root/$1"
    export KADMIUM_INSTALL_DIR
    set -- --version v0.1.0
    . "$repo_root/scripts/install.sh"
)
install_case 'success with spaces'
[ -x "$test_root/success with spaces/releases/v0.1.0-linux-x86_64/kadmium" ]
if install_case 'success with spaces'; then printf 'FAIL: overwrite accepted\n'; exit 1; fi
printf '%064d  kadmium-linux-x86_64.tar.gz\n' 0 > "$fixture/SHA256SUMS"
if install_case checksum; then printf 'FAIL: bad checksum accepted\n'; exit 1; fi
[ ! -e "$test_root/checksum/releases/v0.1.0-linux-x86_64" ]
printf 'extra\n' > "$fixture/unexpected"
tar -czf "$fixture/kadmium-linux-x86_64.tar.gz" -C "$fixture" kadmium unexpected
manifest
if install_case members; then printf 'FAIL: extra archive member accepted\n'; exit 1; fi
fixture_platform=macos-arm64
cp "$fixture/kadmium" "$fixture/kadmium-desktop"
tar -czf "$fixture/kadmium-macos-arm64.tar.gz" -C "$fixture" kadmium kadmium-desktop
manifest
install_case macos
[ -x "$test_root/macos/releases/v0.1.0-macos-arm64/kadmium-desktop" ]
printf 'PASS: Linux/macOS install, spaces, no overwrite, bad checksum and extra member rejection.\n'
