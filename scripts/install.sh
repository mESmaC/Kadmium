#!/bin/sh
# Install a verified tagged release without root or changes to shell startup files.
set -eu

fail() { printf '%s\n' "Kadmium: $*" >&2; exit 1; }
version=''
launch=0
install_root=${KADMIUM_INSTALL_DIR:-"${XDG_DATA_HOME:-$HOME/.local/share}/kadmium"}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --version) [ "$#" -ge 2 ] || fail '--version needs a tag'; version=$2; shift 2 ;;
        --install-dir) [ "$#" -ge 2 ] || fail '--install-dir needs a path'; install_root=$2; shift 2 ;;
        --launch) launch=1; shift ;;
        --help) printf '%s\n' 'Usage: sh install.sh --version vX.Y.Z [--install-dir PATH] [--launch]'; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done
printf '%s\n' "$version" | LC_ALL=C grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9][A-Za-z0-9.-]*)?$' || fail 'supply an explicit release tag with --version vX.Y.Z'
case "$(uname -s):$(uname -m)" in
    Darwin:arm64) platform=macos-arm64 ;;
    Linux:x86_64) platform=linux-x86_64 ;;
    Linux:aarch64|Linux:arm64) platform=linux-arm64 ;;
    *) fail 'supported downloads: Apple Silicon macOS, Linux x86_64/ARM64 (glibc); Windows x64 uses install.ps1' ;;
esac
for required in curl tar awk mktemp; do command -v "$required" >/dev/null 2>&1 || fail "install $required first"; done
if command -v sha256sum >/dev/null 2>&1; then
    hash_tool=sha256sum
elif command -v shasum >/dev/null 2>&1; then
    hash_tool=shasum
else
    fail 'install sha256sum or shasum first'
fi
asset="kadmium-$platform.tar.gz"
base=${KADMIUM_RELEASE_BASE_URL:-"https://github.com/mESmaC/Kadmium/releases/download/$version"}
case "$base" in https://*) ;; *) fail 'release base URL must use HTTPS' ;; esac
mkdir -p "$install_root/releases"
install_root=$(cd "$install_root" && pwd -P)
destination="$install_root/releases/$version-$platform"
[ ! -e "$destination" ] || fail "release already installed at $destination; use its kadmium executable or choose another install directory"
stage=$(mktemp -d "$install_root/.install.XXXXXXXX")
# Only remove the directory created by mktemp within the resolved install root.
cleanup() { case "$stage" in "$install_root"/.install.*) rm -rf -- "$stage" ;; esac; }
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
curl --fail --show-error --silent --location --proto '=https' --proto-redir '=https' --connect-timeout 15 --max-time 900 "$base/$asset" -o "$stage/$asset"
curl --fail --show-error --silent --location --proto '=https' --proto-redir '=https' --connect-timeout 15 --max-time 60 "$base/SHA256SUMS" -o "$stage/SHA256SUMS"
expected=$(awk -v file="$asset" '$2 == file {count++; hash=$1} END {if (count != 1) exit 1; print hash}' "$stage/SHA256SUMS") || fail "SHA256SUMS must name $asset exactly once"
printf '%s\n' "$expected" | LC_ALL=C grep -Eq '^[a-fA-F0-9]{64}$' || fail 'invalid SHA-256 manifest'
if [ "$hash_tool" = sha256sum ]; then
    actual=$(sha256sum "$stage/$asset" | awk '{print $1}')
else
    actual=$(shasum -a 256 "$stage/$asset" | awk '{print $1}')
fi
[ "$(printf '%s' "$expected" | tr A-F a-f)" = "$actual" ] || fail 'download checksum mismatch; nothing installed'
# These archives contain only top-level regular executable files, never arbitrary paths.
tar -tzf "$stage/$asset" > "$stage/members"
count=1
[ "$platform" != macos-arm64 ] || count=2
awk -v expected="$count" '
    $0 != "kadmium" && $0 != "kadmium-desktop" {exit 1}
    seen[$0]++ {exit 1}
    END {if (NR != expected || !seen["kadmium"]) exit 1}
' "$stage/members" || fail 'unexpected archive entries'
tar -tvzf "$stage/$asset" > "$stage/types"
awk 'substr($1, 1, 1) != "-" {exit 1}' "$stage/types" || fail 'archive contains non-regular entries'
mkdir "$stage/bin"
tar -xzf "$stage/$asset" -C "$stage/bin"
chmod 755 "$stage/bin/kadmium"
[ "$platform" != macos-arm64 ] || chmod 755 "$stage/bin/kadmium-desktop"
actual_version=$("$stage/bin/kadmium" --version) || fail 'binary cannot run on this computer (check OS/glibc requirements)'
[ "$actual_version" = "kadmium ${version#v}" ] || fail 'binary version does not match requested release'
# Claim the version directory without replacing an existing installation.
mkdir "$destination" || fail 'another installer already created this release directory'
mv "$stage/bin/kadmium" "$destination/kadmium"
if [ "$platform" = macos-arm64 ]; then mv "$stage/bin/kadmium-desktop" "$destination/kadmium-desktop"; fi
printf '\nInstalled %s\nLaunch: "%s/kadmium" launch\nPreview: "%s/kadmium" launch --plan\n' "$actual_version" "$destination" "$destination"
if [ "$launch" = 1 ]; then "$destination/kadmium" launch; fi
