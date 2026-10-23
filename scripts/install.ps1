# Install a verified tagged release under the current user's profile. No elevation required.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9][A-Za-z0-9.-]*)?$')]
    [string]$Version,
    [string]$InstallDir = $(if ($env:KADMIUM_INSTALL_DIR) { $env:KADMIUM_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Kadmium' }),
    [switch]$Launch
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64' -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'This installer supports Windows x64. Use install.sh for Apple Silicon macOS or Linux x64/ARM64.'
}
$asset = 'kadmium-windows-x86_64.zip'
$base = if ($env:KADMIUM_RELEASE_BASE_URL) { $env:KADMIUM_RELEASE_BASE_URL.TrimEnd('/') } else { "https://github.com/mESmaC/Kadmium/releases/download/$Version" }
$uri = [Uri]$base
if (-not $uri.IsAbsoluteUri -or $uri.Scheme -ne 'https' -or $uri.UserInfo -or $uri.Query -or $uri.Fragment) { throw 'Release base URL must be an absolute HTTPS URL without credentials, query or fragment.' }
$installRoot = [IO.Path]::GetFullPath($InstallDir)
$releases = Join-Path $installRoot 'releases'
[void][IO.Directory]::CreateDirectory($releases)
$destination = Join-Path $releases "$Version-windows-x86_64"
if (Test-Path -LiteralPath $destination) { throw "Release already installed at $destination. Use its kadmium.exe or choose another install directory." }
$stage = Join-Path $installRoot ('.install.' + [Guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($stage)
$oldProtocol = [Net.ServicePointManager]::SecurityProtocol
try {
    [Net.ServicePointManager]::SecurityProtocol = $oldProtocol -bor [Net.SecurityProtocolType]::Tls12
    $archive = Join-Path $stage $asset
    $manifest = Join-Path $stage 'SHA256SUMS'
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$asset" -OutFile $archive -TimeoutSec 900
    Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile $manifest -TimeoutSec 60
    $manifestEntries = @(Get-Content -LiteralPath $manifest | Where-Object { $_ -match ('^[a-fA-F0-9]{64}\s+' + [Regex]::Escape($asset) + '$') })
    if ($manifestEntries.Count -ne 1) { throw "SHA256SUMS must name $asset exactly once." }
    $expected = ($manifestEntries[0] -split '\s+')[0]
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw 'Download checksum mismatch; nothing installed.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    $bin = Join-Path $stage 'bin'
    [void][IO.Directory]::CreateDirectory($bin)
    try {
        $names = @($zip.Entries | ForEach-Object { $_.FullName })
        if ($names.Count -ne 2 -or @($names | Sort-Object -Unique).Count -ne 2 -or $names -notcontains 'kadmium.exe' -or $names -notcontains 'kadmium-desktop.exe') { throw 'Unexpected release archive entries.' }
        foreach ($entry in $zip.Entries) {
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $bin $entry.FullName), $false)
        }
    } finally { $zip.Dispose() }
    $actualVersion = & (Join-Path $bin 'kadmium.exe') --version
    if ($LASTEXITCODE -ne 0 -or $actualVersion -ne "kadmium $($Version.Substring(1))") { throw 'Binary cannot run or its version does not match the requested release.' }
    # Directory.Move fails if the destination appeared concurrently and preserves existing releases.
    [IO.Directory]::Move($bin, $destination)
    $executable = Join-Path $destination 'kadmium.exe'
    Write-Host "Installed $actualVersion"
    Write-Host "Launch: & '$($executable.Replace("'", "''"))' launch"
    Write-Host "Preview: & '$($executable.Replace("'", "''"))' launch --plan"
} finally {
    [Net.ServicePointManager]::SecurityProtocol = $oldProtocol
    # Verify the exact generated staging directory stays within this installation before cleanup.
    $resolvedStage = [IO.Path]::GetFullPath($stage)
    if ($resolvedStage.StartsWith($installRoot.TrimEnd('\') + '\.install.', [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $resolvedStage)) {
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}
if ($Launch) {
    & $executable launch
    if ($LASTEXITCODE -ne 0) { throw "Kadmium exited with code $LASTEXITCODE." }
}
