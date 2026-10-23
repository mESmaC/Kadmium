# Offline installer checks. Run after cargo build --bin kadmium.
param([string]$Executable = 'target/debug/kadmium.exe')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = Split-Path $PSScriptRoot -Parent
$binary = [IO.Path]::GetFullPath((Join-Path $repoRoot $Executable))
$testRoot = Join-Path $repoRoot ('target/installer-tests-' + [Guid]::NewGuid().ToString('N'))
$fixture = Join-Path $testRoot 'fixture'
[void][IO.Directory]::CreateDirectory($fixture)
$version = (& $binary --version) -replace '^kadmium ', 'v'
if ($LASTEXITCODE -ne 0) { throw 'Build the CLI before testing installers.' }
Copy-Item -LiteralPath $binary -Destination (Join-Path $fixture 'kadmium.exe')
Copy-Item -LiteralPath $binary -Destination (Join-Path $fixture 'kadmium-desktop.exe')
$archive = Join-Path $fixture 'kadmium-windows-x86_64.zip'
Compress-Archive -LiteralPath @((Join-Path $fixture 'kadmium.exe'), (Join-Path $fixture 'kadmium-desktop.exe')) -DestinationPath $archive
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
Set-Content -LiteralPath (Join-Path $fixture 'SHA256SUMS') -Value "$digest  kadmium-windows-x86_64.zip"
# Replace network access only in this test scope; production installer is unchanged.
function Invoke-WebRequest {
    param([switch]$UseBasicParsing, [string]$Uri, [string]$OutFile, [int]$TimeoutSec)
    if (-not $Uri.StartsWith('https://github.com/mESmaC/Kadmium/releases/download/')) { throw "Unexpected URL: $Uri" }
    Copy-Item -LiteralPath (Join-Path $fixture ([Uri]$Uri).Segments[-1]) -Destination $OutFile
}
$installer = Join-Path $PSScriptRoot 'install.ps1'
$success = Join-Path $testRoot 'successful install with spaces'
& $installer -Version $version -InstallDir $success
$installed = Join-Path $success "releases/$version-windows-x86_64/kadmium.exe"
if (-not (Test-Path -LiteralPath $installed)) { throw 'Install did not produce a CLI.' }
$rejected = $false
try { & $installer -Version $version -InstallDir $success } catch { $rejected = $_.Exception.Message -like '*already installed*' }
if (-not $rejected) { throw 'Installer overwrote an existing release.' }
Set-Content -LiteralPath (Join-Path $fixture 'SHA256SUMS') -Value ((('0' * 64)) + '  kadmium-windows-x86_64.zip')
$rejected = $false
$bad = Join-Path $testRoot 'bad-checksum'
try { & $installer -Version $version -InstallDir $bad } catch { $rejected = $_.Exception.Message -like '*checksum mismatch*' }
if (-not $rejected -or (Test-Path -LiteralPath (Join-Path $bad "releases/$version-windows-x86_64"))) { throw 'Checksum rejection failed.' }
if (@(Get-ChildItem -LiteralPath $bad -Directory -Filter '.install.*').Count -ne 0) { throw 'Staging files were left behind after rejection.' }
# A checksummed archive still must contain exactly the two supported binaries.
Set-Content -LiteralPath (Join-Path $fixture 'unexpected.txt') -Value 'unexpected'
Compress-Archive -LiteralPath (Join-Path $fixture 'unexpected.txt') -DestinationPath $archive -Update
$digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
Set-Content -LiteralPath (Join-Path $fixture 'SHA256SUMS') -Value "$digest  kadmium-windows-x86_64.zip"
$rejected = $false
try { & $installer -Version $version -InstallDir (Join-Path $testRoot 'bad-members') } catch { $rejected = $_.Exception.Message -like '*Unexpected release archive entries*' }
if (-not $rejected) { throw 'Unexpected archive entry was accepted.' }
Write-Host 'PASS: installation, paths with spaces, no overwrite, checksum rejection, archive member checks, staging cleanup.'
Write-Host "Fixtures retained under $testRoot"
