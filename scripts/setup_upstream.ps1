# Clone the base-predictor code at the commit used for the paper-facing results,
# then verify the files CGSR imports. The base predictor is ReCLIP++ (Wang & Kang);
# its code is not redistributed in this repository.
# Usage: powershell -File scripts\setup_upstream.ps1 [destination]   (default: third_party\base_predictor)
param([string]$Dest = "third_party\base_predictor")
$ErrorActionPreference = "Stop"

$Url = "https://github.com/dogehhh/ReCLIP.git"
$Commit = "46298007fcfa75b0ca41c3a9a8420b2dfa800475"

if (Test-Path $Dest) {
    Write-Host "Using existing directory: $Dest"
} else {
    git clone $Url $Dest
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
git -C $Dest -c advice.detachedHead=false checkout $Commit
if ($LASTEXITCODE -ne 0) { throw "git checkout $Commit failed" }
python tools\verify_assets.py --base-code-root $Dest --skip-candidate
if ($LASTEXITCODE -ne 0) { throw "base-code verification failed" }
