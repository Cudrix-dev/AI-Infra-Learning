param(
    [Parameter(Mandatory)][string]$Remote,
    [Parameter(Mandatory)][ValidateRange(1, 65535)][int]$Port,
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunId,
    [string]$RemoteProject = '/workspace/AI-Infra-Learning',
    [string]$IdentityFile,
    [switch]$LegacyScp,
    [switch]$IncludeLaunchLog
)
# Day 09: download the archive written by run_server_once.sh, check its SHA-256, extract it, and check the internal
# SHA256SUMS manifest. Destination: outputs/day09_downloads/<RunId>/ in this project.
#   experiments/day09_mixed_precision/download_results.ps1 -Remote root@HOST -Port PORT -RunId RUN_ID

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Destination = Join-Path $Project "outputs/day09_downloads/$RunId"
$Name = "day09_$RunId.tar.gz"
$Archive = Join-Path $Destination $Name
$Export = "$RemoteProject/outputs/day09_exports"
$ScpOptions = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', "$Port")
if ($LegacyScp) {
    $ScpOptions += '-O'
}
if ($IdentityFile) {
    $ScpOptions += @('-i', $IdentityFile)
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$Files = @($Name, "$Name.sha256")
if ($IncludeLaunchLog) {
    $Files += "day09_$RunId.log"
}
$Sources = foreach ($File in $Files) { "${Remote}:$Export/$File" }
& scp @ScpOptions @Sources "$Destination/"
if ($LASTEXITCODE -ne 0) { throw 'SCP failed. Check SSH access and whether the requested exports exist.' }

$Expected = ((Get-Content -LiteralPath "$Archive.sha256" -TotalCount 1) -split '\s+')[0]
$Actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash
if ($Expected -ne $Actual) { throw "SHA-256 mismatch for $Archive" }
Write-Output "Archive SHA-256 OK: $Actual"

& tar -xzf $Archive -C $Destination
if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $Archive" }

# Every file listed in the server-side manifest must exist here with the same hash.
$Session = Join-Path $Destination "session_$RunId"
$Bad = 0
$Lines = Get-Content -LiteralPath (Join-Path $Session 'SHA256SUMS')
foreach ($Line in $Lines) {
    $Hash, $Rel = $Line -split '  ', 2
    $Path = Join-Path $Destination $Rel
    if (-not (Test-Path -LiteralPath $Path) -or (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Hash.ToUpper()) {
        Write-Output "MISMATCH: $Rel"
        $Bad++
    }
}
if ($Bad -ne 0) { throw "$Bad file(s) failed the internal manifest check" }
Write-Output "Manifest OK: $($Lines.Count) files verified. Extracted to $Destination"
Get-Content -LiteralPath (Join-Path $Session 'status.txt')
foreach ($Flag in @('CORE_SUCCESS.txt', 'SUCCESS.txt')) {
    if (Test-Path -LiteralPath (Join-Path $Session $Flag)) {
        Get-Content -LiteralPath (Join-Path $Session $Flag)
    }
}
