param(
    [Parameter(Mandatory)][string]$Remote,
    [Parameter(Mandatory)][ValidateRange(1, 65535)][int]$Port,
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunId,
    [string]$RemoteProject = '/workspace/AI-Infra-Learning',
    [string]$IdentityFile,
    [switch]$LegacyScp,
    [switch]$IncludeLaunchLog
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Destination = Join-Path $Project "outputs/day08_downloads/$RunId"
$Name = "day08_$RunId.tar.gz"
$Archive = Join-Path $Destination $Name
$Export = "$RemoteProject/outputs/day08_exports"
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
    $Files += "day08_$RunId.log"
}
$Sources = foreach ($File in $Files) { "${Remote}:$Export/$File" }
& scp @ScpOptions @Sources "$Destination/"
if ($LASTEXITCODE -ne 0) { throw 'SCP failed. Check SSH access and whether the requested exports exist.' }

$Expected = ((Get-Content -LiteralPath "$Archive.sha256" -TotalCount 1) -split '\s+')[0]
$Actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash
if ($Expected -ne $Actual) { throw "SHA-256 mismatch for $Archive" }

& tar -xzf $Archive -C $Destination
if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $Archive" }

Write-Output "Verified and extracted to $Destination"
$Session = Join-Path $Destination "session_$RunId"
Get-Content -LiteralPath (Join-Path $Session 'status.txt')
if (Test-Path -LiteralPath (Join-Path $Session 'CORE_SUCCESS.txt')) {
    Get-Content -LiteralPath (Join-Path $Session 'CORE_SUCCESS.txt')
}
