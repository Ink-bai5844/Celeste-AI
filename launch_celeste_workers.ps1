param(
    [int]$Workers = 4,
    [string]$GameRoot = "$PSScriptRoot\GAME",
    [switch]$Hidden
)

$ErrorActionPreference = "Stop"

for ($i = 1; $i -le $Workers; $i++) {
    $workerDir = Join-Path $GameRoot "Celeste-$i"
    $exePath = Join-Path $workerDir "Celeste.exe"

    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Missing Celeste.exe for worker $i at $exePath"
    }

    $startInfo = @{
        FilePath = $exePath
        WorkingDirectory = $workerDir
    }

    if ($Hidden) {
        $startInfo.WindowStyle = "Hidden"
    }

    Start-Process @startInfo
    Write-Host "Started worker $i from $workerDir"
}
