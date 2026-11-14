param(
    [ValidateSet('plan','batch','catalog')][string]$Mode = 'plan',
    [double]$Seconds = 120,
    [int]$Workers = 4,
    [string]$Map = "$PSScriptRoot\GAME\Celeste-1\Content\Maps\1-ForsakenCity.bin",
    [string]$Room = '',
    [string]$Target = '',
    [string]$Resume = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$searchArgs = @("$PSScriptRoot\Celeste-Search\search.py", $Mode)
if ($Mode -ne 'catalog') {
    $searchArgs += @('--seconds', $Seconds, '--workers', $Workers)
}
if ($Mode -eq 'plan') {
    $searchArgs += @('--map', $Map)
    if ($Room) { $searchArgs += @('--room', $Room) }
    if ($Target) { $searchArgs += @('--target', $Target) }
    if ($Resume) { $searchArgs += @('--resume', $Resume) }
}
if ($Output) { $searchArgs += @('--output', $Output) }
& python @searchArgs
exit $LASTEXITCODE
