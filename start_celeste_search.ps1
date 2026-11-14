param(
    [string]$GameDir = "$PSScriptRoot\GAME\Celeste-1",
    [int]$Port = 25102,
    [int]$Workers = [Math]::Min(12, [Math]::Max(1, [int]([Environment]::ProcessorCount / 2))),
    [double]$PlanSeconds = 8,
    [switch]$LaunchGame
)
# Compatibility entry point: the project now starts the manual-play assist tool.
& "$PSScriptRoot\start_celeste_assist.ps1" @PSBoundParameters
exit $LASTEXITCODE
