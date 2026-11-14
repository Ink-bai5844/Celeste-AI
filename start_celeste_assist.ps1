param(
    [string]$GameDir = "$PSScriptRoot\GAME\Celeste-1",
    [int]$Port = 25102,
    [int]$Workers = [Math]::Min(12, [Math]::Max(1, [int]([Environment]::ProcessorCount / 2))),
    [double]$PlanSeconds = 8,
    [switch]$LaunchGame
)
$ErrorActionPreference = 'Stop'
$project = "$PSScriptRoot\Celeste-Instrumentation\CelesteInstrumentation\CelesteInstrumentation\CelesteInstrumentation.csproj"
& dotnet build $project -c Release "-p:CelesteGameDir=$GameDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$assistArgs = @("$PSScriptRoot\Celeste-Search\search.py",'assist','--game-dir',$GameDir,
    '--port',$Port,'--workers',$Workers,'--plan-seconds',$PlanSeconds)
if ($LaunchGame) { $assistArgs += '--launch' }
& python @assistArgs
exit $LASTEXITCODE
