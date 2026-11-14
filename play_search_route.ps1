param(
    [string]$Route = "$PSScriptRoot\Celeste-Search\outputs\route.json",
    [string]$GameDir = "$PSScriptRoot\GAME\Celeste-1",
    [int]$Port = 25101,
    [double]$Tolerance = 16
)
$ErrorActionPreference = 'Stop'
$project = "$PSScriptRoot\Celeste-Instrumentation\CelesteInstrumentation\CelesteInstrumentation\CelesteInstrumentation.csproj"
& dotnet build $project -c Release "-p:CelesteGameDir=$GameDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "$PSScriptRoot\Celeste-Search\search.py" play $Route --game-dir $GameDir --launch --install-bridge --port $Port --tolerance $Tolerance
exit $LASTEXITCODE
