[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Packet,
    [Parameter(Mandatory=$true)][string]$Results,
    [Parameter(Mandatory=$true)][string]$Ledger,
    [double]$BudgetUsd,
    [ValidateRange(1,16)][int]$Workers = 8,
    [string]$Python,
    [string]$KeyFile,
    [string]$ProxyUrl = 'http://127.0.0.1:12450',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
$names = @('OPENAI_API_KEY','HTTP_PROXY','HTTPS_PROXY','PYTHONPATH','PYTHONUTF8')
$before = @{}
foreach ($name in $names) { $before[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    $env:PYTHONUTF8 = '1'
    $taskArgs = @('-X','utf8','-m','prism.v3.evaluation','run-local','--packet',$Packet,
                  '--results',$Results,'--ledger',$Ledger,'--workers',"$Workers")
    if ($Execute) {
        if ($BudgetUsd -le 0) { throw 'Supply the explicit positive V3 evaluation BudgetUsd.' }
        if (-not $env:OPENAI_API_KEY) {
            if (-not $KeyFile) { throw 'Set OPENAI_API_KEY locally or supply a private KeyFile.' }
            $privateText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $KeyFile).Path)
            $keyMatches = [regex]::Matches($privateText, '(?im)^\s*(?:openai\s+api\s*[:：]\s*)?(sk-[A-Za-z0-9_-]+)\s*$')
            if ($keyMatches.Count -ne 1) { throw 'Expected exactly one OpenAI API key in the private file.' }
            $env:OPENAI_API_KEY = $keyMatches[0].Groups[1].Value
        }
        if ($ProxyUrl) { $env:HTTP_PROXY = $ProxyUrl; $env:HTTPS_PROXY = $ProxyUrl }
        $taskArgs += @('--budget-usd',"$BudgetUsd",'--execute')
    }
    & $Python @taskArgs
    if ($LASTEXITCODE -ne 0) { throw "V3 Judge stopped with exit code $LASTEXITCODE. Preserve ledger, results and billing snapshot." }
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$before[$name],'Process') }
    $privateText = $null
    $keyMatches = $null
}
