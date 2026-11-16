[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Python,
    [string]$KeyFile,
    [string]$ProxyUrl = 'http://127.0.0.1:12450',
    [ValidateRange(0,2)][int]$NetworkRetries = 0,
    [ValidateRange(0,8)][int]$MaxNetworkResumes = 0,
    [switch]$ResponsesStream,
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
    if (-not $Execute) {
        & $Python -X utf8 -m prism.budgeted_judge inspect --manifest $Manifest
    } else {
        if (-not $env:OPENAI_API_KEY) {
            if (-not $KeyFile) { $KeyFile = Join-Path $repoRoot '计算资源\api.md' }
            $privateText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $KeyFile).Path)
            $keyMatches = [regex]::Matches($privateText, '(?im)^\s*openai\s+api\s*[:：]\s*(sk-[A-Za-z0-9_-]+)\s*$')
            if ($keyMatches.Count -ne 1) { throw 'Expected exactly one OpenAI API key in the private key file.' }
            $env:OPENAI_API_KEY = $keyMatches[0].Groups[1].Value
        }
        $env:HTTP_PROXY = $ProxyUrl
        $env:HTTPS_PROXY = $ProxyUrl
        $taskTransportArgs = @()
        if ($ResponsesStream) { $taskTransportArgs += '--responses-stream' }
        & $Python -X utf8 -m prism.budgeted_judge run --manifest $Manifest --execute --network-retries $NetworkRetries --max-network-resumes $MaxNetworkResumes @taskTransportArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "V2 Judge stopped with exit code $LASTEXITCODE. Keep the same ledger and raw results." }
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$before[$name],'Process') }
    $privateText = $null
    $keyMatches = $null
}
