[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Packet,
    [Parameter(Mandatory=$true)][string]$Results,
    [string]$Python,
    [string]$KeyFile,
    [string]$ProxyUrl = 'http://127.0.0.1:12450',
    [ValidateRange(1,16)][int]$Workers = 4,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Python) { $Python = Join-Path $repoRoot 'tmp\assets-venv\Scripts\python.exe' }
if (-not $KeyFile) { $KeyFile = Join-Path $repoRoot '计算资源\api.md' }
$names = @('OPENAI_API_KEY','HTTP_PROXY','HTTPS_PROXY','PYTHONPATH','PYTHONUTF8')
$before = @{}
foreach ($name in $names) { $before[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    if (-not $Execute) { throw 'Local paid API execution requires -Execute.' }
    $packetMetadata = Get-Content -LiteralPath $Packet -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($packetMetadata.role -ne 'evaluation' -or $packetMetadata.settings.backend -ne 'openai') {
        throw 'This local runner is reserved for OpenAI evaluation; DeepSeek training runs on the cluster.'
    }
    $privateText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $KeyFile).Path)
    $keyMatches = [regex]::Matches($privateText, '(?im)^\s*openai\s+api\s*[:：]\s*(sk-[A-Za-z0-9_-]+)\s*$')
    if ($keyMatches.Count -ne 1) { throw 'Expected exactly one OpenAI API key in the private key file.' }
    $env:OPENAI_API_KEY = $keyMatches[0].Groups[1].Value
    $env:HTTP_PROXY = $ProxyUrl
    $env:HTTPS_PROXY = $ProxyUrl
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    $env:PYTHONUTF8 = '1'
    & $Python -m prism.local_judge run --packet $Packet --results $Results --workers $Workers --execute
    if ($LASTEXITCODE -ne 0) { throw "Local Judge stopped with exit code $LASTEXITCODE; existing call logs are preserved." }
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$before[$name],'Process') }
    $privateText = $null
    $keyMatches = $null
}
