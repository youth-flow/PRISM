param(
    [string]$Portal = 'http://223.108.218.59:3001/login',
    [string]$SshHost = '174.0.250.88',
    [string]$Report = 'local/network-report.json'
)
$ErrorActionPreference = 'Stop'
$proxy = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
function Probe-Http([string[]]$ProxyArgs) {
    $result = & curl.exe @ProxyArgs --connect-timeout 5 --max-time 8 --silent --show-error --output NUL --write-out '%{http_code}' $Portal 2>&1
    $exit = $LASTEXITCODE
    [pscustomobject]@{ ExitCode=$exit; Result=($result -join "`n") }
}
$direct = Probe-Http -ProxyArgs @('--noproxy','*')
$viaProxy = $null
if ($proxy.ProxyEnable -eq 1 -and $proxy.ProxyServer) {
    $server = $proxy.ProxyServer
    if ($server -match '(?:^|;)http=([^;]+)') { $server=$Matches[1] }
    if ($server -notmatch '://') { $server="http://$server" }
    $viaProxy = Probe-Http -ProxyArgs @('--proxy',$server,'--noproxy','')
}
$tcp = [Net.Sockets.TcpClient]::new()
try {
    $task = $tcp.ConnectAsync($SshHost,22)
    $connected = $task.Wait(5000) -and $tcp.Connected
} catch { $connected=$false } finally { $tcp.Dispose() }
$data = [ordered]@{ CheckedAt=(Get-Date -Format o); Portal=$Portal; Direct=$direct; SystemProxyEnabled=($proxy.ProxyEnable -eq 1); ViaProxy=$viaProxy; SshPort22Reachable=$connected; SettingsChanged=$false }
$directory=Split-Path -Parent $Report
if ($directory) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
$data | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Report -Encoding utf8
$data | ConvertTo-Json -Depth 5
