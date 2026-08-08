$ErrorActionPreference = "Stop"

$bridgeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$executable = Join-Path $bridgeRoot "bin\acl-fixed\mimic-tear-controller-bridge.exe"
$clientSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value

if (-not (Test-Path -LiteralPath $executable)) {
    throw "Controller bridge is not built. Run bridges\hidmaestro\bootstrap.ps1 first."
}

Start-Process `
    -FilePath $executable `
    -ArgumentList @(
        "serve",
        "--watchdog-ms",
        "250",
        "--client-sid",
        $clientSid
    ) `
    -Verb RunAs
