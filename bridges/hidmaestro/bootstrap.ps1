$ErrorActionPreference = "Stop"

$version = "1.5.0"
$expectedSha256 = "F527A1983898FCD8360F9DBEC8FB20B852A48A018693F82CA2E78B2C84187AA8"
$bridgeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dependencyRoot = Join-Path $bridgeRoot ".deps"
$archivePath = Join-Path $dependencyRoot "HIDMaestro-v$version.zip"
$sdkPath = Join-Path $dependencyRoot "HIDMaestro-v$version"
$sdkDll = Join-Path $sdkPath "HIDMaestro.Core.dll"
$publishPath = Join-Path $bridgeRoot "bin\acl-fixed"
$downloadUrl = "https://github.com/hifihedgehog/HIDMaestro/releases/download/v$version/HIDMaestro-v$version.zip"

New-Item -ItemType Directory -Path $dependencyRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
}

$actualSha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
if ($actualSha256 -ne $expectedSha256) {
    throw "HIDMaestro archive hash mismatch. Expected $expectedSha256, got $actualSha256."
}

if (-not (Test-Path -LiteralPath $sdkDll)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $sdkPath -Force
}

dotnet publish (Join-Path $bridgeRoot "AiPlayer.ControllerBridge.csproj") `
    --configuration Release `
    --runtime win-x64 `
    --self-contained false `
    --output $publishPath
