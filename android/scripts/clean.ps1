param([switch]$LocalNcnn)
. "$PSScriptRoot\common.ps1"
if (Test-Path "$script:AndroidRoot\gradlew.bat") { Invoke-Gradle @("--no-daemon", "clean") "clean.log" }
if ($LocalNcnn) {
    $target = (Resolve-Path (Join-Path $script:AndroidRoot "..\.phase18_ncnn_src") -ErrorAction SilentlyContinue)
    if ($target -and $target.Path.StartsWith((Resolve-Path (Join-Path $script:AndroidRoot "..")).Path)) { Remove-Item -LiteralPath $target.Path -Recurse -Force }
}
