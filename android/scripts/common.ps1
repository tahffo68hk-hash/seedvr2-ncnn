$ErrorActionPreference = "Stop"
$script:AndroidRoot = Split-Path -Parent $PSScriptRoot
$env:ANDROID_HOME = "D:\Android\sdk"
$env:ANDROID_SDK_ROOT = "D:\Android\sdk"
$env:ANDROID_USER_HOME = "D:\Android\user-config"
$env:JAVA_HOME = "D:\Android\jdk"
$env:GRADLE_USER_HOME = "D:\Android\gradle-cache"
$env:Path = @("D:\Android\jdk\bin", "D:\Android\sdk\platform-tools", "D:\Android\sdk\cmdline-tools\latest\bin", "D:\Android\ninja", $env:Path) -join ";"
$env:TEMP = "D:\Android\temp"
$env:TMP = "D:\Android\temp"
New-Item -ItemType Directory -Force -Path $env:ANDROID_USER_HOME, $env:GRADLE_USER_HOME, $env:TEMP, (Join-Path $script:AndroidRoot "..\.phase18_build_logs") | Out-Null
function Assert-Path([string]$Path, [string]$Name) { if (-not (Test-Path -LiteralPath $Path)) { throw "$Name missing: $Path" } }
function Invoke-Gradle([string[]]$Arguments, [string]$LogName) {
    Push-Location $script:AndroidRoot
    try { & .\gradlew.bat @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $script:AndroidRoot "..\.phase18_build_logs\$LogName"); if ($LASTEXITCODE -ne 0) { throw "Gradle failed with exit $LASTEXITCODE" } }
    finally { Pop-Location }
}
