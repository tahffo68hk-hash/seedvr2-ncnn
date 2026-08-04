param([string]$Abi = "arm64-v8a")
. "$PSScriptRoot\common.ps1"
if ($Abi -ne "arm64-v8a") { throw "Unsupported ABI: $Abi" }
Assert-Path $env:ANDROID_HOME "Android SDK"
Assert-Path "$env:ANDROID_HOME\ndk\26.3.11579264" "Android NDK"
Assert-Path "$env:ANDROID_HOME\cmake\3.22.1\bin\cmake.exe" "CMake"
Assert-Path "$env:JAVA_HOME\bin\java.exe" "JDK"
Assert-Path "D:\Android\ninja\ninja.exe" "Ninja"
foreach ($value in @($env:ANDROID_HOME,$env:ANDROID_SDK_ROOT,$env:ANDROID_USER_HOME,$env:JAVA_HOME,$env:GRADLE_USER_HOME,$env:TEMP)) { if (-not $value.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) { throw "Core path is not on D: $value" } }
if ((Resolve-Path $script:AndroidRoot).Path -ne (Resolve-Path (Join-Path $PSScriptRoot "..")).Path) { throw "Unexpected Android project directory" }
& "$env:JAVA_HOME\bin\java.exe" -version
Write-Output "ENVIRONMENT_OK abi=$Abi gradle_user_home=$env:GRADLE_USER_HOME"
