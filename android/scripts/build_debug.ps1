. "$PSScriptRoot\common.ps1"
& "$PSScriptRoot\check_environment.ps1"
& "$PSScriptRoot\prepare_ncnn.ps1"
Invoke-Gradle @("--no-daemon", "clean", ":app:assembleDebug") "debug-build.log"
