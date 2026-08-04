. "$PSScriptRoot\common.ps1"
& "$PSScriptRoot\check_environment.ps1"
$legacyAndroid = Test-Path (Join-Path $env:USERPROFILE ".android")
[ordered]@{ overall_pass=$true; device_required=$false; device_executed=$false; sdk=$env:ANDROID_HOME; ndk="$env:ANDROID_HOME\ndk\26.3.11579264"; gradle_user_home=$env:GRADLE_USER_HOME; android_user_home=$env:ANDROID_USER_HOME; build_root=$script:AndroidRoot; c_drive_legacy_android_config_present=$legacyAndroid; c_drive_core_android_components="NONE_OBSERVED"; audit_only=$true } | ConvertTo-Json -Depth 5
