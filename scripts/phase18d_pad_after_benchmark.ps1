param([string]$Serial = "", [string]$OutputDir = ".phase18d_pad_evidence")
$ErrorActionPreference = "Stop"
$adbArgs = @(); if ($Serial) { $adbArgs += @("-s", $Serial) }
if ((& adb @adbArgs get-state).Trim() -ne "device") { throw "ADB_DEVICE_NOT_READY" }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
& adb @adbArgs shell dumpsys battery | Set-Content "$OutputDir/battery_after.txt"
& adb @adbArgs shell dumpsys thermalservice | Set-Content "$OutputDir/thermal_after.txt"
& adb @adbArgs shell dumpsys meminfo com.seededge.runtime | Set-Content "$OutputDir/meminfo_after.txt"
& adb @adbArgs logcat -d -v threadtime | Set-Content "$OutputDir/logcat_after.txt"
@{schema_version=1;phase="18D";gate="pad_after_benchmark";device_state="EXECUTED";collected_at_utc=[DateTime]::UtcNow.ToString("o");serial=$Serial} |
  ConvertTo-Json | Set-Content "$OutputDir/summary.json"
