param([string]$Serial = "", [string]$OutputDir = ".phase18d_pad_evidence", [string]$TaskId = "")
$ErrorActionPreference = "Stop"
$adbArgs = @(); if ($Serial) { $adbArgs += @("-s", $Serial) }
if ((& adb @adbArgs get-state).Trim() -ne "device") { throw "ADB_DEVICE_NOT_READY" }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
& adb @adbArgs shell dumpsys battery | Set-Content "$OutputDir/battery_after.txt"
& adb @adbArgs shell dumpsys thermalservice | Set-Content "$OutputDir/thermal_after.txt"
& adb @adbArgs shell dumpsys meminfo com.seededge.runtime | Set-Content "$OutputDir/meminfo_after.txt"
& adb @adbArgs shell cat /proc/meminfo | Set-Content "$OutputDir/proc_meminfo_after.txt"
& adb @adbArgs shell df -k /data | Set-Content "$OutputDir/data_disk_after.txt"
& adb @adbArgs shell dumpsys power | Set-Content "$OutputDir/power_after.txt"
& adb @adbArgs shell dumpsys notification --noredact | Set-Content "$OutputDir/notification_after.txt"
& adb @adbArgs logcat -d -v threadtime | Set-Content "$OutputDir/logcat_after.txt"
if ($TaskId) {
  if ($TaskId -notmatch '^[A-Za-z0-9_-]+$') { throw "INVALID_TASK_ID" }
  $taskBase = "files/tasks/$TaskId"
  foreach ($name in @("status.json", "task_report.json", "performance_profile.json", "runtime_progress.json", "stage_major_checkpoint.json")) {
    & adb @adbArgs shell run-as com.seededge.runtime cat "$taskBase/$name" 2>&1 | Set-Content "$OutputDir/$name"
  }
  & adb @adbArgs shell run-as com.seededge.runtime ls -la "$taskBase/checkpoints" 2>&1 |
    Set-Content "$OutputDir/checkpoint_generations.txt"
}
@{schema_version=1;phase="18D";gate="pad_after_benchmark";device_state="EXECUTED";collected_at_utc=[DateTime]::UtcNow.ToString("o");serial=$Serial;task_id=$TaskId;runtime_fields=@("requested_runtime_mode","runtime_mode","runtime_selection_reason","tile_cohort_size","checkpoint_generation","actual_task_disk_bytes","peak_rss_mb")} |
  ConvertTo-Json | Set-Content "$OutputDir/summary.json"
