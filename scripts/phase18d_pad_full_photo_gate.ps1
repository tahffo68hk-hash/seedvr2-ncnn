param([Parameter(Mandatory=$true)][string]$Apk, [string]$Serial = "", [string]$OutputDir = ".phase18d_pad_evidence/full_photo_gate", [string]$TaskId = "")
$ErrorActionPreference = "Stop"
$adbArgs = @(); if ($Serial) { $adbArgs += @("-s", $Serial) }
if (-not (Test-Path -LiteralPath $Apk -PathType Leaf)) { throw "APK_NOT_FOUND" }
if ((& adb @adbArgs get-state).Trim() -ne "device") { throw "ADB_DEVICE_NOT_READY" }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
& adb @adbArgs install -r $Apk; if ($LASTEXITCODE -ne 0) { throw "APK_INSTALL_FAILED" }
& adb @adbArgs shell am instrument -w -r -e class com.seededge.runtime.PhotoTaskDeviceTest com.seededge.runtime.test/androidx.test.runner.AndroidJUnitRunner |
  Set-Content "$OutputDir/instrumentation.txt"
if ($LASTEXITCODE -ne 0) { throw "DEVICE_TEST_FAILED" }
& "$PSScriptRoot/phase18d_pad_after_benchmark.ps1" -Serial $Serial -OutputDir $OutputDir -TaskId $TaskId
