param([Parameter(Mandatory=$true)][string]$Apk, [string]$Variant="debug")
$ErrorActionPreference="Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (-not (Test-Path -LiteralPath $Apk)) { throw "APK missing: $Apk" }
$zip=[System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $Apk))
try {
  $names=@($zip.Entries | ForEach-Object FullName)
  foreach($required in @("classes.dex","AndroidManifest.xml","lib/arm64-v8a/libseededge.so")){if($names -notcontains $required){throw "required APK entry missing: $required"}}
  $abis=@($names | Where-Object {$_ -match '^lib/([^/]+)/'} | ForEach-Object {([regex]::Match($_,'^lib/([^/]+)/')).Groups[1].Value} | Sort-Object -Unique)
  if(($abis.Count -ne 1)-or($abis[0] -ne "arm64-v8a")){throw "unexpected ABI set: $($abis -join ',')"}
  $forbidden=@($names | Where-Object {$_ -match '(?i)(\.pth$|\.ckpt$|keystore|\.jks$|\.git/|models/.+\.(bin|param)$)'})
  if($forbidden.Count){throw "forbidden APK entries: $($forbidden -join ',')"}
  $bytes=[IO.File]::ReadAllBytes((Resolve-Path $Apk));$ascii=[Text.Encoding]::ASCII.GetString($bytes)
  foreach($s in @('D:\AIProjects','C:\Users')){if($ascii.Contains($s)){throw "sensitive path embedded: $s"}}
  [ordered]@{overall_pass=$true;device_required=$false;device_executed=$false;apk=(Split-Path $Apk -Leaf);variant=$Variant;size_bytes=(Get-Item $Apk).Length;abis=$abis;seededge_present=$true;forbidden_entries=@();models_packaged=$false}|ConvertTo-Json -Depth 5
} finally {$zip.Dispose()}
