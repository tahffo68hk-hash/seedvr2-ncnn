. "$PSScriptRoot\common.ps1"
$source = "D:\AIProjects\third_party\ncnn"
$expected = "a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a"
Assert-Path "$source\.git" "Pinned ncnn repository"
$actual = (& git -C $source rev-parse HEAD).Trim()
if ($actual -ne $expected) { throw "ncnn commit mismatch: $actual" }
$target = Join-Path $script:AndroidRoot "..\.phase18_ncnn_src"
if (Test-Path "$target\.phase18_commit") {
    if ((Get-Content -Raw "$target\.phase18_commit").Trim() -ne $expected) { throw "Existing ncnn export has wrong commit; run clean.ps1 -LocalNcnn" }
} else {
    if ((Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) { throw "Incomplete ncnn export exists; run clean.ps1 -LocalNcnn" }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $archive = Join-Path $env:TEMP "phase18-ncnn-$expected.zip"
    & git -C $source archive --format=zip --output=$archive $expected
    if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
    Expand-Archive -LiteralPath $archive -DestinationPath $target
    [System.IO.File]::WriteAllText("$target\.phase18_commit", $expected + [Environment]::NewLine, (New-Object System.Text.UTF8Encoding($false)))
    Remove-Item -LiteralPath $archive
}
$glslangCommit = "fe88f421038e1bb0a25cd5c1b2dfe505db82d08f"
$glslang = Join-Path $target "glslang"
if (-not (Test-Path "$glslang\.git")) {
    if (Test-Path $glslang) { Remove-Item -LiteralPath $glslang -Recurse -Force }
    & git clone --no-checkout https://github.com/nihui/glslang $glslang
    if ($LASTEXITCODE -ne 0) { throw "official glslang clone failed" }
    & git -C $glslang checkout --detach $glslangCommit
    if ($LASTEXITCODE -ne 0) { throw "glslang fixed commit checkout failed" }
}
$actualGlslang = (& git -C $glslang rev-parse HEAD).Trim()
if ($actualGlslang -ne $glslangCommit) { throw "glslang commit mismatch: $actualGlslang" }
Write-Output "NCNN_READY $expected glslang=$glslangCommit"
