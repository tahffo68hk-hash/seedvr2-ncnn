. "$PSScriptRoot\common.ps1"
$build=Join-Path $script:AndroidRoot "..\.phase18_host_tests"
& "D:\Android\sdk\cmake\3.22.1\bin\cmake.exe" -S "$script:AndroidRoot\native\tests" -B $build -G "Visual Studio 17 2022" -A x64
if($LASTEXITCODE -ne 0){throw "host test configure failed"}
& "D:\Android\sdk\cmake\3.22.1\bin\cmake.exe" --build $build --config Release
if($LASTEXITCODE -ne 0){throw "host test build failed"}
& "$build\Release\model_manifest_test.exe"
if($LASTEXITCODE -ne 0){throw "host manifest tests failed"}
& "$build\Release\synthetic_cpu_host_test.exe"
if($LASTEXITCODE -ne 0){throw "host synthetic ncnn CPU test failed"}
