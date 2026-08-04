$ErrorActionPreference="Stop"
Push-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
try { python android\tests\test_phase18a_contracts.py; if($LASTEXITCODE -ne 0){throw "contract tests failed"} }
finally { Pop-Location }
