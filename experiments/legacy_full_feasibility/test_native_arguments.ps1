[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native_arguments.ps1')

$Python = 'C:\Program Files\Python310\python.exe'
$Expected = @(
    'alpha beta',
    'C:\Users\fshuu\Documents\tetris\1313\mainmodel copy 3.jld2',
    'embedded"quote',
    'path with space\',
    '',
    '日本語 path'
)
$PythonCode = 'import json,sys; print(json.dumps(sys.argv[1:]))'
$NativeCommandLine = Join-NativeArguments (@('-c', $PythonCode) + $Expected)
$Stdout = [System.IO.Path]::GetTempFileName()
$Stderr = [System.IO.Path]::GetTempFileName()
try {
    $Process = Start-Process -FilePath $Python -ArgumentList $NativeCommandLine -NoNewWindow -PassThru -Wait -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if ($Process.ExitCode -ne 0) {
        throw "argv probe failed: $(Get-Content -LiteralPath $Stderr -Raw)"
    }
    $Received = @(Get-Content -LiteralPath $Stdout -Raw | ConvertFrom-Json)
    if ($Received.Count -ne $Expected.Count) {
        throw "argv count mismatch: $($Received.Count) != $($Expected.Count)"
    }
    for ($Index = 0; $Index -lt $Expected.Count; $Index += 1) {
        if ($Received[$Index] -cne $Expected[$Index]) {
            throw "argv[$Index] mismatch: '$($Received[$Index])' != '$($Expected[$Index])'"
        }
    }
    Write-Output "native argv boundary OK ($($Expected.Count) exact arguments)"
} finally {
    Remove-Item -LiteralPath $Stdout -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $Stderr -Force -ErrorAction SilentlyContinue
}
