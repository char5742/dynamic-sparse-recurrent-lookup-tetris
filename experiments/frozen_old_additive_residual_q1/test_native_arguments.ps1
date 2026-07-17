[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native_arguments.ps1')

$Python = 'C:\Program Files\Python310\python.exe'
$Expected = @(
    'alpha beta',
    'C:\synthetic Q1\path\',
    'embedded"quote',
    '',
    '日本語 path'
)
$PythonCode = 'import json,sys,time; print(json.dumps(sys.argv[1:])); time.sleep(0.1)'
$NativeCommandLine = Join-NativeArguments (@('-c', $PythonCode) + $Expected)
$TemporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
    'q1-native-test-' + [Guid]::NewGuid().ToString('N')
)
[System.IO.Directory]::CreateDirectory($TemporaryDirectory) | Out-Null
$Stdout = Join-Path $TemporaryDirectory 'child.stdout.log'
$Stderr = Join-Path $TemporaryDirectory 'child.stderr.log'
$Job = [Q1Native.JobObject]::new()
try {
    $Process = Start-Process -FilePath $Python -ArgumentList $NativeCommandLine `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    $Job.Assign($Process)
    if ($Process.Id -notin @($Job.GetProcessIds())) {
        throw 'Q1 Job Object did not report its assigned root process'
    }
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) {
        throw "argv probe failed: $(Get-Content -LiteralPath $Stderr -Raw)"
    }
    $Received = Get-Content -LiteralPath $Stdout -Raw | ConvertFrom-Json
    if ($Received.Count -ne $Expected.Count) {
        throw "argv count mismatch: $($Received.Count) != $($Expected.Count)"
    }
    for ($Index = 0; $Index -lt $Expected.Count; $Index += 1) {
        if ($Received[$Index] -cne $Expected[$Index]) {
            throw "argv[$Index] mismatch"
        }
    }
} finally {
    $Job.Dispose()
    Remove-Item -LiteralPath $Stdout -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $Stderr -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $TemporaryDirectory -PathType Container) {
        [System.IO.Directory]::Delete($TemporaryDirectory, $false)
    }
}

[ordered]@{
    status = 'q1_native_synthetic_checks_passed'
    native_arguments = $Expected.Count
    process_tree_job_object = $true
    actual_input_opened = $false
    marker_read_or_written = $false
    heavy_work_run = $false
} | ConvertTo-Json
