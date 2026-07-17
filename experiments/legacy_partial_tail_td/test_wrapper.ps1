[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native_arguments.ps1')

$WrapperPath = Join-Path $PSScriptRoot 'invoke_once.ps1'
$Tokens = $null
$ParseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
    $WrapperPath,
    [ref]$Tokens,
    [ref]$ParseErrors
)
if ($ParseErrors.Count -ne 0) {
    throw "wrapper parse failure: $($ParseErrors[0].Message)"
}

$Python = 'C:\Program Files\Python310\python.exe'
$Expected = @(
    'alpha beta',
    'C:\Users\fshuu\Documents\tetris\1313\mainmodel copy 3.jld2',
    'embedded"quote',
    'path with space\',
    '',
    '日本語 path'
)
$PythonCode = 'import json,sys,time; print(json.dumps(sys.argv[1:])); time.sleep(0.2)'
$NativeCommandLine = Join-NativeArguments (@('-c', $PythonCode) + $Expected)
$TemporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
    'p1-wrapper-test-' + [Guid]::NewGuid().ToString('N')
)
[System.IO.Directory]::CreateDirectory($TemporaryDirectory) | Out-Null
$Stdout = Join-Path $TemporaryDirectory 'child.stdout.log'
$Stderr = Join-Path $TemporaryDirectory 'child.stderr.log'
$Job = [P1Native.JobObject]::new()
try {
    $Process = Start-Process -FilePath $Python -ArgumentList $NativeCommandLine `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    $Job.Assign($Process)
    $Assigned = @($Job.GetProcessIds())
    if ($Process.Id -notin $Assigned) {
        throw 'Job Object did not report its assigned root process'
    }
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) {
        throw "argv probe failed: $(Get-Content -LiteralPath $Stderr -Raw)"
    }
    # Windows PowerShell 5 returns a top-level JSON array as one Object[]
    # pipeline object; wrapping that expression in @() would nest it.
    $Received = Get-Content -LiteralPath $Stdout -Raw | ConvertFrom-Json
    if ($Received.Count -ne $Expected.Count) {
        throw "argv count mismatch: $($Received.Count) != $($Expected.Count)"
    }
    for ($Index = 0; $Index -lt $Expected.Count; $Index += 1) {
        if ($Received[$Index] -cne $Expected[$Index]) {
            throw "argv[$Index] mismatch: '$($Received[$Index])' != '$($Expected[$Index])'"
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
    status = 'wrapper_synthetic_checks_passed'
    native_arguments = $Expected.Count
    process_tree_job_object = $true
    checkpoint_or_dataset_opened = $false
    marker_read_or_written = $false
    heavy_work_run = $false
} | ConvertTo-Json
