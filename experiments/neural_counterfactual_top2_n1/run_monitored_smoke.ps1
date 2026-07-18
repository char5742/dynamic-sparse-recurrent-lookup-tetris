param(
    [string]$OutputRoot = "D:\tetris-paper-plus\runs"
)

$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\fshuu\Documents\tetris"
$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$Smoke = Join-Path $RepoRoot "experiments\neural_counterfactual_top2_n1\smoke.jl"
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$OutputDirectory = Join-Path $OutputRoot ("n1_engineering_smoke_73200_" + $Timestamp)
New-Item -ItemType Directory -Path $OutputDirectory -ErrorAction Stop | Out-Null

$Stdout = Join-Path $OutputDirectory "stdout.log"
$Stderr = Join-Path $OutputDirectory "stderr.log"
$Telemetry = Join-Path $OutputDirectory "resource_samples.jsonl"
$Monitor = Join-Path $OutputDirectory "monitor_result.json"
$PrivateLimit = [int64](4GB)
$WorkingSetLimit = [int64](2GB)
$TimeoutSeconds = 300.0
$Started = Get-Date
$PeakPrivate = [int64]0
$PeakWorkingSet = [int64]0
$SampleCount = 0
$LimitReason = $null
$CapturedExitCode = $null

$Arguments = @(
    "--startup-file=no",
    "--history-file=no",
    "--project=$RepoRoot",
    $Smoke,
    $OutputDirectory
)
$LaunchedProcess = Start-Process -FilePath $Julia -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot -PassThru `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
# Start-Process's wrapper can expose a null ExitCode after redirected launch on
# Windows PowerShell. Reattach a native Process handle before monitoring.
$Process = [System.Diagnostics.Process]::GetProcessById($LaunchedProcess.Id)

function Get-TreeProcessIds([int]$RootId) {
    $Rows = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    $Known = [System.Collections.Generic.HashSet[int]]::new()
    [void]$Known.Add($RootId)
    $Changed = $true
    while ($Changed) {
        $Changed = $false
        foreach ($Row in $Rows) {
            if ($Known.Contains([int]$Row.ParentProcessId) -and
                -not $Known.Contains([int]$Row.ProcessId)) {
                [void]$Known.Add([int]$Row.ProcessId)
                $Changed = $true
            }
        }
    }
    return @($Known)
}

try {
    while (-not $Process.HasExited) {
        $Elapsed = ((Get-Date) - $Started).TotalSeconds
        $Ids = @(Get-TreeProcessIds $Process.Id)
        $Private = [int64]0
        $WorkingSet = [int64]0
        $Observed = @()
        foreach ($Id in $Ids) {
            try {
                $Current = Get-Process -Id $Id -ErrorAction Stop
                $Private += [int64]$Current.PrivateMemorySize64
                $WorkingSet += [int64]$Current.WorkingSet64
                $Observed += [ordered]@{
                    pid = $Id
                    private_bytes = [int64]$Current.PrivateMemorySize64
                    working_set_bytes = [int64]$Current.WorkingSet64
                }
            } catch {
                # A process can exit between the tree and metric snapshots.
            }
        }
        $PeakPrivate = [Math]::Max($PeakPrivate, $Private)
        $PeakWorkingSet = [Math]::Max($PeakWorkingSet, $WorkingSet)
        $SampleCount += 1
        ([ordered]@{
            elapsed_seconds = $Elapsed
            process_count = $Observed.Count
            private_bytes = $Private
            working_set_bytes = $WorkingSet
            processes = $Observed
        } | ConvertTo-Json -Compress -Depth 4) | Add-Content -Path $Telemetry -Encoding UTF8

        if ($Private -gt $PrivateLimit) {
            $LimitReason = "private committed memory exceeded 4 GiB"
        } elseif ($WorkingSet -gt $WorkingSetLimit) {
            $LimitReason = "working set exceeded 2 GiB"
        } elseif ($Elapsed -gt $TimeoutSeconds) {
            $LimitReason = "wall time exceeded 300 seconds"
        }
        if ($null -ne $LimitReason) {
            foreach ($Id in ($Ids | Sort-Object -Descending)) {
                Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
            }
            break
        }
        Start-Sleep -Milliseconds 200
        $Process.Refresh()
    }
    $Process.WaitForExit()
    $Process.Refresh()
    $CapturedExitCode = $Process.ExitCode
} finally {
    $WallSeconds = ((Get-Date) - $Started).TotalSeconds
    if ($null -eq $CapturedExitCode -and $Process.HasExited) {
        $Process.Refresh()
        $CapturedExitCode = $Process.ExitCode
    }
    $ExitCode = if ($null -ne $LimitReason) { -1 } else { $CapturedExitCode }
    $SmokeResult = Join-Path $OutputDirectory "smoke_result.json"
    $Result = [ordered]@{
        status = if ($ExitCode -eq 0 -and (Test-Path $SmokeResult)) {
            "N1-monitored-smoke-pass"
        } else {
            "N1-monitored-smoke-fail"
        }
        output_directory = $OutputDirectory
        root_pid = $Process.Id
        exit_code = $ExitCode
        wall_seconds = $WallSeconds
        sample_interval_milliseconds = 200
        sample_count = $SampleCount
        peak_process_tree_private_bytes = $PeakPrivate
        peak_process_tree_working_set_bytes = $PeakWorkingSet
        private_limit_bytes = $PrivateLimit
        working_set_limit_bytes = $WorkingSetLimit
        timeout_seconds = $TimeoutSeconds
        limit_reason = $LimitReason
        smoke_result_exists = (Test-Path $SmokeResult)
    }
    $Result | ConvertTo-Json -Depth 4 | Set-Content -Path $Monitor -Encoding UTF8
}

Write-Output $OutputDirectory
if ($CapturedExitCode -ne 0 -or $null -ne $LimitReason -or
    -not (Test-Path (Join-Path $OutputDirectory "smoke_result.json"))) {
    exit 1
}
