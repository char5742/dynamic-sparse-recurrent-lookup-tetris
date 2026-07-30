param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [Parameter(Mandatory = $true)]
    [string]$TrainerRunId,

    [string]$Dataset = 'C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release',

    [ValidateRange(10, 300)]
    [int]$PollSeconds = 30
)

$ErrorActionPreference = 'Stop'

$runRootPath = [System.IO.Path]::GetFullPath($RunRoot)
$workspacePath = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..\..\..')
)
$trainerPath = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot 'train_paper_elm_twin_official_full.jl')
)
$finalizerPath = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot 'finalize_paper_elm_twin_official_full.jl')
)
$juliaPath = (Get-Command julia).Source
$watchdogLog = Join-Path $runRootPath 'watchdog.jsonl'
$artifactPath = Join-Path $runRootPath 'artifacts\paper_elm_twin_official_v2_dev1500.jld2'
$finalizerLock = Join-Path $runRootPath 'artifacts\.finalizer.lock'
$seeds = @(
    '6077687918186389328',
    '6077687918186389329',
    '6077687918186389330'
)

New-Item -ItemType Directory -Path $runRootPath -Force | Out-Null

function Write-WatchdogEvent {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Fields
    )
    $record = [ordered]@{
        timestamp = [DateTime]::UtcNow.ToString('o')
    }
    foreach ($key in $Fields.Keys) {
        $record[$key] = $Fields[$key]
    }
    $line = $record | ConvertTo-Json -Compress -Depth 5
    [System.IO.File]::AppendAllText(
        $watchdogLog,
        $line + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Get-CompletedEpoch {
    param([int]$RestartIndex)
    $checkpointDirectory = Join-Path $runRootPath (
        'checkpoints\restart_{0}' -f $RestartIndex
    )
    if (-not (Test-Path -LiteralPath $checkpointDirectory)) {
        return 0
    }
    $epochs = Get-ChildItem -LiteralPath $checkpointDirectory `
        -Filter 'epoch_*.jld2' -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.BaseName -match '^epoch_(\d{3})$') {
                [int]$Matches[1]
            }
        }
    if ($null -eq $epochs) {
        return 0
    }
    return [int](($epochs | Measure-Object -Maximum).Maximum)
}

function Test-WorkerRunning {
    param([int]$RestartIndex)
    $restartToken = '--restart-index {0}' -f $RestartIndex
    $matches = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq 'julialauncher.exe' -and
            $_.CommandLine -like ('*--trainer-run-id {0}*' -f $TrainerRunId) -and
            $_.CommandLine -like ('*{0}*' -f $restartToken)
        }
    return @($matches).Count -gt 0
}

function Start-Worker {
    param([int]$RestartIndex)
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $stdout = Join-Path $runRootPath (
        'stdout_restart_{0}_resume_{1}.jsonl' -f $RestartIndex, $stamp
    )
    $stderr = Join-Path $runRootPath (
        'stderr_restart_{0}_resume_{1}.log' -f $RestartIndex, $stamp
    )
    $arguments = @(
        '--project=.',
        '--threads=2',
        $trainerPath,
        '--dataset', $Dataset,
        '--run-root', $runRootPath,
        '--trainer-run-id', $TrainerRunId,
        '--run-id', ('elm-dev1500-restart-{0}' -f $RestartIndex),
        '--restart-index', [string]$RestartIndex,
        '--seed', $seeds[$RestartIndex - 1],
        '--epochs', '35',
        '--batch-size', '8',
        '--blas-threads', '6',
        '--resume', 'true'
    )
    $process = Start-Process `
        -FilePath $juliaPath `
        -ArgumentList $arguments `
        -WorkingDirectory $workspacePath `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Write-WatchdogEvent @{
        event = 'worker_restarted'
        restart_index = $RestartIndex
        launcher_pid = $process.Id
        completed_epoch_before_restart = Get-CompletedEpoch $RestartIndex
        stdout = $stdout
        stderr = $stderr
    }
}

function Invoke-Finalizer {
    New-Item -ItemType Directory -Path (Split-Path $artifactPath) -Force |
        Out-Null
    if (Test-Path -LiteralPath $artifactPath) {
        Write-WatchdogEvent @{
            event = 'artifact_already_present'
            artifact_path = $artifactPath
        }
        return $true
    }
    $lockStream = $null
    try {
        $lockStream = [System.IO.File]::Open(
            $finalizerLock,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
    }
    catch [System.IO.IOException] {
        Write-WatchdogEvent @{
            event = 'finalizer_lock_present'
            lock_path = $finalizerLock
        }
        return $false
    }
    finally {
        if ($null -ne $lockStream) {
            $lockStream.Dispose()
        }
    }

    $stdout = Join-Path $runRootPath 'finalizer_stdout.jsonl'
    $stderr = Join-Path $runRootPath 'finalizer_stderr.log'
    $arguments = @(
        '--project=.',
        '--threads=4',
        $finalizerPath,
        '--run-root', $runRootPath,
        '--trainer-run-id', $TrainerRunId,
        '--dataset', $Dataset,
        '--output', $artifactPath
    )
    Write-WatchdogEvent @{
        event = 'finalizer_started'
        artifact_path = $artifactPath
    }
    $process = Start-Process `
        -FilePath $juliaPath `
        -ArgumentList $arguments `
        -WorkingDirectory $workspacePath `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -eq 0 -and (Test-Path -LiteralPath $artifactPath)) {
        Write-WatchdogEvent @{
            event = 'finalizer_completed'
            exit_code = $process.ExitCode
            artifact_path = $artifactPath
        }
        return $true
    }
    Remove-Item -LiteralPath $finalizerLock -Force -ErrorAction SilentlyContinue
    Write-WatchdogEvent @{
        event = 'finalizer_failed'
        exit_code = $process.ExitCode
        stdout = $stdout
        stderr = $stderr
    }
    return $false
}

Write-WatchdogEvent @{
    event = 'watchdog_started'
    watchdog_pid = $PID
    run_root = $runRootPath
    trainer_run_id = $TrainerRunId
    poll_seconds = $PollSeconds
}

while ($true) {
    $completed = @(
        Get-CompletedEpoch 1
        Get-CompletedEpoch 2
        Get-CompletedEpoch 3
    )
    if (($completed | Where-Object { $_ -lt 35 }).Count -eq 0) {
        if (Invoke-Finalizer) {
            Write-WatchdogEvent @{
                event = 'watchdog_completed'
                completed_epochs = $completed
            }
            exit 0
        }
    }
    else {
        for ($restart = 1; $restart -le 3; $restart++) {
            if (
                $completed[$restart - 1] -lt 35 -and
                -not (Test-WorkerRunning $restart)
            ) {
                Start-Worker $restart
            }
        }
    }
    Write-WatchdogEvent @{
        event = 'heartbeat'
        completed_epochs = $completed
    }
    Start-Sleep -Seconds $PollSeconds
}
