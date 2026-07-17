[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][string]$SourceFingerprint,
    [Parameter(Mandatory = $true)][string]$StartGate
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repository = [System.IO.Path]::GetFullPath($Repository)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$SourceFingerprint = [System.IO.Path]::GetFullPath($SourceFingerprint)
$StartGate = [System.IO.Path]::GetFullPath($StartGate)
$ExpectedCheckpointHash = '7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1'
$ExpectedDatasetHash = 'e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d'
$HardWallSeconds = 1500.0
$MaxWorkingSetBytes = [int64](8 * 1024 * 1024 * 1024)
$Checkpoint = Join-Path $Repository '1313\mainmodel copy 3.jld2'
$Dataset = 'D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2'
$Experiment = Join-Path $Repository 'experiments\legacy_full_feasibility'
$Manifest = Join-Path $Repository 'Manifest.toml'
$FreezePath = Join-Path $OutputDirectory 'freeze.json'
$MonitorPath = Join-Path $OutputDirectory 'monitor.json'
$StartedPath = Join-Path $OutputDirectory 'started.json'
$GlobalStartedPath = 'D:\tetris-paper-plus\runs\legacy_full_feasibility_F.started.json'
$SubsetPath = Join-Path $OutputDirectory 'subset.npz'
$SubsetJson = Join-Path $OutputDirectory 'subset.json'
$JuliaPhase = Join-Path $OutputDirectory 'julia_phase.json'
$WeightsPath = Join-Path $OutputDirectory 'temporary_updated_weights.npz'
$ReferencePath = Join-Path $OutputDirectory 'temporary_updated_reference.npz'
$OpenVinoPhase = Join-Path $OutputDirectory 'openvino_phase.json'
$SystemPython = 'C:\Program Files\Python310\python.exe'
$OpenVinoPython = 'D:\tetris-paper-plus\python-env\Scripts\python.exe'
$JuliaExe = (Get-Command julia -ErrorAction Stop).Source

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic([string]$Path, $Value, [int]$Depth = 12) {
    $Temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $Temporary -Encoding utf8
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Write-JsonCreateNew([string]$Path, $Value, [int]$Depth = 12) {
    $Json = $Value | ConvertTo-Json -Depth $Depth
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Json)
    $Stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally {
        $Stream.Dispose()
    }
}

function Get-ProcessSnapshot {
    return @(
        Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.ProcessName -match '^(julia|python|pythonw|openvino)' } |
            ForEach-Object {
                [ordered]@{
                    name = $_.ProcessName
                    id = $_.Id
                    cpu_seconds = $_.CPU
                    working_set_bytes = $_.WorkingSet64
                    private_bytes = $_.PrivateMemorySize64
                    path = try { $_.Path } catch { $null }
                }
            }
    )
}

if (-not $OutputDirectory.StartsWith('D:\tetris-paper-plus\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'F output must be a fresh directory below D:\tetris-paper-plus'
}
if (Test-Path -LiteralPath $OutputDirectory) {
    if ((Get-ChildItem -LiteralPath $OutputDirectory -Force | Measure-Object).Count -ne 0) {
        throw "output directory already exists and is nonempty: $OutputDirectory"
    }
} else {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}
if (Test-Path -LiteralPath $StartedPath) {
    throw 'one-shot F guard already exists; retry is prohibited'
}
if (Test-Path -LiteralPath $GlobalStartedPath) {
    throw "global one-shot F guard already exists; retry is prohibited: $GlobalStartedPath"
}
if (-not (Test-Path -LiteralPath $SourceFingerprint -PathType Leaf)) {
    throw "missing clean source fingerprint: $SourceFingerprint"
}
if ((Get-Sha256 $Checkpoint) -ne $ExpectedCheckpointHash) {
    throw 'checkpoint SHA-256 mismatch before F execution'
}
if ((Get-Sha256 $Dataset) -ne $ExpectedDatasetHash) {
    throw 'dataset SHA-256 mismatch before F execution'
}
$GitStatus = (& git -C $Repository status --porcelain=v1) -join "`n"
if ($GitStatus.Length -ne 0) {
    throw "repository must be clean before freeze:`n$GitStatus"
}
$Commit = (& git -C $Repository rev-parse HEAD).Trim()

$HarnessFiles = @(
    'contract.jl',
    'extract_rows.py',
    'run_benchmark.jl',
    'verify_openvino.py',
    'finalize_result.py',
    'invoke_once.ps1',
    'test_contract.jl',
    'EXPERIMENT.md'
)
$HarnessRecords = @()
$HarnessDigestInput = New-Object System.Text.StringBuilder
foreach ($Name in $HarnessFiles) {
    $Path = Join-Path $Experiment $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "missing harness file: $Path"
    }
    $Hash = Get-Sha256 $Path
    $HarnessRecords += [ordered]@{ path = "experiments/legacy_full_feasibility/$Name"; sha256 = $Hash }
    [void]$HarnessDigestInput.Append("$Name`0$Hash`n")
}
$Sha = [System.Security.Cryptography.SHA256]::Create()
$HarnessBytes = [System.Text.Encoding]::UTF8.GetBytes($HarnessDigestInput.ToString())
$HarnessAggregate = ([BitConverter]::ToString($Sha.ComputeHash($HarnessBytes))).Replace('-', '').ToLowerInvariant()
$Sha.Dispose()

$ProcessesBefore = Get-ProcessSnapshot
$JuliaBefore = @($ProcessesBefore | Where-Object { $_.name -eq 'julia' })
if ($JuliaBefore.Count -ne 0) {
    throw "another Julia process is active; heavy F execution refused"
}
$HeavyPython = @($ProcessesBefore | Where-Object { $_.name -match '^python' -and $_.working_set_bytes -gt 2GB })
if ($HeavyPython.Count -ne 0) {
    throw "a Python process above 2 GiB is active; heavy F execution refused"
}

$OperatingSystem = Get-CimInstance Win32_OperatingSystem
$Processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$Computer = Get-CimInstance Win32_ComputerSystem
$Commands = [ordered]@{
    extract = "`"$SystemPython`" `"$(Join-Path $Experiment 'extract_rows.py')`" `"$Dataset`" `"$SubsetPath`" `"$SubsetJson`""
    julia = "F_OUTPUT_DIRECTORY=`"$OutputDirectory`" F_SUBSET_PATH=`"$SubsetPath`" F_CHECKPOINT_PATH=`"$Checkpoint`" F_FREEZE_PATH=`"$FreezePath`" `"$JuliaExe`" --startup-file=no --project=`"$Repository`" --threads=20 `"$(Join-Path $Experiment 'run_benchmark.jl')`""
    openvino = "`"$OpenVinoPython`" `"$(Join-Path $Experiment 'verify_openvino.py')`" `"$Repository`" `"$WeightsPath`" `"$ReferencePath`" `"$JuliaPhase`" `"$OpenVinoPhase`""
    finalize = "`"$SystemPython`" `"$(Join-Path $Experiment 'finalize_result.py')`" `"$OutputDirectory`""
}
$Freeze = [ordered]@{
    benchmark = 'F legacy full-model continuation feasibility'
    frozen_at = (Get-Date).ToString('o')
    source_commit = $Commit
    repository_clean = $true
    source_fingerprint_path = $SourceFingerprint
    source_fingerprint_sha256 = Get-Sha256 $SourceFingerprint
    source_fingerprint = Get-Content -LiteralPath $SourceFingerprint -Raw | ConvertFrom-Json
    manifest_path = $Manifest
    manifest_sha256 = Get-Sha256 $Manifest
    checkpoint_path = $Checkpoint
    checkpoint_sha256 = Get-Sha256 $Checkpoint
    dataset_path = $Dataset
    dataset_sha256 = Get-Sha256 $Dataset
    harness_sha256 = $HarnessAggregate
    harness_files = $HarnessRecords
    output_directory = $OutputDirectory
    commands = $Commands
    constants = [ordered]@{
        rows = @(1, 251, 501, 751, 1001, 1251)
        episode_ids = @(1, 2, 3, 4, 5, 6)
        seeds = @(5742, 5743, 5744, 5745, 5746, 5747)
        n_step = 3
        gamma = 0.997
        loss = 'selected-action Huber(delta=1) on exact historical selected chunk'
        optimizer = 'AdamW(couple=true)'
        learning_rate = 0.00001
        betas = @(0.9, 0.999)
        weight_decay = 0.0001
        backend = 'Lux+Zygote only'
        julia_threads = 20
        blas_threads = 10
        historical_chunk = 16
        hard_wall_seconds = $HardWallSeconds
        max_working_set_bytes = $MaxWorkingSetBytes
        actor_seconds_per_step = 0.411
        actor_refresh_count = 4
        t1000_limit_seconds = 1800
        global_one_shot_marker = $GlobalStartedPath
    }
    machine = [ordered]@{
        computer_name = $env:COMPUTERNAME
        cpu = $Processor.Name
        logical_processors = $Computer.NumberOfLogicalProcessors
        total_physical_memory_bytes = [int64]$Computer.TotalPhysicalMemory
        free_physical_memory_bytes = [int64]$OperatingSystem.FreePhysicalMemory * 1024
        os = $OperatingSystem.Caption
        os_version = $OperatingSystem.Version
        powershell = $PSVersionTable.PSVersion.ToString()
        julia_executable = $JuliaExe
        system_python = $SystemPython
        openvino_python = $OpenVinoPython
        processes_before = $ProcessesBefore
    }
}
Write-JsonAtomic $FreezePath $Freeze 20
Write-JsonAtomic (Join-Path $OutputDirectory 'ready.json') ([ordered]@{
    status = 'frozen_waiting_for_start_gate'
    wrapper_pid = $PID
    commit = $Commit
    freeze_path = $FreezePath
    start_gate = $StartGate
    created_at = (Get-Date).ToString('o')
})

$WaitStarted = Get-Date
while (-not (Test-Path -LiteralPath $StartGate -PathType Leaf)) {
    if (((Get-Date) - $WaitStarted).TotalSeconds -gt 300) {
        throw 'start gate was not provided within 300 seconds'
    }
    Start-Sleep -Milliseconds 250
}
$GitStatusAtGate = (& git -C $Repository status --porcelain=v1) -join "`n"
if ($GitStatusAtGate.Length -ne 0 -or ((& git -C $Repository rev-parse HEAD).Trim() -ne $Commit)) {
    throw 'repository changed after freeze and before start gate; F execution refused'
}
Write-JsonCreateNew $GlobalStartedPath ([ordered]@{
    benchmark = 'F legacy full-model continuation feasibility'
    started_at = (Get-Date).ToString('o')
    wrapper_pid = $PID
    source_commit = $Commit
    output_directory = $OutputDirectory
    retry_prohibited = $true
})
Write-JsonAtomic $StartedPath ([ordered]@{
    started_at = (Get-Date).ToString('o')
    wrapper_pid = $PID
    source_commit = $Commit
    one_shot = $true
})

$env:F_BLAS_THREADS = '10'
$env:OPENBLAS_NUM_THREADS = '10'
$env:JULIA_NUM_THREADS = '20'
$env:F_OUTPUT_DIRECTORY = $OutputDirectory
$env:F_SUBSET_PATH = $SubsetPath
$env:F_CHECKPOINT_PATH = $Checkpoint
$env:F_FREEZE_PATH = $FreezePath
$BenchmarkStarted = Get-Date
$PeakWorkingSet = [int64]0
$PeakPrivateBytes = [int64]0
$LowMemorySamples = 0
$SustainedPaging = $false
$PhaseRecords = @()
$OverallStopReason = 'completed'

function Invoke-MonitoredPhase(
    [string]$Name,
    [string]$FilePath,
    [string[]]$ArgumentList
) {
    $Stdout = Join-Path $OutputDirectory "$Name.stdout.log"
    $Stderr = Join-Path $OutputDirectory "$Name.stderr.log"
    $PhaseStarted = Get-Date
    $Process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -NoNewWindow -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    Write-JsonAtomic (Join-Path $OutputDirectory "$Name.started.json") ([ordered]@{
        phase = $Name
        pid = $Process.Id
        started_at = $PhaseStarted.ToString('o')
        executable = $FilePath
        arguments = $ArgumentList
    })
    $StopReason = 'completed'
    while (-not $Process.HasExited) {
        Start-Sleep -Milliseconds 250
        $Process.Refresh()
        $script:PeakWorkingSet = [Math]::Max($script:PeakWorkingSet, [int64]$Process.WorkingSet64)
        $script:PeakPrivateBytes = [Math]::Max($script:PeakPrivateBytes, [int64]$Process.PrivateMemorySize64)
        $Elapsed = ((Get-Date) - $BenchmarkStarted).TotalSeconds
        $PerCallStop = $null
        if ($Name -eq 'julia' -and (Test-Path -LiteralPath $JuliaPhase -PathType Leaf)) {
            try {
                $StageState = Get-Content -LiteralPath $JuliaPhase -Raw | ConvertFrom-Json
                $StageElapsed = ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0) - [double]$StageState.stage_started_unix
                if ($StageState.stage -eq 'specialization_running' -and $StageElapsed -ge 300.0) {
                    $PerCallStop = 'external first-specialization timeout reached 300 seconds'
                } elseif ($StageState.stage -match '^warm_update_[1-6]_running$' -and $StageElapsed -ge 120.0) {
                    $PerCallStop = "external $($StageState.stage) timeout reached 120 seconds"
                }
            } catch {
                # Atomic JSON replacement can briefly race path metadata, but a
                # malformed persistent file will still make the Julia phase fail.
            }
        }
        if ($null -ne $PerCallStop) {
            $StopReason = $PerCallStop
        } elseif ($Process.WorkingSet64 -gt $MaxWorkingSetBytes) {
            $StopReason = 'peak working set exceeded 8 GiB'
        } elseif ($Elapsed -ge $HardWallSeconds) {
            $StopReason = 'hard 25-minute wall reached'
        } else {
            $FreeBytes = [int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024
            if ($FreeBytes -lt 4GB) { $script:LowMemorySamples += 1 } else { $script:LowMemorySamples = 0 }
            if ($script:LowMemorySamples -ge 20) {
                $script:SustainedPaging = $true
                $StopReason = 'sustained low-memory/paging risk for 5 seconds'
            }
        }
        if ($StopReason -ne 'completed') {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
            break
        }
    }
    $Process.WaitForExit()
    $ExitCode = if ($StopReason -eq 'completed') { $Process.ExitCode } else { -999 }
    $Record = [ordered]@{
        phase = $Name
        pid = $Process.Id
        started_at = $PhaseStarted.ToString('o')
        ended_at = (Get-Date).ToString('o')
        seconds = ((Get-Date) - $PhaseStarted).TotalSeconds
        exit_code = $ExitCode
        stop_reason = if ($ExitCode -eq 0) { $StopReason } elseif ($StopReason -eq 'completed') { "process exit $ExitCode" } else { $StopReason }
        stdout = $Stdout
        stderr = $Stderr
    }
    $script:PhaseRecords += $Record
    if ($ExitCode -ne 0) {
        $script:OverallStopReason = $Record.stop_reason
        return $false
    }
    return $true
}

try {
    $Ok = Invoke-MonitoredPhase 'extract' $SystemPython @(
        (Join-Path $Experiment 'extract_rows.py'), $Dataset, $SubsetPath, $SubsetJson
    )
    if ($Ok) {
        $Ok = Invoke-MonitoredPhase 'julia' $JuliaExe @(
            '--startup-file=no', "--project=$Repository", '--threads=20',
            (Join-Path $Experiment 'run_benchmark.jl')
        )
    }
    if ($Ok) {
        $Ok = Invoke-MonitoredPhase 'openvino' $OpenVinoPython @(
            (Join-Path $Experiment 'verify_openvino.py'), $Repository, $WeightsPath,
            $ReferencePath, $JuliaPhase, $OpenVinoPhase
        )
    }
} catch {
    $OverallStopReason = "wrapper exception: $($_.Exception.Message)"
}

$Monitor = [ordered]@{
    stop_reason = $OverallStopReason
    started_at = $BenchmarkStarted.ToString('o')
    ended_at = (Get-Date).ToString('o')
    wall_seconds = ((Get-Date) - $BenchmarkStarted).TotalSeconds
    peak_working_set_bytes = $PeakWorkingSet
    peak_private_bytes = $PeakPrivateBytes
    sustained_paging_observed = $SustainedPaging
    low_memory_threshold_bytes = [int64](4GB)
    phases = $PhaseRecords
    one_shot_retry_prohibited = $true
}
Write-JsonAtomic $MonitorPath $Monitor 12

& $SystemPython (Join-Path $Experiment 'finalize_result.py') $OutputDirectory
if ($LASTEXITCODE -ne 0) {
    throw "F finalizer failed with exit code $LASTEXITCODE"
}
if ($OverallStopReason -ne 'completed') {
    exit 2
}
exit 0
