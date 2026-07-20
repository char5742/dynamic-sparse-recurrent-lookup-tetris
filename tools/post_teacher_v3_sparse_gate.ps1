param(
    [Parameter(Mandatory = $true)]
    [int]$PostTeacherGateProcessId,
    [string]$RepositoryRoot = 'C:\Users\fshuu\Documents\tetris',
    [string]$DatasetRoot = 'D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3',
    [Parameter(Mandatory = $true)]
    [string]$PostTeacherGateRoot,
    [Parameter(Mandatory = $true)]
    [string]$RunRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-AtomicJson {
    param([string]$Path, [object]$Value)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function New-HashBindings {
    param([string[]]$Paths)
    return @($Paths | ForEach-Object {
        $resolved = (Resolve-Path -LiteralPath $_ -ErrorAction Stop).Path
        [ordered]@{
            path = $resolved
            sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
}

function Assert-HashBindings {
    param([object[]]$Bindings, [string]$Phase)
    foreach ($binding in $Bindings) {
        $path = [string]$binding.path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "$Phase closure file is missing: $path"
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$binding.sha256).ToLowerInvariant()) {
            throw "$Phase closure changed: $path"
        }
    }
}

function Get-ProcessIdentity {
    param([System.Diagnostics.Process]$Process)
    $executable = $null
    try {
        $executable = $Process.Path
    } catch {
        $executable = $null
    }
    return [ordered]@{
        process_id = [int]$Process.Id
        process_name = [string]$Process.ProcessName
        process_start_utc = $Process.StartTime.ToUniversalTime().ToString('o')
        executable = $executable
    }
}

$script:JuliaMutex = $null
$script:JuliaMutexHeld = $false

function Acquire-JuliaLease {
    $script:JuliaMutex = [System.Threading.Mutex]::new(
        $false,
        'Local\TetrisBeatFirstV1ExclusiveJulia'
    )
    try {
        $script:JuliaMutexHeld = $script:JuliaMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $script:JuliaMutexHeld = $true
    }
    if (-not $script:JuliaMutexHeld) {
        $script:JuliaMutex.Dispose()
        $script:JuliaMutex = $null
        throw 'another patched Tetris Julia controller owns the exclusive Julia lease'
    }
}

function Release-JuliaLease {
    if ($script:JuliaMutexHeld -and $null -ne $script:JuliaMutex) {
        try {
            $script:JuliaMutex.ReleaseMutex()
        } catch {
            # The terminal status remains authoritative if cleanup observes an abandoned lease.
        }
    }
    $script:JuliaMutexHeld = $false
    if ($null -ne $script:JuliaMutex) {
        $script:JuliaMutex.Dispose()
        $script:JuliaMutex = $null
    }
}

function Assert-NoCompetingJulia {
    $competingJulia = @(Get-Process -Name julia -ErrorAction SilentlyContinue)
    if ($competingJulia.Count -ne 0) {
        throw "refusing to overlap sparse gate with Julia PID(s): $($competingJulia.Id -join ',')"
    }
}

function Invoke-LoggedProcess {
    param(
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments
    )
    $stdout = Join-Path $RunRoot "$Name.stdout.log"
    $stderr = Join-Path $RunRoot "$Name.stderr.log"
    $started = Get-Date
    $overlapIds = @()
    Push-Location $RepositoryRoot
    try {
        $process = Start-Process `
            -FilePath $Executable `
            -ArgumentList $Arguments `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $exited = $false
        while (-not $exited) {
            $competitors = @(Get-Process -Name julia -ErrorAction SilentlyContinue | Where-Object {
                $_.Id -ne $process.Id
            })
            if ($competitors.Count -ne 0) {
                $overlapIds = @($overlapIds + $competitors.Id | Select-Object -Unique)
                try {
                    if (-not $process.HasExited) {
                        $process.Kill()
                    }
                } catch {
                    # The overlap is already terminal even if the owned process exited first.
                }
                break
            }
            $exited = $process.WaitForExit(100)
        }
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        $processId = $process.Id
    } finally {
        Pop-Location
    }
    $record = [ordered]@{
        name = $Name
        process_id = $processId
        started_at = $started.ToString('o')
        finished_at = (Get-Date).ToString('o')
        exit_code = $exitCode
        observed_competing_julia_process_ids = $overlapIds
        stdout = $stdout
        stderr = $stderr
    }
    Write-AtomicJson -Path (Join-Path $RunRoot "$Name.json") -Value $record
    if ($overlapIds.Count -ne 0) {
        throw "$Name observed competing Julia PID(s): $($overlapIds -join ',')"
    }
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
    return $record
}

if (Test-Path -LiteralPath $RunRoot) {
    throw "sparse system-gate root already exists; refusing overwrite: $RunRoot"
}
New-Item -ItemType Directory -Path $RunRoot | Out-Null
$statusPath = Join-Path $RunRoot 'status.json'

$producer = Get-ProcessIdentity -Process (Get-Process -Id $PID -ErrorAction Stop)
$gateInvocationId = [guid]::NewGuid().ToString('D')
$canonicalRunRoot = (Resolve-Path -LiteralPath $RunRoot -ErrorAction Stop).Path
$canonicalRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot -ErrorAction Stop).Path
$canonicalDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot -ErrorAction Stop).Path
$canonicalPostTeacherGateRoot = (
    Resolve-Path -LiteralPath $PostTeacherGateRoot -ErrorAction Stop
).Path
$gateControllerBinding = (New-HashBindings -Paths @($PSCommandPath))[0]

function Write-GateStatus {
    param(
        [string]$Status,
        [System.Collections.IDictionary]$Fields = ([ordered]@{})
    )
    $record = [ordered]@{
        status = $Status
        generated_at = (Get-Date).ToString('o')
        invocation_id = $gateInvocationId
        producer = $producer
        run_root = $canonicalRunRoot
        repository_root = $canonicalRepositoryRoot
        dataset_root = $canonicalDatasetRoot
    }
    foreach ($key in $Fields.Keys) {
        $record[$key] = $Fields[$key]
    }
    Write-AtomicJson -Path $statusPath -Value $record
}

trap {
    $failureMessage = $_.Exception.Message
    Release-JuliaLease
    try {
        Write-GateStatus -Status 'failed' -Fields ([ordered]@{
            error = $failureMessage
            training_authorized = $false
        })
    } catch {
        # Preserve the original error if terminal publication itself fails.
    }
    exit 1
}

$postGate = Get-Process -Id $PostTeacherGateProcessId -ErrorAction Stop
$postGateIdentity = Get-ProcessIdentity -Process $postGate
Write-GateStatus -Status 'waiting_for_post_teacher_gate' -Fields ([ordered]@{
    upstream_process_id = $PostTeacherGateProcessId
    upstream_process_observed = $postGateIdentity
    upstream_root = $canonicalPostTeacherGateRoot
    training_authorized = $false
})
Wait-Process -InputObject $postGate

$postGateStatusPath = Join-Path $PostTeacherGateRoot 'status.json'
if (-not (Test-Path -LiteralPath $postGateStatusPath -PathType Leaf)) {
    throw "post-teacher gate status is missing: $postGateStatusPath"
}
$postGateStatus = Get-Content -LiteralPath $postGateStatusPath -Raw | ConvertFrom-Json
if ([string]$postGateStatus.status -ne 'complete') {
    throw "post-teacher gate did not complete successfully: $($postGateStatus.status)"
}
try {
    $postGateGeneratedUtc = [datetimeoffset]::Parse(
        [string]$postGateStatus.generated_at,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [System.Globalization.DateTimeStyles]::RoundtripKind
    ).UtcDateTime
} catch {
    throw 'post-teacher gate status has an invalid generated_at timestamp'
}
$postGateStartedUtc = [datetimeoffset]::Parse(
    [string]$postGateIdentity.process_start_utc,
    [System.Globalization.CultureInfo]::InvariantCulture,
    [System.Globalization.DateTimeStyles]::RoundtripKind
).UtcDateTime
if ($postGateGeneratedUtc -lt $postGateStartedUtc -or
    (Get-Item -LiteralPath $postGateStatusPath).LastWriteTimeUtc -lt $postGateStartedUtc) {
    throw 'post-teacher gate status predates the observed upstream process'
}
$postGateStatusBinding = (New-HashBindings -Paths @($postGateStatusPath))[0]

$project = Join-Path $RepositoryRoot 'experiments\beat_first_v1'
$sparseRoot = Join-Path $project 'sparse_dynamic'
$julia = (Get-Command julia -ErrorAction Stop).Source
$common = @('--startup-file=no', '--threads=20', "--project=$project")
$env:BEAT_ALLOW_PARTIAL_DATASET = 'false'
$env:BEAT_SPARSE_BENCH_SYNTHETIC_SMOKE = 'false'

$productionClosure = New-HashBindings -Paths @(
    (Join-Path $project 'training\core.jl'),
    (Join-Path $sparseRoot 'features.jl'),
    (Join-Path $sparseRoot 'wta_index.jl'),
    (Join-Path $sparseRoot 'model.jl'),
    (Join-Path $sparseRoot 'sparse_optimizer.jl'),
    (Join-Path $sparseRoot 'SparseQ20.jl'),
    (Join-Path $sparseRoot 'sparse_training.jl')
)
$harnessClosure = New-HashBindings -Paths @(
    (Join-Path $sparseRoot 'test_sparse_q20.jl'),
    (Join-Path $sparseRoot 'test_sparse_training.jl'),
    (Join-Path $sparseRoot 'benchmark_sparse_q20.jl')
)
$environmentClosure = New-HashBindings -Paths @(
    (Join-Path $project 'Project.toml'),
    (Join-Path $project 'Manifest.toml')
)
$launcherClosure = @($gateControllerBinding) + @(New-HashBindings -Paths @(
    (Join-Path $sparseRoot 'train_sparse_supervised.jl')
))
$juliaBinding = (New-HashBindings -Paths @($julia))[0]
$datasetEvidencePath = if (Test-Path -LiteralPath $DatasetRoot -PathType Container) {
    Join-Path $DatasetRoot 'manifest.json'
} else {
    $DatasetRoot
}
$datasetBinding = (New-HashBindings -Paths @($datasetEvidencePath))[0]

function Assert-GateClosure {
    param([string]$Phase)
    Assert-HashBindings -Bindings $productionClosure -Phase "$Phase production"
    Assert-HashBindings -Bindings $harnessClosure -Phase "$Phase harness"
    Assert-HashBindings -Bindings $environmentClosure -Phase "$Phase environment"
    Assert-HashBindings -Bindings $launcherClosure -Phase "$Phase launcher"
    Assert-HashBindings -Bindings @($juliaBinding) -Phase "$Phase Julia runtime"
    Assert-HashBindings -Bindings @($datasetBinding) -Phase "$Phase dataset provenance"
    Assert-HashBindings -Bindings @($postGateStatusBinding) -Phase "$Phase upstream status"
}

Acquire-JuliaLease
Assert-NoCompetingJulia
Assert-GateClosure -Phase 'pre-correctness'

Write-GateStatus -Status 'running_sparse_correctness' -Fields ([ordered]@{
    source_hashes = $productionClosure
    harness_hashes = $harnessClosure
    environment_hashes = $environmentClosure
    launcher_hashes = $launcherClosure
    julia = $juliaBinding
    dataset = $datasetBinding
    training_authorized = $false
})
Invoke-LoggedProcess -Name 'test_sparse_q20' -Executable $julia -Arguments (
    $common + @((Join-Path $sparseRoot 'test_sparse_q20.jl'))
) | Out-Null

Assert-GateClosure -Phase 'pre-training-resume'
Assert-NoCompetingJulia
Write-GateStatus -Status 'running_sparse_training_resume' -Fields ([ordered]@{
    source_hashes = $productionClosure
    harness_hashes = $harnessClosure
    environment_hashes = $environmentClosure
    launcher_hashes = $launcherClosure
    julia = $juliaBinding
    dataset = $datasetBinding
    training_authorized = $false
})
Invoke-LoggedProcess -Name 'test_sparse_training' -Executable $julia -Arguments (
    $common + @((Join-Path $sparseRoot 'test_sparse_training.jl'))
) | Out-Null

Assert-GateClosure -Phase 'pre-real-teacher-benchmark'
Assert-NoCompetingJulia
Write-GateStatus -Status 'running_real_teacher_dense_twin_benchmark' -Fields ([ordered]@{
    source_hashes = $productionClosure
    harness_hashes = $harnessClosure
    environment_hashes = $environmentClosure
    launcher_hashes = $launcherClosure
    julia = $juliaBinding
    dataset = $datasetBinding
    training_authorized = $false
})
$benchmark = Invoke-LoggedProcess -Name 'benchmark_sparse_q20_real' -Executable $julia -Arguments (
    $common + @(
        (Join-Path $sparseRoot 'benchmark_sparse_q20.jl'),
        "--dataset=$DatasetRoot"
    )
)
Assert-GateClosure -Phase 'post-real-teacher-benchmark'
Assert-NoCompetingJulia

$benchmarkLines = @(Get-Content -LiteralPath $benchmark.stdout)
$requiredPasses = @(
    "gate`te036_real_teacher_input`tPASS",
    "gate`te036_inference_speed_3x`tPASS",
    "gate`te036_training_speed_2x`tPASS",
    "gate`te036_index_maintenance_under_25pct`tPASS",
    "gate`tindex_scaling`tPASS"
)
$missing = @($requiredPasses | Where-Object { $_ -notin $benchmarkLines })
$explicitFailures = @($benchmarkLines | Where-Object {
    $_ -like 'gate*' -and ($_ -match '\tFAIL(?:\t|$)')
})
$gateParsePath = Join-Path $RunRoot 'gate_parse.json'
if ($missing.Count -ne 0 -or $explicitFailures.Count -ne 0) {
    Write-AtomicJson -Path $gateParsePath -Value ([ordered]@{
        status = 'failed'
        invocation_id = $gateInvocationId
        producer = $producer
        run_root = $canonicalRunRoot
        missing_required_passes = $missing
        explicit_failures = $explicitFailures
        source_hashes = $productionClosure
        harness_hashes = $harnessClosure
        environment_hashes = $environmentClosure
        launcher_hashes = $launcherClosure
        julia = $juliaBinding
        dataset_root = $canonicalDatasetRoot
        dataset = $datasetBinding
        upstream_status = $postGateStatusBinding
    })
    throw 'SparseQ20 real-input systems gate did not pass every frozen criterion'
}

Write-AtomicJson -Path $gateParsePath -Value ([ordered]@{
    status = 'passed'
    invocation_id = $gateInvocationId
    producer = $producer
    run_root = $canonicalRunRoot
    repository_root = $canonicalRepositoryRoot
    dataset_root = $canonicalDatasetRoot
    required_passes = $requiredPasses
    source_hashes = $productionClosure
    harness_hashes = $harnessClosure
    environment_hashes = $environmentClosure
    launcher_hashes = $launcherClosure
    julia = $juliaBinding
    dataset = $datasetBinding
    upstream_status = $postGateStatusBinding
    training_authorized = $false
})
$gateParseSha256 = (
    Get-FileHash -LiteralPath $gateParsePath -Algorithm SHA256
).Hash.ToLowerInvariant()
Write-GateStatus -Status 'complete' -Fields ([ordered]@{
    benchmark_stdout = $benchmark.stdout
    gate_parse = (Resolve-Path -LiteralPath $gateParsePath -ErrorAction Stop).Path
    gate_parse_sha256 = $gateParseSha256
    source_hashes = $productionClosure
    harness_hashes = $harnessClosure
    environment_hashes = $environmentClosure
    launcher_hashes = $launcherClosure
    julia = $juliaBinding
    dataset = $datasetBinding
    upstream_status = $postGateStatusBinding
    training_authorized = $false
})
Release-JuliaLease
