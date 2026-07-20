param(
    [Parameter(Mandatory = $true)]
    [int]$SparseGateProcessId,
    [string]$RepositoryRoot = 'C:\Users\fshuu\Documents\tetris',
    [string]$DatasetRoot = 'D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3',
    [Parameter(Mandatory = $true)]
    [string]$SparseGateRoot,
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [Parameter(Mandatory = $true)]
    [string]$TrainingOutputRoot
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

function Assert-ExactHashBindings {
    param(
        [object[]]$Bindings,
        [string[]]$ExpectedPaths,
        [string]$Name
    )
    $expected = @($ExpectedPaths | ForEach-Object {
        (Resolve-Path -LiteralPath $_ -ErrorAction Stop).Path.ToLowerInvariant()
    })
    $boundPaths = @($Bindings | ForEach-Object {
        if ($null -eq $_ -or [string]::IsNullOrWhiteSpace([string]$_.path)) {
            throw "$Name contains an empty path binding"
        }
        if ([string]$_.sha256 -notmatch '^[0-9a-fA-F]{64}$') {
            throw "$Name contains an invalid SHA-256 binding: $($_.path)"
        }
        [System.IO.Path]::GetFullPath([string]$_.path).ToLowerInvariant()
    })
    if ($Bindings.Count -ne $expected.Count -or
        @($boundPaths | Select-Object -Unique).Count -ne $expected.Count -or
        @(Compare-Object -ReferenceObject $expected -DifferenceObject $boundPaths).Count -ne 0) {
        throw "$Name does not exactly match its required path allowlist"
    }
    Assert-HashBindings -Bindings $Bindings -Phase $Name
}

$numericClrTypes = [type[]]@(
    [System.SByte], [System.Byte],
    [System.Int16], [System.UInt16],
    [System.Int32], [System.UInt32],
    [System.Int64], [System.UInt64],
    [System.Single], [System.Double],
    [System.Decimal]
)

function Convert-RequiredFiniteDouble {
    param([object]$Value, [string]$Name)
    if ($null -eq $Value) {
        throw "required numeric value is missing: $Name"
    }
    if ($Value.GetType() -notin $numericClrTypes) {
        throw "required numeric value is not a JSON number: $Name"
    }
    try {
        $number = [double]$Value
    } catch {
        throw "required numeric value is not representable as a double: $Name"
    }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw "required numeric value is not finite: $Name"
    }
    return $number
}

function Convert-RequiredExactInteger {
    param([object]$Value, [string]$Name)
    $number = Convert-RequiredFiniteDouble -Value $Value -Name $Name
    if ($number -ne [math]::Truncate($number)) {
        throw "required integer value is not integral: $Name"
    }
    if ($number -lt [long]::MinValue -or $number -gt [long]::MaxValue) {
        throw "required integer value is outside Int64 range: $Name"
    }
    return [long]$number
}

function Assert-CanonicalPathEqual {
    param([object]$Actual, [string]$Expected, [string]$Name)
    if ([string]::IsNullOrWhiteSpace([string]$Actual)) {
        throw "required path is missing: $Name"
    }
    $actualFull = [System.IO.Path]::GetFullPath([string]$Actual)
    $expectedFull = [System.IO.Path]::GetFullPath($Expected)
    if (-not $actualFull.Equals($expectedFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Name path mismatch: actual=$actualFull expected=$expectedFull"
    }
}

function Assert-FalseBoolean {
    param([object]$Value, [string]$Name)
    if ($Value -isnot [bool] -or [bool]$Value) {
        throw "$Name must be the JSON boolean false"
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

function Assert-ProducerIdentity {
    param([object]$Published, [object]$Observed, [string]$Name)
    if ($null -eq $Published) {
        throw "$Name producer identity is missing"
    }
    $publishedId = Convert-RequiredExactInteger $Published.process_id "$Name.producer.process_id"
    if ($publishedId -ne [long]$Observed.process_id) {
        throw "$Name producer PID mismatch: published=$publishedId observed=$($Observed.process_id)"
    }
    if ([string]::IsNullOrWhiteSpace([string]$Published.process_name) -or
        -not ([string]$Published.process_name).Equals(
            [string]$Observed.process_name,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
        throw "$Name producer process name mismatch"
    }
    try {
        $publishedStart = [datetimeoffset]::Parse(
            [string]$Published.process_start_utc,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::RoundtripKind
        ).UtcDateTime
    } catch {
        throw "$Name producer start time is invalid"
    }
    $observedStart = [datetimeoffset]::Parse(
        [string]$Observed.process_start_utc,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [System.Globalization.DateTimeStyles]::RoundtripKind
    ).UtcDateTime
    if ($publishedStart -ne $observedStart) {
        throw "$Name producer start time mismatch"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$Observed.executable)) {
        Assert-CanonicalPathEqual `
            -Actual $Published.executable `
            -Expected ([string]$Observed.executable) `
            -Name "$Name.producer.executable"
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
        throw "refusing to overlap sparse signal with Julia PID(s): $($competingJulia.Id -join ',')"
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
    throw "signal gate root already exists; refusing overwrite: $RunRoot"
}
if (Test-Path -LiteralPath $TrainingOutputRoot) {
    throw "signal output root already exists; refusing overwrite: $TrainingOutputRoot"
}
New-Item -ItemType Directory -Path $RunRoot | Out-Null
$statusPath = Join-Path $RunRoot 'status.json'

trap {
    $failureMessage = $_.Exception.Message
    Release-JuliaLease
    try {
        Write-AtomicJson -Path $statusPath -Value ([ordered]@{
            status = 'failed'
            generated_at = (Get-Date).ToString('o')
            error = $failureMessage
            full_epoch_authorized = $false
        })
    } catch {
        # Preserve the original error if terminal publication itself fails.
    }
    exit 1
}

$project = Join-Path $RepositoryRoot 'experiments\beat_first_v1'
$sparseRoot = Join-Path $project 'sparse_dynamic'
$canonicalRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot -ErrorAction Stop).Path
$canonicalDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot -ErrorAction Stop).Path
$canonicalSparseGateRoot = (Resolve-Path -LiteralPath $SparseGateRoot -ErrorAction Stop).Path
$canonicalRunRoot = (Resolve-Path -LiteralPath $RunRoot -ErrorAction Stop).Path
$watcherBinding = (New-HashBindings -Paths @($PSCommandPath))[0]

Write-AtomicJson -Path $statusPath -Value ([ordered]@{
    status = 'waiting_for_sparse_system_gate'
    generated_at = (Get-Date).ToString('o')
    process_id = $SparseGateProcessId
    sparse_gate_root = $canonicalSparseGateRoot
    watcher = $watcherBinding
    full_epoch_authorized = $false
})

$sparseGate = Get-Process -Id $SparseGateProcessId -ErrorAction Stop
$observedProducer = Get-ProcessIdentity -Process $sparseGate
Wait-Process -InputObject $sparseGate

$sparseStatusPath = Join-Path $SparseGateRoot 'status.json'
$gateParsePath = Join-Path $SparseGateRoot 'gate_parse.json'
if (-not (Test-Path -LiteralPath $sparseStatusPath -PathType Leaf)) {
    throw "sparse system-gate status is missing: $sparseStatusPath"
}
if (-not (Test-Path -LiteralPath $gateParsePath -PathType Leaf)) {
    throw "sparse system-gate parse record is missing: $gateParsePath"
}
$sparseStatus = Get-Content -LiteralPath $sparseStatusPath -Raw | ConvertFrom-Json
if ([string]$sparseStatus.status -ne 'complete') {
    throw "sparse system gate did not complete: $($sparseStatus.status)"
}
Assert-FalseBoolean -Value $sparseStatus.training_authorized -Name 'status.training_authorized'
Assert-CanonicalPathEqual -Actual $sparseStatus.run_root -Expected $canonicalSparseGateRoot -Name 'status.run_root'
Assert-CanonicalPathEqual -Actual $sparseStatus.repository_root -Expected $canonicalRepositoryRoot -Name 'status.repository_root'
Assert-CanonicalPathEqual -Actual $sparseStatus.dataset_root -Expected $canonicalDatasetRoot -Name 'status.dataset_root'
Assert-ProducerIdentity -Published $sparseStatus.producer -Observed $observedProducer -Name 'status'

$parsedInvocationId = [guid]::Empty
if (-not [guid]::TryParse([string]$sparseStatus.invocation_id, [ref]$parsedInvocationId)) {
    throw 'sparse system-gate status invocation_id is not a GUID'
}
$expectedGateParsePath = [System.IO.Path]::GetFullPath($gateParsePath)
Assert-CanonicalPathEqual -Actual $sparseStatus.gate_parse -Expected $expectedGateParsePath -Name 'status.gate_parse'
if ([string]$sparseStatus.gate_parse_sha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'sparse system-gate status has an invalid gate_parse_sha256'
}
$actualGateParseSha256 = (
    Get-FileHash -LiteralPath $gateParsePath -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($actualGateParseSha256 -ne ([string]$sparseStatus.gate_parse_sha256).ToLowerInvariant()) {
    throw 'sparse system-gate parse record does not match the status-bound SHA-256'
}

$gateParse = Get-Content -LiteralPath $gateParsePath -Raw | ConvertFrom-Json
if ([string]$gateParse.status -ne 'passed') {
    throw "sparse system-gate parse record did not pass: $($gateParse.status)"
}
if ([string]$gateParse.invocation_id -ne [string]$sparseStatus.invocation_id) {
    throw 'sparse system-gate status and parse invocation IDs differ'
}
Assert-FalseBoolean -Value $gateParse.training_authorized -Name 'gate_parse.training_authorized'
Assert-CanonicalPathEqual -Actual $gateParse.run_root -Expected $canonicalSparseGateRoot -Name 'gate_parse.run_root'
Assert-CanonicalPathEqual -Actual $gateParse.repository_root -Expected $canonicalRepositoryRoot -Name 'gate_parse.repository_root'
Assert-CanonicalPathEqual -Actual $gateParse.dataset_root -Expected $canonicalDatasetRoot -Name 'gate_parse.dataset_root'
Assert-ProducerIdentity -Published $gateParse.producer -Observed $observedProducer -Name 'gate_parse'
$acceptedSparseStatusSha256 = (
    Get-FileHash -LiteralPath $sparseStatusPath -Algorithm SHA256
).Hash.ToLowerInvariant()
$acceptedGateParseSha256 = (
    Get-FileHash -LiteralPath $gateParsePath -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($acceptedGateParseSha256 -ne $actualGateParseSha256) {
    throw 'sparse system-gate parse record changed while it was being validated'
}

$productionPaths = @(
    (Join-Path $project 'training\core.jl'),
    (Join-Path $sparseRoot 'features.jl'),
    (Join-Path $sparseRoot 'wta_index.jl'),
    (Join-Path $sparseRoot 'model.jl'),
    (Join-Path $sparseRoot 'sparse_optimizer.jl'),
    (Join-Path $sparseRoot 'SparseQ20.jl'),
    (Join-Path $sparseRoot 'sparse_training.jl')
)
$harnessPaths = @(
    (Join-Path $sparseRoot 'test_sparse_q20.jl'),
    (Join-Path $sparseRoot 'test_sparse_training.jl'),
    (Join-Path $sparseRoot 'benchmark_sparse_q20.jl')
)
$environmentPaths = @(
    (Join-Path $project 'Project.toml'),
    (Join-Path $project 'Manifest.toml')
)
$launcherPaths = @(
    (Join-Path $RepositoryRoot 'tools\post_teacher_v3_sparse_gate.ps1'),
    (Join-Path $sparseRoot 'train_sparse_supervised.jl')
)
$sourceBindings = @($gateParse.source_hashes)
$harnessBindings = @($gateParse.harness_hashes)
$environmentBindings = @($gateParse.environment_hashes)
$launcherBindings = @($gateParse.launcher_hashes)
Assert-ExactHashBindings -Bindings $sourceBindings -ExpectedPaths $productionPaths -Name 'gate production closure'
Assert-ExactHashBindings -Bindings $harnessBindings -ExpectedPaths $harnessPaths -Name 'gate harness closure'
Assert-ExactHashBindings -Bindings $environmentBindings -ExpectedPaths $environmentPaths -Name 'gate environment closure'
Assert-ExactHashBindings -Bindings $launcherBindings -ExpectedPaths $launcherPaths -Name 'gate launcher closure'
Assert-ExactHashBindings -Bindings @($sparseStatus.source_hashes) -ExpectedPaths $productionPaths -Name 'status production closure'
Assert-ExactHashBindings -Bindings @($sparseStatus.harness_hashes) -ExpectedPaths $harnessPaths -Name 'status harness closure'
Assert-ExactHashBindings -Bindings @($sparseStatus.environment_hashes) -ExpectedPaths $environmentPaths -Name 'status environment closure'
Assert-ExactHashBindings -Bindings @($sparseStatus.launcher_hashes) -ExpectedPaths $launcherPaths -Name 'status launcher closure'

$expectedPasses = @(
    "gate`te036_real_teacher_input`tPASS",
    "gate`te036_inference_speed_3x`tPASS",
    "gate`te036_training_speed_2x`tPASS",
    "gate`te036_index_maintenance_under_25pct`tPASS",
    "gate`tindex_scaling`tPASS"
)
$publishedPasses = @($gateParse.required_passes)
if ($publishedPasses.Count -ne $expectedPasses.Count -or
    @($publishedPasses | Select-Object -Unique).Count -ne $expectedPasses.Count -or
    @(Compare-Object -ReferenceObject $expectedPasses -DifferenceObject $publishedPasses).Count -ne 0) {
    throw 'sparse system-gate required PASS set is not exact'
}

$julia = (Get-Command julia -ErrorAction Stop).Source
Assert-ExactHashBindings -Bindings @($gateParse.julia) -ExpectedPaths @($julia) -Name 'gate Julia runtime'
Assert-ExactHashBindings -Bindings @($sparseStatus.julia) -ExpectedPaths @($julia) -Name 'status Julia runtime'
$juliaBinding = $gateParse.julia

$datasetEvidencePath = if (Test-Path -LiteralPath $DatasetRoot -PathType Container) {
    Join-Path $DatasetRoot 'manifest.json'
} else {
    $DatasetRoot
}
Assert-ExactHashBindings -Bindings @($gateParse.dataset) -ExpectedPaths @($datasetEvidencePath) -Name 'gate dataset provenance'
Assert-ExactHashBindings -Bindings @($sparseStatus.dataset) -ExpectedPaths @($datasetEvidencePath) -Name 'status dataset provenance'
$datasetBinding = $gateParse.dataset
$gateUpstreamBindings = @($gateParse.upstream_status)
$statusUpstreamBindings = @($sparseStatus.upstream_status)
if ($gateUpstreamBindings.Count -ne 1 -or $statusUpstreamBindings.Count -ne 1) {
    throw 'sparse system-gate evidence must contain exactly one upstream status binding'
}
if ([string]$gateUpstreamBindings[0].sha256 -notmatch '^[0-9a-fA-F]{64}$' -or
    [string]$statusUpstreamBindings[0].sha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'sparse system-gate upstream status binding has an invalid SHA-256'
}
Assert-CanonicalPathEqual `
    -Actual $gateUpstreamBindings[0].path `
    -Expected ([string]$statusUpstreamBindings[0].path) `
    -Name 'gate upstream status'
if (-not ([string]$gateUpstreamBindings[0].sha256).Equals(
    [string]$statusUpstreamBindings[0].sha256,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw 'sparse system-gate status and parse upstream hashes differ'
}
Assert-HashBindings -Bindings $gateUpstreamBindings -Phase 'gate upstream status'

Acquire-JuliaLease
Assert-NoCompetingJulia
New-Item -ItemType Directory -Path $TrainingOutputRoot | Out-Null

$trainer = Join-Path $sparseRoot 'train_sparse_supervised.jl'
Remove-Item Env:BEAT_SPARSE_RESUME -ErrorAction SilentlyContinue
Remove-Item Env:BEAT_TEACHER_DATASET -ErrorAction SilentlyContinue
$env:BEAT_ALLOW_PARTIAL_DATASET = 'false'
$env:BEAT_SPARSE_DATASET = $DatasetRoot
$env:BEAT_SPARSE_OUTPUT = $TrainingOutputRoot
$env:BEAT_SPARSE_SEED = '20260718'
$env:BEAT_SPARSE_MODEL_SEED = '20260719'
$env:BEAT_SPARSE_SPLIT_SEED = '20260720'
$env:BEAT_SPARSE_SAMPLER_SEED = '20260721'
$env:BEAT_SPARSE_CANDIDATE_WIDTH = '208'
$env:BEAT_SPARSE_EPOCHS = '0.01'
$env:BEAT_SPARSE_MAX_UPDATES = '250'
$env:BEAT_SPARSE_EVAL_INTERVAL = '250'
$env:BEAT_SPARSE_CHECKPOINT_INTERVAL = '250'
$env:BEAT_SPARSE_TRAIN_EVAL_STATES = '64'
$env:BEAT_SPARSE_VALIDATION_EVAL_STATES = '128'
$env:BEAT_SPARSE_EVALUATE_INITIAL = 'true'
$env:BEAT_SPARSE_VALIDATION_FRACTION = '0.20'
$env:BEAT_SPARSE_BANK_LR = '0.01'
$env:BEAT_SPARSE_HEAD_LR = '0.001'
$env:BEAT_SPARSE_HEAD_WEIGHT_DECAY = '0.0001'
$env:BEAT_SPARSE_WTA_M = '8'
$env:BEAT_SPARSE_WTA_K = '4'
$env:BEAT_SPARSE_WTA_L = '16'
$env:BEAT_SPARSE_WTA_SEED = '2026071801'

Write-AtomicJson -Path $statusPath -Value ([ordered]@{
    status = 'running_250_update_signal'
    generated_at = (Get-Date).ToString('o')
    sparse_gate_invocation_id = [string]$gateParse.invocation_id
    sparse_gate_status = $sparseStatusPath
    sparse_gate_parse = $gateParsePath
    maximum_updates = 250
    watcher = $watcherBinding
    full_epoch_authorized = $false
})
$run = Invoke-LoggedProcess -Name 'sparse_q20_signal_250' -Executable $julia -Arguments @(
    '--startup-file=no',
    '--threads=20',
    "--project=$project",
    $trainer
)
Assert-NoCompetingJulia
$postSignalSparseStatusSha256 = (
    Get-FileHash -LiteralPath $sparseStatusPath -Algorithm SHA256
).Hash.ToLowerInvariant()
$postSignalGateParseSha256 = (
    Get-FileHash -LiteralPath $gateParsePath -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($postSignalSparseStatusSha256 -ne $acceptedSparseStatusSha256 -or
    $postSignalGateParseSha256 -ne $acceptedGateParseSha256) {
    throw 'sparse system-gate evidence changed during the signal run'
}
Assert-HashBindings -Bindings $sourceBindings -Phase 'post-signal production'
Assert-HashBindings -Bindings $harnessBindings -Phase 'post-signal harness'
Assert-HashBindings -Bindings $environmentBindings -Phase 'post-signal environment'
Assert-HashBindings -Bindings $launcherBindings -Phase 'post-signal launcher'
Assert-HashBindings -Bindings @($juliaBinding) -Phase 'post-signal Julia runtime'
Assert-HashBindings -Bindings @($datasetBinding) -Phase 'post-signal dataset provenance'
Assert-HashBindings -Bindings $gateUpstreamBindings -Phase 'post-signal upstream status'
Assert-HashBindings -Bindings @($watcherBinding) -Phase 'post-signal watcher'

$runDirectories = @(Get-ChildItem -LiteralPath $TrainingOutputRoot -Directory)
if ($runDirectories.Count -ne 1) {
    throw "expected exactly one sparse signal run directory; found $($runDirectories.Count)"
}
$trainingRun = $runDirectories[0].FullName
$metricsPath = Join-Path $trainingRun 'metrics.jsonl'
$latestPath = Join-Path $trainingRun 'latest.json'
if (-not (Test-Path -LiteralPath $metricsPath -PathType Leaf)) {
    throw "signal metrics are missing: $metricsPath"
}
if (-not (Test-Path -LiteralPath $latestPath -PathType Leaf)) {
    throw "signal latest checkpoint pointer is missing: $latestPath"
}
$metricLines = @(Get-Content -LiteralPath $metricsPath | Where-Object { $_.Trim().Length -gt 0 })
if ($metricLines.Count -ne 2) {
    throw "signal run must publish exactly initial and final metrics; found $($metricLines.Count)"
}
$initial = $metricLines[0] | ConvertFrom-Json
$final = $metricLines[-1] | ConvertFrom-Json
$initialUpdate = Convert-RequiredExactInteger $initial.update 'initial.update'
$finalUpdate = Convert-RequiredExactInteger $final.update 'final.update'
if ($initialUpdate -ne 0) {
    throw "signal initial update is not zero: $initialUpdate"
}
if ($finalUpdate -ne 250) {
    throw "signal final update is not 250: $finalUpdate"
}

$latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
$checkpointPath = [string]$latest.path
if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "signal checkpoint is missing: $checkpointPath"
}
$expectedCheckpointPath = [System.IO.Path]::GetFullPath(
    (Join-Path $trainingRun 'checkpoints\checkpoint_000000250.jld2')
)
$actualCheckpointPath = [System.IO.Path]::GetFullPath($checkpointPath)
if (-not $actualCheckpointPath.Equals($expectedCheckpointPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "signal latest pointer is not the exact run-local update-250 checkpoint: $checkpointPath"
}
$checkpointBytes = Convert-RequiredExactInteger $latest.bytes 'latest.bytes'
$actualCheckpointBytes = (Get-Item -LiteralPath $checkpointPath).Length
if ($checkpointBytes -le 0 -or $checkpointBytes -ne $actualCheckpointBytes) {
    throw "signal checkpoint byte count mismatch: latest=$checkpointBytes actual=$actualCheckpointBytes"
}
if ([string]$latest.sha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'signal latest pointer has an invalid checkpoint sha256'
}
$checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointSha256 -ne ([string]$latest.sha256).ToLowerInvariant()) {
    throw 'signal checkpoint sha256 does not match latest pointer'
}

$initialValidationTop1 = Convert-RequiredFiniteDouble $initial.validation.top1_agreement 'initial.validation.top1_agreement'
$finalValidationTop1 = Convert-RequiredFiniteDouble $final.validation.top1_agreement 'final.validation.top1_agreement'
$initialValidationNdcg = Convert-RequiredFiniteDouble $initial.validation.ndcg 'initial.validation.ndcg'
$finalValidationNdcg = Convert-RequiredFiniteDouble $final.validation.ndcg 'final.validation.ndcg'
$initialValidationPairwise = Convert-RequiredFiniteDouble $initial.validation.pairwise_accuracy 'initial.validation.pairwise_accuracy'
$finalValidationPairwise = Convert-RequiredFiniteDouble $final.validation.pairwise_accuracy 'final.validation.pairwise_accuracy'
$initialValidationLoss = Convert-RequiredFiniteDouble $initial.validation.composite_loss 'initial.validation.composite_loss'
$finalValidationLoss = Convert-RequiredFiniteDouble $final.validation.composite_loss 'final.validation.composite_loss'
$epochEquivalent = Convert-RequiredFiniteDouble $final.epoch_equivalent 'final.epoch_equivalent'
$updatesPerSecond = Convert-RequiredFiniteDouble $final.throughput.updates_per_second 'final.throughput.updates_per_second'
$candidatesPerSecond = Convert-RequiredFiniteDouble $final.throughput.candidates_per_second 'final.throughput.candidates_per_second'
if ($epochEquivalent -le 0.0 -or $epochEquivalent -ge 1.0 -or
    $updatesPerSecond -le 0.0 -or $candidatesPerSecond -le 0.0) {
    throw 'signal epoch must be partial and throughput metrics must be positive'
}

$summary = [ordered]@{
    status = 'complete'
    update = $finalUpdate
    epoch_equivalent = $epochEquivalent
    initial_validation_top1 = $initialValidationTop1
    final_validation_top1 = $finalValidationTop1
    initial_validation_ndcg = $initialValidationNdcg
    final_validation_ndcg = $finalValidationNdcg
    initial_validation_pairwise = $initialValidationPairwise
    final_validation_pairwise = $finalValidationPairwise
    initial_validation_loss = $initialValidationLoss
    final_validation_loss = $finalValidationLoss
    updates_per_second = $updatesPerSecond
    candidates_per_second = $candidatesPerSecond
    checkpoint = $checkpointPath
    checkpoint_sha256 = $checkpointSha256
    checkpoint_bytes = $checkpointBytes
    metrics = $metricsPath
    training_stdout = $run.stdout
    training_stderr = $run.stderr
    sparse_gate_invocation_id = [string]$gateParse.invocation_id
    sparse_gate_status_sha256 = $acceptedSparseStatusSha256
    sparse_gate_parse_sha256 = $acceptedGateParseSha256
    source_closure = $sourceBindings
    environment_closure = $environmentBindings
    launcher_closure = $launcherBindings
    julia = $juliaBinding
    dataset = $datasetBinding
    upstream_status = $gateUpstreamBindings[0]
    watcher = $watcherBinding
    full_epoch_authorized = $false
}
Write-AtomicJson -Path (Join-Path $RunRoot 'signal_summary.json') -Value $summary
Write-AtomicJson -Path $statusPath -Value ([ordered]@{
    status = 'complete'
    generated_at = (Get-Date).ToString('o')
    signal_summary = (Join-Path $RunRoot 'signal_summary.json')
    sparse_gate_invocation_id = [string]$gateParse.invocation_id
    full_epoch_authorized = $false
})
Release-JuliaLease
