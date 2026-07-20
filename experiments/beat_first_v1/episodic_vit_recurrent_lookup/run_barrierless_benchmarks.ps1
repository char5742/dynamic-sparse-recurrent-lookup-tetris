param(
    [string]$OutputRoot = (Join-Path 'C:\tmp' ("evrl_barrierless_bench_" + (Get-Date -Format 'yyyyMMddTHHmmss'))),
    [string[]]$ResetFailedArm = @()
)

$ErrorActionPreference = 'Stop'

$Julia = 'C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe'
$Project = 'C:\Users\fshuu\Documents\tetris\experiments\beat_first_v1'
$Runner = Join-Path $Project 'episodic_vit_recurrent_lookup\run_teacher_signal.jl'
$Checkpoint = 'D:\tetris-paper-plus\runs\beat_first_v1\episodic_vit_recurrent_lookup\evrl_routed_workspace_relation_d128_r2_a2_c3k2_wta16t13_u1000_t16_20260720_r1\checkpoints\checkpoint_000001000.jls'
$CheckpointSha = 'cd745a8449c81c36c7d8ee471a9a6ac64b6670b788399c7ed1c869b68a5c5f23'

$env:JULIA_NUM_THREADS = '20,0'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'

# Immutable checkpoint geometry. Optimizer/loss settings are inherited from
# the checkpoint by teacher_training.jl and are intentionally not overridden.
$env:DSRL_CARRIER_DIM = '128'
$env:DSRL_TABLES_PER_BLOCK = '13'
$env:DSRL_WTA_CHOICES = '16'
$env:DSRL_ROWS_PER_TABLE_LOOKUP = '1'
$env:EVRL_ATTENTION_DIM = '32'
$env:EVRL_ATTENTION_HEADS = '4'
$env:EVRL_REGISTERS = '2'
$env:EVRL_ROUTER_TABLES = '2'
$env:EVRL_ROUTER_BITS = '4'
$env:EVRL_ROUTER_BUCKET_CAP = '16'
$env:EVRL_EPISODIC_SHORTLIST = '4'
$env:EVRL_EPISODIC_CANDIDATE_CAP = '16'
$env:EVRL_SPATIAL_ANCHORS = '2'
$env:EVRL_SPATIAL_SHORTLIST = '2'
$env:EVRL_SPATIAL_CANDIDATE_CAP = '3'
$env:EVRL_FFN_DIM = '32'
$env:EVRL_INITIAL_HALT_PROBABILITY = '0.8'

$env:EVRL_OUTPUT = $OutputRoot
$env:EVRL_RESUME_CHECKPOINT = $Checkpoint
$env:EVRL_RESUME_SHA256 = $CheckpointSha
$env:EVRL_MAX_UPDATES = '1110'
$env:EVRL_STATE_BATCH = '4'
$env:EVRL_BENCHMARK_ONLY = '1'
$env:EVRL_BENCHMARK_WARMUP_UPDATES = '10'
$env:EVRL_WRITE_CHECKPOINTS = '0'
$env:EVRL_CHECKPOINT_INTERVAL = '1110'
$env:EVRL_EVAL_INTERVAL = '1110'

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$ResolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path.TrimEnd('\')
$summaries = [ordered]@{}

function Invoke-Arm {
    param(
        [string]$Name,
        [string]$Scheduler,
        [string]$CpuSets,
        [int]$Chunk,
        [bool]$Adaptive,
        [bool]$Detailed
    )

    if ($Name -notmatch '^[0-9]{2}_[a-z0-9_]+$') {
        throw "Unsafe benchmark arm name: $Name"
    }
    $armPath = [IO.Path]::GetFullPath((Join-Path $ResolvedOutputRoot $Name))
    $armParent = [IO.Directory]::GetParent($armPath).FullName.TrimEnd('\')
    if ($armParent -ne $ResolvedOutputRoot) {
        throw "Benchmark arm escaped output root: $armPath"
    }
    $summaryPath = Join-Path $armPath 'summary.json'
    if (Test-Path -LiteralPath $summaryPath) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $summaryRunDir = [IO.Path]::GetFullPath([string]$summary.run_dir).TrimEnd('\')
        if (
            [string]$summary.status -ne 'complete' -or
            [int]$summary.update -ne 1110 -or
            [int]$summary.measured_updates -ne 100 -or
            $summaryRunDir -ne $armPath
        ) {
            throw "Existing summary for $Name is not a valid completed benchmark"
        }
        $script:summaries[$Name] = $summary
        return $summary
    }

    $existing = Get-Process -Name julia -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Refusing to start $Name while another Julia process is running"
    }
    if (Test-Path -LiteralPath $armPath) {
        if ($ResetFailedArm -notcontains $Name) {
            throw "Incomplete arm $Name already exists; pass -ResetFailedArm $Name to archive and retry it"
        }
        $archiveName = "$Name.failed-$(Get-Date -Format 'yyyyMMddTHHmmss')"
        $archivePath = [IO.Path]::GetFullPath((Join-Path $ResolvedOutputRoot $archiveName))
        $archiveParent = [IO.Directory]::GetParent($archivePath).FullName.TrimEnd('\')
        if ($archiveParent -ne $ResolvedOutputRoot -or (Test-Path -LiteralPath $archivePath)) {
            throw "Unsafe or occupied failed-arm archive path: $archivePath"
        }
        Move-Item -LiteralPath $armPath -Destination $archivePath
    }

    $env:EVRL_RUN_ID = $Name
    $env:EVRL_SCHEDULER = $Scheduler
    $env:EVRL_CPUSET_MODE = $CpuSets
    $env:EVRL_QUEUE_CHUNK = [string]$Chunk
    $env:EVRL_ADAPTIVE_TAIL = if ($Adaptive) { '1' } else { '0' }
    $env:EVRL_DETAILED_BENCHMARK = if ($Detailed) { '1' } else { '0' }

    & $Julia --startup-file=no --history-file=no --threads=20,0 `
        --project=$Project $Runner
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "$Name did not produce $summaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $script:summaries[$Name] = $summary
    return $summary
}

Invoke-Arm '01_current_static_none_c8' 'static' 'none' 8 $false $false | Out-Null
Invoke-Arm '02_barriered_dynamic_none_c8' 'dynamic' 'none' 8 $false $false | Out-Null

$fixedNames = @()
foreach ($chunk in 8, 16, 32) {
    $name = "03_barrierless_none_c$chunk"
    $fixedNames += $name
    Invoke-Arm $name 'barrierless' 'none' $chunk $false $true | Out-Null
}

$fixedWinner = $fixedNames | ForEach-Object {
    [pscustomobject]@{
        Name = $_
        Chunk = [int]($_ -replace '^.*_c', '')
        UpdatesPerSecond = [double]$summaries[$_].segment_throughput.updates_per_second
    }
} | Sort-Object UpdatesPerSecond -Descending | Select-Object -First 1

$winnerChunk = $fixedWinner.Chunk
Invoke-Arm '04_barrierless_all_fixed' 'barrierless' 'all' $winnerChunk $false $true | Out-Null
Invoke-Arm '05_barrierless_ponly_fixed' 'barrierless' 'p_only' $winnerChunk $false $true | Out-Null
Invoke-Arm '06_barrierless_all_adaptive' 'barrierless' 'all' $winnerChunk $true $true | Out-Null

$records = foreach ($entry in $summaries.GetEnumerator()) {
    $summary = $entry.Value
    [ordered]@{
        name = $entry.Key
        update = [int]$summary.update
        measured_updates = [int]$summary.measured_updates
        training_seconds = [double]$summary.segment_throughput.training_seconds
        updates_per_second = [double]$summary.segment_throughput.updates_per_second
        average_cpu_percent = [double]$summary.segment_throughput.average_cpu_percent
        candidate_cpu_percent = [double]$summary.segment_throughput.candidate_cpu_percent
        candidate_assigned_cpu_percent = [double]$summary.segment_throughput.candidate_assigned_cpu_percent
        scheduler_benchmark = $summary.scheduler_benchmark
        run_dir = [string]$summary.run_dir
    }
}

$report = [ordered]@{
    output_root = $OutputRoot
    fixed_winner = $fixedWinner
    records = $records
}
$reportPath = Join-Path $OutputRoot 'benchmark_report.json'
$report | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $reportPath -Encoding utf8
$report | ConvertTo-Json -Depth 100
