[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("one_layer", "three_layer")]
    [string]$Model,

    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_learning_signal_1l3l_k64_v1",
    [string]$Julia = "julia"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$comparisonRoot = $PSScriptRoot
$experimentRoot = Split-Path -Parent $comparisonRoot
$contractPath = Join-Path $comparisonRoot "paired_100_update_contract.toml"
$manifestPath = Join-Path $Dataset "manifest.json"
$expectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"

if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf)) {
    throw "Pairing contract is missing: $contractPath"
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Teacher manifest is missing: $manifestPath"
}
$manifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestSha256 -ne $expectedManifestSha256) {
    throw "Teacher manifest SHA-256 differs from the frozen pairing contract"
}

$juliaMutex = [System.Threading.Mutex]::new(
    $false,
    'Local\TetrisBeatFirstV1ExclusiveJulia'
)
$juliaMutexHeld = $false
try {
    $juliaMutexHeld = $juliaMutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $juliaMutexHeld = $true
}
if (-not $juliaMutexHeld) {
    $juliaMutex.Dispose()
    throw "Another patched Tetris controller owns the exclusive Julia lease"
}

try {
    # Lease acquisition, process-list validation, launch, overlap monitoring,
    # and child exit form one critical section shared with the other patched
    # Tetris Julia controllers.
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        $pids = ($liveJulia | ForEach-Object { $_.Id }) -join ","
        throw "Refusing to start while Julia PID(s) are live: $pids"
    }

$contractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
$env:BEAT_SPARSE_PAIRING_CONTRACT_SHA256 = $contractSha256
$env:BEAT_TEACHER_DATASET = [IO.Path]::GetFullPath($Dataset)
$env:BEAT_ALLOW_PARTIAL_DATASET = "false"
$env:JULIA_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

if ($Model -eq "one_layer") {
    $env:BEAT_SPARSE_DATASET = $env:BEAT_TEACHER_DATASET
    $env:BEAT_SPARSE_OUTPUT = Join-Path ([IO.Path]::GetFullPath($OutputRoot)) "one_layer"
    $env:BEAT_SPARSE_RESUME = ""
    $env:BEAT_SPARSE_SEED = "2026071900"
    $env:BEAT_SPARSE_MODEL_SEED = "2026071901"
    $env:BEAT_SPARSE_SPLIT_SEED = "2026071902"
    $env:BEAT_SPARSE_SAMPLER_SEED = "2026071903"
    $env:BEAT_SPARSE_CANDIDATE_WIDTH = "80"
    $env:BEAT_SPARSE_EPOCHS = "1.0"
    $env:BEAT_SPARSE_MAX_UPDATES = "100"
    $env:BEAT_SPARSE_EVAL_INTERVAL = "25"
    $env:BEAT_SPARSE_CHECKPOINT_INTERVAL = "100"
    $env:BEAT_SPARSE_TRAIN_EVAL_STATES = "16"
    $env:BEAT_SPARSE_VALIDATION_EVAL_STATES = "32"
    $env:BEAT_SPARSE_EVALUATE_INITIAL = "true"
    $env:BEAT_SPARSE_VALIDATION_FRACTION = "0.20"
    $env:BEAT_SPARSE_BANK_LR = "0.0001"
    $env:BEAT_SPARSE_HEAD_LR = "0.0001"
    $env:BEAT_SPARSE_HEAD_WEIGHT_DECAY = "0.0"
    $env:BEAT_SPARSE_HEAD_BETA1 = "0.9"
    $env:BEAT_SPARSE_HEAD_BETA2 = "0.999"
    $env:BEAT_SPARSE_HEAD_EPSILON = "1.0e-8"
    $env:BEAT_SPARSE_WTA_M = "8"
    $env:BEAT_SPARSE_WTA_K = "4"
    $env:BEAT_SPARSE_WTA_L = "16"
    $env:BEAT_SPARSE_WTA_SEED = "2026071904"
    $entrypoint = Join-Path $experimentRoot "sparse_dynamic\train_sparse_supervised.jl"
} else {
    $env:BEAT_3L_DATASET = $env:BEAT_TEACHER_DATASET
    $env:BEAT_3L_OUTPUT = Join-Path ([IO.Path]::GetFullPath($OutputRoot)) "three_layer"
    $env:BEAT_3L_RESUME = ""
    $env:BEAT_3L_SEED = "2026071900"
    $env:BEAT_3L_MODEL_SEED = "2026071901"
    $env:BEAT_3L_SPLIT_SEED = "2026071902"
    $env:BEAT_3L_SAMPLER_SEED = "2026071903"
    $env:BEAT_3L_VARIANT = "k64"
    $env:BEAT_3L_EPOCHS = "1.0"
    $env:BEAT_3L_MAX_UPDATES = "100"
    $env:BEAT_3L_EVAL_INTERVAL = "25"
    $env:BEAT_3L_CHECKPOINT_INTERVAL = "100"
    $env:BEAT_3L_TRAIN_EVAL_STATES = "16"
    $env:BEAT_3L_VALIDATION_EVAL_STATES = "32"
    $env:BEAT_3L_EVALUATE_INITIAL = "true"
    $env:BEAT_3L_VALIDATION_FRACTION = "0.20"
    $env:BEAT_3L_LR = "0.0001"
    $env:BEAT_3L_WEIGHT_DECAY = "0.0"
    $env:BEAT_3L_BETA1 = "0.9"
    $env:BEAT_3L_BETA2 = "0.999"
    $env:BEAT_3L_EPSILON = "1.0e-8"
    $entrypoint = Join-Path $experimentRoot "sparse_dynamic_3layer\train_teacher_supervised.jl"
}

if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
    throw "Trainer entry point is missing: $entrypoint"
}

    $ownedJulia = $null
    $overlapIds = @()
    try {
        $ownedJulia = Start-Process `
            -FilePath $Julia `
            -ArgumentList @("--project=$experimentRoot", $entrypoint) `
            -WorkingDirectory $experimentRoot `
            -WindowStyle Hidden `
            -PassThru
        $exited = $false
        while (-not $exited) {
            $competitors = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue |
                Where-Object { $_.Id -ne $ownedJulia.Id })
            if ($competitors.Count -ne 0) {
                $overlapIds = @($overlapIds + $competitors.Id | Select-Object -Unique)
                if (-not $ownedJulia.HasExited) {
                    $ownedJulia.Kill()
                }
                break
            }
            $exited = $ownedJulia.WaitForExit(100)
        }
        $ownedJulia.WaitForExit()
        $competitors = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue |
            Where-Object { $_.Id -ne $ownedJulia.Id })
        if ($competitors.Count -ne 0) {
            $overlapIds = @($overlapIds + $competitors.Id | Select-Object -Unique)
        }
    } finally {
        if ($null -ne $ownedJulia -and -not $ownedJulia.HasExited) {
            $ownedJulia.Kill()
            $ownedJulia.WaitForExit()
        }
    }
    if ($overlapIds.Count -ne 0) {
        throw "$Model observed competing Julia PID(s): $($overlapIds -join ',')"
    }
    if ($ownedJulia.ExitCode -ne 0) {
        throw "$Model trainer exited with code $($ownedJulia.ExitCode)"
    }
} finally {
    if ($juliaMutexHeld) {
        try {
            $juliaMutex.ReleaseMutex()
        } catch {
            # The scientific arm has already terminated; cleanup cannot promote it.
        }
    }
    $juliaMutex.Dispose()
}
