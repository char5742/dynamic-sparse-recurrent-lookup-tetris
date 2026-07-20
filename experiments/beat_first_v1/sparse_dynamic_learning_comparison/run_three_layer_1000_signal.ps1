[CmdletBinding()]
param(
    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_1000_v1",
    [string]$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$comparisonRoot = $PSScriptRoot
$experimentRoot = Split-Path -Parent $comparisonRoot
$contractPath = Join-Path $comparisonRoot "three_layer_1000_update_contract.toml"
$manifestPath = Join-Path $Dataset "manifest.json"
$entrypoint = Join-Path $experimentRoot "sparse_dynamic_3layer\train_teacher_supervised.jl"
$expectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"

foreach ($required in @($contractPath, $manifestPath, $entrypoint, $Julia)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}
$manifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestSha256 -ne $expectedManifestSha256) {
    throw "Teacher manifest SHA-256 differs from the frozen signal contract"
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Signal output root already exists: $resolvedOutput"
}

$juliaMutex = [System.Threading.Mutex]::new($false, 'Local\TetrisBeatFirstV1ExclusiveJulia')
$juliaMutexHeld = $false
try {
    try {
        $juliaMutexHeld = $juliaMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $juliaMutexHeld = $true
    }
    if (-not $juliaMutexHeld) {
        throw "Another patched Tetris controller owns the exclusive Julia lease"
    }
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Refusing to start while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
    }

    New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
    $stdoutPath = Join-Path $resolvedOutput "controller.stdout.log"
    $stderrPath = Join-Path $resolvedOutput "controller.stderr.log"
    $statusPath = Join-Path $resolvedOutput "controller_status.json"
    $contractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()

    $env:BEAT_SPARSE_PAIRING_CONTRACT_SHA256 = $contractSha256
    $env:BEAT_TEACHER_DATASET = [IO.Path]::GetFullPath($Dataset)
    $env:BEAT_3L_DATASET = $env:BEAT_TEACHER_DATASET
    $env:BEAT_3L_OUTPUT = $resolvedOutput
    $env:BEAT_3L_RESUME = ""
    $env:BEAT_ALLOW_PARTIAL_DATASET = "false"
    $env:BEAT_3L_SEED = "2026071900"
    $env:BEAT_3L_MODEL_SEED = "2026071901"
    $env:BEAT_3L_SPLIT_SEED = "2026071902"
    $env:BEAT_3L_SAMPLER_SEED = "2026071903"
    $env:BEAT_3L_VARIANT = "k64"
    $env:BEAT_3L_EPOCHS = "1.0"
    $env:BEAT_3L_MAX_UPDATES = "1000"
    $env:BEAT_3L_EVAL_INTERVAL = "100"
    $env:BEAT_3L_CHECKPOINT_INTERVAL = "1000"
    $env:BEAT_3L_TRAIN_EVAL_STATES = "16"
    $env:BEAT_3L_VALIDATION_EVAL_STATES = "32"
    $env:BEAT_3L_EVALUATE_INITIAL = "true"
    $env:BEAT_3L_VALIDATION_FRACTION = "0.20"
    $env:BEAT_3L_LR = "0.0001"
    $env:BEAT_3L_WEIGHT_DECAY = "0.0"
    $env:BEAT_3L_BETA1 = "0.9"
    $env:BEAT_3L_BETA2 = "0.999"
    $env:BEAT_3L_EPSILON = "1.0e-8"
    $env:JULIA_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"

    $ownedJulia = Start-Process `
        -FilePath $Julia `
        -ArgumentList @("--project=$experimentRoot", $entrypoint) `
        -WorkingDirectory $experimentRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    [ordered]@{
        status = "running"
        pid = $ownedJulia.Id
        started_at = [DateTimeOffset]::Now.ToString("o")
        contract = $contractPath
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        output_root = $resolvedOutput
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM

    $overlapIds = @()
    $exited = $false
    $deadline = [DateTimeOffset]::Now.AddMinutes(10)
    while (-not $exited) {
        if ([DateTimeOffset]::Now -gt $deadline) {
            if (-not $ownedJulia.HasExited) { $ownedJulia.Kill() }
            throw "Three-layer signal exceeded the frozen ten-minute wall-time limit"
        }
        $competitors = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue |
            Where-Object { $_.Id -ne $ownedJulia.Id })
        if ($competitors.Count -ne 0) {
            $overlapIds = @($competitors.Id)
            if (-not $ownedJulia.HasExited) { $ownedJulia.Kill() }
            throw "Competing Julia PID(s) appeared: $($overlapIds -join ',')"
        }
        $exited = $ownedJulia.WaitForExit(100)
    }
    $ownedJulia.WaitForExit()
    if ($ownedJulia.ExitCode -ne 0) {
        throw "Three-layer signal trainer exited with code $($ownedJulia.ExitCode)"
    }
    [ordered]@{
        status = "completed"
        pid = $ownedJulia.Id
        completed_at = [DateTimeOffset]::Now.ToString("o")
        exit_code = $ownedJulia.ExitCode
        contract = $contractPath
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        output_root = $resolvedOutput
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM
} catch {
    if (Test-Path -LiteralPath $resolvedOutput) {
        [ordered]@{
            status = "failed"
            failed_at = [DateTimeOffset]::Now.ToString("o")
            error = $_.Exception.Message
            output_root = $resolvedOutput
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedOutput "controller_status.json") -Encoding utf8NoBOM
    }
    throw
} finally {
    if ($juliaMutexHeld) {
        try { $juliaMutex.ReleaseMutex() } catch {}
    }
    $juliaMutex.Dispose()
}
