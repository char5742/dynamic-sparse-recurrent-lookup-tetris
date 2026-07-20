[CmdletBinding()]
param(
    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "",
    [ValidateSet(2500, 5000, 20000)]
    [int]$MaximumUpdates = 2500,
    [ValidateSet("k64", "k128", "k256")]
    [string]$Variant = "k64",
    [ValidateSet("baseline015", "margin100", "hardneg015", "rawgap100")]
    [string]$ObjectiveProfile = "baseline015",
    [string]$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe",
    [string]$RawGapOneShotNonce = "",
    [string]$RawGapOneShotReservation = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-FiniteJsonValue {
    param($Value, [string]$Path)
    if ($null -eq $Value) { return }
    if ($Value -is [double] -or $Value -is [single]) {
        if ([double]::IsNaN([double]$Value) -or [double]::IsInfinity([double]$Value)) {
            throw "Non-finite metric at $Path"
        }
        return
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        foreach ($property in $Value.PSObject.Properties) {
            Assert-FiniteJsonValue -Value $property.Value -Path "$Path.$($property.Name)"
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        $index = 0
        foreach ($item in $Value) {
            Assert-FiniteJsonValue -Value $item -Path "$Path[$index]"
            $index += 1
        }
    }
}

function Assert-RunLeaf {
    param([string]$Path, [string]$Root)
    $resolved = [IO.Path]::GetFullPath($Path)
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Run artifact escapes the fresh output root: $resolved"
    }
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Run artifact is not a file: $resolved"
    }
    $item = Get-Item -LiteralPath $resolved -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Run artifact is a reparse point: $resolved"
    }
    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $cursor = $item.Directory
    while ($null -ne $cursor) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Run artifact traverses a reparse point: $($cursor.FullName)"
        }
        if ($cursor.FullName.Equals($rootPath, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        if (-not $cursor.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Run artifact ancestry escapes the fresh output root: $resolved"
        }
        $cursor = $cursor.Parent
    }
    return $resolved
}
$comparisonRoot = $PSScriptRoot
$experimentRoot = Split-Path -Parent $comparisonRoot
$marginPivot = $ObjectiveProfile -eq "margin100"
$hardNegativePivot = $ObjectiveProfile -eq "hardneg015"
$rawGapPivot = $ObjectiveProfile -eq "rawgap100"
if (-not $rawGapPivot -and
    (-not [string]::IsNullOrWhiteSpace($RawGapOneShotNonce) -or
     -not [string]::IsNullOrWhiteSpace($RawGapOneShotReservation))) {
    throw "Raw-gap one-shot credentials are forbidden for other CPF profiles"
}
if (($marginPivot -or $hardNegativePivot -or $rawGapPivot) -and
    ($Variant -ne "k128" -or $MaximumUpdates -ne 20000)) {
    throw "The objective pivot profiles are frozen to fresh k128 20000-update runs"
}
$contractName = if ($marginPivot) {
    "three_layer_cpf_k128_margin1_20000_update_contract.toml"
} elseif ($hardNegativePivot) {
    "three_layer_cpf_k128_hardneg_20000_update_contract.toml"
} elseif ($rawGapPivot) {
    "three_layer_cpf_k128_rawgap_20000_update_contract.toml"
} elseif ($Variant -ne "k64") {
    if ($Variant -eq "k128" -and $MaximumUpdates -eq 20000) {
        "three_layer_cpf_k128_20000_update_contract.toml"
    } elseif ($MaximumUpdates -ne 2500) {
        throw "Only the promoted k128 arm permits the frozen 20000-update extension"
    } elseif ($Variant -eq "k128") {
        "three_layer_cpf_k128_2500_update_contract.toml"
    } else {
        "three_layer_cpf_k256_2500_update_contract.toml"
    }
} elseif ($MaximumUpdates -eq 2500) {
    "three_layer_cpf_2500_update_contract.toml"
} elseif ($MaximumUpdates -eq 5000) {
    "three_layer_cpf_5000_update_contract.toml"
} else {
    throw "The frozen k64 arms are exactly 2500 or 5000 updates"
}
$contractPath = Join-Path $comparisonRoot $contractName
$defaultRunName = if ($marginPivot) {
    "sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1"
} elseif ($hardNegativePivot) {
    "sparse_3l_k128_hardneg_teacher_signal_cpf_20000_v1"
} elseif ($rawGapPivot) {
    "sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1"
} elseif ($Variant -ne "k64") {
    if ($Variant -eq "k128" -and $MaximumUpdates -eq 20000) {
        "sparse_3l_k128_teacher_signal_cpf_20000_v1"
    } elseif ($Variant -eq "k128") {
        "sparse_3l_k128_teacher_signal_cpf_2500_v1"
    } else {
        "sparse_3l_k256_teacher_signal_cpf_2500_v1"
    }
} elseif ($MaximumUpdates -eq 2500) {
    "sparse_3l_teacher_signal_cpf_2500_v1"
} else {
    "sparse_3l_teacher_signal_cpf_5000_v1"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\$defaultRunName"
}
$manifestPath = Join-Path $Dataset "manifest.json"
$entrypoint = Join-Path $experimentRoot "sparse_dynamic_3layer\train_teacher_supervised.jl"
$expectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"

foreach ($required in @($contractPath, $manifestPath, $entrypoint, $Julia)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}
if ($rawGapPivot) {
    $frozenDataset = [IO.Path]::GetFullPath("D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3")
    $frozenOutput = [IO.Path]::GetFullPath("D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1")
    $frozenJulia = [IO.Path]::GetFullPath("C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe")
    $frozenReservation = [IO.Path]::GetFullPath("D:\tetris-paper-plus\runs\beat_first_v1\.sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1.reservation.json")
    $frozenConsumed = [IO.Path]::GetFullPath("D:\tetris-paper-plus\runs\beat_first_v1\.sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1.consumed.json")
    if (-not [IO.Path]::GetFullPath($Dataset).Equals($frozenDataset, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Raw-gap one-shot dataset root changed"
    }
    if (-not [IO.Path]::GetFullPath($OutputRoot).Equals($frozenOutput, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Raw-gap one-shot output root changed; retry/alternate roots are forbidden"
    }
    if (-not [IO.Path]::GetFullPath($Julia).Equals($frozenJulia, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Raw-gap one-shot Julia path changed"
    }
    if ([string]::IsNullOrWhiteSpace($RawGapOneShotReservation) -or
        -not [IO.Path]::GetFullPath($RawGapOneShotReservation).Equals($frozenReservation, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Raw-gap one-shot reservation path is missing or changed"
    }
    if ($RawGapOneShotNonce -cnotmatch '^[0-9a-f]{32}$') {
        throw "Raw-gap one-shot nonce must be exactly 32 lowercase hexadecimal characters"
    }
    if (Test-Path -LiteralPath $frozenConsumed) {
        throw "Raw-gap one-shot was already consumed"
    }
    if (-not (Test-Path -LiteralPath $frozenReservation -PathType Leaf)) {
        throw "Raw-gap one-shot reservation is missing"
    }
    $reservationItem = Get-Item -LiteralPath $frozenReservation -Force
    if (($reservationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Raw-gap one-shot reservation must not be a reparse point"
    }
    $rawGapContractItem = Get-Item -LiteralPath $contractPath
    if ([int64]$rawGapContractItem.Length -ne 5125) {
        throw "Raw-gap preregistration contract byte count changed"
    }
    $rawGapContractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($rawGapContractSha256 -cne "5a766947e9e34b1de4822b6622880db0892ed6fb4326cbff10c415ba4c393263") {
        throw "Raw-gap preregistration contract SHA-256 changed"
    }
    $reservation = Get-Content -LiteralPath $frozenReservation -Raw -Encoding utf8 |
        ConvertFrom-Json -ErrorAction Stop
    $expectedReservationProperties = @(
        "format_version", "experiment_id", "status", "nonce",
        "dataset_root", "output_root", "julia_path", "reservation_path",
        "consumed_path", "contract_sha256", "dataset_manifest_sha256"
    )
    $actualReservationProperties = @($reservation.PSObject.Properties.Name)
    if ($actualReservationProperties.Count -ne $expectedReservationProperties.Count) {
        throw "Raw-gap one-shot reservation schema changed"
    }
    foreach ($propertyName in $expectedReservationProperties) {
        if ($actualReservationProperties -cnotcontains $propertyName) {
            throw "Raw-gap one-shot reservation lacks $propertyName"
        }
    }
    if ([int]$reservation.format_version -ne 1 -or
        [string]$reservation.experiment_id -cne "sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1" -or
        [string]$reservation.status -cne "reserved_for_atomic_consume" -or
        [string]$reservation.nonce -cne $RawGapOneShotNonce -or
        -not [IO.Path]::GetFullPath([string]$reservation.dataset_root).Equals($frozenDataset, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$reservation.output_root).Equals($frozenOutput, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$reservation.julia_path).Equals($frozenJulia, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$reservation.reservation_path).Equals($frozenReservation, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$reservation.consumed_path).Equals($frozenConsumed, [StringComparison]::OrdinalIgnoreCase) -or
        [string]$reservation.contract_sha256 -cne $rawGapContractSha256 -or
        [string]$reservation.dataset_manifest_sha256 -cne $expectedManifestSha256) {
        throw "Raw-gap one-shot reservation does not match the frozen invocation"
    }
    $rawGapReservationSha256 = (Get-FileHash -LiteralPath $frozenReservation -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestSha256 -ne $expectedManifestSha256) {
    throw "Teacher manifest SHA-256 differs from the frozen CPF1 contract"
}
$comparisonBaselinePath = ""
$comparisonBaselineSha256 = ""
if ($Variant -ne "k64") {
    if ($marginPivot) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_20000_v1\sparse_3l_20260719T135857\metrics.jsonl"
        $comparisonBaselineSha256 = "8ed7ecba6eb01e916ce85c6b7164628b3b0121696e00004c6e6d8bbb540a012d"
    } elseif ($hardNegativePivot) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1_retry1\sparse_3l_20260719T143553\metrics.jsonl"
        $comparisonBaselineSha256 = "2a5149ff47a4d16053679162b71c2ea589efdd3c5a5d5a80814eb0fdf474d597"
    } elseif ($rawGapPivot) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1_retry1\sparse_3l_20260719T143553\metrics.jsonl"
        $comparisonBaselineSha256 = "2a5149ff47a4d16053679162b71c2ea589efdd3c5a5d5a80814eb0fdf474d597"
    } elseif ($Variant -eq "k128" -and $MaximumUpdates -eq 20000) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_2500_v1\sparse_3l_20260719T134221\metrics.jsonl"
        $comparisonBaselineSha256 = "b6298c1ba512414782309e69e018f35a698f28821fe5a9681ec1acc127c9bff6"
    } elseif ($Variant -eq "k128") {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_cpf_2500_v1\sparse_3l_20260719T131935\metrics.jsonl"
        $comparisonBaselineSha256 = "ff685ea03abdefd9dc6da0fe35152b35573903e4130faabb4074ceae1905c0c3"
    } else {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_2500_v1\sparse_3l_20260719T134221\metrics.jsonl"
        $comparisonBaselineSha256 = "b6298c1ba512414782309e69e018f35a698f28821fe5a9681ec1acc127c9bff6"
    }
    if (-not (Test-Path -LiteralPath $comparisonBaselinePath -PathType Leaf)) {
        throw "Pinned $Variant comparison-baseline metrics are missing"
    }
    $observedBaselineSha256 = (Get-FileHash -LiteralPath $comparisonBaselinePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observedBaselineSha256 -cne $comparisonBaselineSha256) {
        throw "Pinned $Variant comparison-baseline metrics SHA-256 changed"
    }
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "CPF1 output root already exists: $resolvedOutput"
}

$juliaMutex = [System.Threading.Mutex]::new($false, 'Local\TetrisBeatFirstV1ExclusiveJulia')
$held = $false
try {
    try { $held = $juliaMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $held = $true }
    if (-not $held) { throw "Another patched Tetris controller owns the exclusive Julia lease" }
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Refusing to start while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
    }

    if ($rawGapPivot) {
        if (Test-Path -LiteralPath $frozenConsumed) {
            throw "Raw-gap one-shot was consumed before launch"
        }
        if (-not (Test-Path -LiteralPath $frozenReservation -PathType Leaf)) {
            throw "Raw-gap one-shot reservation disappeared before launch"
        }
        $recheckedReservationSha256 = (Get-FileHash -LiteralPath $frozenReservation -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($recheckedReservationSha256 -cne $rawGapReservationSha256) {
            throw "Raw-gap one-shot reservation changed before launch"
        }
        # Same-volume File.Move is the atomic consume point.  Every failure
        # after this statement intentionally burns the sole registered arm.
        [IO.File]::Move($frozenReservation, $frozenConsumed)
        if ((Test-Path -LiteralPath $frozenReservation) -or
            -not (Test-Path -LiteralPath $frozenConsumed -PathType Leaf)) {
            throw "Raw-gap one-shot reservation was not consumed atomically"
        }
        $rawGapConsumedSha256 = (Get-FileHash -LiteralPath $frozenConsumed -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($rawGapConsumedSha256 -cne $rawGapReservationSha256) {
            throw "Raw-gap consumed sentinel differs from its nonce-bound reservation"
        }
    }

    New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
    $stdoutPath = Join-Path $resolvedOutput "controller.stdout.log"
    $stderrPath = Join-Path $resolvedOutput "controller.stderr.log"
    $statusPath = Join-Path $resolvedOutput "controller_status.json"
    $contractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($rawGapPivot -and $contractSha256 -cne $rawGapContractSha256) {
        throw "Raw-gap preregistration contract changed between preflight and launch"
    }

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
    $env:BEAT_3L_VARIANT = $Variant
    $env:BEAT_3L_ROUTING_MODE = "fixed_wta"
    foreach ($staleRoutingVariable in @(
        "BEAT_3L_MONGOOSE_LR", "BEAT_3L_MONGOOSE_BETA",
        "BEAT_3L_MONGOOSE_SEED", "BEAT_3L_MONGOOSE_WARMUP_UPDATES",
        "BEAT_3L_MONGOOSE_REFRESH_INTERVAL"
    )) {
        Remove-Item -LiteralPath "Env:$staleRoutingVariable" -ErrorAction SilentlyContinue
    }
    $env:BEAT_3L_EPOCHS = "1.0"
    $env:BEAT_3L_MAX_UPDATES = [string]$MaximumUpdates
    $evaluationInterval = if ($MaximumUpdates -eq 20000) { 2000 } else { 500 }
    $env:BEAT_3L_EVAL_INTERVAL = [string]$evaluationInterval
    $env:BEAT_3L_EVAL_SCHEDULE = if ($rawGapPivot) { "0,2000,5000,10000,20000" } else { "" }
    $env:BEAT_3L_CHECKPOINT_INTERVAL = if ($MaximumUpdates -eq 2500) { "500" } elseif ($MaximumUpdates -eq 5000) { "1000" } else { "10000" }
    $env:BEAT_3L_CHECKPOINT_SCHEDULE = if ($rawGapPivot) { "5000,10000,20000" } else { "" }
    $env:BEAT_3L_TRAIN_EVAL_STATES = "128"
    $env:BEAT_3L_VALIDATION_EVAL_STATES = "512"
    $env:BEAT_3L_EVALUATE_INITIAL = "true"
    $env:BEAT_3L_VALIDATION_FRACTION = "0.20"
    $env:BEAT_3L_LR = "0.0001"
    $env:BEAT_3L_WEIGHT_DECAY = "0.0"
    $env:BEAT_3L_BETA1 = "0.9"
    $env:BEAT_3L_BETA2 = "0.999"
    $env:BEAT_3L_EPSILON = "1.0e-8"
    $expectedMarginWeight = if ($marginPivot -or $rawGapPivot) { 1.0 } else { 0.15 }
    # The trainer intentionally stores the effective objective weight as
    # Float32 and publishes Float64(trainer.objective_margin_weight).  Compare
    # metrics against that exact effective value while keeping the controller
    # and frozen contract value at the requested decimal (0.15 or 1.0).
    $expectedMetricMarginWeight = [double][single]$expectedMarginWeight
    $env:BEAT_3L_MARGIN_WEIGHT = [string]$expectedMarginWeight
    $expectedMarginMode = if ($hardNegativePivot) { "student_hard_negative" } else { "fixed_teacher_top2" }
    $env:BEAT_3L_MARGIN_MODE = $expectedMarginMode
    $expectedObjectiveMode = if ($rawGapPivot) { "raw_teacher_top_gap_huber" } else { "standardized_listnet_plus_margin" }
    $env:BEAT_3L_OBJECTIVE_MODE = $expectedObjectiveMode
    $env:JULIA_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"

    $ownedJulia = Start-Process -FilePath $Julia `
        -ArgumentList @("--startup-file=no", "--project=$experimentRoot", $entrypoint) `
        -WorkingDirectory $experimentRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [ordered]@{
        status = "running"; pid = $ownedJulia.Id; started_at = [DateTimeOffset]::Now.ToString("o")
        variant = $Variant; maximum_updates = $MaximumUpdates
        objective_profile = $ObjectiveProfile
        objective_margin_weight = $expectedMarginWeight
        objective_margin_mode = $expectedMarginMode
        objective_mode = $expectedObjectiveMode
        contract = $contractPath; contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256; output_root = $resolvedOutput
        comparison_baseline_metrics_path = $comparisonBaselinePath
        comparison_baseline_metrics_sha256 = $comparisonBaselineSha256
        one_shot_nonce = if ($rawGapPivot) { $RawGapOneShotNonce } else { "" }
        one_shot_consumed_path = if ($rawGapPivot) { $frozenConsumed } else { "" }
        one_shot_consumed_sha256 = if ($rawGapPivot) { $rawGapConsumedSha256 } else { "" }
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM

    $deadlineMinutes = if ($rawGapPivot) { 30 } elseif ($MaximumUpdates -eq 20000) { 20 } else { 15 }
    $deadline = [DateTimeOffset]::Now.AddMinutes($deadlineMinutes)
    while (-not $ownedJulia.WaitForExit(100)) {
        if ([DateTimeOffset]::Now -gt $deadline) {
            $ownedJulia.Kill(); $ownedJulia.WaitForExit()
            throw "CPF1 signal exceeded the frozen $deadlineMinutes-minute wall-time limit"
        }
        $competitors = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue |
            Where-Object { $_.Id -ne $ownedJulia.Id })
        if ($competitors.Count -ne 0) {
            $ownedJulia.Kill(); $ownedJulia.WaitForExit()
            throw "Competing Julia PID(s) appeared: $(($competitors.Id) -join ',')"
        }
    }
    if ($ownedJulia.ExitCode -ne 0) {
        throw "CPF1 signal trainer exited with code $($ownedJulia.ExitCode)"
    }

    $metricFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "metrics.jsonl")
    if ($metricFiles.Count -ne 1) {
        throw "CPF1 run must publish exactly one metrics.jsonl; got $($metricFiles.Count)"
    }
    $metricsPath = Assert-RunLeaf -Path $metricFiles[0].FullName -Root $resolvedOutput
    $metricLines = @(Get-Content -LiteralPath $metricsPath | Where-Object { $_.Trim().Length -gt 0 })
    $expectedUpdates = if ($rawGapPivot) {
        @(0, 2000, 5000, 10000, 20000)
    } else {
        @(0..([int]($MaximumUpdates / $evaluationInterval)) | ForEach-Object { $_ * $evaluationInterval })
    }
    if ($metricLines.Count -ne $expectedUpdates.Count) {
        throw "CPF1 metrics do not contain the exact frozen evaluation sequence"
    }
    $metrics = @($metricLines | ForEach-Object { $_ | ConvertFrom-Json -ErrorAction Stop })
    for ($i = 0; $i -lt $expectedUpdates.Count; $i += 1) {
        if ([int]$metrics[$i].update -ne $expectedUpdates[$i]) {
            throw "CPF1 metric update sequence changed at index $i"
        }
        if (-not ($metrics[$i].PSObject.Properties.Name -contains "objective_margin_weight") -or
            [double]$metrics[$i].objective_margin_weight -ne $expectedMetricMarginWeight) {
            throw "CPF1 metric objective margin weight changed at index $i"
        }
        if (-not ($metrics[$i].PSObject.Properties.Name -contains "objective_margin_mode") -or
            [string]$metrics[$i].objective_margin_mode -cne $expectedMarginMode) {
            throw "CPF1 metric objective margin mode changed at index $i"
        }
        if (-not ($metrics[$i].PSObject.Properties.Name -contains "objective_mode") -or
            [string]$metrics[$i].objective_mode -cne $expectedObjectiveMode) {
            throw "CPF1 metric objective mode changed at index $i"
        }
        if ($rawGapPivot) {
            if ([double]$metrics[$i].effective_listnet_weight -ne 0.0 -or
                [double]$metrics[$i].effective_margin_weight -ne 1.0 -or
                [double]$metrics[$i].effective_raw_top_gap_weight -ne 1.0) {
                throw "CPF1 raw-gap effective objective weights changed at index $i"
            }
            foreach ($splitName in @("training", "validation")) {
                $splitMetric = $metrics[$i].$splitName
                if ([double]$splitMetric.listnet_loss -ne 0.0 -or
                    [double]$splitMetric.effective_listnet_weight -ne 0.0 -or
                    [double]$splitMetric.effective_raw_top_gap_weight -ne 1.0) {
                    throw "CPF1 raw-gap $splitName objective contribution changed at index $i"
                }
                if (-not ($splitMetric.PSObject.Properties.Name -contains "diagnostic_listnet_loss")) {
                    throw "CPF1 raw-gap $splitName lacks its unweighted ListNet diagnostic"
                }
            }
        }
        Assert-FiniteJsonValue -Value $metrics[$i] -Path "metrics[$i]"
    }
    $baseline = $metrics[0]
    $final = $metrics[-1]
    if ([string]$final.variant -ne $Variant) { throw "CPF1 final variant differs from $Variant" }
    if ([string]$final.routing_policy -ne "wta-collision-count-stable-id-prefilter-v1") {
        throw "CPF1 final routing policy differs from the frozen contract"
    }
    if ([int]$final.throughput.updates -ne $MaximumUpdates) {
        throw "CPF1 throughput did not authoritatively reach update $MaximumUpdates"
    }
    if ([int]$final.parameter_contract.total_parameters -ne 19924022) {
        throw "CPF1 total parameter count changed"
    }
    $expectedActiveParameters = if ($Variant -eq "k64") { 31934 } elseif ($Variant -eq "k128") { 58214 } else { 110774 }
    $expectedActiveCounts = if ($Variant -eq "k64") { @(24, 20, 20) } elseif ($Variant -eq "k128") { @(48, 40, 40) } else { @(96, 80, 80) }
    if ([int]$final.parameter_contract.active_parameters -ne $expectedActiveParameters) {
        throw "CPF1 active parameter count changed for $Variant"
    }
    if ((@($final.parameter_contract.active_counts) -join ',') -ne ($expectedActiveCounts -join ',')) {
        throw "CPF1 active widths changed for $Variant"
    }
    if ([string]$final.parameter_contract.routing_policy -ne [string]$final.routing_policy) {
        throw "CPF1 metric and parameter-contract routing policies differ"
    }
    $ndcgDelta = [double]$final.validation.ndcg - [double]$baseline.validation.ndcg
    $pairwiseDelta = [double]$final.validation.pairwise_accuracy - [double]$baseline.validation.pairwise_accuracy
    $top1Delta = [double]$final.validation.top1_agreement - [double]$baseline.validation.top1_agreement
    $lossFraction = [double]$final.validation.composite_loss / [double]$baseline.validation.composite_loss
    if ($Variant -eq "k64") {
        $minimumNdcgDelta = if ($MaximumUpdates -eq 2500) { 0.015 } else { 0.025 }
        $minimumPairwiseDelta = if ($MaximumUpdates -eq 2500) { 0.015 } else { 0.025 }
        $minimumTop1Delta = if ($MaximumUpdates -eq 2500) { 0.0 } else { 0.02 }
        $maximumLossFraction = if ($MaximumUpdates -eq 2500) { 0.95 } else { 0.90 }
        if ($ndcgDelta -lt $minimumNdcgDelta) { throw "CPF1 held NDCG delta missed the frozen gate: $ndcgDelta" }
        if ($pairwiseDelta -lt $minimumPairwiseDelta) { throw "CPF1 held pairwise delta missed the frozen gate: $pairwiseDelta" }
        if ($top1Delta -lt $minimumTop1Delta) { throw "CPF1 held top-1 delta missed the frozen gate: $top1Delta" }
        if ($lossFraction -gt $maximumLossFraction) { throw "CPF1 held composite loss fraction missed the frozen gate: $lossFraction" }
    }
    if ($MaximumUpdates -eq 5000) {
        if ([double]$final.validation.listnet_loss -ge [double]$baseline.validation.listnet_loss) {
            throw "CPF1 held ListNet loss did not improve"
        }
        if ([double]$final.validation.q_huber_loss -ge [double]$baseline.validation.q_huber_loss) {
            throw "CPF1 held Q Huber loss did not improve"
        }
        if ([double]$final.validation.q_std -lt 0.15) { throw "CPF1 held Q std collapsed" }
        if ([double]$final.validation.action_margin -lt 0.02) { throw "CPF1 held action margin collapsed" }
        $tail = @($metrics | Select-Object -Last 3)
        for ($i = 1; $i -lt $tail.Count; $i += 1) {
            $priorLoss = [double]$tail[$i - 1].validation.composite_loss
            $rebound = ([double]$tail[$i].validation.composite_loss - $priorLoss) / [math]::Abs($priorLoss)
            if ($rebound -gt 0.02) { throw "CPF1 final-three loss rebound exceeded the frozen gate: $rebound" }
        }
        $coverageMinima = @(1.0, 0.999, 0.999)
        for ($i = 0; $i -lt 3; $i += 1) {
            if ([double]$final.bank_coverage[$i].ever_updated_fraction -lt $coverageMinima[$i]) {
                throw "CPF1 layer $($i + 1) bank coverage missed the frozen gate"
            }
        }
        $midpoint = @($metrics | Where-Object { [int]$_.update -eq 2500 })
        if ($midpoint.Count -ne 1) { throw "CPF1 5000 run lacks its midpoint record" }
        $midNdcg = [double]$midpoint[0].validation.ndcg - [double]$baseline.validation.ndcg
        $midPair = [double]$midpoint[0].validation.pairwise_accuracy - [double]$baseline.validation.pairwise_accuracy
        $midLoss = [double]$midpoint[0].validation.composite_loss / [double]$baseline.validation.composite_loss
        if ($midNdcg -lt 0.015 -or $midPair -lt 0.015 -or $midLoss -gt 0.95) {
            throw "CPF1 update-2500 midpoint gate failed"
        }
    }
    if ($Variant -ne "k64") {
        $observedBaselineSha256 = (Get-FileHash -LiteralPath $comparisonBaselinePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($observedBaselineSha256 -cne $comparisonBaselineSha256) {
            throw "Pinned $Variant comparison-baseline metrics changed during the run"
        }
        $finalTop1 = [double]$final.validation.top1_agreement
        $finalNdcg = [double]$final.validation.ndcg
        $finalPairwise = [double]$final.validation.pairwise_accuracy
        $finalLoss = [double]$final.validation.composite_loss
        if ($Variant -eq "k128" -and $MaximumUpdates -eq 20000) {
            if ($rawGapPivot) {
                $observation = @($metrics | Where-Object { [int]$_.update -eq 5000 })
                if ($observation.Count -ne 1) { throw "Raw-gap run lacks exact update-5000 observation" }
                if ($finalTop1 -lt 0.234375 -or
                    [double]$final.validation.action_margin -lt 0.02 -or
                    $finalNdcg -lt 0.9374244614677424 -or
                    $finalPairwise -lt 0.7407546814419289 -or
                    [double]$final.validation.q_std -lt 0.15 -or
                    [double]$final.validation.raw_top_gap_loss -ge 0.023125343285737676) {
                    throw "Raw-gap update-20000 preregistered gate failed"
                }
                if ([double]$final.validation.effective_listnet_weight -ne 0.0 -or
                    [double]$final.validation.effective_raw_top_gap_weight -ne 1.0) {
                    throw "Raw-gap final held objective weights changed"
                }
                $coverageMinima = @(1.0, 0.999, 0.999)
                for ($i = 0; $i -lt 3; $i += 1) {
                    if ([double]$final.bank_coverage[$i].ever_updated_fraction -lt $coverageMinima[$i]) {
                        throw "Raw-gap k128 layer $($i + 1) bank coverage missed the frozen gate"
                    }
                }
                $materialGain = $true
            } else {
            $midpoint = @($metrics | Where-Object { [int]$_.update -eq 10000 })
            if ($midpoint.Count -ne 1) { throw "CPF1 k128 long run lacks update 10000" }
            if ([double]$midpoint[0].validation.top1_agreement -lt 0.11 -or
                [double]$midpoint[0].validation.ndcg -lt 0.88 -or
                [double]$midpoint[0].validation.pairwise_accuracy -lt 0.62 -or
                [double]$midpoint[0].validation.composite_loss -gt 5.50) {
                throw "CPF1 k128 update-10000 midpoint gate failed"
            }
            if ($finalTop1 -lt 0.15 -or $finalNdcg -lt 0.90 -or
                $finalPairwise -lt 0.65 -or $finalLoss -gt 5.25) {
                throw "CPF1 k128 update-20000 convergence gate failed"
            }
            if ([double]$final.validation.q_std -lt 0.15) { throw "CPF1 k128 held Q std collapsed" }
            if ([double]$final.validation.action_margin -lt 0.02) { throw "CPF1 k128 held action margin collapsed" }
            $tail = @($metrics | Select-Object -Last 3)
            for ($i = 1; $i -lt $tail.Count; $i += 1) {
                $priorLoss = [double]$tail[$i - 1].validation.composite_loss
                $rebound = ([double]$tail[$i].validation.composite_loss - $priorLoss) / [math]::Abs($priorLoss)
                if ($rebound -gt 0.03) { throw "CPF1 k128 final-three loss rebound exceeded the frozen gate: $rebound" }
            }
            $coverageMinima = @(1.0, 0.999, 0.999)
            for ($i = 0; $i -lt 3; $i += 1) {
                if ([double]$final.bank_coverage[$i].ever_updated_fraction -lt $coverageMinima[$i]) {
                    throw "CPF1 k128 layer $($i + 1) bank coverage missed the frozen gate"
                }
            }
            $materialGain = $true
            }
        } elseif ($Variant -eq "k128") {
            if ($finalTop1 -lt 0.048828125 -or $finalNdcg -lt 0.8434740281749898 -or
                $finalPairwise -lt 0.537651329761083 -or $finalLoss -gt 6.439771521836519) {
                throw "CPF1 k128 is inferior to the paired k64 update-2500 checkpoint"
            }
            $materialGain = $finalTop1 -ge 0.0546875 -or
                $finalNdcg -ge 0.8484740281749898 -or
                $finalPairwise -ge 0.547651329761083
        } else {
            if ($finalTop1 -lt 0.0703125 -or $finalNdcg -lt 0.8440925775521915 -or
                $finalPairwise -lt 0.5407089182563002 -or $finalLoss -gt 6.333081512246281) {
                throw "CPF1 k256 is inferior to the paired k128 update-2500 checkpoint"
            }
            $materialGain = $finalTop1 -ge 0.076171875 -or
                $finalNdcg -ge 0.8490925775521915 -or
                $finalPairwise -ge 0.5507089182563002
        }
        if (-not $materialGain) { throw "CPF1 $Variant reached no frozen material-gain margin" }
    }
    if ($hardNegativePivot) {
        if ([int64]$final.validation.hard_negative_valid_selections -le 0) {
            throw "Hard-negative final held evaluation contains no valid selections"
        }
        if ([int64]$final.validation.hard_negative_differs_from_teacher_top2 -le 0) {
            throw "Hard-negative final gate observed no selection differing from teacher top-2"
        }
    }
    $prefilterActivations = [int64]$final.throughput.prefilter_activation_routes
    $prefilterDrops = [int64](($final.throughput.prefilter_dropped_rows | Measure-Object -Sum).Sum)
    if ($prefilterActivations -le 0 -or $prefilterDrops -le 0) {
        throw "CPF1 never exercised the overflow prefilter"
    }
    $stderrText = Get-Content -LiteralPath $stderrPath -Raw
    if ($stderrText -match '(?im)^ERROR:|Stacktrace:|bucket-entry cap .* exceeded|exact-score cap .* exceeded') {
        throw "CPF1 stderr contains a forbidden failure record"
    }

    $latestFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "latest.json")
    if ($latestFiles.Count -ne 1) {
        throw "CPF1 run must publish exactly one latest.json; got $($latestFiles.Count)"
    }
    $latestPath = Assert-RunLeaf -Path $latestFiles[0].FullName -Root $resolvedOutput
    $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json -ErrorAction Stop
    if ([int]$latest.update -ne $MaximumUpdates -or [string]$latest.variant -ne $Variant) {
        throw "CPF1 latest checkpoint identity is not update $MaximumUpdates $Variant"
    }
    $checkpointPath = Assert-RunLeaf -Path ([string]$latest.path) -Root $resolvedOutput
    $checkpoint = Get-Item -LiteralPath $checkpointPath
    if ([int64]$latest.bytes -ne [int64]$checkpoint.Length) {
        throw "CPF1 latest checkpoint byte count differs from disk"
    }
    $checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$latest.sha256 -cne $checkpointSha256) {
        throw "CPF1 latest checkpoint SHA-256 differs from disk"
    }
    $metricsSha256 = (Get-FileHash -LiteralPath $metricsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $gatePath = Join-Path $resolvedOutput "gate_status.json"
    $gatePendingPath = Join-Path $resolvedOutput "gate_status.pending.json"
    [ordered]@{
        status = "PASS"
        final_update = $MaximumUpdates
        variant = $Variant
        objective_profile = $ObjectiveProfile
        objective_margin_weight = $expectedMarginWeight
        objective_margin_mode = $expectedMarginMode
        objective_mode = $expectedObjectiveMode
        effective_listnet_weight = [double]$final.effective_listnet_weight
        effective_margin_weight = [double]$final.effective_margin_weight
        effective_raw_top_gap_weight = [double]$final.effective_raw_top_gap_weight
        hard_negative_valid_selections = [int64]$final.validation.hard_negative_valid_selections
        hard_negative_differs_from_teacher_top2 = [int64]$final.validation.hard_negative_differs_from_teacher_top2
        routing_policy = [string]$final.routing_policy
        held_ndcg_delta = $ndcgDelta
        held_pairwise_delta = $pairwiseDelta
        held_top1_delta = $top1Delta
        held_composite_loss_fraction = $lossFraction
        prefilter_activation_routes = $prefilterActivations
        prefilter_dropped_rows = $prefilterDrops
        maximum_retrieved_rows = @($final.throughput.maximum_retrieved_rows)
        metrics_path = $metricsPath
        metrics_sha256 = $metricsSha256
        checkpoint_path = $checkpointPath
        checkpoint_bytes = [int64]$checkpoint.Length
        checkpoint_sha256 = $checkpointSha256
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        comparison_baseline_metrics_path = $comparisonBaselinePath
        comparison_baseline_metrics_sha256 = $comparisonBaselineSha256
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gatePendingPath -Encoding utf8NoBOM
    $gateSha256 = (Get-FileHash -LiteralPath $gatePendingPath -Algorithm SHA256).Hash.ToLowerInvariant()
    # Publish the authoritative gate before any completed status can reference
    # it.  This generic ordering applies to every future CPF profile.
    Move-Item -LiteralPath $gatePendingPath -Destination $gatePath
    [ordered]@{
        status = "completed"; pid = $ownedJulia.Id; completed_at = [DateTimeOffset]::Now.ToString("o")
        variant = $Variant; maximum_updates = $MaximumUpdates
        objective_profile = $ObjectiveProfile
        objective_margin_weight = $expectedMarginWeight
        objective_margin_mode = $expectedMarginMode
        objective_mode = $expectedObjectiveMode
        exit_code = $ownedJulia.ExitCode; contract = $contractPath; contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256; output_root = $resolvedOutput
        gate_status = $gatePath; gate_status_sha256 = $gateSha256
        comparison_baseline_metrics_path = $comparisonBaselinePath
        comparison_baseline_metrics_sha256 = $comparisonBaselineSha256
        one_shot_nonce = if ($rawGapPivot) { $RawGapOneShotNonce } else { "" }
        one_shot_consumed_path = if ($rawGapPivot) { $frozenConsumed } else { "" }
        one_shot_consumed_sha256 = if ($rawGapPivot) { $rawGapConsumedSha256 } else { "" }
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM
} catch {
    if (Test-Path -LiteralPath $resolvedOutput) {
        [ordered]@{
            status="failed"; failed_at=[DateTimeOffset]::Now.ToString("o")
            error=$_.Exception.Message; output_root=$resolvedOutput
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedOutput "controller_status.json") -Encoding utf8NoBOM
    }
    throw
} finally {
    if ($held) { try { $juliaMutex.ReleaseMutex() } catch {} }
    $juliaMutex.Dispose()
}
