[CmdletBinding()]
param(
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_hardneg_teacher_signal_cpf_20000_v1",
    [string]$DatasetManifest = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3\manifest.json",
    [string]$Contract = (Join-Path $PSScriptRoot "three_layer_cpf_k128_hardneg_20000_update_contract.toml"),
    [string]$ComparisonBaselineMetrics = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1_retry1\sparse_3l_20260719T143553\metrics.jsonl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
$expectedContractSha256 = "2f48a0e40fed65dbb627cb73ff19c5df91484fb6bc8f7280ae8ff980b49589bc"
$expectedParentSha256 = "2a5149ff47a4d16053679162b71c2ea589efdd3c5a5d5a80814eb0fdf474d597"
$expectedRoutingPolicy = "wta-collision-count-stable-id-prefilter-v1"
$expectedMarginMode = "student_hard_negative"
$requestedMarginWeight = 0.15
$expectedEffectiveMarginWeight = [double][single]$requestedMarginWeight
$expectedUpdates = @(0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)

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
    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $prefix = $rootPath + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Run artifact escapes the output root: $resolved"
    }
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Run artifact is not a file: $resolved"
    }
    $item = Get-Item -LiteralPath $resolved -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Run artifact is a reparse point: $resolved"
    }
    $cursor = $item.Directory
    while ($null -ne $cursor) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Run artifact traverses a reparse point: $($cursor.FullName)"
        }
        if ($cursor.FullName.Equals($rootPath, [StringComparison]::OrdinalIgnoreCase)) {
            return $resolved
        }
        if (-not $cursor.FullName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Run artifact ancestry escapes the output root: $resolved"
        }
        $cursor = $cursor.Parent
    }
    throw "Run artifact ancestry never reaches the output root: $resolved"
}

function Get-ExactSha256 {
    param([string]$Path, [string]$Expected, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $observed = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -cne $Expected) {
        throw "$Label SHA-256 changed: $observed"
    }
    return $observed
}

$juliaMutex = [System.Threading.Mutex]::new($false, 'Local\TetrisBeatFirstV1ExclusiveJulia')
$held = $false
try {
    try { $held = $juliaMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $held = $true }
    if (-not $held) { throw "Recovery gate refuses to overlap the active Tetris Julia controller" }
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Recovery gate refuses to run while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
    }

    $root = [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "Hard-negative output root is missing: $root"
    }
    $rootItem = Get-Item -LiteralPath $root -Force
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Hard-negative output root is a reparse point: $root"
    }

    $manifestSha256 = Get-ExactSha256 -Path $DatasetManifest -Expected $expectedManifestSha256 -Label "Dataset manifest"
    $contractSha256 = Get-ExactSha256 -Path $Contract -Expected $expectedContractSha256 -Label "Hard-negative contract"
    $parentSha256 = Get-ExactSha256 -Path $ComparisonBaselineMetrics -Expected $expectedParentSha256 -Label "Pinned parent metrics"

    $resultPath = Join-Path $root "hardneg_post_gate_recovery_status.json"
    $pendingPath = Join-Path $root "hardneg_post_gate_recovery_status.pending.json"
    foreach ($forbidden in @($resultPath, $pendingPath, (Join-Path $root "gate_status.json"), (Join-Path $root "gate_status.pending.json"))) {
        if (Test-Path -LiteralPath $forbidden) {
            throw "Recovery requires a fresh post-gate output slot; found: $forbidden"
        }
    }

    $controllerPath = Assert-RunLeaf -Path (Join-Path $root "controller_status.json") -Root $root
    $controllerSha256 = (Get-FileHash -LiteralPath $controllerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $controller = Get-Content -LiteralPath $controllerPath -Raw | ConvertFrom-Json -ErrorAction Stop
    if ([string]$controller.status -cne "failed") {
        throw "Recovery is authorized only after the controller records failure"
    }
    if ([string]$controller.error -notmatch '^CPF1 metric objective margin weight changed at index [0-9]+$') {
        throw "Recovery is authorized only for the known Float32 objective-weight post-gate failure"
    }
    if ([IO.Path]::GetFullPath([string]$controller.output_root).TrimEnd('\') -cne $root) {
        throw "Failed controller output root differs from the requested recovery root"
    }
    $controllerFailedAt = [DateTimeOffset]::Parse([string]$controller.failed_at)

    $stdoutPath = Assert-RunLeaf -Path (Join-Path $root "controller.stdout.log") -Root $root
    $stderrPath = Assert-RunLeaf -Path (Join-Path $root "controller.stderr.log") -Root $root
    $stdoutSha256 = (Get-FileHash -LiteralPath $stdoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $stderrSha256 = (Get-FileHash -LiteralPath $stderrPath -Algorithm SHA256).Hash.ToLowerInvariant()

    $metricFiles = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "metrics.jsonl")
    if ($metricFiles.Count -ne 1) { throw "Expected exactly one metrics.jsonl; got $($metricFiles.Count)" }
    $metricsPath = Assert-RunLeaf -Path $metricFiles[0].FullName -Root $root
    $runDirectory = Split-Path -Parent $metricsPath
    if ([IO.Path]::GetFullPath((Split-Path -Parent $runDirectory)).TrimEnd('\') -cne $root) {
        throw "Metrics run directory is not a direct fresh child of the output root"
    }
    $runDirectories = @(Get-ChildItem -LiteralPath $root -Directory -Force)
    if ($runDirectories.Count -ne 1 -or
        [IO.Path]::GetFullPath($runDirectories[0].FullName).TrimEnd('\') -cne $runDirectory) {
        throw "Fresh output root must contain exactly the one metrics-bearing run directory"
    }
    if (($runDirectories[0].Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Fresh run directory is a reparse point"
    }
    $runLeaf = Split-Path -Leaf $runDirectory
    if ($runLeaf -notmatch '^sparse_3l_[0-9]{8}T[0-9]{6}$') {
        throw "Run directory does not carry the trainer's fresh-run identity: $runLeaf"
    }
    $runTimestamp = [DateTime]::ParseExact(
        $runLeaf.Substring(10),
        'yyyyMMddTHHmmss',
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::AssumeLocal
    )
    if ([DateTimeOffset]$runTimestamp -gt $controllerFailedAt) {
        throw "Fresh run identity timestamp is later than the controller failure"
    }

    $metricLines = @(Get-Content -LiteralPath $metricsPath | Where-Object { $_.Trim().Length -gt 0 })
    if ($metricLines.Count -ne $expectedUpdates.Count) {
        throw "Hard-negative metrics must contain exactly $($expectedUpdates.Count) evaluation records"
    }
    $metrics = @($metricLines | ForEach-Object { $_ | ConvertFrom-Json -ErrorAction Stop })
    for ($i = 0; $i -lt $expectedUpdates.Count; $i += 1) {
        $metric = $metrics[$i]
        if ([int]$metric.update -ne $expectedUpdates[$i]) {
            throw "Hard-negative metric update sequence changed at index $i"
        }
        if (-not ($metric.PSObject.Properties.Name -contains "objective_margin_mode") -or
            [string]$metric.objective_margin_mode -cne $expectedMarginMode) {
            throw "Hard-negative metric objective mode changed at index $i"
        }
        if (-not ($metric.PSObject.Properties.Name -contains "objective_margin_weight") -or
            [double]$metric.objective_margin_weight -ne $expectedEffectiveMarginWeight) {
            throw "Hard-negative metric effective objective weight changed at index $i"
        }
        Assert-FiniteJsonValue -Value $metric -Path "metrics[$i]"
    }

    $baseline = $metrics[0]
    $midpointRecords = @($metrics | Where-Object { [int]$_.update -eq 10000 })
    if ($midpointRecords.Count -ne 1) { throw "Hard-negative run lacks its unique update-10000 midpoint" }
    $midpoint = $midpointRecords[0]
    $final = $metrics[-1]

    if ([string]$final.variant -cne "k128") { throw "Final variant differs from k128" }
    if ([string]$final.routing_policy -cne $expectedRoutingPolicy) { throw "Final routing policy changed" }
    if ([int]$final.throughput.updates -ne 20000) { throw "Run did not authoritatively reach update 20000" }
    if ([int]$final.parameter_contract.total_parameters -ne 19924022) { throw "Total parameter count changed" }
    if ([int]$final.parameter_contract.active_parameters -ne 58214) { throw "Active parameter count changed" }
    if ((@($final.parameter_contract.active_counts) -join ',') -cne '48,40,40') { throw "Active widths changed" }
    if ([string]$final.parameter_contract.routing_policy -cne $expectedRoutingPolicy) {
        throw "Parameter-contract routing policy changed"
    }

    if ([double]$midpoint.validation.top1_agreement -lt 0.11 -or
        [double]$midpoint.validation.ndcg -lt 0.88 -or
        [double]$midpoint.validation.pairwise_accuracy -lt 0.62 -or
        [double]$midpoint.validation.composite_loss -gt 5.50) {
        throw "Hard-negative update-10000 strength gate failed"
    }
    if ([double]$final.validation.top1_agreement -lt 0.15 -or
        [double]$final.validation.ndcg -lt 0.90 -or
        [double]$final.validation.pairwise_accuracy -lt 0.65 -or
        [double]$final.validation.composite_loss -gt 5.25) {
        throw "Hard-negative update-20000 strength gate failed"
    }
    if ([double]$final.validation.q_std -lt 0.15) { throw "Final held Q std collapsed" }
    if ([double]$final.validation.action_margin -lt 0.02) { throw "Final held action margin collapsed" }

    $tail = @($metrics | Select-Object -Last 3)
    for ($i = 1; $i -lt $tail.Count; $i += 1) {
        $priorLoss = [double]$tail[$i - 1].validation.composite_loss
        $rebound = ([double]$tail[$i].validation.composite_loss - $priorLoss) / [math]::Abs($priorLoss)
        if ($rebound -gt 0.03) { throw "Final-three held loss rebound exceeded 3%: $rebound" }
    }
    $coverageMinima = @(1.0, 0.999, 0.999)
    for ($i = 0; $i -lt 3; $i += 1) {
        if ([double]$final.bank_coverage[$i].ever_updated_fraction -lt $coverageMinima[$i]) {
            throw "Layer $($i + 1) bank coverage missed its frozen gate"
        }
    }
    $hardNegativeValid = [int64]$final.validation.hard_negative_valid_selections
    $hardNegativeDifferent = [int64]$final.validation.hard_negative_differs_from_teacher_top2
    if ($hardNegativeValid -le 0) { throw "Final held evaluation contains no valid hard-negative selections" }
    if ($hardNegativeDifferent -le 0) { throw "Final held evaluation contains no hard negative differing from teacher top-2" }

    $prefilterActivations = [int64]$final.throughput.prefilter_activation_routes
    $prefilterDropsByLayer = @($final.throughput.prefilter_dropped_rows | ForEach-Object { [int64]$_ })
    $prefilterDrops = [int64](($prefilterDropsByLayer | Measure-Object -Sum).Sum)
    if ($prefilterActivations -le 0 -or $prefilterDrops -le 0) {
        throw "Overflow prefilter was never exercised"
    }

    $stderrText = Get-Content -LiteralPath $stderrPath -Raw
    if ($stderrText -match '(?im)^ERROR:|Stacktrace:|bucket-entry cap .* exceeded|exact-score cap .* exceeded') {
        throw "Trainer stderr contains a forbidden failure record"
    }

    $latestFiles = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "latest.json")
    if ($latestFiles.Count -ne 1) { throw "Expected exactly one latest.json; got $($latestFiles.Count)" }
    $latestPath = Assert-RunLeaf -Path $latestFiles[0].FullName -Root $root
    if ((Split-Path -Parent $latestPath) -cne $runDirectory) {
        throw "latest.json and metrics.jsonl do not belong to the same fresh run directory"
    }
    $latestSha256 = (Get-FileHash -LiteralPath $latestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json -ErrorAction Stop
    if ([int]$latest.update -ne 20000 -or [string]$latest.variant -cne "k128") {
        throw "Latest checkpoint is not update 20000 k128"
    }
    $checkpointPath = Assert-RunLeaf -Path ([string]$latest.path) -Root $root
    $expectedCheckpointDirectory = Join-Path $runDirectory "checkpoints"
    if ([IO.Path]::GetFullPath((Split-Path -Parent $checkpointPath)).TrimEnd('\') -cne
        [IO.Path]::GetFullPath($expectedCheckpointDirectory).TrimEnd('\')) {
        throw "Latest checkpoint is not inside the fresh run's checkpoints directory"
    }
    $checkpoint = Get-Item -LiteralPath $checkpointPath
    if ([int64]$latest.bytes -ne [int64]$checkpoint.Length) { throw "Checkpoint byte count differs from disk" }
    $checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$latest.sha256 -cne $checkpointSha256) { throw "Checkpoint SHA-256 differs from disk" }

    $metricsSha256 = (Get-FileHash -LiteralPath $metricsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-FileHash -LiteralPath $controllerPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $controllerSha256) {
        throw "Failed controller status changed during recovery validation"
    }
    Get-ExactSha256 -Path $DatasetManifest -Expected $expectedManifestSha256 -Label "Dataset manifest" | Out-Null
    Get-ExactSha256 -Path $Contract -Expected $expectedContractSha256 -Label "Hard-negative contract" | Out-Null
    Get-ExactSha256 -Path $ComparisonBaselineMetrics -Expected $expectedParentSha256 -Label "Pinned parent metrics" | Out-Null
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Julia appeared during recovery validation: $(($liveJulia.Id) -join ',')"
    }

    $invokedAt = [DateTimeOffset]::Now
    $scriptPath = [IO.Path]::GetFullPath($MyInvocation.MyCommand.Path)
    $scriptSha256 = (Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $ndcgDelta = [double]$final.validation.ndcg - [double]$baseline.validation.ndcg
    $pairwiseDelta = [double]$final.validation.pairwise_accuracy - [double]$baseline.validation.pairwise_accuracy
    $top1Delta = [double]$final.validation.top1_agreement - [double]$baseline.validation.top1_agreement
    $lossFraction = [double]$final.validation.composite_loss / [double]$baseline.validation.composite_loss
    [ordered]@{
        status = "PASS_RECOVERED_FLOAT32_WEIGHT_POST_GATE"
        recovery_invocation_id = [guid]::NewGuid().ToString("D")
        recovery_invoked_at = $invokedAt.ToString("o")
        recovery_script_path = $scriptPath
        recovery_script_sha256 = $scriptSha256
        recovery_reason = "trainer exited zero; original post-gate compared Float64(Float32(0.15)) against decimal Float64(0.15)"
        recovered_controller_status_path = $controllerPath
        recovered_controller_status_sha256 = $controllerSha256
        recovered_controller_failed_at = $controllerFailedAt.ToString("o")
        recovered_controller_error = [string]$controller.error
        fresh_run_directory = $runDirectory
        final_update = 20000
        variant = "k128"
        objective_profile = "hardneg015"
        requested_objective_margin_weight = $requestedMarginWeight
        effective_objective_margin_weight = $expectedEffectiveMarginWeight
        objective_margin_mode = $expectedMarginMode
        hard_negative_valid_selections = $hardNegativeValid
        hard_negative_differs_from_teacher_top2 = $hardNegativeDifferent
        routing_policy = $expectedRoutingPolicy
        held_ndcg_delta = $ndcgDelta
        held_pairwise_delta = $pairwiseDelta
        held_top1_delta = $top1Delta
        held_composite_loss_fraction = $lossFraction
        prefilter_activation_routes = $prefilterActivations
        prefilter_dropped_rows_by_layer = $prefilterDropsByLayer
        maximum_retrieved_rows = @($final.throughput.maximum_retrieved_rows)
        controller_stdout_path = $stdoutPath
        controller_stdout_sha256 = $stdoutSha256
        controller_stderr_path = $stderrPath
        controller_stderr_sha256 = $stderrSha256
        metrics_path = $metricsPath
        metrics_sha256 = $metricsSha256
        latest_path = $latestPath
        latest_sha256 = $latestSha256
        checkpoint_path = $checkpointPath
        checkpoint_bytes = [int64]$checkpoint.Length
        checkpoint_sha256 = $checkpointSha256
        contract_path = [IO.Path]::GetFullPath($Contract)
        contract_sha256 = $contractSha256
        dataset_manifest_path = [IO.Path]::GetFullPath($DatasetManifest)
        dataset_manifest_sha256 = $manifestSha256
        comparison_baseline_metrics_path = [IO.Path]::GetFullPath($ComparisonBaselineMetrics)
        comparison_baseline_metrics_sha256 = $parentSha256
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $pendingPath -Encoding utf8NoBOM
    Move-Item -LiteralPath $pendingPath -Destination $resultPath
    Get-Content -LiteralPath $resultPath -Raw
} finally {
    if ($held) { try { $juliaMutex.ReleaseMutex() } catch {} }
    $juliaMutex.Dispose()
}
