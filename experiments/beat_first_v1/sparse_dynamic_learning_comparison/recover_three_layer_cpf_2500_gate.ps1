[CmdletBinding()]
param(
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_teacher_signal_cpf_2500_v1",
    [string]$DatasetManifest = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3\manifest.json",
    [string]$Contract = (Join-Path $PSScriptRoot "three_layer_cpf_2500_update_contract.toml")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$expectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
$expectedRoutingPolicy = "wta-collision-count-stable-id-prefilter-v1"

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

$liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
if ($liveJulia.Count -ne 0) {
    throw "Recovery gate refuses to run while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
}
$root = [IO.Path]::GetFullPath($OutputRoot)
foreach ($required in @($root, $DatasetManifest, $Contract)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required recovery input is missing: $required" }
}
$manifestSha256 = (Get-FileHash -LiteralPath $DatasetManifest -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestSha256 -ne $expectedManifestSha256) { throw "Dataset manifest SHA-256 changed" }
$contractSha256 = (Get-FileHash -LiteralPath $Contract -Algorithm SHA256).Hash.ToLowerInvariant()
$controllerPath = Assert-RunLeaf -Path (Join-Path $root "controller_status.json") -Root $root
$controllerSha256 = (Get-FileHash -LiteralPath $controllerPath -Algorithm SHA256).Hash.ToLowerInvariant()
$controller = Get-Content -LiteralPath $controllerPath -Raw | ConvertFrom-Json -ErrorAction Stop
if ([string]$controller.status -ne "failed" -or [string]$controller.error -notmatch "property 'Parent' cannot be found") {
    throw "Recovery gate is authorized only for the observed post-gate FileInfo.Parent failure"
}

$metricFiles = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "metrics.jsonl")
if ($metricFiles.Count -ne 1) { throw "Expected exactly one metrics.jsonl; got $($metricFiles.Count)" }
$metricsPath = Assert-RunLeaf -Path $metricFiles[0].FullName -Root $root
$lines = @(Get-Content -LiteralPath $metricsPath | Where-Object { $_.Trim().Length -gt 0 })
if ($lines.Count -ne 6) { throw "Metrics must contain exactly six evaluation records" }
$metrics = @($lines | ForEach-Object { $_ | ConvertFrom-Json -ErrorAction Stop })
$expectedUpdates = @(0, 500, 1000, 1500, 2000, 2500)
for ($i = 0; $i -lt $expectedUpdates.Count; $i += 1) {
    if ([int]$metrics[$i].update -ne $expectedUpdates[$i]) { throw "Metric update sequence changed at $i" }
    Assert-FiniteJsonValue -Value $metrics[$i] -Path "metrics[$i]"
}
$baseline = $metrics[0]
$final = $metrics[-1]
if ([string]$final.variant -ne "k64") { throw "Final variant differs from k64" }
if ([string]$final.routing_policy -ne $expectedRoutingPolicy) { throw "Final routing policy changed" }
if ([int]$final.throughput.updates -ne 2500) { throw "Run did not reach update 2500" }
if ([int]$final.parameter_contract.total_parameters -ne 19924022) { throw "Parameter count changed" }
if ([string]$final.parameter_contract.routing_policy -ne $expectedRoutingPolicy) {
    throw "Parameter contract routing policy changed"
}
$ndcgDelta = [double]$final.validation.ndcg - [double]$baseline.validation.ndcg
$pairwiseDelta = [double]$final.validation.pairwise_accuracy - [double]$baseline.validation.pairwise_accuracy
$top1Delta = [double]$final.validation.top1_agreement - [double]$baseline.validation.top1_agreement
$lossFraction = [double]$final.validation.composite_loss / [double]$baseline.validation.composite_loss
if ($ndcgDelta -lt 0.015) { throw "Held NDCG gate failed: $ndcgDelta" }
if ($pairwiseDelta -lt 0.015) { throw "Held pairwise gate failed: $pairwiseDelta" }
if ($top1Delta -lt 0.0) { throw "Held top-1 gate failed: $top1Delta" }
if ($lossFraction -gt 0.95) { throw "Held composite-loss gate failed: $lossFraction" }
$prefilterActivations = [int64]$final.throughput.prefilter_activation_routes
$prefilterDropsByLayer = @($final.throughput.prefilter_dropped_rows | ForEach-Object { [int64]$_ })
$prefilterDrops = [int64](($prefilterDropsByLayer | Measure-Object -Sum).Sum)
if ($prefilterActivations -le 0 -or $prefilterDrops -le 0) { throw "Overflow prefilter was never exercised" }

$stderrPath = Assert-RunLeaf -Path (Join-Path $root "controller.stderr.log") -Root $root
$stderrText = Get-Content -LiteralPath $stderrPath -Raw
if ($stderrText -match '(?im)^ERROR:|Stacktrace:|bucket-entry cap .* exceeded|exact-score cap .* exceeded') {
    throw "Trainer stderr contains a forbidden failure record"
}
$latestFiles = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "latest.json")
if ($latestFiles.Count -ne 1) { throw "Expected exactly one latest.json; got $($latestFiles.Count)" }
$latestPath = Assert-RunLeaf -Path $latestFiles[0].FullName -Root $root
$latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json -ErrorAction Stop
if ([int]$latest.update -ne 2500 -or [string]$latest.variant -ne "k64") {
    throw "Latest checkpoint is not update 2500 k64"
}
$checkpointPath = Assert-RunLeaf -Path ([string]$latest.path) -Root $root
$checkpoint = Get-Item -LiteralPath $checkpointPath
if ([int64]$latest.bytes -ne [int64]$checkpoint.Length) { throw "Checkpoint byte count changed" }
$checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ([string]$latest.sha256 -cne $checkpointSha256) { throw "Checkpoint SHA-256 changed" }
$metricsSha256 = (Get-FileHash -LiteralPath $metricsPath -Algorithm SHA256).Hash.ToLowerInvariant()
$resultPath = Join-Path $root "post_gate_recovery_status.json"
if (Test-Path -LiteralPath $resultPath) { throw "Recovery result already exists: $resultPath" }
[ordered]@{
    status = "PASS_RECOVERED_POST_GATE"
    recovered_controller_status_sha256 = $controllerSha256
    recovery_reason = "launcher FileInfo.Parent post-gate bug; trainer process had already exited zero"
    final_update = 2500
    routing_policy = $expectedRoutingPolicy
    held_ndcg_delta = $ndcgDelta
    held_pairwise_delta = $pairwiseDelta
    held_top1_delta = $top1Delta
    held_composite_loss_fraction = $lossFraction
    prefilter_activation_routes = $prefilterActivations
    prefilter_dropped_rows_by_layer = $prefilterDropsByLayer
    maximum_retrieved_rows = @($final.throughput.maximum_retrieved_rows)
    metrics_path = $metricsPath
    metrics_sha256 = $metricsSha256
    checkpoint_path = $checkpointPath
    checkpoint_bytes = [int64]$checkpoint.Length
    checkpoint_sha256 = $checkpointSha256
    contract_path = [IO.Path]::GetFullPath($Contract)
    contract_sha256 = $contractSha256
    dataset_manifest_path = [IO.Path]::GetFullPath($DatasetManifest)
    dataset_manifest_sha256 = $manifestSha256
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding utf8NoBOM
Get-Content -LiteralPath $resultPath -Raw
