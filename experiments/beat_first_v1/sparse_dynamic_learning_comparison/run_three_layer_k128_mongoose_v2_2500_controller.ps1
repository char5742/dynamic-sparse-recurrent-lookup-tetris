[CmdletBinding()]
param(
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# One fresh, bounded MONGOOSE-v2 fixed-k signal only.  This controller exposes
# no path, seed, objective, width, cadence, retry, resume, or routing switch.
# It deliberately fails before any filesystem mutation while the v2 runtime
# source freeze below is unresolved.
$ExperimentId = "sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1"
$RoutingMode = "mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2"
$RoutingPolicy = "mongoose-simhash-k7-l2-bounded-lanes16-fixed-k128-live-pending-bce-v2"
$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\$ExperimentId"
$RunParent = "D:\tetris-paper-plus\runs\beat_first_v1"
$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$MaximumUpdates = 2500
$Variant = "k128"
$ObjectiveMarginWeight = 0.15
$ObjectiveMarginMode = "fixed_teacher_top2"
$ObjectiveMode = "standardized_listnet_plus_margin"
$EvaluationUpdates = @(0, 2000, 2500)
$CheckpointUpdates = @(2000, 2500)
$ActiveCounts = @(48, 40, 40)
$TrainingProbes = @(6, 5, 5)
$GlobalVisitCaps = @(1536, 1280, 1280)
$TableVisitCaps = @(1536, 1280, 1280)
$LaneVisitCaps = @(96, 80, 80)
$ScoreCaps = @(384, 640, 640)
$ShortlistCaps = @(378, 635, 635)
$FillProbeCaps = @(42, 35, 35)
$TrainingProbeAttemptCaps = @(48, 40, 40)
$FocusedAndRegressionTestsPassed = 109444
$Tables = 2
$LanesPerTable = 16
$LaneSlots = 32
$Unresolved = "UNRESOLVED_AFTER_V2_IMPLEMENTATION_AND_TEST_FREEZE"

$comparisonRoot = $PSScriptRoot
$experimentRoot = Split-Path -Parent $comparisonRoot
$contractPath = Join-Path $comparisonRoot "three_layer_cpf_k128_mongoose_v2_2500_update_contract.toml"
$entrypoint = Join-Path $experimentRoot "sparse_dynamic_3layer\train_teacher_supervised.jl"
$manifestPath = Join-Path $Dataset "manifest.json"
$reservationPath = Join-Path $RunParent ".$ExperimentId.reservation.json"
$consumedPath = Join-Path $RunParent ".$ExperimentId.consumed.json"
$SmokeRoot = "C:\tmp\tetris_mongoose_v2_real_teacher_smoke_20260719T223000"
$SmokeSummaryPath = Join-Path $SmokeRoot "summary.json"
$SmokeCheckpointPath = Join-Path $SmokeRoot "checkpoint_000000002.jls"

# Filled only after the exact implementation, tests, real two-update smoke,
# and clean prelaunch review are frozen.  Do not substitute hashes observed
# while the implementation is still moving.
$ExpectedContractBytes = 15421
$ExpectedContractSha256 = "6e8402bbea3dd981cc2a7bcf503c301b11e5b8a4b4927253d08301285252a450"
$PinnedImplementationFiles = @(
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\teacher_training.jl"); Bytes=140468; Sha256="e1cc8d21a8398922222aa999430a2c4fc38ee713ba9d14841c2f598f04ffb641"; Label="teacher training source" },
    [pscustomobject]@{ Path=$entrypoint; Bytes=149; Sha256="41893978e984c0c696470aa15d73ae54eae01deba05e0db776a9269e25897b5d"; Label="teacher entrypoint" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\SparseDynamic3Layer.jl"); Bytes=3187; Sha256="c8d44a3f8537b44e1c129e1c2c83b39c317bdb33fb8c38dbf5e0b9b97e796cc1"; Label="three-layer module" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\dynamic_k64_k256.jl"); Bytes=28207; Sha256="2ff6d17c59cc616caa90878ac3ff451fbbf8f02d61440ab49d8d1e3f6f1a5d97"; Label="included but forbidden dynamic-k source" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\mongoose_simhash_overlay.jl"); Bytes=92399; Sha256="abe5ad0e1d63b9306d668257f77a7e8c486c0e4a21309743b8961e4b6c2ea4fb"; Label="bounded MONGOOSE-v2 overlay" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\geometry.jl"); Bytes=12006; Sha256="5ec92d85be3f888554c1f2700e06a1e3dc5531d25b354677bb208eb6978d3a27"; Label="three-layer geometry" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\model.jl"); Bytes=23135; Sha256="ff94f5771d1cbb09532348b8603019c5aa3126979dfef74ac277d223a749e7ac"; Label="three-layer model" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\optimizer.jl"); Bytes=20897; Sha256="a5c057e015e3b9677510faaf07e3c26ca993233e09ab9f602296fbd3299b5790"; Label="sparse optimizer" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\runtime.jl"); Bytes=48536; Sha256="bea7e0573f71fd491f8c246a3bd59cb97c9da68483a01bac536551f53d9f2bc3"; Label="three-layer runtime" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\checkpoint.jl"); Bytes=7449; Sha256="9c8da9aece786396ac304b1ab8aa58e8c408316838effb48c4d12c25a618ccbf"; Label="checkpoint source" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic\features.jl"); Bytes=9360; Sha256="0a96497f69271bd900054be72c33c13a244b7909631b75dc95183730872197b7"; Label="feature adapter" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic\wta_index.jl"); Bytes=24011; Sha256="be270e8e36fe356be496e834edac1b3c6a13416fcd517a3e651319eb5b581ceb"; Label="fixed-WTA witness index" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "training\core.jl"); Bytes=51813; Sha256="5c5dd8e51bb8480424400f309d51de73e17c6b29e1552486c065aaf9b7135217"; Label="teacher loss core" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\test_mongoose_simhash_overlay.jl"); Bytes=13963; Sha256="acb11999f1e61bf4788ddfd0afe79a4a69f5b7eb53881a6f975387421ab10c8b"; Label="v1 overlay regression test source" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\test_mongoose_v2_bounded_core.jl"); Bytes=23151; Sha256="f51b7d6692be646cbf1f3dc544cf6c91bb4f7501abd466d9065b4e6b4955817e"; Label="v2 bounded-core test source" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\test_mongoose_v2_trainer_runtime_integration.jl"); Bytes=21353; Sha256="4d495bb2adabb9e1f5ae10177778c22f6c8808a2c59cab334b4210240e3f5cfa"; Label="v2 trainer/runtime integration test source" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\test_sparse_dynamic_3layer.jl"); Bytes=37881; Sha256="35594809d9dba6229435d3c2a99d932faee8cd108a1017680430d11a77dcb3a0"; Label="base runtime regression test source" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\smoke_mongoose_v2_real_teacher.jl"); Bytes=26098; Sha256="47731cce38326de37f7fa3bafe4c76e1def447096518c9a94a9d6b1489795bf3"; Label="real-teacher two-update smoke driver" }
)
$PinnedStableFiles = @(
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "Project.toml"); Bytes=1221; Sha256="62bbd9b624a4e2963e0382c86c8b4ff56b865c539e41ce1e8535cc7416ae4c7a"; Label="Project.toml" },
    [pscustomobject]@{ Path=(Join-Path $experimentRoot "Manifest.toml"); Bytes=47731; Sha256="6ef9ed11797722e72e62d1d4e500da59d0bc14124f96344f93a14a322ec9a10a"; Label="Manifest.toml" },
    [pscustomobject]@{ Path=$Julia; Bytes=170952; Sha256="4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043"; Label="Julia runtime" }
)
$ExpectedManifestBytes = 1268581
$ExpectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
$ExpectedSmokeSummaryBytes = 5018
$ExpectedSmokeSummarySha256 = "197aded6320d07c97c3bf330f4513a62bfe921a3e4077509da927d23b8a795bb"
$ExpectedSmokeSourceClosureSha256 = "752f66fe8738561471dde37a3507bf2e4eec459541dab8996fb5f0f220bb7a2a"
$ExpectedSmokeCheckpointBytes = 249870121
$ExpectedSmokeCheckpointSha256 = "8b15558fa06b8f6ae2572079badcbf280c8b77c9186ae99f332e9a88d43e03d9"
$ParentMetricsPath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_2500_v1\sparse_3l_20260719T134221\metrics.jsonl"
$ParentMetricsBytes = 36712
$ParentMetricsSha256 = "b6298c1ba512414782309e69e018f35a698f28821fe5a9681ec1acc127c9bff6"
$ParentCheckpointPath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_2500_v1\sparse_3l_20260719T134221\checkpoints\checkpoint_000002500.jls"
$ParentCheckpointBytes = 248673984
$ParentCheckpointSha256 = "de9aac395fc9406f2a3c77de4fa2408ada62716155d1d1915a1aaedb62670b85"

function Assert-FreezeResolved {
    $unresolvedBindings = [System.Collections.Generic.List[string]]::new()
    if ($ExpectedContractBytes -le 0) { $unresolvedBindings.Add("contract bytes") }
    if ($ExpectedContractSha256 -ceq $Unresolved -or
        $ExpectedContractSha256 -notmatch '^[0-9a-f]{64}$') {
        $unresolvedBindings.Add("contract SHA-256")
    }
    foreach ($pin in $PinnedImplementationFiles) {
        if ([int64]$pin.Bytes -le 0) { $unresolvedBindings.Add("$($pin.Label) bytes") }
        if ([string]$pin.Sha256 -ceq $Unresolved -or
            [string]$pin.Sha256 -notmatch '^[0-9a-f]{64}$') {
            $unresolvedBindings.Add("$($pin.Label) SHA-256")
        }
    }
    if ($unresolvedBindings.Count -ne 0) {
        throw "MONGOOSE-v2 source freeze is unresolved; no validation or launch is allowed: $($unresolvedBindings -join ', ')"
    }
}

function Assert-PinnedFile {
    param(
        [string]$Path,
        [int64]$ExpectedBytes,
        [string]$ExpectedSha256,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Pinned $Label is missing: $Path"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Pinned $Label is a reparse point: $Path"
    }
    if ([int64]$item.Length -ne $ExpectedBytes) {
        throw "Pinned $Label byte count changed: $($item.Length) != $ExpectedBytes"
    }
    $observed = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -cne $ExpectedSha256) {
        throw "Pinned $Label SHA-256 changed: $observed"
    }
    return $item.FullName
}

function Assert-ExactIntArray {
    param($Value, [int[]]$Expected, [string]$Label)
    $actual = @($Value)
    if ($actual.Count -ne $Expected.Count) {
        throw "$Label length changed: $($actual.Count) != $($Expected.Count)"
    }
    for ($index = 0; $index -lt $Expected.Count; $index += 1) {
        $item = $actual[$index]
        $isExactIntegerType =
            $item -is [sbyte] -or $item -is [byte] -or
            $item -is [int16] -or $item -is [uint16] -or
            $item -is [int32] -or $item -is [uint32] -or
            $item -is [int64] -or $item -is [uint64]
        if (-not $isExactIntegerType -or [decimal]$item -ne [decimal]$Expected[$index]) {
            $typeName = if ($null -eq $item) { "null" } else { $item.GetType().FullName }
            throw "$Label is not the exact integer vector at index $index`: value='$item' type='$typeName' expected=$($Expected[$index])"
        }
    }
}

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
        throw "Run artifact escapes output root: $resolved"
    }
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Run artifact is missing: $resolved"
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
            break
        }
        $cursor = $cursor.Parent
    }
    throw "Run artifact ancestry did not terminate at output root: $resolved"
}

function Assert-V2AccountingTelemetry {
    param($Accounting, [int]$Update, [int[]]$MaxLaneEntries)
    $required = @(
        "bucket_entries", "scored_rows", "retrieved_rows",
        "prefilter_dropped_rows", "mongoose_v2_bucket_entries_available",
        "mongoose_v2_truncated_bucket_entries", "mongoose_v2_fill_probe_attempts",
        "mongoose_v2_training_probe_attempts", "mongoose_v2_overloaded_routes",
        "mongoose_v2_bucket_entries_visited", "mongoose_v2_key_rows_scored",
        "mongoose_v2_bucket_entry_caps", "mongoose_v2_exact_score_caps",
        "mongoose_v2_lane_entry_caps",
        "mongoose_v2_table_entries_available", "mongoose_v2_table_entries_visited",
        "mongoose_v2_lane_entries_available", "mongoose_v2_lane_entries_visited"
    )
    foreach ($name in $required) {
        if (-not ($Accounting.PSObject.Properties.Name -contains $name)) {
            throw "MONGOOSE-v2 accounting field '$name' is missing at update $Update"
        }
    }
    Assert-ExactIntArray -Value $Accounting.mongoose_v2_bucket_entry_caps -Expected $GlobalVisitCaps -Label "MONGOOSE-v2 step global caps at update $Update"
    Assert-ExactIntArray -Value $Accounting.mongoose_v2_exact_score_caps -Expected $ScoreCaps -Label "MONGOOSE-v2 step exact-score caps at update $Update"
    Assert-ExactIntArray -Value $Accounting.mongoose_v2_lane_entry_caps -Expected $MaxLaneEntries -Label "MONGOOSE-v2 step max_lane_entries at update $Update"
    $routes = [int64]$Accounting.valid_candidates
    if ($routes -le 0) { throw "MONGOOSE-v2 accounting contains no candidate routes" }
    foreach ($name in @(
        "bucket_entries", "scored_rows", "retrieved_rows",
        "prefilter_dropped_rows", "mongoose_v2_bucket_entries_available",
        "mongoose_v2_truncated_bucket_entries", "mongoose_v2_fill_probe_attempts",
        "mongoose_v2_training_probe_attempts", "mongoose_v2_overloaded_routes",
        "mongoose_v2_bucket_entries_visited", "mongoose_v2_key_rows_scored"
    )) {
        if (@($Accounting.$name).Count -ne 3) {
            throw "MONGOOSE-v2 accounting field '$name' is not three layers"
        }
    }
    if (@($Accounting.mongoose_v2_table_entries_available).Count -ne 3 -or
        @($Accounting.mongoose_v2_table_entries_visited).Count -ne 3 -or
        @($Accounting.mongoose_v2_lane_entries_available).Count -ne 3 -or
        @($Accounting.mongoose_v2_lane_entries_visited).Count -ne 3) {
        throw "MONGOOSE-v2 table/lane telemetry is not three layers"
    }
    for ($layer = 0; $layer -lt 3; $layer += 1) {
        $available = [int64]$Accounting.mongoose_v2_bucket_entries_available[$layer]
        $visited = [int64]$Accounting.bucket_entries[$layer]
        $truncated = [int64]$Accounting.mongoose_v2_truncated_bucket_entries[$layer]
        $scored = [int64]$Accounting.scored_rows[$layer]
        $retrieved = [int64]$Accounting.retrieved_rows[$layer]
        $dropped = [int64]$Accounting.prefilter_dropped_rows[$layer]
        $fillAttempts = [int64]$Accounting.mongoose_v2_fill_probe_attempts[$layer]
        $trainingAttempts = [int64]$Accounting.mongoose_v2_training_probe_attempts[$layer]
        $overloadedRoutes = [int64]$Accounting.mongoose_v2_overloaded_routes[$layer]
        $publishedVisited = [int64]$Accounting.mongoose_v2_bucket_entries_visited[$layer]
        $publishedScored = [int64]$Accounting.mongoose_v2_key_rows_scored[$layer]
        foreach ($value in @($available, $visited, $truncated, $scored, $retrieved, $dropped, $fillAttempts, $trainingAttempts, $overloadedRoutes, $publishedVisited, $publishedScored)) {
            if ($value -lt 0) { throw "MONGOOSE-v2 telemetry is negative at update $Update layer $($layer + 1)" }
        }
        if ($available -ne $visited + $truncated) {
            throw "MONGOOSE-v2 available != visited + truncated at update $Update layer $($layer + 1)"
        }
        if ($publishedVisited -ne $visited -or $publishedScored -ne $scored) {
            throw "MONGOOSE-v2 published visited/scored aliases disagree at update $Update layer $($layer + 1)"
        }
        if ($dropped -gt $retrieved) {
            throw "MONGOOSE-v2 prefilter_dropped_rows exceeds retrieved_rows at update $Update layer $($layer + 1)"
        }
        if (($overloadedRoutes -eq 0) -ne ($truncated -eq 0)) {
            throw "MONGOOSE-v2 overload/truncation telemetry is inconsistent at update $Update layer $($layer + 1)"
        }
        if ($visited -gt $routes * $GlobalVisitCaps[$layer] -or
            $scored -gt $routes * $ScoreCaps[$layer] -or
            ($retrieved - $dropped) -gt $routes * $ShortlistCaps[$layer] -or
            $fillAttempts -gt $routes * $FillProbeCaps[$layer] -or
            $trainingAttempts -gt $routes * $TrainingProbeAttemptCaps[$layer] -or
            $overloadedRoutes -gt $routes) {
            throw "MONGOOSE-v2 global/score/shortlist/fill/probe cap failed at update $Update layer $($layer + 1)"
        }
        $tableAvailable = @($Accounting.mongoose_v2_table_entries_available[$layer])
        $tableVisited = @($Accounting.mongoose_v2_table_entries_visited[$layer])
        if ($tableAvailable.Count -ne $Tables -or $tableVisited.Count -ne $Tables) {
            throw "MONGOOSE-v2 table telemetry width is not $Tables"
        }
        if ([int64](($tableAvailable | Measure-Object -Sum).Sum) -ne $available -or
            [int64](($tableVisited | Measure-Object -Sum).Sum) -ne $visited) {
            throw "MONGOOSE-v2 table totals do not close at update $Update layer $($layer + 1)"
        }
        foreach ($value in $tableAvailable) {
            if ([int64]$value -lt 0) { throw "MONGOOSE-v2 table availability is negative" }
        }
        for ($table = 0; $table -lt $Tables; $table += 1) {
            $value = [int64]$tableVisited[$table]
            if ($value -lt 0 -or $value -gt [int64]$tableAvailable[$table] -or
                $value -gt $routes * $TableVisitCaps[$layer]) {
                throw "MONGOOSE-v2 table visit cap failed"
            }
        }
        $laneAvailable = @($Accounting.mongoose_v2_lane_entries_available[$layer])
        $laneVisited = @($Accounting.mongoose_v2_lane_entries_visited[$layer])
        if ($laneAvailable.Count -ne $LaneSlots -or $laneVisited.Count -ne $LaneSlots) {
            throw "MONGOOSE-v2 lane telemetry width is not $LaneSlots"
        }
        if ([int64](($laneAvailable | Measure-Object -Sum).Sum) -ne $available -or
            [int64](($laneVisited | Measure-Object -Sum).Sum) -ne $visited) {
            throw "MONGOOSE-v2 lane totals do not close at update $Update layer $($layer + 1)"
        }
        foreach ($value in $laneAvailable) {
            if ([int64]$value -lt 0) { throw "MONGOOSE-v2 lane availability is negative" }
        }
        $laneAggregateCap = $overloadedRoutes * $MaxLaneEntries[$layer] +
            ($routes - $overloadedRoutes) * $GlobalVisitCaps[$layer]
        for ($lane = 0; $lane -lt $LaneSlots; $lane += 1) {
            $value = [int64]$laneVisited[$lane]
            if ($value -lt 0 -or $value -gt [int64]$laneAvailable[$lane] -or
                $value -gt $laneAggregateCap) {
                throw "MONGOOSE-v2 lane visit cap failed"
            }
        }
    }
}

function Assert-V2CumulativePhaseTelemetry {
    param($WarmupThroughput, $FinalThroughput)

    $scalarFields = @(
        "mongoose_v2_bucket_entries_available",
        "mongoose_v2_bucket_entries_visited",
        "mongoose_v2_key_rows_scored",
        "mongoose_v2_truncated_bucket_entries",
        "mongoose_v2_fill_probe_attempts",
        "mongoose_v2_training_probe_attempts",
        "mongoose_v2_overloaded_routes"
    )
    $nestedFields = @(
        [pscustomobject]@{ Name="mongoose_v2_table_entries_available"; Width=$Tables },
        [pscustomobject]@{ Name="mongoose_v2_table_entries_visited"; Width=$Tables },
        [pscustomobject]@{ Name="mongoose_v2_lane_entries_available"; Width=$LaneSlots },
        [pscustomobject]@{ Name="mongoose_v2_lane_entries_visited"; Width=$LaneSlots }
    )
    foreach ($snapshot in @(
        [pscustomobject]@{ Value=$WarmupThroughput; Update=2000; RequireZero=$true },
        [pscustomobject]@{ Value=$FinalThroughput; Update=2500; RequireZero=$false }
    )) {
        $throughput = $snapshot.Value
        if ($null -eq $throughput -or [int]$throughput.updates -ne [int]$snapshot.Update) {
            throw "MONGOOSE-v2 cumulative throughput update identity failed at $($snapshot.Update)"
        }
        foreach ($name in @("retrieved_rows", "prefilter_dropped_rows") + $scalarFields) {
            if (-not ($throughput.PSObject.Properties.Name -contains $name) -or
                @($throughput.$name).Count -ne 3) {
                throw "MONGOOSE-v2 cumulative field '$name' is not three layers at update $($snapshot.Update)"
            }
            foreach ($value in @($throughput.$name)) {
                if ([int64]$value -lt 0) {
                    throw "MONGOOSE-v2 cumulative field '$name' is negative at update $($snapshot.Update)"
                }
                if ($snapshot.RequireZero -and $scalarFields -contains $name -and [int64]$value -ne 0) {
                    throw "MONGOOSE-v2 warmup-boundary cumulative field '$name' is not zero"
                }
            }
        }
        for ($layer = 0; $layer -lt 3; $layer += 1) {
            if ([int64]$throughput.prefilter_dropped_rows[$layer] -gt
                [int64]$throughput.retrieved_rows[$layer]) {
                throw "MONGOOSE-v2 cumulative prefilter_dropped_rows exceeds retrieved_rows at update $($snapshot.Update) layer $($layer + 1)"
            }
        }
        foreach ($spec in $nestedFields) {
            $name = [string]$spec.Name
            $outer = @($throughput.$name)
            if ($outer.Count -ne 3) {
                throw "MONGOOSE-v2 cumulative field '$name' is not three layers at update $($snapshot.Update)"
            }
            for ($layer = 0; $layer -lt 3; $layer += 1) {
                $inner = @($outer[$layer])
                if ($inner.Count -ne [int]$spec.Width) {
                    throw "MONGOOSE-v2 cumulative field '$name' inner width is not $($spec.Width) at update $($snapshot.Update) layer $($layer + 1)"
                }
                foreach ($value in $inner) {
                    if ([int64]$value -lt 0) {
                        throw "MONGOOSE-v2 cumulative field '$name' is negative at update $($snapshot.Update) layer $($layer + 1)"
                    }
                    if ($snapshot.RequireZero -and [int64]$value -ne 0) {
                        throw "MONGOOSE-v2 warmup-boundary cumulative field '$name' is not zero"
                    }
                }
            }
        }
        Assert-ExactIntArray -Value $throughput.mongoose_v2_bucket_entry_caps -Expected $GlobalVisitCaps -Label "MONGOOSE-v2 cumulative global caps at update $($snapshot.Update)"
        Assert-ExactIntArray -Value $throughput.mongoose_v2_exact_score_caps -Expected $ScoreCaps -Label "MONGOOSE-v2 cumulative exact-score caps at update $($snapshot.Update)"
        Assert-ExactIntArray -Value $throughput.mongoose_v2_lane_entry_caps -Expected $LaneVisitCaps -Label "MONGOOSE-v2 cumulative max_lane_entries at update $($snapshot.Update)"
    }

    $routeDelta = [int64]$FinalThroughput.candidates - [int64]$WarmupThroughput.candidates
    if ($routeDelta -le 0) {
        throw "MONGOOSE-v2 cumulative learned-routing candidate delta is not positive"
    }
    for ($layer = 0; $layer -lt 3; $layer += 1) {
        $available = [int64]$FinalThroughput.mongoose_v2_bucket_entries_available[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_bucket_entries_available[$layer]
        $truncated = [int64]$FinalThroughput.mongoose_v2_truncated_bucket_entries[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_truncated_bucket_entries[$layer]
        $fillAttempts = [int64]$FinalThroughput.mongoose_v2_fill_probe_attempts[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_fill_probe_attempts[$layer]
        $trainingAttempts = [int64]$FinalThroughput.mongoose_v2_training_probe_attempts[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_training_probe_attempts[$layer]
        $overloadedRoutes = [int64]$FinalThroughput.mongoose_v2_overloaded_routes[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_overloaded_routes[$layer]
        $bucketVisited = [int64]$FinalThroughput.mongoose_v2_bucket_entries_visited[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_bucket_entries_visited[$layer]
        $scored = [int64]$FinalThroughput.mongoose_v2_key_rows_scored[$layer] -
            [int64]$WarmupThroughput.mongoose_v2_key_rows_scored[$layer]
        $retrieved = [int64]$FinalThroughput.retrieved_rows[$layer] -
            [int64]$WarmupThroughput.retrieved_rows[$layer]
        $dropped = [int64]$FinalThroughput.prefilter_dropped_rows[$layer] -
            [int64]$WarmupThroughput.prefilter_dropped_rows[$layer]
        foreach ($value in @($available, $truncated, $fillAttempts, $trainingAttempts, $overloadedRoutes, $bucketVisited, $scored, $retrieved, $dropped)) {
            if ($value -lt 0) {
                throw "MONGOOSE-v2 cumulative phase delta is negative at layer $($layer + 1)"
            }
        }
        if ($dropped -gt $retrieved) {
            throw "MONGOOSE-v2 cumulative prefilter_dropped_rows exceeds retrieved_rows at layer $($layer + 1)"
        }

        $tableAvailable = @(for ($table = 0; $table -lt $Tables; $table += 1) {
            [int64]$FinalThroughput.mongoose_v2_table_entries_available[$layer][$table] -
                [int64]$WarmupThroughput.mongoose_v2_table_entries_available[$layer][$table]
        })
        $tableVisited = @(for ($table = 0; $table -lt $Tables; $table += 1) {
            [int64]$FinalThroughput.mongoose_v2_table_entries_visited[$layer][$table] -
                [int64]$WarmupThroughput.mongoose_v2_table_entries_visited[$layer][$table]
        })
        $laneAvailable = @(for ($lane = 0; $lane -lt $LaneSlots; $lane += 1) {
            [int64]$FinalThroughput.mongoose_v2_lane_entries_available[$layer][$lane] -
                [int64]$WarmupThroughput.mongoose_v2_lane_entries_available[$layer][$lane]
        })
        $laneVisited = @(for ($lane = 0; $lane -lt $LaneSlots; $lane += 1) {
            [int64]$FinalThroughput.mongoose_v2_lane_entries_visited[$layer][$lane] -
                [int64]$WarmupThroughput.mongoose_v2_lane_entries_visited[$layer][$lane]
        })
        foreach ($value in @($tableAvailable) + @($tableVisited) + @($laneAvailable) + @($laneVisited)) {
            if ([int64]$value -lt 0) {
                throw "MONGOOSE-v2 cumulative table/lane phase delta is negative at layer $($layer + 1)"
            }
        }
        $tableAvailableTotal = [int64](($tableAvailable | Measure-Object -Sum).Sum)
        $tableVisitedTotal = [int64](($tableVisited | Measure-Object -Sum).Sum)
        $laneAvailableTotal = [int64](($laneAvailable | Measure-Object -Sum).Sum)
        $laneVisitedTotal = [int64](($laneVisited | Measure-Object -Sum).Sum)
        if ($tableAvailableTotal -ne $available -or $laneAvailableTotal -ne $available -or
            $tableVisitedTotal -ne $laneVisitedTotal -or $tableVisitedTotal -ne $bucketVisited -or
            $available -ne $bucketVisited + $truncated) {
            throw "MONGOOSE-v2 cumulative table/lane totals do not close at layer $($layer + 1)"
        }
        if (($overloadedRoutes -eq 0) -ne ($truncated -eq 0) -or
            $overloadedRoutes -gt $routeDelta -or
            $bucketVisited -gt $routeDelta * $GlobalVisitCaps[$layer] -or
            $scored -gt $routeDelta * $ScoreCaps[$layer] -or
            ($retrieved - $dropped) -gt $routeDelta * $ShortlistCaps[$layer] -or
            $fillAttempts -gt $routeDelta * $FillProbeCaps[$layer] -or
            $trainingAttempts -gt $routeDelta * $TrainingProbeAttemptCaps[$layer]) {
            throw "MONGOOSE-v2 cumulative global/shortlist/fill/probe cap failed at layer $($layer + 1)"
        }
        for ($table = 0; $table -lt $Tables; $table += 1) {
            $value = [int64]$tableVisited[$table]
            if ($value -gt [int64]$tableAvailable[$table] -or
                $value -gt $routeDelta * $TableVisitCaps[$layer]) {
                throw "MONGOOSE-v2 cumulative table visit cap failed at layer $($layer + 1)"
            }
        }
        $laneAggregateCap = $overloadedRoutes * $LaneVisitCaps[$layer] +
            ($routeDelta - $overloadedRoutes) * $GlobalVisitCaps[$layer]
        for ($lane = 0; $lane -lt $LaneSlots; $lane += 1) {
            $value = [int64]$laneVisited[$lane]
            if ($value -gt [int64]$laneAvailable[$lane] -or $value -gt $laneAggregateCap) {
                throw "MONGOOSE-v2 cumulative max_lane_entries cap failed at layer $($layer + 1)"
            }
        }
    }
}

function Assert-V2Metric {
    param($Metric)
    $update = [int]$Metric.update
    if ([string]$Metric.variant -cne $Variant) { throw "MONGOOSE-v2 variant changed" }
    if ([string]$Metric.routing_policy -cne $RoutingPolicy) {
        throw "MONGOOSE-v2 routing policy changed at update $update"
    }
    if ([int]$Metric.learner_width -ne 80 -or [int]$Metric.observed_max_candidates -ne 76) {
        throw "MONGOOSE-v2 fixed candidate width changed"
    }
    if ([double]$Metric.objective_margin_weight -ne [double][single]$ObjectiveMarginWeight -or
        [string]$Metric.objective_margin_mode -cne $ObjectiveMarginMode -or
        [string]$Metric.objective_mode -cne $ObjectiveMode) {
        throw "MONGOOSE-v2 teacher objective changed"
    }
    $router = $Metric.router_state
    if ($null -eq $router -or [string]$router.configured_mode -cne $RoutingMode) {
        throw "MONGOOSE-v2 router state/configured mode is missing"
    }
    if ($update -eq 0) {
        if ([bool]$router.active -or [string]$router.serving_mode -cne "fixed_wta_warmup") {
            throw "MONGOOSE-v2 update-0 routing is not the frozen WTA warmup"
        }
        return
    }
    if (-not [bool]$router.active -or [string]$router.serving_mode -cne $RoutingMode) {
        throw "MONGOOSE-v2 learned router is not live at update $update"
    }
    if ([int]$router.route_dim -ne 64 -or [int]$router.bits_per_table -ne 7 -or
        [int]$router.tables -ne 2 -or [int]$router.mongoose_v2_load_balance_lanes -ne 16 -or
        [int]$router.trainable_parameters -ne 2688 -or [bool]$router.column_normalization) {
        throw "MONGOOSE-v2 router geometry changed at update $update"
    }
    if ([int]$router.mongoose_v2_query_version -ne 2 -or
        [int]$router.mongoose_v2_index_version -ne 3 -or
        [string]$router.mongoose_v2_lane_identity -cne "splitmix-domain-separated-router-seed-layer-table-neuron-v1" -or
        [string]$router.mongoose_v2_table_lane_order -cne "fixed-table-lane-round-robin-shared-bucket-budget" -or
        [string]$router.mongoose_v2_shortlist_order -cne "collision-count-descending-then-stable-neuron-id-ascending" -or
        [string]$router.mongoose_v2_rerank_order -cne "exact-raw-dot-descending-then-collision-count-descending-then-stable-neuron-id-ascending" -or
        [string]$router.mongoose_v2_fill_policy -cne "splitmix-full-cycle-bounded-fill-to-ranked-target" -or
        [string]$router.mongoose_v2_training_probe_policy -cne "splitmix-full-cycle-bounded-fixed-slot-probes" -or
        [bool]$router.mongoose_v2_dense_fallback) {
        throw "MONGOOSE-v2 bounded policy identity changed at update $update"
    }
    Assert-ExactIntArray -Value $router.mongoose_v2_bucket_entry_caps -Expected $GlobalVisitCaps -Label "MONGOOSE-v2 global visit caps"
    Assert-ExactIntArray -Value $router.mongoose_v2_exact_score_caps -Expected $ScoreCaps -Label "MONGOOSE-v2 exact score caps"
    $maxLaneEntries = @($router.mongoose_v2_lane_entry_caps)
    Assert-ExactIntArray -Value $maxLaneEntries -Expected $LaneVisitCaps -Label "MONGOOSE-v2 max_lane_entries"
    Assert-ExactIntArray -Value $router.mongoose_v2_active_counts -Expected $ActiveCounts -Label "MONGOOSE-v2 fixed active widths"
    Assert-ExactIntArray -Value $router.mongoose_v2_training_probes -Expected $TrainingProbes -Label "MONGOOSE-v2 fixed probe widths"
    if ([int]$router.warmup_updates -ne 2000 -or [int]$router.refresh_interval -ne 2000 -or
        [int]$router.last_refresh_update -ne 2000 -or [int]$router.refresh_count -ne 1 -or
        [bool]$router.refresh_due) {
        throw "MONGOOSE-v2 refresh clock changed at update $update"
    }
    Assert-ExactIntArray -Value $router.optimizer_steps -Expected @($update, $update, $update) -Label "MONGOOSE-v2 router optimizer clocks"
    $accounting = $Metric.last_step.accounting
    if ($null -eq $accounting -or [string]$accounting.routing_mode -cne $RoutingMode) {
        throw "MONGOOSE-v2 active-only accounting mode changed"
    }
    foreach ($name in @(
        "mongoose_v2_bucket_entry_caps", "mongoose_v2_exact_score_caps",
        "mongoose_v2_lane_entry_caps"
    )) {
        if (-not ($accounting.PSObject.Properties.Name -contains $name)) {
            throw "MONGOOSE-v2 step cap publication '$name' is missing at update $update"
        }
    }
    Assert-ExactIntArray -Value $accounting.mongoose_v2_bucket_entry_caps -Expected $GlobalVisitCaps -Label "MONGOOSE-v2 step global caps at update $update"
    Assert-ExactIntArray -Value $accounting.mongoose_v2_exact_score_caps -Expected $ScoreCaps -Label "MONGOOSE-v2 step exact-score caps at update $update"
    Assert-ExactIntArray -Value $accounting.mongoose_v2_lane_entry_caps -Expected $maxLaneEntries -Label "MONGOOSE-v2 step max_lane_entries at update $update"
    Assert-ExactIntArray -Value $accounting.active_counts -Expected $ActiveCounts -Label "MONGOOSE-v2 active widths"
    Assert-ExactIntArray -Value $accounting.training_probes -Expected $TrainingProbes -Label "MONGOOSE-v2 exploration widths"
    if (@($accounting.retrieved_rows).Count -ne 3 -or
        @($accounting.prefilter_dropped_rows).Count -ne 3) {
        throw "MONGOOSE-v2 step shortlist telemetry is not three layers at update $update"
    }
    for ($layer = 0; $layer -lt 3; $layer += 1) {
        $retrieved = [int64]$accounting.retrieved_rows[$layer]
        $dropped = [int64]$accounting.prefilter_dropped_rows[$layer]
        if ($retrieved -lt 0 -or $dropped -lt 0 -or $dropped -gt $retrieved) {
            throw "MONGOOSE-v2 step prefilter_dropped_rows exceeds retrieved_rows at update $update layer $($layer + 1)"
        }
    }
    if ($null -ne $accounting.dynamic_k) { throw "Dynamic-k appeared in fixed-k MONGOOSE-v2" }
    if ([int]$accounting.total_parameters -ne 19924022 -or
        [int]$accounting.mongoose_trainable_parameters -ne 2688 -or
        [int]$accounting.total_parameters_with_router -ne 19926710 -or
        [int]$accounting.mongoose_active_parameter_touches -ne 2688 -or
        [double]$accounting.mongoose_active_touch_fraction -gt 0.05) {
        throw "MONGOOSE-v2 parameter/touch accounting failed"
    }
    if (@($accounting.optimizer).Count -ne 3) {
        throw "MONGOOSE-v2 sparse optimizer telemetry must contain three layers"
    }
    $rowDimensions = @(560, 321, 321)
    for ($layer = 0; $layer -lt 3; $layer += 1) {
        $optimizer = $accounting.optimizer[$layer]
        if ([int]$optimizer.global_step -ne $update -or
            [int]$optimizer.active_rows -ne [int]$accounting.unique_active_rows[$layer] -or
            [int64]$optimizer.active_elements -ne [int64]$optimizer.active_rows * $rowDimensions[$layer] -or
            [int]$optimizer.dirty_route_rows -ne [int]$optimizer.active_rows -or
            [int64]$optimizer.theta_elements_read -ne [int64]$optimizer.theta_elements_written -or
            [int64]$optimizer.theta_elements_read -ne [int64]$optimizer.active_elements -or
            [int64]$optimizer.moment_elements_read -ne [int64]$optimizer.moment_elements_written -or
            [int64]$optimizer.moment_elements_read -ne 2 * [int64]$optimizer.theta_elements_read) {
            throw "MONGOOSE-v2 active-only optimizer gate failed at update $update layer $($layer + 1)"
        }
    }
    if ($null -eq $accounting.mongoose_pair -or
        [int64]$accounting.mongoose_pair.parameter_touches -ne 2688) {
        throw "MONGOOSE-v2 router update escaped its projection parameters"
    }
    if ($update -eq 2000) {
        # Update 2000's training step is still fixed-WTA; the atomic learned
        # index publication happens immediately after that step and before the
        # update-2000 evaluation.  V2-specific step counters must therefore be
        # present but exactly zero, while router_state above must already show
        # the live v2 evaluation/checkpoint state.
        foreach ($name in @(
            "mongoose_v2_bucket_entries_available",
            "mongoose_v2_truncated_bucket_entries",
            "mongoose_v2_fill_probe_attempts",
            "mongoose_v2_training_probe_attempts",
            "mongoose_v2_overloaded_routes",
            "mongoose_v2_bucket_entries_visited",
            "mongoose_v2_key_rows_scored"
        )) {
            if (@($accounting.$name).Count -ne 3 -or
                [int64](($accounting.$name | Measure-Object -Sum).Sum) -ne 0) {
                throw "MONGOOSE-v2 warmup-boundary field '$name' is not exactly zero"
            }
        }
        foreach ($spec in @(
            [pscustomobject]@{ Name="mongoose_v2_table_entries_available"; Width=$Tables },
            [pscustomobject]@{ Name="mongoose_v2_table_entries_visited"; Width=$Tables },
            [pscustomobject]@{ Name="mongoose_v2_lane_entries_available"; Width=$LaneSlots },
            [pscustomobject]@{ Name="mongoose_v2_lane_entries_visited"; Width=$LaneSlots }
        )) {
            $name = [string]$spec.Name
            if (@($accounting.$name).Count -ne 3) {
                throw "MONGOOSE-v2 warmup-boundary field '$name' is not three layers"
            }
            foreach ($layerValues in @($accounting.$name)) {
                if (@($layerValues).Count -ne [int]$spec.Width -or
                    [int64](($layerValues | Measure-Object -Sum).Sum) -ne 0) {
                    throw "MONGOOSE-v2 warmup-boundary field '$name' is not exactly zero"
                }
            }
        }
    } else {
        Assert-V2AccountingTelemetry -Accounting $accounting -Update $update -MaxLaneEntries $maxLaneEntries
    }
}

function Confirm-OwnedJuliaTermination {
    param(
        [System.Diagnostics.Process]$Process,
        [switch]$ForceStop
    )
    if ($null -eq $Process) { return }
    $Process.Refresh()
    if (-not $Process.HasExited -and $ForceStop) {
        Stop-Process -InputObject $Process -Force -ErrorAction Stop
        $exitedWithinLimit = $Process.WaitForExit(30000)
        $Process.Refresh()
        if (-not $exitedWithinLimit -or -not $Process.HasExited) {
            throw "Owned Julia PID $($Process.Id) did not terminate within 30 seconds"
        }
    }
    $Process.Refresh()
    if (-not $Process.HasExited) {
        throw "Owned Julia PID $($Process.Id) termination is not confirmed"
    }
    # Complete redirected-stream draining after the positive process-exit check.
    $Process.WaitForExit()
}

# Fail before any launch-time mutation until every final implementation binding
# is supplied by the integration owner.
Assert-FreezeResolved
if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf)) {
    throw "MONGOOSE-v2 preregistration contract is missing"
}
$contractFreezeText = Get-Content -LiteralPath $contractPath -Raw
if ($contractFreezeText.Contains($Unresolved) -or
    $contractFreezeText -match '(?m)^(?:[A-Za-z0-9_]+_bytes|focused_and_regression_tests_passed)\s*=\s*-1\s*$') {
    throw "MONGOOSE-v2 preregistration contract still contains unresolved freeze bindings"
}

$controllerPath = [IO.Path]::GetFullPath($PSCommandPath)
$controllerItem = Get-Item -LiteralPath $controllerPath -Force
if (($controllerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "MONGOOSE-v2 controller is a reparse point"
}
$controllerInitialBytes = [int64]$controllerItem.Length
$controllerInitialSha256 = (Get-FileHash -LiteralPath $controllerPath -Algorithm SHA256).Hash.ToLowerInvariant()
function Assert-ControllerUnchanged {
    [void](Assert-PinnedFile -Path $script:controllerPath -ExpectedBytes $script:controllerInitialBytes -ExpectedSha256 $script:controllerInitialSha256 -Label "MONGOOSE-v2 controller")
}

$expectedDataset = [IO.Path]::GetFullPath("D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3")
$expectedOutput = [IO.Path]::GetFullPath("D:\tetris-paper-plus\runs\beat_first_v1\$ExperimentId")
$expectedJulia = [IO.Path]::GetFullPath("C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe")
if ([IO.Path]::GetFullPath($Dataset) -cne $expectedDataset -or
    [IO.Path]::GetFullPath($OutputRoot) -cne $expectedOutput -or
    [IO.Path]::GetFullPath($Julia) -cne $expectedJulia) {
    throw "MONGOOSE-v2 frozen dataset/output/Julia identity changed"
}

[void](Assert-PinnedFile -Path $contractPath -ExpectedBytes $ExpectedContractBytes -ExpectedSha256 $ExpectedContractSha256 -Label "MONGOOSE-v2 preregistration contract")
[void](Assert-PinnedFile -Path $manifestPath -ExpectedBytes $ExpectedManifestBytes -ExpectedSha256 $ExpectedManifestSha256 -Label "teacher_v3 manifest")
foreach ($pin in @($PinnedImplementationFiles) + @($PinnedStableFiles)) {
    [void](Assert-PinnedFile -Path $pin.Path -ExpectedBytes ([int64]$pin.Bytes) -ExpectedSha256 ([string]$pin.Sha256) -Label ([string]$pin.Label))
}
[void](Assert-PinnedFile -Path $ParentMetricsPath -ExpectedBytes $ParentMetricsBytes -ExpectedSha256 $ParentMetricsSha256 -Label "fixed-WTA parent metrics")
[void](Assert-PinnedFile -Path $ParentCheckpointPath -ExpectedBytes $ParentCheckpointBytes -ExpectedSha256 $ParentCheckpointSha256 -Label "fixed-WTA parent checkpoint")
[void](Assert-PinnedFile -Path $SmokeSummaryPath -ExpectedBytes $ExpectedSmokeSummaryBytes -ExpectedSha256 $ExpectedSmokeSummarySha256 -Label "MONGOOSE-v2 real-teacher smoke summary")
[void](Assert-PinnedFile -Path $SmokeCheckpointPath -ExpectedBytes $ExpectedSmokeCheckpointBytes -ExpectedSha256 $ExpectedSmokeCheckpointSha256 -Label "MONGOOSE-v2 real-teacher smoke checkpoint")
$resolvedSmokeRoot = [IO.Path]::GetFullPath($SmokeRoot).TrimEnd('\')
if (-not (Test-Path -LiteralPath $resolvedSmokeRoot -PathType Container)) {
    throw "MONGOOSE-v2 real-teacher smoke root is missing"
}
$smokeRootItem = Get-Item -LiteralPath $resolvedSmokeRoot -Force
if (($smokeRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "MONGOOSE-v2 real-teacher smoke root is a reparse point"
}
[void](Assert-RunLeaf -Path $SmokeSummaryPath -Root $resolvedSmokeRoot)
[void](Assert-RunLeaf -Path $SmokeCheckpointPath -Root $resolvedSmokeRoot)
$smokeFiles = @(Get-ChildItem -LiteralPath $resolvedSmokeRoot -File -Force)
if ($smokeFiles.Count -ne 2 -or
    ($smokeFiles.Name | Sort-Object) -join ',' -cne 'checkpoint_000000002.jls,summary.json') {
    throw "MONGOOSE-v2 accepted smoke root does not contain exactly its frozen summary and checkpoint"
}
$smoke = Get-Content -LiteralPath $SmokeSummaryPath -Raw | ConvertFrom-Json -ErrorAction Stop
Assert-FiniteJsonValue -Value $smoke -Path "real_teacher_smoke"
if ([string]$smoke.schema -cne "mongoose-v2-real-teacher-two-update-smoke-v1" -or
    [string]$smoke.status -cne "PASS" -or
    -not [bool]$smoke.bounded_test_contract -or
    [IO.Path]::GetFullPath([string]$smoke.dataset_root) -cne $expectedDataset -or
    [string]$smoke.dataset_manifest_sha256 -cne $ExpectedManifestSha256 -or
    [IO.Path]::GetFullPath([string]$smoke.output_root).TrimEnd('\') -cne $resolvedSmokeRoot -or
    [string]$smoke.julia_version -cne "1.12.6" -or
    [string]$smoke.source_sha256 -cne $ExpectedSmokeSourceClosureSha256 -or
    [string]$smoke.project_sha256 -cne "62bbd9b624a4e2963e0382c86c8b4ff56b865c539e41ce1e8535cc7416ae4c7a" -or
    [string]$smoke.manifest_sha256 -cne "6ef9ed11797722e72e62d1d4e500da59d0bc14124f96344f93a14a322ec9a10a") {
    throw "MONGOOSE-v2 real-teacher smoke identity/closure changed"
}
if ([string]$smoke.driver_sha256 -cne "47731cce38326de37f7fa3bafe4c76e1def447096518c9a94a9d6b1489795bf3" -or
    [int]$smoke.total_parameters -ne 19924022 -or
    [int]$smoke.learner_width -ne 80 -or [int]$smoke.observed_max_candidates -ne 76 -or
    [string]$smoke.objective_mode -cne $ObjectiveMode -or
    [string]$smoke.objective_margin_mode -cne $ObjectiveMarginMode -or
    [double]$smoke.objective_margin_weight -ne [double][single]$ObjectiveMarginWeight -or
    [string]$smoke.routing_mode -cne $RoutingMode -or
    [int]$smoke.checkpoint.update -ne 2 -or
    [int64]$smoke.checkpoint.bytes -ne $ExpectedSmokeCheckpointBytes -or
    [string]$smoke.checkpoint.sha256 -cne $ExpectedSmokeCheckpointSha256 -or
    [IO.Path]::GetFullPath([string]$smoke.checkpoint.path) -cne [IO.Path]::GetFullPath($SmokeCheckpointPath)) {
    throw "MONGOOSE-v2 real-teacher smoke scientific/checkpoint contract changed"
}
Assert-ExactIntArray -Value $smoke.active_counts -Expected $ActiveCounts -Label "MONGOOSE-v2 smoke active widths"
Assert-ExactIntArray -Value $smoke.training_probes -Expected $TrainingProbes -Label "MONGOOSE-v2 smoke probe widths"
foreach ($name in @("next_row", "next_seed_id", "loss", "runtime_sha256_before", "overlay_sha256_before", "sampler_sha256_before", "runtime_sha256_after", "overlay_sha256_after", "sampler_sha256_after")) {
    if (-not ($smoke.exact_continuation.PSObject.Properties.Name -contains $name)) {
        throw "MONGOOSE-v2 real-teacher smoke exact-continuation witness '$name' is missing"
    }
}

$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
$resolvedRunParent = [IO.Path]::GetFullPath($RunParent).TrimEnd('\')
if (-not [IO.Path]::GetDirectoryName($resolvedOutput).Equals($resolvedRunParent, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetFileName($resolvedOutput) -cne $ExperimentId) {
    throw "MONGOOSE-v2 output is not the one direct frozen run child"
}
foreach ($directory in @($expectedDataset, $resolvedRunParent)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Pinned directory is missing: $directory"
    }
    $item = Get-Item -LiteralPath $directory -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Pinned directory is a reparse point: $directory"
    }
}
if ((Test-Path -LiteralPath $resolvedOutput) -or
    (Test-Path -LiteralPath $reservationPath) -or
    (Test-Path -LiteralPath $consumedPath)) {
    throw "MONGOOSE-v2 fresh one-shot output/reservation has already been consumed"
}

$contractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ValidateOnly) {
    [pscustomobject]@{
        validated = $true
        mutation = $false
        launched = $false
        experiment_id = $ExperimentId
        routing_mode = $RoutingMode
        variant = $Variant
        maximum_updates = $MaximumUpdates
        evaluation_updates = $EvaluationUpdates
        checkpoint_updates = $CheckpointUpdates
        active_counts = $ActiveCounts
        training_probes = $TrainingProbes
        load_balance_lanes_per_table = $LanesPerTable
        global_visit_caps = $GlobalVisitCaps
        table_visit_caps = $TableVisitCaps
        lane_visit_caps = $LaneVisitCaps
        score_caps = $ScoreCaps
        shortlist_caps = $ShortlistCaps
        fill_probe_caps = $FillProbeCaps
        training_probe_attempt_caps = $TrainingProbeAttemptCaps
        focused_and_regression_tests_passed = $FocusedAndRegressionTestsPassed
        smoke_summary_path = $SmokeSummaryPath
        smoke_summary_sha256 = $ExpectedSmokeSummarySha256
        smoke_checkpoint_path = $SmokeCheckpointPath
        smoke_checkpoint_sha256 = $ExpectedSmokeCheckpointSha256
        smoke_source_closure_sha256 = $ExpectedSmokeSourceClosureSha256
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        output_root = $resolvedOutput
    }
    return
}

$juliaMutex = [System.Threading.Mutex]::new($false, 'Local\TetrisBeatFirstV1ExclusiveJulia')
$held = $false
$ownedJulia = $null
$ownedJuliaLaunchAttempted = $false
$ownedJuliaTerminationConfirmed = $true
try {
    try { $held = $juliaMutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $held = $true }
    if (-not $held) { throw "Another Tetris controller owns the exclusive Julia lease" }
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Refusing to start while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
    }
    if ((Test-Path -LiteralPath $resolvedOutput) -or
        (Test-Path -LiteralPath $reservationPath) -or
        (Test-Path -LiteralPath $consumedPath)) {
        throw "MONGOOSE-v2 one-shot identity changed while acquiring the Julia lease"
    }

    $nonce = [Guid]::NewGuid().ToString("N")
    $reservationDocument = [ordered]@{
        format_version = 1
        experiment_id = $ExperimentId
        status = "reserved_for_atomic_consume"
        nonce = $nonce
        dataset_root = $expectedDataset
        output_root = $resolvedOutput
        routing_mode = $RoutingMode
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        controller_sha256 = $controllerInitialSha256
    } | ConvertTo-Json -Compress
    $reservationBytes = [Text.UTF8Encoding]::new($false).GetBytes($reservationDocument + [Environment]::NewLine)
    $reservationStream = [IO.FileStream]::new($reservationPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $reservationStream.Write($reservationBytes, 0, $reservationBytes.Length)
        $reservationStream.Flush($true)
    } finally {
        $reservationStream.Dispose()
    }
    Move-Item -LiteralPath $reservationPath -Destination $consumedPath
    Assert-ControllerUnchanged

    New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
    $stdoutPath = Join-Path $resolvedOutput "controller.stdout.log"
    $stderrPath = Join-Path $resolvedOutput "controller.stderr.log"
    $statusPath = Join-Path $resolvedOutput "controller_status.json"

    $env:BEAT_SPARSE_PAIRING_CONTRACT_SHA256 = $contractSha256
    $env:BEAT_TEACHER_DATASET = $expectedDataset
    $env:BEAT_3L_DATASET = $expectedDataset
    $env:BEAT_3L_OUTPUT = $resolvedOutput
    $env:BEAT_3L_RESUME = ""
    $env:BEAT_ALLOW_PARTIAL_DATASET = "false"
    $env:BEAT_3L_SEED = "2026071900"
    $env:BEAT_3L_MODEL_SEED = "2026071901"
    $env:BEAT_3L_SPLIT_SEED = "2026071902"
    $env:BEAT_3L_SAMPLER_SEED = "2026071903"
    $env:BEAT_3L_VARIANT = $Variant
    $env:BEAT_3L_EPOCHS = "1.0"
    $env:BEAT_3L_MAX_UPDATES = "2500"
    $env:BEAT_3L_EVAL_INTERVAL = "2500"
    $env:BEAT_3L_EVAL_SCHEDULE = "0,2000,2500"
    $env:BEAT_3L_CHECKPOINT_INTERVAL = "2500"
    $env:BEAT_3L_CHECKPOINT_SCHEDULE = "2000,2500"
    $env:BEAT_3L_TRAIN_EVAL_STATES = "128"
    $env:BEAT_3L_VALIDATION_EVAL_STATES = "512"
    $env:BEAT_3L_EVALUATE_INITIAL = "true"
    $env:BEAT_3L_VALIDATION_FRACTION = "0.20"
    $env:BEAT_3L_LR = "0.0001"
    $env:BEAT_3L_WEIGHT_DECAY = "0.0"
    $env:BEAT_3L_BETA1 = "0.9"
    $env:BEAT_3L_BETA2 = "0.999"
    $env:BEAT_3L_EPSILON = "1.0e-8"
    $env:BEAT_3L_MARGIN_WEIGHT = "0.15"
    $env:BEAT_3L_MARGIN_MODE = $ObjectiveMarginMode
    $env:BEAT_3L_OBJECTIVE_MODE = $ObjectiveMode
    $env:BEAT_3L_ROUTING_MODE = $RoutingMode
    $env:BEAT_3L_MONGOOSE_LR = "0.0001"
    $env:BEAT_3L_MONGOOSE_BETA = "1.0"
    $env:BEAT_3L_MONGOOSE_SEED = "1297043015"
    $env:BEAT_3L_MONGOOSE_WARMUP_UPDATES = "2000"
    $env:BEAT_3L_MONGOOSE_REFRESH_INTERVAL = "2000"
    foreach ($name in @(
        "BEAT_3L_DYNAMIC_K_MARGIN_THRESHOLD", "BEAT_3L_DYNAMIC_K_SCOUT_COUNTS",
        "BEAT_3L_DYNAMIC_K_EXPANDED_COUNTS", "BEAT_3L_DYNAMIC_K_SCOUT_TRAINING_PROBES",
        "BEAT_3L_DYNAMIC_K_EXPANDED_TRAINING_PROBES"
    )) {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
    $env:JULIA_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"

    # Mark the launch outcome unconfirmed before Start-Process can create the
    # child.  If the cmdlet is interrupted after creation but before returning
    # its handle, finally refuses to release the mutex rather than treating a
    # null handle as proof that no Julia exists.
    $ownedJuliaLaunchAttempted = $true
    $ownedJuliaTerminationConfirmed = $false
    $ownedJulia = Start-Process -FilePath $Julia `
        -ArgumentList @("--startup-file=no", "--project=$experimentRoot", $entrypoint) `
        -WorkingDirectory $experimentRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [ordered]@{
        status = "running"
        experiment_id = $ExperimentId
        pid = $ownedJulia.Id
        started_at = [DateTimeOffset]::Now.ToString("o")
        variant = $Variant
        maximum_updates = $MaximumUpdates
        routing_mode = $RoutingMode
        active_counts = $ActiveCounts
        evaluation_updates = $EvaluationUpdates
        checkpoint_updates = $CheckpointUpdates
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract = $contractPath
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        output_root = $resolvedOutput
        one_shot_nonce = $nonce
        consumed_sentinel = $consumedPath
        focused_and_regression_tests_passed = $FocusedAndRegressionTestsPassed
        smoke_summary_sha256 = $ExpectedSmokeSummarySha256
        smoke_checkpoint_sha256 = $ExpectedSmokeCheckpointSha256
        smoke_source_closure_sha256 = $ExpectedSmokeSourceClosureSha256
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM

    $deadline = [DateTimeOffset]::Now.AddMinutes(20)
    while (-not $ownedJulia.HasExited) {
        if ([DateTimeOffset]::Now -ge $deadline) {
            Stop-Process -Id $ownedJulia.Id -Force -ErrorAction SilentlyContinue
            throw "MONGOOSE-v2 signal exceeded its 20-minute wall limit"
        }
        $otherJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne $ownedJulia.Id })
        if ($otherJulia.Count -ne 0) {
            Stop-Process -Id $ownedJulia.Id -Force -ErrorAction SilentlyContinue
            throw "A second Julia process overlapped the owned signal: $(($otherJulia.Id) -join ',')"
        }
        Start-Sleep -Milliseconds 500
        $ownedJulia.Refresh()
    }
    Confirm-OwnedJuliaTermination -Process $ownedJulia
    $ownedJuliaTerminationConfirmed = $true
    if ($ownedJulia.ExitCode -ne 0) {
        throw "MONGOOSE-v2 trainer exited with code $($ownedJulia.ExitCode)"
    }

    Assert-ControllerUnchanged
    $metricFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "metrics.jsonl")
    if ($metricFiles.Count -ne 1) { throw "MONGOOSE-v2 must publish exactly one metrics.jsonl" }
    $metricsPath = Assert-RunLeaf -Path $metricFiles[0].FullName -Root $resolvedOutput
    $metrics = @(
        Get-Content -LiteralPath $metricsPath |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_ | ConvertFrom-Json -ErrorAction Stop }
    )
    Assert-ExactIntArray -Value @($metrics | ForEach-Object { [int]$_.update }) -Expected $EvaluationUpdates -Label "MONGOOSE-v2 metric updates"
    foreach ($metric in $metrics) {
        Assert-FiniteJsonValue -Value $metric -Path "metric[$($metric.update)]"
        Assert-V2Metric -Metric $metric
    }
    Assert-V2CumulativePhaseTelemetry -WarmupThroughput $metrics[1].throughput -FinalThroughput $metrics[2].throughput
    $final = $metrics[-1]
    if (@($final.throughput.mongoose_v2_overloaded_routes).Count -ne 3) {
        throw "MONGOOSE-v2 cumulative overload telemetry is not three layers"
    }
    $overloadObserved = [int64](($final.throughput.mongoose_v2_overloaded_routes | Measure-Object -Sum).Sum) -gt 0
    if ([double]$final.validation.top1_agreement -lt 0.0703125 -or
        [double]$final.validation.ndcg -lt 0.8440925775521915 -or
        [double]$final.validation.pairwise_accuracy -lt 0.5407089182563002 -or
        [double]$final.validation.composite_loss -gt 6.333081512246281) {
        throw "MONGOOSE-v2 learned-routing teacher signal regressed below its frozen floor"
    }

    if ((Get-Item -LiteralPath $stdoutPath).Length -ne 0) {
        throw "MONGOOSE-v2 stdout must remain empty"
    }
    $stderrText = Get-Content -LiteralPath $stderrPath -Raw
    if ($stderrText -match '(?im)^ERROR:|Stacktrace:|exact-score cap .* exceeded|bucket-entry cap .* exceeded|(?<![A-Za-z])(?:NaN|[+-]?Inf(?:inity)?)(?![A-Za-z])') {
        throw "MONGOOSE-v2 stderr contains a forbidden failure record"
    }
    $stderrLines = @(Get-Content -LiteralPath $stderrPath)
    foreach ($line in $stderrLines) {
        if ($line.Length -gt 0 -and $line -notmatch '^(┌ Info:|│|└)') {
            throw "MONGOOSE-v2 stderr contains an unexpected non-Info line: $line"
        }
    }
    if (@($stderrLines | Where-Object { $_ -eq '┌ Info: Three-layer initial teacher evaluation' }).Count -ne 1 -or
        @($stderrLines | Where-Object { $_ -eq '┌ Info: Three-layer teacher progress' }).Count -ne 2 -or
        @($stderrLines | Where-Object { $_ -eq '┌ Info: Three-layer teacher checkpoint' }).Count -ne 2) {
        throw "MONGOOSE-v2 stderr Info record counts changed"
    }

    $runDirectory = $metricFiles[0].Directory
    if ($runDirectory.Name -notmatch '^sparse_3l_[0-9]{8}T[0-9]{6}$') {
        throw "MONGOOSE-v2 authoritative run directory identity changed"
    }
    $expectedCheckpointDirectory = [IO.Path]::GetFullPath((Join-Path $runDirectory.FullName "checkpoints"))
    $expectedCheckpointPaths = @(
        [IO.Path]::GetFullPath((Join-Path $expectedCheckpointDirectory "checkpoint_000002000.jls")),
        [IO.Path]::GetFullPath((Join-Path $expectedCheckpointDirectory "checkpoint_000002500.jls"))
    )
    $checkpointFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "checkpoint_*.jls" | Sort-Object Name)
    if ($checkpointFiles.Count -ne 2 -or
        ($checkpointFiles.Name -join ',') -cne 'checkpoint_000002000.jls,checkpoint_000002500.jls') {
        throw "MONGOOSE-v2 did not publish exactly the update-2000 and update-2500 checkpoints"
    }
    for ($index = 0; $index -lt $expectedCheckpointPaths.Count; $index += 1) {
        $checkpointFilePath = Assert-RunLeaf -Path $checkpointFiles[$index].FullName -Root $resolvedOutput
        if (-not $checkpointFilePath.Equals($expectedCheckpointPaths[$index], [StringComparison]::OrdinalIgnoreCase)) {
            throw "MONGOOSE-v2 checkpoint is not at its exact authoritative enumerated path"
        }
    }
    $latestFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "latest.json")
    if ($latestFiles.Count -ne 1) { throw "MONGOOSE-v2 must publish exactly one latest.json" }
    $latestPath = Assert-RunLeaf -Path $latestFiles[0].FullName -Root $resolvedOutput
    $expectedLatestPath = [IO.Path]::GetFullPath((Join-Path $runDirectory.FullName "latest.json"))
    if (-not $latestPath.Equals($expectedLatestPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "MONGOOSE-v2 latest.json is not in the authoritative run directory"
    }
    $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json -ErrorAction Stop
    if ([int]$latest.update -ne 2500 -or [string]$latest.variant -cne $Variant) {
        throw "MONGOOSE-v2 latest checkpoint identity changed"
    }
    $latestDeclaredPath = [string]$latest.path
    if (-not [IO.Path]::IsPathFullyQualified($latestDeclaredPath) -or
        -not [IO.Path]::GetFullPath($latestDeclaredPath).Equals($expectedCheckpointPaths[1], [StringComparison]::OrdinalIgnoreCase)) {
        throw "MONGOOSE-v2 latest.path is not the exact expected final checkpoint"
    }
    $checkpointPath = Assert-RunLeaf -Path $latestDeclaredPath -Root $resolvedOutput
    if (@($checkpointFiles.FullName | Where-Object {
        [IO.Path]::GetFullPath($_).Equals($checkpointPath, [StringComparison]::OrdinalIgnoreCase)
    }).Count -ne 1) {
        throw "MONGOOSE-v2 latest.path does not resolve to exactly one enumerated checkpoint"
    }
    $checkpoint = Get-Item -LiteralPath $checkpointPath
    if ([int64]$latest.bytes -ne [int64]$checkpoint.Length) {
        throw "MONGOOSE-v2 latest checkpoint byte count differs from disk"
    }
    $checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$latest.sha256 -cne $checkpointSha256) {
        throw "MONGOOSE-v2 latest checkpoint SHA-256 differs from disk"
    }
    $checkpointArtifacts = @($checkpointFiles | ForEach-Object {
        [ordered]@{
            path = $_.FullName
            bytes = [int64]$_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    if (-not $latestFiles[0].Directory.FullName.Equals($runDirectory.FullName, [StringComparison]::OrdinalIgnoreCase) -or
        -not $checkpoint.Directory.FullName.Equals($expectedCheckpointDirectory, [StringComparison]::OrdinalIgnoreCase) -or
        -not $checkpoint.Directory.Parent.FullName.Equals($runDirectory.FullName, [StringComparison]::OrdinalIgnoreCase)) {
        throw "MONGOOSE-v2 run artifacts do not share one authoritative run directory"
    }

    $metricsSha256 = (Get-FileHash -LiteralPath $metricsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $gatePath = Join-Path $resolvedOutput "gate_status.json"
    $gatePendingPath = Join-Path $resolvedOutput "gate_status.pending.json"
    [ordered]@{
        status = "PASS"
        experiment_id = $ExperimentId
        final_update = 2500
        variant = $Variant
        routing_mode = $RoutingMode
        routing_policy = $RoutingPolicy
        fixed_active_counts = $ActiveCounts
        fixed_training_probes = $TrainingProbes
        bounded_overload_observed = $overloadObserved
        max_lane_entries = $LaneVisitCaps
        final_held_top1 = [double]$final.validation.top1_agreement
        final_held_ndcg = [double]$final.validation.ndcg
        final_held_pairwise = [double]$final.validation.pairwise_accuracy
        final_held_action_margin = [double]$final.validation.action_margin
        final_step_bucket_entries_visited = @($final.last_step.accounting.bucket_entries)
        final_step_key_rows_scored = @($final.last_step.accounting.scored_rows)
        final_step_fill_probe_attempts = @($final.last_step.accounting.mongoose_v2_fill_probe_attempts)
        final_step_training_probe_attempts = @($final.last_step.accounting.mongoose_v2_training_probe_attempts)
        metrics_path = $metricsPath
        metrics_sha256 = $metricsSha256
        checkpoint_path = $checkpointPath
        checkpoint_bytes = [int64]$checkpoint.Length
        checkpoint_sha256 = $checkpointSha256
        checkpoints = $checkpointArtifacts
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        parent_metrics_path = $ParentMetricsPath
        parent_metrics_sha256 = $ParentMetricsSha256
        parent_checkpoint_path = $ParentCheckpointPath
        parent_checkpoint_sha256 = $ParentCheckpointSha256
        focused_and_regression_tests_passed = $FocusedAndRegressionTestsPassed
        smoke_summary_path = $SmokeSummaryPath
        smoke_summary_sha256 = $ExpectedSmokeSummarySha256
        smoke_checkpoint_path = $SmokeCheckpointPath
        smoke_checkpoint_sha256 = $ExpectedSmokeCheckpointSha256
        smoke_source_closure_sha256 = $ExpectedSmokeSourceClosureSha256
        validation_game_seeds_touched = $false
        sealed_game_seeds_touched = $false
        owned_julia_termination_confirmed = $ownedJuliaTerminationConfirmed
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $gatePendingPath -Encoding utf8NoBOM
    $gateSha256 = (Get-FileHash -LiteralPath $gatePendingPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-ControllerUnchanged
    [ordered]@{
        status = "completed"
        experiment_id = $ExperimentId
        pid = $ownedJulia.Id
        completed_at = [DateTimeOffset]::Now.ToString("o")
        exit_code = $ownedJulia.ExitCode
        variant = $Variant
        maximum_updates = $MaximumUpdates
        routing_mode = $RoutingMode
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract = $contractPath
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        output_root = $resolvedOutput
        gate_status = $gatePath
        gate_status_sha256 = $gateSha256
        one_shot_nonce = $nonce
        consumed_sentinel = $consumedPath
        focused_and_regression_tests_passed = $FocusedAndRegressionTestsPassed
        smoke_summary_sha256 = $ExpectedSmokeSummarySha256
        smoke_checkpoint_sha256 = $ExpectedSmokeCheckpointSha256
        smoke_source_closure_sha256 = $ExpectedSmokeSourceClosureSha256
        owned_julia_termination_confirmed = $ownedJuliaTerminationConfirmed
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM
    Move-Item -LiteralPath $gatePendingPath -Destination $gatePath
} catch {
    $failureMessage = $_.Exception.Message
    if ($null -ne $ownedJulia -and -not $ownedJuliaTerminationConfirmed) {
        try {
            Confirm-OwnedJuliaTermination -Process $ownedJulia -ForceStop
            $ownedJuliaTerminationConfirmed = $true
        } catch {
            $failureMessage += "; owned Julia cleanup failed: $($_.Exception.Message)"
        }
    }
    if (Test-Path -LiteralPath $resolvedOutput) {
        $controllerIntegrity = "PASS"
        try { Assert-ControllerUnchanged } catch { $controllerIntegrity = "FAIL: $($_.Exception.Message)" }
        [ordered]@{
            status = "failed"
            experiment_id = $ExperimentId
            failed_at = [DateTimeOffset]::Now.ToString("o")
            error = $failureMessage
            output_root = $resolvedOutput
            controller_path = $controllerPath
            controller_bytes = $controllerInitialBytes
            controller_sha256 = $controllerInitialSha256
            controller_integrity = $controllerIntegrity
            consumed_sentinel = $consumedPath
            owned_julia_launch_attempted = $ownedJuliaLaunchAttempted
            owned_julia_handle_captured = ($null -ne $ownedJulia)
            owned_julia_termination_confirmed = $ownedJuliaTerminationConfirmed
        } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutput "controller_status.json") -Encoding utf8NoBOM
    }
    throw
} finally {
    if ($ownedJuliaLaunchAttempted -and $null -eq $ownedJulia -and
        -not $ownedJuliaTerminationConfirmed) {
        throw "Julia launch outcome is unconfirmed; refusing to release the exclusive mutex without an owned process handle"
    }
    if ($null -ne $ownedJulia -and -not $ownedJuliaTerminationConfirmed) {
        Confirm-OwnedJuliaTermination -Process $ownedJulia -ForceStop
        $ownedJuliaTerminationConfirmed = $true
    }
    if ($held -and -not $ownedJuliaTerminationConfirmed) {
        throw "Refusing to release the exclusive Julia mutex before owned Julia termination is confirmed"
    }
    if ($held) { try { $juliaMutex.ReleaseMutex() } catch {} }
    $juliaMutex.Dispose()
}
