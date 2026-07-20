[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$StaticParseOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This is a distinct one-shot controller for the cutoff-boundary pair-mining
# causal arm.  The already consumed witness-pair controller is immutable and is
# used only as a byte-pinned governance template.  Every transformation below
# must match its expected source fragment exactly; template drift or a missing
# cutoff-specific binding fails before launch-time filesystem mutation.
$ExperimentId =
    "sparse_3l_k128_mongoose_v2_cutoff_pair_teacher_signal_cpf_2500_v1"
$PairMiningMode = "fixed_wta_exploitation_cutoff_boundary_v1"
$PairMiningVersion = 1
$PairMiningIdentity =
    "bounded-fixed-wta-natural-exploitation-cutoff-pair-v1"
$PairMiningScoreOrder =
    "exact-raw-score-descending-then-stable-neuron-id-ascending"
$PairMiningEligibility =
    "retrieved-and-exact-scored-positive-collision-only"
$PairMiningExclusion =
    "deterministic-fill-and-training-probe-collision-zero-excluded"
$PairMiningOutsidePolicy =
    "fail-closed-without-rank-k-plus-one-bounded-witness"
$ExploitationTargets = @(42, 35, 35)
$PositiveRanks = @(42, 35, 35)
$NegativeRanks = @(43, 36, 36)
$Unresolved =
    "UNRESOLVED_AFTER_CUTOFF_PAIR_IMPLEMENTATION_TEST_SMOKE_FREEZE"

$comparisonRoot = $PSScriptRoot
$templateControllerPath = Join-Path $comparisonRoot `
    "run_three_layer_k128_mongoose_v2_2500_controller.ps1"
$templateContractPath = Join-Path $comparisonRoot `
    "three_layer_cpf_k128_mongoose_v2_2500_update_contract.toml"
$contractPath = Join-Path $comparisonRoot `
    "three_layer_cpf_k128_mongoose_v2_cutoff_pair_2500_update_contract.toml"
$controllerPath = [IO.Path]::GetFullPath($PSCommandPath)

# Immutable final witness-pair governance template.  These are provenance
# bindings, not mutable freeze placeholders.
$TemplateControllerBytes = 69232
$TemplateControllerSha256 =
    "42d2d18710edbe3431f813c57793b7d3d5b9bf1db2e7caf63ded9801c7e33d59"
$TemplateContractBytes = 15421
$TemplateContractSha256 =
    "6e8402bbea3dd981cc2a7bcf503c301b11e5b8a4b4927253d08301285252a450"

# Frozen after the implementation, focused and regression tests, and the
# separate collision-observable cutoff smoke all passed authoritatively.  The
# contract digest is filled last so it binds the complete immutable freeze.
$ExpectedContractBytes = 20520
$ExpectedContractSha256 =
    "67caff644d75f88faf33d987f119183d9209fc2ff7e0969c3780aba178d1af16"
$FocusedAndRegressionTestsPassed = 110311
$ExpectedTeacherTrainingBytes = 154817
$ExpectedTeacherTrainingSha256 =
    "3691b4f741bc1212a2e74f2478173f7404e6d43b58d0516bed9b4ec4da644f2b"
$ExpectedCutoffPairTestBytes = 9299
$ExpectedCutoffPairTestSha256 =
    "4c8fbf530084614fe8d0fc7ba1719a072cab7d9f15f382fed333d195f2b00a76"
$ExpectedCutoffPairDesignBytes = 2631
$ExpectedCutoffPairDesignSha256 =
    "d4032444c677bc4a8253af428fd0b007d2f95143ddbe3bbe808158f0c24f60eb"
$ExpectedCutoffSmokeDriverBytes = 30378
$ExpectedCutoffSmokeDriverSha256 =
    "6b0d0e3872a0e17d28cdb27a1d2bd93ec620198f13efec39eedf8b4cd48b002d"
$CutoffSmokeRoot =
    "C:\tmp\tetris_mongoose_v2_cutoff_collision_real_teacher_smoke_20260719T234300"
$ExpectedCutoffSmokeSummaryBytes = 7827
$ExpectedCutoffSmokeSummarySha256 =
    "9e7a43db38fd3223954ce2f42b8d511248e1d89627cde91b45b85b41b579c1b9"
$ExpectedCutoffSmokeSourceClosureSha256 =
    "f3bb9ebc4f759b1e8939c3a5ae00fe636dd8034726f19bc2b0be66ac1e64a6bf"
$ExpectedCutoffSmokeCheckpointBytes = 249871141
$ExpectedCutoffSmokeCheckpointSha256 =
    "285589c2ff58c9c090567e1c80a22c2b03161e64cc8c3c6b4690253f68cd1a4c"

# The failed easy-extrema v2 signal remains comparison telemetry only.  It is
# deliberately not a performance or efficiency floor for this causal arm.
$WitnessPairMetricsPath =
    "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1\sparse_3l_20260719T213725\metrics.jsonl"
$WitnessPairMetricsBytes = 36596
$WitnessPairMetricsSha256 =
    "e63b16a22971dc37461cb358894f82d56ef1e8729b2e41f1c2e68e52dcd153ea"
$WitnessPairCheckpointPath =
    "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1\sparse_3l_20260719T213725\checkpoints\checkpoint_000002500.jls"
$WitnessPairCheckpointBytes = 249896561
$WitnessPairCheckpointSha256 =
    "c0b8b350be2357c39c27a4cf73fb8efba9b4403ba4de8cee57a2248166eee2af"

function Get-LowerSha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-PinnedTemplateFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][int64]$Bytes,
        [Parameter(Mandatory)][string]$Sha256,
        [Parameter(Mandatory)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Label is a reparse point: $Path"
    }
    if ([int64]$item.Length -ne $Bytes) {
        throw "$Label bytes changed: $($item.Length) != $Bytes"
    }
    $observed = Get-LowerSha256 -Path $item.FullName
    if ($observed -cne $Sha256) {
        throw "$Label SHA-256 changed: $observed"
    }
    return $item.FullName
}

function Assert-ResolvedSha256 {
    param([string]$Value, [string]$Label)
    if ($Value -ceq $Unresolved -or $Value -notmatch '^[0-9a-f]{64}$') {
        throw "$Label remains unresolved"
    }
}

function Assert-CutoffFreezeResolved {
    $unresolvedBindings = [System.Collections.Generic.List[string]]::new()
    foreach ($binding in @(
        [pscustomobject]@{ Label="contract"; Bytes=$ExpectedContractBytes; Sha=$ExpectedContractSha256 },
        [pscustomobject]@{ Label="teacher training source"; Bytes=$ExpectedTeacherTrainingBytes; Sha=$ExpectedTeacherTrainingSha256 },
        [pscustomobject]@{ Label="cutoff-pair focused test"; Bytes=$ExpectedCutoffPairTestBytes; Sha=$ExpectedCutoffPairTestSha256 },
        [pscustomobject]@{ Label="cutoff-pair design"; Bytes=$ExpectedCutoffPairDesignBytes; Sha=$ExpectedCutoffPairDesignSha256 },
        [pscustomobject]@{ Label="cutoff-pair smoke driver"; Bytes=$ExpectedCutoffSmokeDriverBytes; Sha=$ExpectedCutoffSmokeDriverSha256 },
        [pscustomobject]@{ Label="cutoff-pair smoke summary"; Bytes=$ExpectedCutoffSmokeSummaryBytes; Sha=$ExpectedCutoffSmokeSummarySha256 },
        [pscustomobject]@{ Label="cutoff-pair smoke checkpoint"; Bytes=$ExpectedCutoffSmokeCheckpointBytes; Sha=$ExpectedCutoffSmokeCheckpointSha256 }
    )) {
        if ([int64]$binding.Bytes -le 0) {
            $unresolvedBindings.Add("$($binding.Label) bytes")
        }
        if ([string]$binding.Sha -ceq $Unresolved -or
            [string]$binding.Sha -notmatch '^[0-9a-f]{64}$') {
            $unresolvedBindings.Add("$($binding.Label) SHA-256")
        }
    }
    if ($FocusedAndRegressionTestsPassed -le 0) {
        $unresolvedBindings.Add("focused/regression test count")
    }
    if ($CutoffSmokeRoot -ceq $Unresolved -or
        -not [IO.Path]::IsPathFullyQualified($CutoffSmokeRoot)) {
        $unresolvedBindings.Add("cutoff-pair smoke root")
    }
    if ($ExpectedCutoffSmokeSourceClosureSha256 -ceq $Unresolved -or
        $ExpectedCutoffSmokeSourceClosureSha256 -notmatch '^[0-9a-f]{64}$') {
        $unresolvedBindings.Add("cutoff-pair smoke source closure SHA-256")
    }
    if ($unresolvedBindings.Count -ne 0) {
        throw "Cutoff-pair source/test/smoke freeze is unresolved; no validation or launch is allowed: $($unresolvedBindings -join ', ')"
    }
}

function Replace-ExactCount {
    param(
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][string]$Old,
        [AllowEmptyString()][string]$New,
        [int]$ExpectedCount = 1,
        [Parameter(Mandatory)][string]$Label
    )
    $parts = $Text.Split([string[]]@($Old), [StringSplitOptions]::None)
    $observed = $parts.Count - 1
    if ($observed -ne $ExpectedCount) {
        throw "Governance-template transform '$Label' matched $observed times, expected $ExpectedCount"
    }
    return [string]::Join($New, $parts)
}

function Quote-PowerShellLiteral {
    param([Parameter(Mandatory)][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Build-DerivedControllerSource {
    $source = Get-Content -LiteralPath $templateControllerPath -Raw
    $newline = [Environment]::NewLine

    $source = Replace-ExactCount -Text $source `
        -Old '$ExperimentId = "sparse_3l_k128_mongoose_v2_teacher_signal_cpf_2500_v1"' `
        -New ('$ExperimentId = "' + $ExperimentId + '"') `
        -Label "experiment identity"
    $source = Replace-ExactCount -Text $source `
        -Old '$ObjectiveMode = "standardized_listnet_plus_margin"' `
        -New ('$ObjectiveMode = "standardized_listnet_plus_margin"' + $newline +
            '$PairMiningMode = "' + $PairMiningMode + '"' + $newline +
            '$PairMiningVersion = 1' + $newline +
            '$PairMiningIdentity = "' + $PairMiningIdentity + '"' + $newline +
            '$PairMiningScoreOrder = "' + $PairMiningScoreOrder + '"' + $newline +
            '$PairMiningEligibility = "' + $PairMiningEligibility + '"' + $newline +
            '$PairMiningExclusion = "' + $PairMiningExclusion + '"' + $newline +
            '$PairMiningOutsidePolicy = "' + $PairMiningOutsidePolicy + '"' + $newline +
            '$ExploitationTargets = @(42, 35, 35)' + $newline +
            '$PositiveRanks = @(42, 35, 35)' + $newline +
            '$NegativeRanks = @(43, 36, 36)') `
        -Label "cutoff constants"
    $source = Replace-ExactCount -Text $source `
        -Old '$FocusedAndRegressionTestsPassed = 109444' `
        -New ('$FocusedAndRegressionTestsPassed = ' + $FocusedAndRegressionTestsPassed) `
        -Label "focused test count"
    $source = Replace-ExactCount -Text $source `
        -Old '$Unresolved = "UNRESOLVED_AFTER_V2_IMPLEMENTATION_AND_TEST_FREEZE"' `
        -New ('$Unresolved = "' + $Unresolved + '"') `
        -Label "unresolved token"
    $source = Replace-ExactCount -Text $source `
        -Old '$comparisonRoot = $PSScriptRoot' `
        -New ('$comparisonRoot = ' + (Quote-PowerShellLiteral -Value $comparisonRoot)) `
        -Label "derived comparison root"
    $source = Replace-ExactCount -Text $source `
        -Old '$contractPath = Join-Path $comparisonRoot "three_layer_cpf_k128_mongoose_v2_2500_update_contract.toml"' `
        -New ('$contractPath = Join-Path $comparisonRoot "three_layer_cpf_k128_mongoose_v2_cutoff_pair_2500_update_contract.toml"') `
        -Label "cutoff contract path"
    $source = Replace-ExactCount -Text $source `
        -Old '$controllerPath = [IO.Path]::GetFullPath($PSCommandPath)' `
        -New ('$controllerPath = [IO.Path]::GetFullPath(' +
            (Quote-PowerShellLiteral -Value $controllerPath) + ')') `
        -Label "wrapper self-integrity path"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedContractBytes = 15421' `
        -New ('$ExpectedContractBytes = ' + $ExpectedContractBytes) `
        -Label "contract bytes"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedContractSha256 = "6e8402bbea3dd981cc2a7bcf503c301b11e5b8a4b4927253d08301285252a450"' `
        -New ('$ExpectedContractSha256 = "' + $ExpectedContractSha256 + '"') `
        -Label "contract digest"

    $source = Replace-ExactCount -Text $source `
        -Old 'Bytes=140468; Sha256="e1cc8d21a8398922222aa999430a2c4fc38ee713ba9d14841c2f598f04ffb641"; Label="teacher training source"' `
        -New ('Bytes=' + $ExpectedTeacherTrainingBytes + '; Sha256="' +
            $ExpectedTeacherTrainingSha256 + '"; Label="teacher training source"') `
        -Label "teacher source pin"
    $integrationTestPin =
        '[pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\test_mongoose_v2_trainer_runtime_integration.jl"); Bytes=21353; Sha256="4d495bb2adabb9e1f5ae10177778c22f6c8808a2c59cab334b4210240e3f5cfa"; Label="v2 trainer/runtime integration test source" },'
    $cutoffTestPin =
        '[pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\test_mongoose_v2_cutoff_boundary_pair.jl"); Bytes=' +
        $ExpectedCutoffPairTestBytes + '; Sha256="' + $ExpectedCutoffPairTestSha256 +
        '"; Label="cutoff-boundary pair focused test source" },' + $newline +
        '    [pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_DESIGN.md"); Bytes=' +
        $ExpectedCutoffPairDesignBytes + '; Sha256="' + $ExpectedCutoffPairDesignSha256 +
        '"; Label="cutoff-boundary pair design" },'
    $source = Replace-ExactCount -Text $source -Old $integrationTestPin `
        -New ($integrationTestPin + $newline + '    ' + $cutoffTestPin) `
        -Label "cutoff test and design pins"
    $source = Replace-ExactCount -Text $source `
        -Old '[pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\smoke_mongoose_v2_real_teacher.jl"); Bytes=26098; Sha256="47731cce38326de37f7fa3bafe4c76e1def447096518c9a94a9d6b1489795bf3"; Label="real-teacher two-update smoke driver" }' `
        -New ('[pscustomobject]@{ Path=(Join-Path $experimentRoot "sparse_dynamic_3layer\smoke_mongoose_v2_real_teacher.jl"); Bytes=' +
            $ExpectedCutoffSmokeDriverBytes + '; Sha256="' +
            $ExpectedCutoffSmokeDriverSha256 +
            '"; Label="cutoff-pair real-teacher two-update smoke driver" }') `
        -Label "cutoff smoke driver pin"

    $source = Replace-ExactCount -Text $source `
        -Old '$SmokeRoot = "C:\tmp\tetris_mongoose_v2_real_teacher_smoke_20260719T223000"' `
        -New ('$SmokeRoot = ' + (Quote-PowerShellLiteral -Value $CutoffSmokeRoot)) `
        -Label "cutoff smoke root"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedSmokeSummaryBytes = 5018' `
        -New ('$ExpectedSmokeSummaryBytes = ' + $ExpectedCutoffSmokeSummaryBytes) `
        -Label "cutoff smoke summary bytes"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedSmokeSummarySha256 = "197aded6320d07c97c3bf330f4513a62bfe921a3e4077509da927d23b8a795bb"' `
        -New ('$ExpectedSmokeSummarySha256 = "' + $ExpectedCutoffSmokeSummarySha256 + '"') `
        -Label "cutoff smoke summary digest"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedSmokeSourceClosureSha256 = "752f66fe8738561471dde37a3507bf2e4eec459541dab8996fb5f0f220bb7a2a"' `
        -New ('$ExpectedSmokeSourceClosureSha256 = "' +
            $ExpectedCutoffSmokeSourceClosureSha256 + '"') `
        -Label "cutoff smoke source closure"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedSmokeCheckpointBytes = 249870121' `
        -New ('$ExpectedSmokeCheckpointBytes = ' + $ExpectedCutoffSmokeCheckpointBytes) `
        -Label "cutoff smoke checkpoint bytes"
    $source = Replace-ExactCount -Text $source `
        -Old '$ExpectedSmokeCheckpointSha256 = "8b15558fa06b8f6ae2572079badcbf280c8b77c9186ae99f332e9a88d43e03d9"' `
        -New ('$ExpectedSmokeCheckpointSha256 = "' +
            $ExpectedCutoffSmokeCheckpointSha256 + '"') `
        -Label "cutoff smoke checkpoint digest"
    $source = Replace-ExactCount -Text $source `
        -Old '"mongoose-v2-real-teacher-two-update-smoke-v1"' `
        -New '"mongoose-v2-real-teacher-two-update-smoke-cutoff-v1"' `
        -Label "cutoff smoke schema"
    $source = Replace-ExactCount -Text $source `
        -Old '"47731cce38326de37f7fa3bafe4c76e1def447096518c9a94a9d6b1489795bf3"' `
        -New ('"' + $ExpectedCutoffSmokeDriverSha256 + '"') -ExpectedCount 1 `
        -Label "cutoff smoke driver semantic digest"

    $pairTelemetryFunction = @'
function Assert-CutoffBoundaryPairTelemetry {
    param($Pair, [int]$Update)
    if ($null -eq $Pair) {
        throw "Cutoff-boundary pair telemetry is missing at update $Update"
    }
    foreach ($name in @(
        "pair_mining_mode", "positive_ids", "negative_ids",
        "positive_scores", "negative_scores", "cutoff_score_gaps",
        "positive_collisions", "negative_collisions",
        "positive_ranks", "negative_ranks", "positive_in_exploitation",
        "negative_in_exploitation", "exploitation_targets",
        "eligible_witness_rows", "retrieved_rows", "scored_rows",
        "losses", "positive_logits", "negative_logits", "pair_macs",
        "parameter_touches", "column_normalization"
    )) {
        if (-not ($Pair.PSObject.Properties.Name -contains $name)) {
            throw "Cutoff-boundary pair field '$name' is missing at update $Update"
        }
    }
    if ([string]$Pair.pair_mining_mode -cne $PairMiningMode) {
        throw "Cutoff-boundary pair-mining mode changed at update $Update"
    }
    Assert-ExactIntArray -Value $Pair.exploitation_targets -Expected $ExploitationTargets -Label "cutoff exploitation targets at update $Update"
    Assert-ExactIntArray -Value $Pair.positive_ranks -Expected $PositiveRanks -Label "cutoff positive ranks at update $Update"
    Assert-ExactIntArray -Value $Pair.negative_ranks -Expected $NegativeRanks -Label "cutoff negative ranks at update $Update"
    foreach ($name in @(
        "positive_ids", "negative_ids", "positive_scores", "negative_scores",
        "cutoff_score_gaps", "positive_collisions", "negative_collisions",
        "positive_in_exploitation",
        "negative_in_exploitation", "eligible_witness_rows",
        "retrieved_rows", "scored_rows", "losses", "positive_logits",
        "negative_logits", "pair_macs"
    )) {
        if (@($Pair.$name).Count -ne 3) {
            throw "Cutoff-boundary pair field '$name' is not three layers at update $Update"
        }
    }
    Assert-ExactIntArray -Value $Pair.pair_macs -Expected @(6300, 6300, 6300) -Label "one-pair-per-layer BCE MACs at update $Update"
    if ([int64]$Pair.parameter_touches -ne 2688 -or
        [bool]$Pair.column_normalization) {
        throw "Cutoff-boundary pair BCE parameter contract changed at update $Update"
    }
    for ($layer = 0; $layer -lt 3; $layer += 1) {
        $positiveId = [int64]$Pair.positive_ids[$layer]
        $negativeId = [int64]$Pair.negative_ids[$layer]
        $positiveScore = [double]$Pair.positive_scores[$layer]
        $negativeScore = [double]$Pair.negative_scores[$layer]
        $gap = [double]$Pair.cutoff_score_gaps[$layer]
        $positiveCollisions = [int64]$Pair.positive_collisions[$layer]
        $negativeCollisions = [int64]$Pair.negative_collisions[$layer]
        $eligible = [int64]$Pair.eligible_witness_rows[$layer]
        $retrieved = [int64]$Pair.retrieved_rows[$layer]
        $scored = [int64]$Pair.scored_rows[$layer]
        $loss = [double]$Pair.losses[$layer]
        $positiveLogit = [double]$Pair.positive_logits[$layer]
        $negativeLogit = [double]$Pair.negative_logits[$layer]
        if ($positiveId -lt 1 -or $negativeId -lt 1 -or $positiveId -eq $negativeId) {
            throw "Cutoff-boundary pair IDs are invalid at update $Update layer $($layer + 1)"
        }
        if ($positiveCollisions -lt 1 -or $negativeCollisions -lt 1) {
            throw "Cutoff-boundary pair includes a collision-zero fill/probe at update $Update layer $($layer + 1)"
        }
        if (-not [bool]$Pair.positive_in_exploitation[$layer] -or
            [bool]$Pair.negative_in_exploitation[$layer]) {
            throw "Cutoff-boundary exploitation membership changed at update $Update layer $($layer + 1)"
        }
        if ($eligible -lt $NegativeRanks[$layer] -or
            $eligible -gt $retrieved -or $eligible -gt $scored) {
            throw "Cutoff-boundary eligible-row accounting is invalid at update $Update layer $($layer + 1)"
        }
        if (-not [double]::IsFinite($positiveScore) -or
            -not [double]::IsFinite($negativeScore) -or
            -not [double]::IsFinite($gap) -or $gap -lt 0.0 -or
            $positiveScore -lt $negativeScore) {
            throw "Cutoff-boundary raw-score ordering is invalid at update $Update layer $($layer + 1)"
        }
        if ($positiveScore -eq $negativeScore -and $positiveId -gt $negativeId) {
            throw "Cutoff-boundary equal-score stable-ID order changed at update $Update layer $($layer + 1)"
        }
        if (-not [double]::IsFinite($loss) -or $loss -lt 0.0 -or
            -not [double]::IsFinite($positiveLogit) -or
            -not [double]::IsFinite($negativeLogit)) {
            throw "Cutoff-boundary one-pair BCE telemetry is invalid at update $Update layer $($layer + 1)"
        }
        $positiveBce = [Math]::Max($positiveLogit, 0.0) - $positiveLogit +
            [Math]::Log(1.0 + [Math]::Exp(-[Math]::Abs($positiveLogit)))
        $negativeBce = [Math]::Max($negativeLogit, 0.0) +
            [Math]::Log(1.0 + [Math]::Exp(-[Math]::Abs($negativeLogit)))
        $recomputedLoss = 0.5 * ($positiveBce + $negativeBce)
        $lossTolerance = 2.0e-6 * [Math]::Max(1.0, [Math]::Abs($recomputedLoss))
        if ([Math]::Abs($loss - $recomputedLoss) -gt $lossTolerance) {
            throw "Cutoff-boundary BCE loss/logit telemetry does not close at update $Update layer $($layer + 1)"
        }
        $recomputed = $positiveScore - $negativeScore
        $tolerance = 1.0e-10 * [Math]::Max(1.0, [Math]::Max([Math]::Abs($gap), [Math]::Abs($recomputed)))
        if ([Math]::Abs($gap - $recomputed) -gt $tolerance) {
            throw "Cutoff-boundary score-gap telemetry does not close at update $Update layer $($layer + 1)"
        }
    }
}

'@
    $source = Replace-ExactCount -Text $source -Old 'function Assert-V2Metric {' `
        -New ($pairTelemetryFunction + 'function Assert-V2Metric {') `
        -Label "cutoff telemetry validator"

    $routerAnchor = @'
    if ($null -eq $router -or [string]$router.configured_mode -cne $RoutingMode) {
        throw "MONGOOSE-v2 router state/configured mode is missing"
    }
    if ($update -eq 0) {
'@
    $routerGate = @'
    if ($null -eq $router -or [string]$router.configured_mode -cne $RoutingMode) {
        throw "MONGOOSE-v2 router state/configured mode is missing"
    }
    if ([string]$Metric.mongoose_pair_mining_mode -cne $PairMiningMode -or
        [string]$router.pair_mining_mode -cne $PairMiningMode -or
        [int]$router.mongoose_pair_mining_version -ne $PairMiningVersion -or
        [string]$router.mongoose_pair_mining_identity -cne $PairMiningIdentity -or
        [string]$router.mongoose_pair_mining_score_order -cne $PairMiningScoreOrder -or
        [string]$router.mongoose_pair_mining_eligibility -cne $PairMiningEligibility -or
        [string]$router.mongoose_pair_mining_exclusion -cne $PairMiningExclusion -or
        [string]$router.mongoose_pair_mining_outside_policy -cne $PairMiningOutsidePolicy -or
        -not [bool]$router.mongoose_pair_mining_no_bank_scan -or
        -not [bool]$router.mongoose_pair_mining_fail_closed) {
        throw "Cutoff-boundary pair policy identity changed at update $update"
    }
    Assert-ExactIntArray -Value $router.mongoose_pair_mining_exploitation_targets -Expected $ExploitationTargets -Label "cutoff router exploitation targets at update $update"
    if ($update -eq 0) {
'@
    $source = Replace-ExactCount -Text $source -Old $routerAnchor -New $routerGate `
        -Label "cutoff router policy gate"

    $pairAnchor = @'
    if ($null -eq $accounting.mongoose_pair -or
        [int64]$accounting.mongoose_pair.parameter_touches -ne 2688) {
        throw "MONGOOSE-v2 router update escaped its projection parameters"
    }
    if ($update -eq 2000) {
'@
    $pairGate = @'
    if ($null -eq $accounting.mongoose_pair -or
        [int64]$accounting.mongoose_pair.parameter_touches -ne 2688) {
        throw "MONGOOSE-v2 router update escaped its projection parameters"
    }
    Assert-CutoffBoundaryPairTelemetry -Pair $accounting.mongoose_pair -Update $update
    if ($update -eq 2000) {
'@
    $source = Replace-ExactCount -Text $source -Old $pairAnchor -New $pairGate `
        -Label "cutoff per-step pair gate"

    $parentPinAnchor = @'
[void](Assert-PinnedFile -Path $ParentCheckpointPath -ExpectedBytes $ParentCheckpointBytes -ExpectedSha256 $ParentCheckpointSha256 -Label "fixed-WTA parent checkpoint")
'@
    $comparisonPins = $parentPinAnchor + $newline +
        '[void](Assert-PinnedFile -Path ' + (Quote-PowerShellLiteral -Value $WitnessPairMetricsPath) +
        ' -ExpectedBytes ' + $WitnessPairMetricsBytes + ' -ExpectedSha256 "' +
        $WitnessPairMetricsSha256 + '" -Label "easy-extrema v2 comparison metrics")' + $newline +
        '[void](Assert-PinnedFile -Path ' + (Quote-PowerShellLiteral -Value $WitnessPairCheckpointPath) +
        ' -ExpectedBytes ' + $WitnessPairCheckpointBytes + ' -ExpectedSha256 "' +
        $WitnessPairCheckpointSha256 + '" -Label "easy-extrema v2 comparison checkpoint")'
    $source = Replace-ExactCount -Text $source -Old $parentPinAnchor `
        -New $comparisonPins -Label "comparison telemetry pins"

    $smokeAnchor = @'
Assert-ExactIntArray -Value $smoke.training_probes -Expected $TrainingProbes -Label "MONGOOSE-v2 smoke probe widths"
'@
    $smokeGate = $smokeAnchor + $newline + @'
if ([string]$smoke.pair_mining_mode -cne $PairMiningMode) {
    throw "Cutoff-pair real-teacher smoke pair-mining identity changed"
}
Assert-CutoffBoundaryPairTelemetry -Pair $smoke.update2_pair_telemetry -Update 2
'@
    $source = Replace-ExactCount -Text $source -Old $smokeAnchor -New $smokeGate `
        -Label "cutoff smoke pair gate"

    $source = Replace-ExactCount -Text $source `
        -Old '$env:BEAT_3L_ROUTING_MODE = $RoutingMode' `
        -New ('$env:BEAT_3L_ROUTING_MODE = $RoutingMode' + $newline +
            '$env:BEAT_3L_MONGOOSE_PAIR_MINING_MODE = $PairMiningMode') `
        -Label "cutoff pair environment"

    $gateAnchor = @'
        routing_policy = $RoutingPolicy
        fixed_active_counts = $ActiveCounts
'@
    $gateFields = @'
        routing_policy = $RoutingPolicy
        pair_mining_mode = $PairMiningMode
        pair_mining_identity = $PairMiningIdentity
        pair_mining_exploitation_targets = $ExploitationTargets
        final_pair_positive_ranks = @($final.last_step.accounting.mongoose_pair.positive_ranks)
        final_pair_negative_ranks = @($final.last_step.accounting.mongoose_pair.negative_ranks)
        final_pair_eligible_witness_rows = @($final.last_step.accounting.mongoose_pair.eligible_witness_rows)
        final_pair_cutoff_score_gaps = @($final.last_step.accounting.mongoose_pair.cutoff_score_gaps)
        easy_extrema_v2_comparison_metrics_path = $WitnessPairMetricsPath
        easy_extrema_v2_comparison_metrics_sha256 = $WitnessPairMetricsSha256
        easy_extrema_v2_comparison_is_gate = $false
        fixed_active_counts = $ActiveCounts
'@
    $source = Replace-ExactCount -Text $source -Old $gateAnchor -New $gateFields `
        -Label "cutoff gate publication"

    return $source
}

if ($ValidateOnly -and $StaticParseOnly) {
    throw "ValidateOnly and StaticParseOnly are mutually exclusive"
}

[void](Assert-PinnedTemplateFile -Path $templateControllerPath `
    -Bytes $TemplateControllerBytes -Sha256 $TemplateControllerSha256 `
    -Label "final witness-pair controller template")
[void](Assert-PinnedTemplateFile -Path $templateContractPath `
    -Bytes $TemplateContractBytes -Sha256 $TemplateContractSha256 `
    -Label "final witness-pair contract provenance")

$derivedSource = Build-DerivedControllerSource
$tokens = $null
$parseErrors = $null
[void][Management.Automation.Language.Parser]::ParseInput(
    $derivedSource,
    [ref]$tokens,
    [ref]$parseErrors
)
if (@($parseErrors).Count -ne 0) {
    $messages = @($parseErrors | ForEach-Object { $_.Message }) -join '; '
    throw "Derived cutoff-pair controller does not parse: $messages"
}

if ($StaticParseOnly) {
    [pscustomobject]@{
        static_parse = $true
        mutation = $false
        launched = $false
        experiment_id = $ExperimentId
        pair_mining_mode = $PairMiningMode
        template_controller_sha256 = $TemplateControllerSha256
        template_contract_sha256 = $TemplateContractSha256
        freeze_resolved = $true
        note = "Static parse only; authoritative source/test/smoke bindings are frozen"
    }
    return
}

# The explicit cutoff-specific freeze gate runs before the derived controller's
# own inherited pin, path, process, reservation, or output logic.
Assert-CutoffFreezeResolved

# The inherited controller intentionally references its top-level identity via
# `$script:`.  A dynamically created scriptblock executes in a child scope, so
# seed those three script-scope witnesses explicitly before invocation.  The
# inherited controller independently recomputes the same values and continues
# to use its unchanged mid-run integrity checks.
$controllerSelfItem = Get-Item -LiteralPath $controllerPath -Force
if (($controllerSelfItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Cutoff-pair controller is a reparse point"
}
$script:controllerPath = $controllerPath
$script:controllerInitialBytes = [int64]$controllerSelfItem.Length
$script:controllerInitialSha256 = Get-LowerSha256 -Path $controllerPath

$derivedController = [scriptblock]::Create($derivedSource)
& $derivedController -ValidateOnly:$ValidateOnly.IsPresent
