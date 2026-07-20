[CmdletBinding()]
param(
    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_dynamic_k64_k256_teacher_20000_v1",
    [string]$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe",
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This controller deliberately exposes no scientific switches. `ValidateOnly`
# is a read-only preflight: it neither creates the output nor takes the lease nor
# launches Julia. The execution path authorizes one fresh dynamic-k64/k256 run,
# never resume/retry/rescue, and never selects a checkpoint after its score.
$MaximumUpdates = 20000
$EvaluationInterval = 2000
$CheckpointInterval = 2000
$WallTimeLimitMinutes = 90
$Variant = "k128"
$RoutingMode = "dynamic_k64_k256_v1"
$RoutingPolicy = "fixed-wta-state-margin-dynamic-k64-k256-v1"
$ExpectedContractBytes = 8861
$ExpectedContractSha256 = "63a6150fa2157e5f124d5095ff8ed80e17b703aa558c86679c5c786020f9d48b"
$ExpectedManifestBytes = 1268581
$ExpectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"
$ExpectedSourceClosureSha256 = "c1b4862a95ed893ccb1a75a6f3f2aac9b4b265606fc5171ab7bdcbaaff2c01af"
$ExpectedMetricMarginWeight = [double][single]0.15
$ExpectedFloat32Threshold = [double][single]0.02
$ExpansionCap = 0.285052563012286
$CoreMacBudget = [int64]141520

function Get-Sha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)]$Value,
        [int]$Depth = 12
    )
    $temporary = "$Path.pending.$PID"
    try {
        $Value | ConvertTo-Json -Depth $Depth |
            Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Assert-FiniteJsonValue {
    param($Value, [string]$Path)
    if ($null -eq $Value) { return }
    if ($Value -is [double] -or $Value -is [single]) {
        if ([double]::IsNaN([double]$Value) -or
            [double]::IsInfinity([double]$Value)) {
            throw "Non-finite value at $Path"
        }
        return
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        foreach ($property in $Value.PSObject.Properties) {
            Assert-FiniteJsonValue -Value $property.Value `
                -Path "$Path.$($property.Name)"
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable] -and
        $Value -isnot [string]) {
        $index = 0
        foreach ($item in $Value) {
            Assert-FiniteJsonValue -Value $item -Path "$Path[$index]"
            $index += 1
        }
    }
}

function Assert-PinnedFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][int64]$ExpectedBytes,
        [Parameter(Mandatory)][string]$ExpectedSha256,
        [Parameter(Mandatory)][string]$Label
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
    $sha256 = Get-Sha256 -Path $item.FullName
    if ($sha256 -cne $ExpectedSha256) {
        throw "Pinned $Label SHA-256 changed: $sha256"
    }
    return $item.FullName
}

function Assert-NoReparseAncestry {
    param([Parameter(Mandatory)][string]$Path)
    $item = Get-Item -LiteralPath $Path -Force
    while ($null -ne $item) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Pinned path traverses a reparse point: $($item.FullName)"
        }
        if ($item -isnot [IO.DirectoryInfo]) { $item = $item.Directory }
        else { $item = $item.Parent }
    }
}

function Assert-RunLeaf {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root
    )
    $resolved = [IO.Path]::GetFullPath($Path)
    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $prefix = $rootPath + '\'
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
    $cursor = $item.Directory
    while ($null -ne $cursor) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Run artifact traverses a reparse point: $($cursor.FullName)"
        }
        if ($cursor.FullName.Equals(
                $rootPath, [StringComparison]::OrdinalIgnoreCase)) { break }
        if (-not $cursor.FullName.StartsWith(
                $prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Run artifact ancestry escapes the fresh output root: $resolved"
        }
        $cursor = $cursor.Parent
    }
    return $resolved
}

function Get-SourceClosureSha256 {
    param([Parameter(Mandatory)][string[]]$Paths)
    $hasher = [Security.Cryptography.IncrementalHash]::CreateHash(
        [Security.Cryptography.HashAlgorithmName]::SHA256
    )
    try {
        foreach ($path in @($Paths | Sort-Object -CaseSensitive)) {
            $full = [IO.Path]::GetFullPath($path)
            $hasher.AppendData([Text.Encoding]::UTF8.GetBytes($full))
            $hasher.AppendData([IO.File]::ReadAllBytes($full))
        }
        return [Convert]::ToHexString($hasher.GetHashAndReset()).ToLowerInvariant()
    } finally {
        $hasher.Dispose()
    }
}

function Get-MetricRecords {
    param(
        [Parameter(Mandatory)][string]$Root,
        [switch]$RequireComplete
    )
    $files = @(Get-ChildItem -LiteralPath $Root -Recurse -File `
        -Filter "metrics.jsonl" -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return @() }
    if ($files.Count -ne 1) {
        throw "Expected at most one metrics.jsonl while running; got $($files.Count)"
    }
    try {
        $metricsPath = Assert-RunLeaf -Path $files[0].FullName -Root $Root
    } catch {
        if ($RequireComplete) { throw }
        return @()
    }
    $stream = $null
    try {
        $stream = [IO.FileStream]::new(
            $metricsPath,
            [IO.FileMode]::Open,
            [IO.FileAccess]::Read,
            [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
        )
        $memory = [IO.MemoryStream]::new()
        try {
            $stream.CopyTo($memory)
            $bytes = $memory.ToArray()
        } finally {
            $memory.Dispose()
        }
    } catch {
        if ($RequireComplete) { throw }
        return @()
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
    }
    $utf8 = [Text.UTF8Encoding]::new($false, $true)
    try {
        $text = $utf8.GetString($bytes)
    } catch {
        if ($RequireComplete) { throw }
        return @()
    }
    if ($RequireComplete -and $text.Length -gt 0 -and
        -not $text.EndsWith("`n", [StringComparison]::Ordinal)) {
        throw "Final metrics.jsonl is not newline-terminated"
    }
    $lastNewline = $text.LastIndexOf("`n", [StringComparison]::Ordinal)
    if ($lastNewline -lt 0) {
        if ($RequireComplete -and $text.Length -gt 0) {
            throw "Final metrics.jsonl contains no complete record"
        }
        return @()
    }
    # While Julia owns the file, bytes after the last newline are an
    # uncommitted append and are deliberately ignored. A complete newline is
    # the publication boundary for each JSON record.
    $completeText = $text.Substring(0, $lastNewline + 1)
    $records = @()
    foreach ($lineWithCarriageReturn in @($completeText -split "`n")) {
        $line = $lineWithCarriageReturn.TrimEnd("`r")
        if ($line.Length -eq 0) { continue }
        $records += ($line | ConvertFrom-Json -ErrorAction Stop)
    }
    return @($records)
}

function Assert-DynamicMetric {
    param([Parameter(Mandatory)]$Metric)
    Assert-FiniteJsonValue -Value $Metric -Path "metric[$($Metric.update)]"
    $update = [int]$Metric.update
    if ($update -lt 2000 -or $update % 2000 -ne 0 -or $update -gt 20000) {
        throw "Unexpected dynamic-k metric update $update"
    }
    if ([string]$Metric.variant -cne "k128" -or
        [string]$Metric.routing_policy -cne $script:RoutingPolicy -or
        [double]$Metric.objective_margin_weight -ne $script:ExpectedMetricMarginWeight -or
        [string]$Metric.objective_margin_mode -cne "fixed_teacher_top2" -or
        [int]$Metric.learner_width -ne 80 -or
        [int]$Metric.observed_max_candidates -ne 76) {
        throw "Dynamic-k metric identity changed at update $update"
    }
    $parameter = $Metric.parameter_contract
    if ($null -eq $parameter -or [int64]$parameter.total_parameters -ne 19924022 -or
        (@($parameter.active_counts) -join ',') -cne '48,40,40' -or
        [string]$parameter.routing_policy -cne 'wta-collision-count-stable-id-prefilter-v1' -or
        [string]$parameter.active_width_policy -cne $script:RoutingMode -or
        (@($parameter.scout_active_counts) -join ',') -cne '24,20,20' -or
        (@($parameter.expanded_active_counts) -join ',') -cne '96,80,80' -or
        [int64]$parameter.scout_forward_macs_per_candidate -ne 32020 -or
        [int64]$parameter.clear_training_macs_per_candidate -ne 78504 -or
        [int64]$parameter.expanded_training_macs_per_candidate -ne 299572 -or
        [int64]$parameter.k128_budget_macs_per_candidate -ne 141520) {
        throw "Dynamic-k parameter/serialized-owner contract changed at update $update"
    }

    $state = $Metric.router_state
    if ($null -eq $state -or [string]$state.mode -cne $script:RoutingMode -or
        [string]$state.routing_policy -cne $script:RoutingPolicy -or
        [double]$state.threshold -ne $script:ExpectedFloat32Threshold -or
        (@($state.scout_counts) -join ',') -cne '24,20,20' -or
        (@($state.expanded_counts) -join ',') -cne '96,80,80' -or
        (@($state.scout_training_probes) -join ',') -cne '3,2,2' -or
        (@($state.expanded_training_probes) -join ',') -cne '12,10,10' -or
        [int64]$state.states_total -ne $update) {
        throw "Dynamic-k cumulative policy identity changed at update $update"
    }
    $total = [int64]$state.candidates_total
    $expanded = [int64]$state.candidates_expanded
    $coreMacs = [int64]$state.core_training_macs
    if ($total -le 0 -or $expanded -lt 0 -or $expanded -gt $total) {
        throw "Dynamic-k cumulative candidate counts are invalid at update $update"
    }
    $fraction = [double]$expanded / [double]$total
    $meanCore = [double]$coreMacs / [double]$total
    if ([math]::Abs(
            [double]$state.candidate_weighted_expansion_fraction - $fraction) -gt 1.0e-12 -or
        [math]::Abs(
            [double]$state.mean_core_training_macs_per_candidate - $meanCore) -gt 1.0e-9) {
        throw "Dynamic-k published cumulative ratios are inconsistent at update $update"
    }
    if ($fraction -gt $script:ExpansionCap -or
        [double]$state.candidate_weighted_expansion_cap -ne $script:ExpansionCap -or
        $meanCore -gt [double]$script:CoreMacBudget -or
        [double]$state.mean_core_training_macs_cap -ne [double]$script:CoreMacBudget -or
        $coreMacs -gt $script:CoreMacBudget * $total) {
        throw "Dynamic-k selected-core compute gate failed at update $update"
    }

    $accounting = $Metric.last_step.accounting
    $dynamic = $accounting.dynamic_k
    if ($null -eq $accounting -or $null -eq $dynamic -or
        [string]$accounting.routing_mode -cne $script:RoutingMode -or
        [int64]$accounting.total_parameters -ne 19924022 -or
        $null -ne $accounting.mongoose_pair -or
        [int64]$accounting.mongoose_trainable_parameters -ne 0 -or
        [int64]$accounting.mongoose_active_parameter_touches -ne 0) {
        throw "Dynamic-k update accounting escaped its frozen model at update $update"
    }
    $candidateCount = [int64]$dynamic.candidate_count
    $expandedState = [bool]$dynamic.expanded
    $expectedCounts = if ($expandedState) { '96,80,80' } else { '24,20,20' }
    $expectedCore = $candidateCount * $(if ($expandedState) { 299572 } else { 78504 })
    $expectedChosen = $candidateCount * $(if ($expandedState) { 267552 } else { 46484 })
    if ($candidateCount -ne [int64]$accounting.valid_candidates -or
        (@($dynamic.scout_active_counts) -join ',') -cne '24,20,20' -or
        (@($dynamic.chosen_active_counts) -join ',') -cne $expectedCounts -or
        [int64]$dynamic.scout_forward_macs -ne 32020 * $candidateCount -or
        [int64]$dynamic.chosen_training_macs -ne $expectedChosen -or
        [int64]$dynamic.core_training_macs -ne $expectedCore -or
        [int64]$dynamic.candidate_weighted_expansion_denominator -ne $candidateCount -or
        [int64]$dynamic.candidate_weighted_expansion_numerator -ne
            $(if ($expandedState) { $candidateCount } else { 0 })) {
        throw "Dynamic-k per-state selected-core accounting changed at update $update"
    }
    $cumulative = $dynamic.cumulative
    if ([int64]$cumulative.states_total -ne $update -or
        [int64]$cumulative.candidates_total -ne $total -or
        [int64]$cumulative.candidates_expanded -ne $expanded -or
        [int64]$cumulative.core_training_macs -ne $coreMacs) {
        throw "Dynamic-k per-step/cumulative snapshots disagree at update $update"
    }

    $rowDimensions = @(560, 321, 321)
    if (@($accounting.optimizer).Count -ne 3) {
        throw "Dynamic-k sparse optimizer telemetry does not contain three layers"
    }
    for ($layer = 0; $layer -lt 3; $layer += 1) {
        $optimizer = $accounting.optimizer[$layer]
        if ([int64]$optimizer.global_step -ne $update -or
            [int64]$optimizer.active_rows -ne [int64]$accounting.unique_active_rows[$layer] -or
            [int64]$optimizer.active_elements -ne
                [int64]$optimizer.active_rows * $rowDimensions[$layer] -or
            [int64]$optimizer.dirty_route_rows -lt 0 -or
            [int64]$optimizer.dirty_route_rows -gt [int64]$optimizer.active_rows -or
            [int64]$optimizer.theta_elements_read -ne [int64]$optimizer.active_elements -or
            [int64]$optimizer.theta_elements_written -ne [int64]$optimizer.active_elements -or
            [int64]$optimizer.moment_elements_read -ne
                2 * [int64]$optimizer.active_elements -or
            [int64]$optimizer.moment_elements_written -ne
                2 * [int64]$optimizer.active_elements) {
            throw "Dynamic-k layer $($layer + 1) active-only optimizer gate failed at update $update"
        }
    }
    # `executed_training_macs` intentionally includes irregular WTA reranking.
    # It is recorded but never substituted for the preregistered core-selected
    # compute budget above.
}

function Get-PollMilliseconds {
    param(
        [int]$CurrentUpdate,
        [double]$UpdatesPerSecond
    )
    $nextMilestone = if ($CurrentUpdate -lt 10000) { 10000 } else { 20000 }
    if ($UpdatesPerSecond -le 0.0) {
        return $(if ($CurrentUpdate -eq 0) { 2000 } else { 5000 })
    }
    $seconds = ($nextMilestone - $CurrentUpdate) / $UpdatesPerSecond
    if ($seconds -le 30.0) { return 500 }
    if ($seconds -le 120.0) { return 1000 }
    if ($seconds -le 600.0) { return 3000 }
    return 10000
}

function Stop-OwnedProcess {
    param(
        [System.Diagnostics.Process]$Process,
        [int64]$ExpectedStartTicks
    )
    if ($null -eq $Process) { return }
    try {
        if ($Process.HasExited) { return }
        $live = Get-Process -Id $Process.Id -ErrorAction Stop
        if ($live.StartTime.ToUniversalTime().Ticks -ne $ExpectedStartTicks) {
            throw "Owned PID identity changed; refusing to terminate PID $($Process.Id)"
        }
        $live.Kill()
        $live.WaitForExit()
    } catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
        return
    }
}

$controllerPath = [IO.Path]::GetFullPath($PSCommandPath)
$controllerItem = Get-Item -LiteralPath $controllerPath -Force
if (($controllerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Dynamic-k controller is a reparse point"
}
$controllerInitialBytes = [int64]$controllerItem.Length
$controllerInitialSha256 = Get-Sha256 -Path $controllerPath
function Assert-ControllerUnchanged {
    [void](Assert-PinnedFile -Path $script:controllerPath `
        -ExpectedBytes $script:controllerInitialBytes `
        -ExpectedSha256 $script:controllerInitialSha256 `
        -Label "dynamic-k controller")
}

$comparisonRoot = $PSScriptRoot
$experimentRoot = Split-Path -Parent $comparisonRoot
$contractPath = Join-Path $comparisonRoot `
    "three_layer_dynamic_k64_k256_20000_update_contract.toml"
$entrypoint = Join-Path $experimentRoot `
    "sparse_dynamic_3layer\train_teacher_supervised.jl"
$manifestPath = Join-Path $Dataset "manifest.json"
$expectedDataset = [IO.Path]::GetFullPath(
    "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
)
$expectedJulia = [IO.Path]::GetFullPath(
    "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)
$expectedOutput = [IO.Path]::GetFullPath(
    "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_dynamic_k64_k256_teacher_20000_v1"
)
$allowedRunParent = [IO.Path]::GetFullPath(
    "D:\tetris-paper-plus\runs\beat_first_v1"
)
if ([IO.Path]::GetFullPath($Dataset) -cne $expectedDataset -or
    [IO.Path]::GetFullPath($Julia) -cne $expectedJulia -or
    [IO.Path]::GetFullPath($OutputRoot) -cne $expectedOutput) {
    throw "Dataset, Julia, and output must be the exact one-shot preregistered paths"
}
if (-not (Test-Path -LiteralPath $allowedRunParent -PathType Container)) {
    throw "Preregistered run parent is missing"
}
Assert-NoReparseAncestry -Path $expectedDataset
Assert-NoReparseAncestry -Path $allowedRunParent
if (Test-Path -LiteralPath $expectedOutput) {
    throw "One-shot dynamic-k output already exists; retry/rescue is forbidden"
}

$pinnedFiles = @(
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\teacher_training.jl"), 104571, "478c1adc7321474857663f9f5ed0c916bcef545f2f1f54d91d0e926f771c3e22", "teacher training source"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\train_teacher_supervised.jl"), 149, "41893978e984c0c696470aa15d73ae54eae01deba05e0db776a9269e25897b5d", "teacher entrypoint"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\SparseDynamic3Layer.jl"), 3031, "8756a5f99c0f18554985fb60a2dd565300ef7c2fb67159f64ddf41a808ef36bd", "three-layer module"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\dynamic_k64_k256.jl"), 28207, "2ff6d17c59cc616caa90878ac3ff451fbbf8f02d61440ab49d8d1e3f6f1a5d97", "dynamic-k source"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\test_dynamic_k64_k256.jl"), 20670, "e276cc95940fb2b1c55b25d3b54acde834823ea3fede8c006f9dc840b53ac097", "dynamic-k focused test"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\mongoose_simhash_overlay.jl"), 47913, "0cbd4b31740a7573380dd0737ed3a5f9544094f428fbf18e3ef89e8c64d36416", "disabled MONGOOSE source"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\geometry.jl"), 12006, "5ec92d85be3f888554c1f2700e06a1e3dc5531d25b354677bb208eb6978d3a27", "three-layer geometry"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\model.jl"), 23135, "ff94f5771d1cbb09532348b8603019c5aa3126979dfef74ac277d223a749e7ac", "three-layer model"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\optimizer.jl"), 20897, "a5c057e015e3b9677510faaf07e3c26ca993233e09ab9f602296fbd3299b5790", "sparse optimizer"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\runtime.jl"), 38151, "9552adefc3bad185146cc1053f6358c3f40c5e5a81a76ecbb0f3b08e6bb6be4a", "three-layer runtime"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\checkpoint.jl"), 6888, "425fd756af3b1d2f029a4f5a0241e588acb563863a5da8be6131bd3e3a661340", "checkpoint source"),
    @((Join-Path $experimentRoot "sparse_dynamic\features.jl"), 9360, "0a96497f69271bd900054be72c33c13a244b7909631b75dc95183730872197b7", "feature adapter"),
    @((Join-Path $experimentRoot "sparse_dynamic\wta_index.jl"), 24011, "be270e8e36fe356be496e834edac1b3c6a13416fcd517a3e651319eb5b581ceb", "WTA index"),
    @((Join-Path $experimentRoot "training\core.jl"), 47644, "05a8592019d302e951828035c252f3a80f5567f45a533fec82ec2c6631ca2aa6", "teacher loss core"),
    @((Join-Path $experimentRoot "Project.toml"), 1221, "62bbd9b624a4e2963e0382c86c8b4ff56b865c539e41ce1e8535cc7416ae4c7a", "Project.toml"),
    @((Join-Path $experimentRoot "Manifest.toml"), 47731, "6ef9ed11797722e72e62d1d4e500da59d0bc14124f96344f93a14a322ec9a10a", "Manifest.toml"),
    @($expectedJulia, 170952, "4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043", "Julia runtime"),
    @($manifestPath, $ExpectedManifestBytes, $ExpectedManifestSha256, "teacher_v3 manifest"),
    @($contractPath, $ExpectedContractBytes, $ExpectedContractSha256, "dynamic-k preregistration contract")
)
foreach ($pin in $pinnedFiles) {
    [void](Assert-PinnedFile -Path $pin[0] -ExpectedBytes $pin[1] `
        -ExpectedSha256 $pin[2] -Label $pin[3])
}
$sourceClosurePaths = @($pinnedFiles[0..3] + $pinnedFiles[5..15] |
    ForEach-Object { [string]$_[0] })
$sourceClosureSha256 = Get-SourceClosureSha256 -Paths $sourceClosurePaths
if ($sourceClosureSha256 -cne $ExpectedSourceClosureSha256) {
    throw "Dynamic-k source closure SHA-256 changed: $sourceClosureSha256"
}
Assert-ControllerUnchanged

if ($ValidateOnly) {
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Validation refuses to pass while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
    }
    [ordered]@{
        status = "VALIDATED_READ_ONLY"
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract_path = $contractPath
        contract_bytes = $ExpectedContractBytes
        contract_sha256 = $ExpectedContractSha256
        source_closure_sha256 = $sourceClosureSha256
        dataset_manifest_sha256 = $ExpectedManifestSha256
        julia_path = $expectedJulia
        fresh_output_path = $expectedOutput
        output_absent = -not (Test-Path -LiteralPath $expectedOutput)
        julia_process_count = 0
        mutation_performed = $false
        julia_launched = $false
    } | ConvertTo-Json -Depth 4
    return
}

$resolvedOutput = $expectedOutput
$ownedJulia = $null
$ownedStartTicks = [int64]0
$held = $false
$juliaMutex = [System.Threading.Mutex]::new(
    $false, 'Local\TetrisBeatFirstV1ExclusiveJulia'
)
try {
    try { $held = $juliaMutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $held = $true }
    if (-not $held) {
        throw "Another Tetris controller owns the exclusive Julia lease"
    }
    $liveJulia = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue)
    if ($liveJulia.Count -ne 0) {
        throw "Refusing to start while Julia PID(s) are live: $(($liveJulia.Id) -join ',')"
    }

    New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
    $stdoutPath = Join-Path $resolvedOutput "controller.stdout.log"
    $stderrPath = Join-Path $resolvedOutput "controller.stderr.log"
    $statusPath = Join-Path $resolvedOutput "controller_status.json"
    $contractSha256 = Get-Sha256 -Path $contractPath
    if ($contractSha256 -cne $ExpectedContractSha256) {
        throw "Dynamic-k contract changed between preflight and launch"
    }

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
    $env:BEAT_3L_VARIANT = "k128"
    $env:BEAT_3L_ROUTING_MODE = $RoutingMode
    $env:BEAT_3L_EPOCHS = "1.0"
    $env:BEAT_3L_MAX_UPDATES = "20000"
    $env:BEAT_3L_EVAL_INTERVAL = "2000"
    $env:BEAT_3L_CHECKPOINT_INTERVAL = "2000"
    $env:BEAT_3L_TRAIN_EVAL_STATES = "128"
    $env:BEAT_3L_VALIDATION_EVAL_STATES = "512"
    $env:BEAT_3L_EVALUATE_INITIAL = "false"
    $env:BEAT_3L_VALIDATION_FRACTION = "0.20"
    $env:BEAT_3L_LR = "0.0001"
    $env:BEAT_3L_WEIGHT_DECAY = "0.0"
    $env:BEAT_3L_BETA1 = "0.9"
    $env:BEAT_3L_BETA2 = "0.999"
    $env:BEAT_3L_EPSILON = "1.0e-8"
    $env:BEAT_3L_MARGIN_WEIGHT = "0.15"
    $env:BEAT_3L_MARGIN_MODE = "fixed_teacher_top2"
    foreach ($name in @(
        'BEAT_3L_MONGOOSE_LR', 'BEAT_3L_MONGOOSE_BETA',
        'BEAT_3L_MONGOOSE_SEED', 'BEAT_3L_MONGOOSE_WARMUP_UPDATES',
        'BEAT_3L_MONGOOSE_REFRESH_INTERVAL'
    )) {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
    $env:JULIA_NUM_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"

    $ownedJulia = Start-Process -FilePath $expectedJulia `
        -ArgumentList @(
            "--startup-file=no", "--project=$experimentRoot", $entrypoint
        ) `
        -WorkingDirectory $experimentRoot -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath -PassThru
    $ownedStartTicks = $ownedJulia.StartTime.ToUniversalTime().Ticks
    Write-JsonAtomic -Path $statusPath -Value ([ordered]@{
        status = "running"
        pid = $ownedJulia.Id
        process_start_utc_ticks = $ownedStartTicks
        started_at = [DateTimeOffset]::Now.ToString("o")
        experiment_id = "sparse_3l_dynamic_k64_k256_teacher_20000_v1"
        variant = $Variant
        routing_mode = $RoutingMode
        maximum_updates = $MaximumUpdates
        evaluation_interval = $EvaluationInterval
        checkpoint_interval = $CheckpointInterval
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract_path = $contractPath
        contract_sha256 = $contractSha256
        source_closure_sha256 = $sourceClosureSha256
        dataset_manifest_sha256 = $ExpectedManifestSha256
        output_root = $resolvedOutput
    })

    $startedAt = [DateTimeOffset]::Now
    $deadline = $startedAt.AddMinutes($WallTimeLimitMinutes)
    $lastObservedUpdate = 0
    $lastObservedAt = $startedAt
    $updatesPerSecond = 0.0
    $midpointChecked = $false
    $pollMilliseconds = 2000
    while (-not $ownedJulia.WaitForExit($pollMilliseconds)) {
        if ([DateTimeOffset]::Now -gt $deadline) {
            Stop-OwnedProcess -Process $ownedJulia `
                -ExpectedStartTicks $ownedStartTicks
            throw "Dynamic-k one-shot exceeded the 90-minute wall-time limit"
        }
        $competitors = @(Get-Process -Name "julia" -ErrorAction SilentlyContinue |
            Where-Object { $_.Id -ne $ownedJulia.Id })
        if ($competitors.Count -ne 0) {
            Stop-OwnedProcess -Process $ownedJulia `
                -ExpectedStartTicks $ownedStartTicks
            throw "Competing Julia PID(s) appeared: $(($competitors.Id) -join ',')"
        }
        Assert-ControllerUnchanged
        $records = @(Get-MetricRecords -Root $resolvedOutput)
        for ($index = 0; $index -lt $records.Count; $index += 1) {
            $expectedUpdate = ($index + 1) * 2000
            if ([int]$records[$index].update -ne $expectedUpdate) {
                throw "Dynamic-k metric cadence changed at index $index"
            }
            Assert-DynamicMetric -Metric $records[$index]
        }
        if ($records.Count -gt 0) {
            $currentUpdate = [int]$records[-1].update
            if ($currentUpdate -gt $lastObservedUpdate) {
                $now = [DateTimeOffset]::Now
                $seconds = ($now - $lastObservedAt).TotalSeconds
                if ($seconds -gt 0.0) {
                    $updatesPerSecond =
                        ($currentUpdate - $lastObservedUpdate) / $seconds
                }
                $lastObservedUpdate = $currentUpdate
                $lastObservedAt = $now
            }
        }
        if (-not $midpointChecked -and $records.Count -ge 5 -and
            [int]$records[4].update -eq 10000) {
            $checkpointFiles = @(Get-ChildItem -LiteralPath $resolvedOutput `
                -Recurse -File -Filter "checkpoint_000010000.jls")
            if ($checkpointFiles.Count -eq 1) {
                $checkpoint10k = Assert-RunLeaf `
                    -Path $checkpointFiles[0].FullName -Root $resolvedOutput
                $midpointChecked = $true
                $midpoint = $records[4]
                if ([double]$midpoint.validation.top1_agreement -lt 0.166015625 -or
                    [double]$midpoint.validation.action_margin -lt 0.02) {
                    $metricFile = @(Get-ChildItem -LiteralPath $resolvedOutput `
                        -Recurse -File -Filter "metrics.jsonl")[0].FullName
                    $evidencePath = Join-Path $resolvedOutput `
                        "midpoint_gate_failure.json"
                    Write-JsonAtomic -Path $evidencePath -Value ([ordered]@{
                        status = "TERMINAL_MIDPOINT_FAILURE"
                        observed_at = [DateTimeOffset]::Now.ToString("o")
                        update = 10000
                        held_top1 = [double]$midpoint.validation.top1_agreement
                        required_top1 = 0.166015625
                        held_action_margin = [double]$midpoint.validation.action_margin
                        required_action_margin = 0.02
                        metrics_path = $metricFile
                        metrics_sha256 = Get-Sha256 -Path $metricFile
                        checkpoint_path = $checkpoint10k
                        checkpoint_bytes = (Get-Item -LiteralPath $checkpoint10k).Length
                        checkpoint_sha256 = Get-Sha256 -Path $checkpoint10k
                    })
                    Stop-OwnedProcess -Process $ownedJulia `
                        -ExpectedStartTicks $ownedStartTicks
                    throw "Dynamic-k update-10000 pruning gate failed after metric and checkpoint capture"
                }
            } elseif ($checkpointFiles.Count -gt 1) {
                throw "Multiple update-10000 checkpoints appeared"
            }
        }
        $pollMilliseconds = Get-PollMilliseconds `
            -CurrentUpdate $lastObservedUpdate `
            -UpdatesPerSecond $updatesPerSecond
    }
    if ($ownedJulia.ExitCode -ne 0) {
        throw "Dynamic-k trainer exited with code $($ownedJulia.ExitCode)"
    }
    if (-not $midpointChecked) {
        throw "Dynamic-k process completed without a live update-10000 metric/checkpoint gate"
    }

    foreach ($pin in $pinnedFiles) {
        [void](Assert-PinnedFile -Path $pin[0] -ExpectedBytes $pin[1] `
            -ExpectedSha256 $pin[2] -Label $pin[3])
    }
    if ((Get-SourceClosureSha256 -Paths $sourceClosurePaths) -cne
        $ExpectedSourceClosureSha256) {
        throw "Dynamic-k source closure changed during the run"
    }
    Assert-ControllerUnchanged

    $metricFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse `
        -File -Filter "metrics.jsonl")
    if ($metricFiles.Count -ne 1) {
        throw "Run must publish exactly one metrics.jsonl"
    }
    $metricsPath = Assert-RunLeaf -Path $metricFiles[0].FullName `
        -Root $resolvedOutput
    $metrics = @(Get-MetricRecords -Root $resolvedOutput -RequireComplete)
    if ($metrics.Count -ne 10) {
        throw "Run does not contain the exact ten evaluation records"
    }
    for ($index = 0; $index -lt 10; $index += 1) {
        if ([int]$metrics[$index].update -ne ($index + 1) * 2000) {
            throw "Final metric cadence changed at index $index"
        }
        Assert-DynamicMetric -Metric $metrics[$index]
    }
    $midpoint = $metrics[4]
    if ([double]$midpoint.validation.top1_agreement -lt 0.166015625 -or
        [double]$midpoint.validation.action_margin -lt 0.02) {
        throw "Completed process missed the update-10000 pruning gate"
    }
    $final = $metrics[9]
    if ([int]$final.throughput.updates -ne 20000 -or
        [double]$final.validation.top1_agreement -lt 0.25 -or
        [double]$final.validation.ndcg -le 0.9374244614677424 -or
        [double]$final.validation.pairwise_accuracy -le 0.7407546814419289 -or
        [double]$final.validation.action_margin -lt 0.02) {
        throw "Dynamic-k update-20000 success gate failed"
    }

    $checkpointFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse `
        -File -Filter "checkpoint_*.jls" | Sort-Object Name)
    $expectedCheckpointNames = @(1..10 | ForEach-Object {
        "checkpoint_" + (($_ * 2000).ToString().PadLeft(9, '0')) + ".jls"
    })
    if ($checkpointFiles.Count -ne 10 -or
        ($checkpointFiles.Name -join ',') -cne
            ($expectedCheckpointNames -join ',')) {
        throw "Dynamic-k checkpoint cadence is not exactly every 2000 updates"
    }
    $checkpointArtifacts = @()
    foreach ($checkpointFile in $checkpointFiles) {
        $path = Assert-RunLeaf -Path $checkpointFile.FullName `
            -Root $resolvedOutput
        $checkpointArtifacts += [ordered]@{
            path = $path
            bytes = [int64]$checkpointFile.Length
            sha256 = Get-Sha256 -Path $path
        }
    }
    $latestFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse `
        -File -Filter "latest.json")
    if ($latestFiles.Count -ne 1) {
        throw "Run must publish exactly one latest.json"
    }
    $latestPath = Assert-RunLeaf -Path $latestFiles[0].FullName `
        -Root $resolvedOutput
    $latest = Get-Content -LiteralPath $latestPath -Raw -Encoding utf8 |
        ConvertFrom-Json -ErrorAction Stop
    if ([int]$latest.update -ne 20000 -or [string]$latest.variant -cne "k128") {
        throw "Latest checkpoint identity is not update-20000 k128"
    }
    $latestCheckpoint = Assert-RunLeaf -Path ([string]$latest.path) `
        -Root $resolvedOutput
    if ([int64]$latest.bytes -ne
            (Get-Item -LiteralPath $latestCheckpoint).Length -or
        [string]$latest.sha256 -cne (Get-Sha256 -Path $latestCheckpoint)) {
        throw "Latest checkpoint bytes/SHA-256 differ from disk"
    }

    if ((Get-Item -LiteralPath $stdoutPath).Length -ne 0) {
        throw "Dynamic-k trainer stdout must remain empty"
    }
    $stderrLines = @(Get-Content -LiteralPath $stderrPath -Encoding utf8)
    foreach ($line in $stderrLines) {
        if ($line.Length -gt 0 -and $line -notmatch '^(┌ Info:|│|└)') {
            throw "Dynamic-k stderr contains an unexpected line: $line"
        }
    }
    if (@($stderrLines | Where-Object {
            $_ -eq '┌ Info: Three-layer initial teacher evaluation'
        }).Count -ne 0 -or
        @($stderrLines | Where-Object {
            $_ -eq '┌ Info: Three-layer teacher progress'
        }).Count -ne 10 -or
        @($stderrLines | Where-Object {
            $_ -eq '┌ Info: Three-layer teacher checkpoint'
        }).Count -ne 10) {
        throw "Dynamic-k stderr Info-block cadence changed"
    }

    $scientificArtifacts = @($metricsPath, $latestPath, $stdoutPath, $stderrPath) +
        @($checkpointFiles.FullName)
    $inventoryEntries = @($scientificArtifacts | ForEach-Object {
        $path = Assert-RunLeaf -Path $_ -Root $resolvedOutput
        $item = Get-Item -LiteralPath $path
        [ordered]@{
            path = $path
            bytes = [int64]$item.Length
            sha256 = Get-Sha256 -Path $path
        }
    })
    $inventoryPath = Join-Path $resolvedOutput "artifact_inventory.json"
    Write-JsonAtomic -Path $inventoryPath -Value ([ordered]@{
        format = "dynamic-k64-k256-20000-artifact-inventory-v1"
        created_at = [DateTimeOffset]::Now.ToString("o")
        controller = [ordered]@{
            path = $controllerPath
            bytes = $controllerInitialBytes
            sha256 = $controllerInitialSha256
        }
        contract = [ordered]@{
            path = $contractPath
            bytes = $ExpectedContractBytes
            sha256 = $ExpectedContractSha256
        }
        source_closure_sha256 = $ExpectedSourceClosureSha256
        dataset_manifest_sha256 = $ExpectedManifestSha256
        artifacts = $inventoryEntries
    })
    $inventorySha256 = Get-Sha256 -Path $inventoryPath

    $gatePendingPath = Join-Path $resolvedOutput "gate_status.pending.json"
    $gatePath = Join-Path $resolvedOutput "gate_status.json"
    Write-JsonAtomic -Path $gatePendingPath -Value ([ordered]@{
        status = "PASS"
        experiment_id = "sparse_3l_dynamic_k64_k256_teacher_20000_v1"
        final_update = 20000
        held_top1 = [double]$final.validation.top1_agreement
        held_ndcg = [double]$final.validation.ndcg
        held_pairwise = [double]$final.validation.pairwise_accuracy
        held_action_margin = [double]$final.validation.action_margin
        candidate_weighted_expansion_fraction =
            [double]$final.router_state.candidate_weighted_expansion_fraction
        mean_core_training_macs_per_candidate =
            [double]$final.router_state.mean_core_training_macs_per_candidate
        cumulative_core_training_macs =
            [int64]$final.router_state.core_training_macs
        cumulative_candidates = [int64]$final.router_state.candidates_total
        metrics_path = $metricsPath
        metrics_sha256 = Get-Sha256 -Path $metricsPath
        checkpoints = $checkpointArtifacts
        latest_path = $latestPath
        latest_sha256 = Get-Sha256 -Path $latestPath
        stdout_sha256 = Get-Sha256 -Path $stdoutPath
        stderr_sha256 = Get-Sha256 -Path $stderrPath
        inventory_path = $inventoryPath
        inventory_sha256 = $inventorySha256
        controller_sha256 = $controllerInitialSha256
        contract_sha256 = $ExpectedContractSha256
        source_closure_sha256 = $ExpectedSourceClosureSha256
        dataset_manifest_sha256 = $ExpectedManifestSha256
        interpretation = "teacher-ranking and selected-core dynamic-width evidence only"
    })
    $gateSha256 = Get-Sha256 -Path $gatePendingPath
    Assert-ControllerUnchanged
    Move-Item -LiteralPath $gatePendingPath -Destination $gatePath
    [void](Assert-RunLeaf -Path $gatePath -Root $resolvedOutput)
    if ((Get-Sha256 -Path $gatePath) -cne $gateSha256) {
        throw "Published gate_status.json differs from the validated pending bytes"
    }
    Write-JsonAtomic -Path $statusPath -Value ([ordered]@{
        status = "completed"
        completed_at = [DateTimeOffset]::Now.ToString("o")
        pid = $ownedJulia.Id
        process_start_utc_ticks = $ownedStartTicks
        exit_code = $ownedJulia.ExitCode
        experiment_id = "sparse_3l_dynamic_k64_k256_teacher_20000_v1"
        output_root = $resolvedOutput
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract_path = $contractPath
        contract_sha256 = $ExpectedContractSha256
        source_closure_sha256 = $ExpectedSourceClosureSha256
        dataset_manifest_sha256 = $ExpectedManifestSha256
        artifact_inventory_path = $inventoryPath
        artifact_inventory_sha256 = $inventorySha256
        gate_status_path = $gatePath
        gate_status_sha256 = $gateSha256
    })
} catch {
    $failureMessage = $_.Exception.Message
    $ownedStopError = $null
    if ($null -ne $ownedJulia -and -not $ownedJulia.HasExited) {
        try {
            Stop-OwnedProcess -Process $ownedJulia `
                -ExpectedStartTicks $ownedStartTicks
        } catch {
            $ownedStopError = $_.Exception.Message
        }
    }
    if (Test-Path -LiteralPath $resolvedOutput -PathType Container) {
        $statusPath = Join-Path $resolvedOutput "controller_status.json"
        $controllerIntegrity = "PASS"
        try { Assert-ControllerUnchanged }
        catch { $controllerIntegrity = "FAIL: $($_.Exception.Message)" }
        $logInventory = @()
        foreach ($name in @("controller.stdout.log", "controller.stderr.log")) {
            $path = Join-Path $resolvedOutput $name
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                $item = Get-Item -LiteralPath $path
                $logInventory += [ordered]@{
                    path = $path
                    bytes = [int64]$item.Length
                    sha256 = Get-Sha256 -Path $path
                }
            }
        }
        Write-JsonAtomic -Path $statusPath -Value ([ordered]@{
            status = "failed"
            failed_at = [DateTimeOffset]::Now.ToString("o")
            error = $failureMessage
            owned_process_stop_error = $ownedStopError
            pid = if ($null -eq $ownedJulia) { $null } else { $ownedJulia.Id }
            process_start_utc_ticks = $ownedStartTicks
            output_root = $resolvedOutput
            controller_path = $controllerPath
            controller_bytes = $controllerInitialBytes
            controller_sha256 = $controllerInitialSha256
            controller_integrity = $controllerIntegrity
            contract_path = $contractPath
            contract_sha256 = $ExpectedContractSha256
            source_closure_sha256 = $ExpectedSourceClosureSha256
            dataset_manifest_sha256 = $ExpectedManifestSha256
            logs = $logInventory
        })
    }
    throw
} finally {
    if ($held) {
        try { $juliaMutex.ReleaseMutex() } catch {}
    }
    $juliaMutex.Dispose()
}
