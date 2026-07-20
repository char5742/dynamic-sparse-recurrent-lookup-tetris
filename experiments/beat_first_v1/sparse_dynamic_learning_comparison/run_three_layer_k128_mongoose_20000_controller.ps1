[CmdletBinding()]
param(
    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_mongoose_teacher_signal_cpf_20000_v1",
    [string]$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This controller intentionally exposes no switches for the model, objective,
# routing mode, cadence, seeds, or resume.  It is the single preregistered
# MONGOOSE K7/L2 k128 20k pilot and fails closed on every frozen field below.
$MaximumUpdates = 20000
$Variant = "k128"
$ObjectiveProfile = "mongoose015"

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
    $observedSha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observedSha256 -cne $ExpectedSha256) {
        throw "Pinned $Label SHA-256 changed: $observedSha256"
    }
    return $item.FullName
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
$controllerPath = [IO.Path]::GetFullPath($PSCommandPath)
$controllerItem = Get-Item -LiteralPath $controllerPath -Force
if (($controllerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "MONGOOSE controller is a reparse point"
}
$controllerInitialBytes = [int64]$controllerItem.Length
$controllerInitialSha256 = (Get-FileHash -LiteralPath $controllerPath -Algorithm SHA256).Hash.ToLowerInvariant()
function Assert-ControllerUnchanged {
    [void](Assert-PinnedFile `
        -Path $script:controllerPath `
        -ExpectedBytes $script:controllerInitialBytes `
        -ExpectedSha256 $script:controllerInitialSha256 `
        -Label "MONGOOSE controller")
}
$comparisonRoot = $PSScriptRoot
$experimentRoot = Split-Path -Parent $comparisonRoot
$expectedDataset = [IO.Path]::GetFullPath("D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3")
$expectedJulia = [IO.Path]::GetFullPath("C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe")
if ([IO.Path]::GetFullPath($Dataset) -cne $expectedDataset) {
    throw "MONGOOSE dataset path is not the preregistered teacher_v3 root"
}
if ([IO.Path]::GetFullPath($Julia) -cne $expectedJulia) {
    throw "MONGOOSE Julia path is not the preregistered 1.12.6 runtime"
}

$pinnedFiles = @(
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\teacher_training.jl"), 88732, "c28b89c4fb48e70e788f809d82af46b841f8ca0199659c03295846c6fc31e2a7", "teacher training source"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\train_teacher_supervised.jl"), 149, "41893978e984c0c696470aa15d73ae54eae01deba05e0db776a9269e25897b5d", "teacher entrypoint"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\SparseDynamic3Layer.jl"), 3031, "8756a5f99c0f18554985fb60a2dd565300ef7c2fb67159f64ddf41a808ef36bd", "three-layer module"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\mongoose_simhash_overlay.jl"), 47913, "0cbd4b31740a7573380dd0737ed3a5f9544094f428fbf18e3ef89e8c64d36416", "MONGOOSE overlay"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\geometry.jl"), 12006, "5ec92d85be3f888554c1f2700e06a1e3dc5531d25b354677bb208eb6978d3a27", "three-layer geometry"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\model.jl"), 23135, "ff94f5771d1cbb09532348b8603019c5aa3126979dfef74ac277d223a749e7ac", "three-layer model"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\optimizer.jl"), 20897, "a5c057e015e3b9677510faaf07e3c26ca993233e09ab9f602296fbd3299b5790", "sparse optimizer"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\runtime.jl"), 38151, "9552adefc3bad185146cc1053f6358c3f40c5e5a81a76ecbb0f3b08e6bb6be4a", "three-layer runtime"),
    @((Join-Path $experimentRoot "sparse_dynamic_3layer\checkpoint.jl"), 6888, "425fd756af3b1d2f029a4f5a0241e588acb563863a5da8be6131bd3e3a661340", "checkpoint source"),
    @((Join-Path $experimentRoot "sparse_dynamic\features.jl"), 9360, "0a96497f69271bd900054be72c33c13a244b7909631b75dc95183730872197b7", "feature adapter"),
    @((Join-Path $experimentRoot "sparse_dynamic\wta_index.jl"), 24011, "be270e8e36fe356be496e834edac1b3c6a13416fcd517a3e651319eb5b581ceb", "fixed-WTA witness index"),
    @((Join-Path $experimentRoot "training\core.jl"), 47644, "05a8592019d302e951828035c252f3a80f5567f45a533fec82ec2c6631ca2aa6", "teacher loss core"),
    @((Join-Path $experimentRoot "Project.toml"), 1221, "62bbd9b624a4e2963e0382c86c8b4ff56b865c539e41ce1e8535cc7416ae4c7a", "Project.toml"),
    @((Join-Path $experimentRoot "Manifest.toml"), 47731, "6ef9ed11797722e72e62d1d4e500da59d0bc14124f96344f93a14a322ec9a10a", "Manifest.toml"),
    @($expectedJulia, 170952, "4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043", "Julia runtime")
)
foreach ($pin in $pinnedFiles) {
    [void](Assert-PinnedFile -Path $pin[0] -ExpectedBytes $pin[1] -ExpectedSha256 $pin[2] -Label $pin[3])
}
$marginPivot = $ObjectiveProfile -eq "margin100"
$hardNegativePivot = $ObjectiveProfile -eq "hardneg015"
$mongoosePivot = $ObjectiveProfile -eq "mongoose015"
if (($marginPivot -or $hardNegativePivot -or $mongoosePivot) -and
    ($Variant -ne "k128" -or $MaximumUpdates -ne 20000)) {
    throw "The objective pivot profiles are frozen to fresh k128 20000-update runs"
}
$contractName = if ($marginPivot) {
    "three_layer_cpf_k128_margin1_20000_update_contract.toml"
} elseif ($hardNegativePivot) {
    "three_layer_cpf_k128_hardneg_20000_update_contract.toml"
} elseif ($mongoosePivot) {
    "three_layer_cpf_k128_mongoose_20000_update_contract.toml"
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
} elseif ($mongoosePivot) {
    "sparse_3l_k128_mongoose_teacher_signal_cpf_20000_v1"
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
$expectedContractSha256 = "90c291aca34b8ddb4d434df47889d69b27b69c3a8c203370f6f1fe4cf4510284"

foreach ($required in @($contractPath, $manifestPath, $entrypoint, $Julia)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}
[void](Assert-PinnedFile -Path $contractPath -ExpectedBytes 7861 -ExpectedSha256 $expectedContractSha256 -Label "MONGOOSE preregistration contract")
[void](Assert-PinnedFile -Path $manifestPath -ExpectedBytes 1268581 -ExpectedSha256 $expectedManifestSha256 -Label "teacher_v3 manifest")
$manifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestSha256 -ne $expectedManifestSha256) {
    throw "Teacher manifest SHA-256 differs from the frozen CPF1 contract"
}
$comparisonBaselinePath = ""
$comparisonBaselineSha256 = ""
$comparisonBaselineCheckpointPath = ""
$comparisonBaselineCheckpointSha256 = ""
if ($Variant -ne "k64") {
    if ($marginPivot) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_20000_v1\sparse_3l_20260719T135857\metrics.jsonl"
        $comparisonBaselineSha256 = "8ed7ecba6eb01e916ce85c6b7164628b3b0121696e00004c6e6d8bbb540a012d"
    } elseif ($hardNegativePivot) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1_retry1\sparse_3l_20260719T143553\metrics.jsonl"
        $comparisonBaselineSha256 = "2a5149ff47a4d16053679162b71c2ea589efdd3c5a5d5a80814eb0fdf474d597"
    } elseif ($mongoosePivot) {
        $comparisonBaselinePath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_20000_v1\sparse_3l_20260719T135857\metrics.jsonl"
        $comparisonBaselineSha256 = "8ed7ecba6eb01e916ce85c6b7164628b3b0121696e00004c6e6d8bbb540a012d"
        $comparisonBaselineCheckpointPath = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_teacher_signal_cpf_20000_v1\sparse_3l_20260719T135857\checkpoints\checkpoint_000020000.jls"
        $comparisonBaselineCheckpointSha256 = "0d642c446d89e5d06c482f4c260daad1f60b5a854055093175d17891c3136f3c"
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
    if ($mongoosePivot) {
        [void](Assert-PinnedFile -Path $comparisonBaselinePath -ExpectedBytes 70674 -ExpectedSha256 $comparisonBaselineSha256 -Label "MONGOOSE parent metrics")
        [void](Assert-PinnedFile -Path $comparisonBaselineCheckpointPath -ExpectedBytes 248680789 -ExpectedSha256 $comparisonBaselineCheckpointSha256 -Label "MONGOOSE parent checkpoint")
    }
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
$allowedRunParent = [IO.Path]::GetFullPath("D:\tetris-paper-plus\runs\beat_first_v1")
$resolvedOutputParent = [IO.Path]::GetDirectoryName($resolvedOutput)
if (-not $resolvedOutputParent.Equals($allowedRunParent, [StringComparison]::OrdinalIgnoreCase)) {
    throw "MONGOOSE output must be a direct child of the preregistered run parent"
}
if ([IO.Path]::GetFileName($resolvedOutput) -notmatch '^sparse_3l_k128_mongoose_teacher_signal_cpf_20000_v1(?:_retry[0-9]+)?$') {
    throw "MONGOOSE output leaf is not the preregistered run identity"
}
foreach ($directory in @($expectedDataset, $allowedRunParent)) {
    $item = Get-Item -LiteralPath $directory -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Pinned directory is a reparse point: $directory"
    }
}
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

    New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
    $stdoutPath = Join-Path $resolvedOutput "controller.stdout.log"
    $stderrPath = Join-Path $resolvedOutput "controller.stderr.log"
    $statusPath = Join-Path $resolvedOutput "controller_status.json"
    $contractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($contractSha256 -cne $expectedContractSha256) {
        throw "MONGOOSE contract changed between preflight and launch"
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
    $env:BEAT_3L_EPOCHS = "1.0"
    $env:BEAT_3L_MAX_UPDATES = [string]$MaximumUpdates
    $evaluationInterval = if ($MaximumUpdates -eq 20000) { 2000 } else { 500 }
    $env:BEAT_3L_EVAL_INTERVAL = [string]$evaluationInterval
    $env:BEAT_3L_CHECKPOINT_INTERVAL = if ($MaximumUpdates -eq 2500) { "500" } elseif ($MaximumUpdates -eq 5000) { "1000" } else { "10000" }
    $env:BEAT_3L_TRAIN_EVAL_STATES = "128"
    $env:BEAT_3L_VALIDATION_EVAL_STATES = "512"
    $env:BEAT_3L_EVALUATE_INITIAL = "true"
    $env:BEAT_3L_VALIDATION_FRACTION = "0.20"
    $env:BEAT_3L_LR = "0.0001"
    $env:BEAT_3L_WEIGHT_DECAY = "0.0"
    $env:BEAT_3L_BETA1 = "0.9"
    $env:BEAT_3L_BETA2 = "0.999"
    $env:BEAT_3L_EPSILON = "1.0e-8"
    $expectedMarginWeight = if ($marginPivot) { 1.0 } else { 0.15 }
    # The trainer intentionally stores the effective objective weight as
    # Float32 and publishes Float64(trainer.objective_margin_weight).  Compare
    # metrics against that exact effective value while keeping the controller
    # and frozen contract value at the requested decimal (0.15 or 1.0).
    $expectedMetricMarginWeight = [double][single]$expectedMarginWeight
    $env:BEAT_3L_MARGIN_WEIGHT = [string]$expectedMarginWeight
    $expectedMarginMode = if ($hardNegativePivot) { "student_hard_negative" } else { "fixed_teacher_top2" }
    $env:BEAT_3L_MARGIN_MODE = $expectedMarginMode
    $env:BEAT_3L_ROUTING_MODE = if ($mongoosePivot) { "mongoose_simhash_k7_l2_v1" } else { "fixed_wta" }
    $env:BEAT_3L_MONGOOSE_LR = if ($mongoosePivot) { "0.0001" } else { "" }
    $env:BEAT_3L_MONGOOSE_BETA = if ($mongoosePivot) { "1.0" } else { "" }
    $env:BEAT_3L_MONGOOSE_SEED = if ($mongoosePivot) { "1297043015" } else { "" }
    $env:BEAT_3L_MONGOOSE_WARMUP_UPDATES = if ($mongoosePivot) { "2000" } else { "" }
    $env:BEAT_3L_MONGOOSE_REFRESH_INTERVAL = if ($mongoosePivot) { "2000" } else { "" }
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
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract = $contractPath; contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256; output_root = $resolvedOutput
        comparison_baseline_metrics_path = $comparisonBaselinePath
        comparison_baseline_metrics_sha256 = $comparisonBaselineSha256
        comparison_baseline_checkpoint_path = $comparisonBaselineCheckpointPath
        comparison_baseline_checkpoint_sha256 = $comparisonBaselineCheckpointSha256
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM

    $deadlineMinutes = if ($mongoosePivot) { 30 } elseif ($MaximumUpdates -eq 20000) { 20 } else { 15 }
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

    foreach ($pin in $pinnedFiles) {
        [void](Assert-PinnedFile -Path $pin[0] -ExpectedBytes $pin[1] -ExpectedSha256 $pin[2] -Label $pin[3])
    }
    [void](Assert-PinnedFile -Path $contractPath -ExpectedBytes 7861 -ExpectedSha256 $expectedContractSha256 -Label "MONGOOSE preregistration contract")
    [void](Assert-PinnedFile -Path $manifestPath -ExpectedBytes 1268581 -ExpectedSha256 $expectedManifestSha256 -Label "teacher_v3 manifest")
    Assert-ControllerUnchanged

    $metricFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "metrics.jsonl")
    if ($metricFiles.Count -ne 1) {
        throw "CPF1 run must publish exactly one metrics.jsonl; got $($metricFiles.Count)"
    }
    $metricsPath = Assert-RunLeaf -Path $metricFiles[0].FullName -Root $resolvedOutput
    $metricLines = @(Get-Content -LiteralPath $metricsPath | Where-Object { $_.Trim().Length -gt 0 })
    $expectedUpdates = @(0..([int]($MaximumUpdates / $evaluationInterval)) | ForEach-Object { $_ * $evaluationInterval })
    if ($metricLines.Count -ne $expectedUpdates.Count) {
        throw "CPF1 metrics do not contain the exact frozen 500-update sequence"
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
        if ($mongoosePivot) {
            $routerState = $metrics[$i].router_state
            if ($null -eq $routerState -or
                [string]$routerState.configured_mode -cne "mongoose_simhash_k7_l2_v1" -or
                [int]$routerState.trainable_parameters -ne 2688 -or
                (@($routerState.optimizer_steps) -join ',') -cne "$($expectedUpdates[$i]),$($expectedUpdates[$i]),$($expectedUpdates[$i])") {
                throw "MONGOOSE router identity/optimizer clock changed at metric index $i"
            }
            if ($expectedUpdates[$i] -eq 0) {
                if ([bool]$routerState.active -or [string]$routerState.serving_mode -cne "fixed_wta_warmup" -or
                    [int]$routerState.refresh_count -ne 0 -or [int]$routerState.last_refresh_update -ne 0) {
                    throw "MONGOOSE update-0 warmup state changed"
                }
            } else {
                if (-not [bool]$routerState.active -or
                    [string]$routerState.serving_mode -cne "mongoose_simhash_k7_l2_v1" -or
                    [int]$routerState.refresh_count -ne [int]($expectedUpdates[$i] / 2000) -or
                    [int]$routerState.last_refresh_update -ne $expectedUpdates[$i] -or
                    [bool]$routerState.refresh_due) {
                    throw "MONGOOSE refresh/serving cadence changed at metric index $i"
                }
            }
        }
        Assert-FiniteJsonValue -Value $metrics[$i] -Path "metrics[$i]"
    }
    $baseline = $metrics[0]
    $final = $metrics[-1]
    if ([string]$final.variant -ne $Variant) { throw "CPF1 final variant differs from $Variant" }
    $expectedRoutingPolicy = if ($mongoosePivot) {
        "mongoose-simhash-k7-l2-live-pending-bce-v1"
    } else {
        "wta-collision-count-stable-id-prefilter-v1"
    }
    if ([string]$final.routing_policy -cne $expectedRoutingPolicy) {
        throw "CPF1 final routing policy differs from the frozen contract"
    }
    if ([int]$final.throughput.updates -ne $MaximumUpdates) {
        throw "CPF1 throughput did not authoritatively reach update $MaximumUpdates"
    }
    if ([int]$final.learner_width -ne 80 -or [int]$final.observed_max_candidates -ne 76) {
        throw "MONGOOSE learner width or teacher_v3 maximum candidate count changed"
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
    if ([string]$final.parameter_contract.routing_policy -cne "wta-collision-count-stable-id-prefilter-v1") {
        throw "CPF1 base sparse-bank routing contract changed"
    }
    if (-not $mongoosePivot -and [string]$final.parameter_contract.routing_policy -cne [string]$final.routing_policy) {
        throw "CPF1 fixed-WTA metric and parameter-contract routing policies differ"
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
        if ($mongoosePivot) {
            $observedParentCheckpointSha256 = (Get-FileHash -LiteralPath $comparisonBaselineCheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($observedParentCheckpointSha256 -cne $comparisonBaselineCheckpointSha256) {
                throw "Pinned MONGOOSE parent checkpoint changed during the run"
            }
        }
        $finalTop1 = [double]$final.validation.top1_agreement
        $finalNdcg = [double]$final.validation.ndcg
        $finalPairwise = [double]$final.validation.pairwise_accuracy
        $finalLoss = [double]$final.validation.composite_loss
        if ($mongoosePivot) {
            $parentEntropy = @(0.9780863300520932, 0.9687321438167865, 0.9658671084173591)
            $minimumEntropy = @(0.8802776970468839, 0.8718589294351079, 0.8692803975756232)
            if ($finalTop1 -lt 0.25) { throw "MONGOOSE held top-1 missed the preregistered gate" }
            if ($finalNdcg -le 0.9374244614677424) { throw "MONGOOSE held NDCG did not strictly exceed the exact parent threshold" }
            if ($finalPairwise -le 0.7407546814419289) { throw "MONGOOSE held pairwise did not strictly exceed the exact parent threshold" }
            if ([double]$final.validation.action_margin -lt 0.0125) {
                throw "MONGOOSE held action margin missed the preregistered gate"
            }
            if (@($final.bank_coverage).Count -ne 3) { throw "MONGOOSE bank coverage must contain three layers" }
            for ($layer = 0; $layer -lt 3; $layer += 1) {
                $coverage = $final.bank_coverage[$layer]
                if (-not ($coverage.PSObject.Properties.Name -contains "usage_normalized_shannon_entropy")) {
                    throw "MONGOOSE layer $($layer + 1) lacks normalized entropy telemetry"
                }
                $observedEntropy = [double]$coverage.usage_normalized_shannon_entropy
                if ($observedEntropy -lt $minimumEntropy[$layer]) {
                    throw "MONGOOSE layer $($layer + 1) normalized entropy $observedEntropy is below 90% of parent $($parentEntropy[$layer])"
                }
            }
            if ($null -eq $final.router_state -or -not [bool]$final.router_state.active) {
                throw "MONGOOSE learned router is not live at the final gate"
            }
            if ([string]$final.router_state.configured_mode -cne "mongoose_simhash_k7_l2_v1" -or
                [string]$final.router_state.serving_mode -cne "mongoose_simhash_k7_l2_v1") {
                throw "MONGOOSE final router mode changed"
            }
            if ([int]$final.router_state.route_dim -ne 64 -or
                [int]$final.router_state.bits_per_table -ne 7 -or
                [int]$final.router_state.tables -ne 2 -or
                [int]$final.router_state.trainable_parameters -ne 2688 -or
                [bool]$final.router_state.column_normalization) {
                throw "MONGOOSE final router geometry changed"
            }
            if ((@($final.router_state.optimizer_steps) -join ',') -cne '20000,20000,20000' -or
                [int]$final.router_state.warmup_updates -ne 2000 -or
                [int]$final.router_state.refresh_interval -ne 2000 -or
                [int]$final.router_state.last_refresh_update -ne 20000 -or
                [int]$final.router_state.refresh_count -ne 10 -or
                [bool]$final.router_state.refresh_due) {
                throw "MONGOOSE final optimizer/refresh clocks changed"
            }
            $routerParameterFraction = 2688.0 / 19924022.0
            if ($routerParameterFraction -gt 0.01) { throw "MONGOOSE router exceeds one percent of base parameters" }
            foreach ($metric in @($metrics | Select-Object -Skip 1)) {
                $accounting = $metric.last_step.accounting
                if ($null -eq $accounting) { throw "MONGOOSE trained metric lacks active-only accounting" }
                if ([string]$accounting.routing_mode -cne "mongoose_simhash_k7_l2_v1") {
                    throw "MONGOOSE trained metric routing mode changed"
                }
                if ([int]$accounting.mongoose_trainable_parameters -ne 2688 -or
                    [int]$accounting.total_parameters_with_router -ne 19926710 -or
                    [int]$accounting.mongoose_active_parameter_touches -ne 2688 -or
                    [double]$accounting.mongoose_active_touch_fraction -gt 0.05) {
                    throw "MONGOOSE router parameter/touch accounting gate failed"
                }
                if (@($accounting.optimizer).Count -ne 3) { throw "MONGOOSE sparse optimizer telemetry must contain three layers" }
                $rowDimensions = @(560, 321, 321)
                $scoreCaps = @(384, 640, 640)
                for ($layer = 0; $layer -lt 3; $layer += 1) {
                    $optimizer = $accounting.optimizer[$layer]
                    if ([int]$optimizer.global_step -ne [int]$metric.update -or
                        [int]$optimizer.active_rows -ne [int]$accounting.unique_active_rows[$layer] -or
                        [int64]$optimizer.active_elements -ne [int64]$optimizer.active_rows * $rowDimensions[$layer] -or
                        [int]$optimizer.dirty_route_rows -ne [int]$optimizer.active_rows -or
                        [int64]$optimizer.theta_elements_read -ne [int64]$optimizer.theta_elements_written -or
                        [int64]$optimizer.theta_elements_read -ne [int64]$optimizer.active_elements -or
                        [int64]$optimizer.moment_elements_read -ne [int64]$optimizer.moment_elements_written -or
                        [int64]$optimizer.moment_elements_read -ne 2 * [int64]$optimizer.theta_elements_read) {
                        throw "MONGOOSE layer $($layer + 1) active-only sparse optimizer accounting failed at update $($metric.update)"
                    }
                    if ([int]$accounting.maximum_retrieved_rows[$layer] -gt $scoreCaps[$layer]) {
                        throw "MONGOOSE layer $($layer + 1) exact-score cap telemetry failed at update $($metric.update)"
                    }
                }
                if ($null -eq $accounting.mongoose_pair -or
                    [int64]$accounting.mongoose_pair.parameter_touches -ne 2688) {
                    throw "MONGOOSE router optimizer escaped its 2,688 projection parameters"
                }
                $pair = $accounting.mongoose_pair
                if ((@($pair.pair_macs) -join ',') -cne '6300,6300,6300' -or
                    (@($pair.unique_input_bytes) -join ',') -cne '4352,4352,4352' -or
                    [bool]$pair.column_normalization) {
                    throw "MONGOOSE pair projection accounting changed at update $($metric.update)"
                }
                for ($layer = 0; $layer -lt 3; $layer += 1) {
                    if ([int]$pair.positive_ids[$layer] -eq [int]$pair.negative_ids[$layer]) {
                        throw "MONGOOSE layer $($layer + 1) witness pair collapsed at update $($metric.update)"
                    }
                }
                if ([int]$metric.update -ge 4000 -and
                    [int64](($accounting.routing_projection_macs | Measure-Object -Sum).Sum) -le 0) {
                    throw "MONGOOSE learned SimHash was not exercised before update $($metric.update)"
                }
                if ($null -eq $metric.last_step.mongoose_refresh -or
                    [int]$metric.last_step.mongoose_refresh.update -ne [int]$metric.update -or
                    [int64]$metric.last_step.mongoose_refresh.total_macs -ne 143203200) {
                    throw "MONGOOSE full-index refresh artifact changed at update $($metric.update)"
                }
            }
            if ([int64]$final.throughput.mongoose_refresh_macs -ne 1432032000 -or
                [int64]$final.throughput.mongoose_refresh_key_bytes -ne 409152000 -or
                [int64]$final.throughput.mongoose_refresh_projection_bytes -ne 322560) {
                throw "MONGOOSE cumulative refresh accounting changed"
            }
            $materialGain = $true
        } elseif ($Variant -eq "k128" -and $MaximumUpdates -eq 20000) {
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
    if (-not $mongoosePivot -and ($prefilterActivations -le 0 -or $prefilterDrops -le 0)) {
        throw "CPF1 never exercised the overflow prefilter"
    }
    $stderrText = Get-Content -LiteralPath $stderrPath -Raw
    if ($stderrText -match '(?im)^ERROR:|Stacktrace:|bucket-entry cap .* exceeded|exact-score cap .* exceeded') {
        throw "CPF1 stderr contains a forbidden failure record"
    }
    if ($mongoosePivot) {
        if ((Get-Item -LiteralPath $stdoutPath).Length -ne 0) {
            throw "MONGOOSE stdout must remain empty"
        }
        $stderrLines = @(Get-Content -LiteralPath $stderrPath)
        foreach ($line in $stderrLines) {
            if ($line.Length -gt 0 -and $line -notmatch '^(┌ Info:|│|└)') {
                throw "MONGOOSE stderr contains an unexpected non-Info line: $line"
            }
        }
        if (@($stderrLines | Where-Object { $_ -eq '┌ Info: Three-layer initial teacher evaluation' }).Count -ne 1 -or
            @($stderrLines | Where-Object { $_ -eq '┌ Info: Three-layer teacher progress' }).Count -ne 10 -or
            @($stderrLines | Where-Object { $_ -eq '┌ Info: Three-layer teacher checkpoint' }).Count -ne 2) {
            throw "MONGOOSE stderr does not contain the exact initial/progress/checkpoint Info record counts"
        }
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
    if ($mongoosePivot) {
        $checkpointFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "checkpoint_*.jls" | Sort-Object Name)
        if ($checkpointFiles.Count -ne 2 -or
            ($checkpointFiles.Name -join ',') -cne 'checkpoint_000010000.jls,checkpoint_000020000.jls') {
            throw "MONGOOSE must publish exactly the preregistered update-10000 and update-20000 checkpoints"
        }
        foreach ($checkpointFile in $checkpointFiles) {
            [void](Assert-RunLeaf -Path $checkpointFile.FullName -Root $resolvedOutput)
        }
    }
    $checkpoint = Get-Item -LiteralPath $checkpointPath
    if ([int64]$latest.bytes -ne [int64]$checkpoint.Length) {
        throw "CPF1 latest checkpoint byte count differs from disk"
    }
    $checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$latest.sha256 -cne $checkpointSha256) {
        throw "CPF1 latest checkpoint SHA-256 differs from disk"
    }
    $checkpointArtifacts = @($checkpointFiles | ForEach-Object {
        [ordered]@{
            path = $_.FullName
            bytes = [int64]$_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    $preGateFiles = @(Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File)
    if ($preGateFiles.Count -ne 7) {
        throw "MONGOOSE output contains unexpected pre-gate artifacts: $($preGateFiles.Count) files"
    }
    $runDirectory = $metricFiles[0].Directory
    if ($runDirectory.Name -notmatch '^sparse_3l_[0-9]{8}T[0-9]{6}$' -or
        -not $latestFiles[0].Directory.FullName.Equals($runDirectory.FullName, [StringComparison]::OrdinalIgnoreCase) -or
        -not $checkpoint.Directory.Parent.FullName.Equals($runDirectory.FullName, [StringComparison]::OrdinalIgnoreCase)) {
        throw "MONGOOSE run artifacts do not share the one authoritative run directory"
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
        routing_policy = [string]$final.routing_policy
        router_state = $final.router_state
        final_held_top1 = [double]$final.validation.top1_agreement
        final_held_ndcg = [double]$final.validation.ndcg
        final_held_pairwise = [double]$final.validation.pairwise_accuracy
        final_held_action_margin = [double]$final.validation.action_margin
        final_layer_normalized_entropy = @($final.bank_coverage | ForEach-Object { [double]$_.usage_normalized_shannon_entropy })
        router_parameter_fraction_of_base = 2688.0 / 19924022.0
        router_touch_fraction_of_base_active = [double]$final.last_step.accounting.mongoose_active_touch_fraction
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
        checkpoints = $checkpointArtifacts
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        comparison_baseline_metrics_path = $comparisonBaselinePath
        comparison_baseline_metrics_sha256 = $comparisonBaselineSha256
        comparison_baseline_checkpoint_path = $comparisonBaselineCheckpointPath
        comparison_baseline_checkpoint_sha256 = $comparisonBaselineCheckpointSha256
        parent_normalized_entropy = if ($mongoosePivot) { @(0.9780863300520932, 0.9687321438167865, 0.9658671084173591) } else { @() }
        minimum_normalized_entropy = if ($mongoosePivot) { @(0.8802776970468839, 0.8718589294351079, 0.8692803975756232) } else { @() }
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gatePendingPath -Encoding utf8NoBOM
    $gateSha256 = (Get-FileHash -LiteralPath $gatePendingPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-ControllerUnchanged
    [ordered]@{
        status = "completed"; pid = $ownedJulia.Id; completed_at = [DateTimeOffset]::Now.ToString("o")
        variant = $Variant; maximum_updates = $MaximumUpdates
        objective_profile = $ObjectiveProfile
        objective_margin_weight = $expectedMarginWeight
        objective_margin_mode = $expectedMarginMode
        controller_path = $controllerPath
        controller_bytes = $controllerInitialBytes
        controller_sha256 = $controllerInitialSha256
        exit_code = $ownedJulia.ExitCode; contract = $contractPath; contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256; output_root = $resolvedOutput
        gate_status = $gatePath; gate_status_sha256 = $gateSha256
        comparison_baseline_metrics_path = $comparisonBaselinePath
        comparison_baseline_metrics_sha256 = $comparisonBaselineSha256
        comparison_baseline_checkpoint_path = $comparisonBaselineCheckpointPath
        comparison_baseline_checkpoint_sha256 = $comparisonBaselineCheckpointSha256
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8NoBOM
    Move-Item -LiteralPath $gatePendingPath -Destination $gatePath
} catch {
    $failureMessage = $_.Exception.Message
    if (Test-Path -LiteralPath $resolvedOutput) {
        $controllerIntegrity = "PASS"
        try { Assert-ControllerUnchanged } catch { $controllerIntegrity = "FAIL: $($_.Exception.Message)" }
        [ordered]@{
            status="failed"; failed_at=[DateTimeOffset]::Now.ToString("o")
            error=$failureMessage; output_root=$resolvedOutput
            controller_path=$controllerPath
            controller_bytes=$controllerInitialBytes
            controller_sha256=$controllerInitialSha256
            controller_integrity=$controllerIntegrity
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedOutput "controller_status.json") -Encoding utf8NoBOM
    }
    throw
} finally {
    if ($held) { try { $juliaMutex.ReleaseMutex() } catch {} }
    $juliaMutex.Dispose()
}
