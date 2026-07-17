[CmdletBinding()]
param(
    [string]$Repository,
    [string]$OutputDirectory,
    [string]$SourceFingerprint,
    [string]$StartGate,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExperimentId = 'legacy_partial_tail_td_P1'
$ExpectedCheckpointHash = '7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1'
$ExpectedDatasetHash = 'e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d'
$ExpectedBaselineWeightsHash = '2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba'
$ExpectedAuthorizationHash = 'a079330917571824fdbb0dd92d37db92dc1df9701012206bb27bc672d24ca906'
$HardWallSeconds = 35 * 60
$FirstUpdateSeconds = 180.0
$WarmUpdateSeconds = 15.0
$MaxWorkingSetBytes = [int64](8GB)
$StartGateTimeoutSeconds = 600.0
$RequiredStartText = "START $ExperimentId"
$GlobalStartedPath = 'D:\tetris-paper-plus\runs\legacy_partial_tail_td_P1.started.json'
$OutputRoot = [System.IO.Path]::GetFullPath('D:\tetris-paper-plus\runs')
$SystemPython = 'C:\Program Files\Python310\python.exe'
$OpenVinoPython = 'D:\tetris-paper-plus\python-env\Scripts\python.exe'

if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Repository = Join-Path $PSScriptRoot '..\..'
}
$Repository = [System.IO.Path]::GetFullPath($Repository)
$Experiment = Join-Path $Repository 'experiments\legacy_partial_tail_td'
. (Join-Path $Experiment 'native_arguments.ps1')

$RequiredHarnessFiles = @(
    'contract.jl',
    'tail_model.jl',
    'extract_dataset.py',
    'select_rows.jl',
    'train_partial.jl',
    'verify_openvino.py',
    'weighted_inference.py',
    'offline_gate.jl',
    'evaluate_development.jl',
    'finalize_result.py',
    'invoke_once.ps1',
    'native_arguments.ps1',
    'test_wrapper.ps1',
    'test_contract.jl',
    'test_static.py',
    'EXPERIMENT.md'
)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic([string]$Path, $Value, [int]$Depth = 24) {
    $Temporary = "$Path.tmp"
    $Json = $Value | ConvertTo-Json -Depth $Depth
    [System.IO.File]::WriteAllText(
        $Temporary,
        $Json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Write-JsonCreateNew([string]$Path, $Value, [int]$Depth = 24) {
    $Json = $Value | ConvertTo-Json -Depth $Depth
    $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Json + [Environment]::NewLine)
    $Stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally {
        $Stream.Dispose()
    }
}

function Get-HarnessRecords {
    $Records = @()
    $Builder = [System.Text.StringBuilder]::new()
    $Files = @(Get-ChildItem -LiteralPath $Experiment -File | Sort-Object Name)
    foreach ($Name in $RequiredHarnessFiles) {
        $Path = Join-Path $Experiment $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "missing required P1 harness file: $Path"
        }
    }
    foreach ($File in $Files) {
        $Hash = Get-Sha256 $File.FullName
        $Records += [ordered]@{
            path = "experiments/legacy_partial_tail_td/$($File.Name)"
            sha256 = $Hash
        }
        [void]$Builder.Append("$($File.Name)`0$Hash`n")
    }
    $Sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Builder.ToString())
        $Aggregate = ([BitConverter]::ToString($Sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $Sha.Dispose()
    }
    return [ordered]@{ aggregate_sha256 = $Aggregate; files = $Records }
}

function Test-HarnessSyntax {
    foreach ($Name in $RequiredHarnessFiles) {
        $Path = Join-Path $Experiment $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "missing required P1 harness file: $Path"
        }
    }

    $PowerShellFiles = @(Get-ChildItem -LiteralPath $Experiment -Filter '*.ps1' -File)
    foreach ($File in $PowerShellFiles) {
        $Tokens = $null
        $Errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile(
            $File.FullName,
            [ref]$Tokens,
            [ref]$Errors
        )
        if ($Errors.Count -ne 0) {
            throw "PowerShell parse failure in $($File.FullName): $($Errors[0].Message)"
        }
    }

    if (-not (Test-Path -LiteralPath $SystemPython -PathType Leaf)) {
        throw "missing fixed system Python: $SystemPython"
    }
    if (-not (Test-Path -LiteralPath $OpenVinoPython -PathType Leaf)) {
        throw "missing fixed OpenVINO Python: $OpenVinoPython"
    }
    $PythonFiles = @(Get-ChildItem -LiteralPath $Experiment -Filter '*.py' -File |
        ForEach-Object { $_.FullName })
    $PythonParseCode = 'import ast,pathlib,sys;[ast.parse(pathlib.Path(p).read_bytes(),filename=p) for p in sys.argv[1:]]'
    & $SystemPython -c $PythonParseCode @PythonFiles
    if ($LASTEXITCODE -ne 0) {
        throw 'Python syntax-only harness parse failed'
    }

    $JuliaExe = (Get-Command julia -ErrorAction Stop).Source
    $JuliaFiles = @(Get-ChildItem -LiteralPath $Experiment -Filter '*.jl' -File |
        ForEach-Object { $_.FullName })
    $JuliaParseCode = 'for path in ARGS; Meta.parseall(read(path, String); filename=path); end'
    & $JuliaExe --startup-file=no --history-file=no -e $JuliaParseCode @JuliaFiles
    if ($LASTEXITCODE -ne 0) {
        throw 'Julia syntax-only harness parse failed'
    }

    return [ordered]@{
        powershell_files = $PowerShellFiles.Count
        python_files = $PythonFiles.Count
        julia_files = $JuliaFiles.Count
        checkpoint_or_dataset_opened = $false
        marker_read_or_written = $false
    }
}

if ($ValidateOnly) {
    $Syntax = Test-HarnessSyntax
    [ordered]@{
        status = 'validation_only_passed'
        experiment = $ExperimentId
        repository = $Repository
        syntax = $Syntax
        note = 'syntax/files only; no checkpoint or dataset hash/read, marker access, output creation, training, inference, or game'
    } | ConvertTo-Json -Depth 8
    exit 0
}

foreach ($InputName in @('OutputDirectory', 'SourceFingerprint', 'StartGate')) {
    $InputValue = Get-Variable -Name $InputName -ValueOnly
    if ([string]::IsNullOrWhiteSpace([string]$InputValue)) {
        throw "-$InputName is required unless -ValidateOnly is used"
    }
}

$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$SourceFingerprint = [System.IO.Path]::GetFullPath($SourceFingerprint)
$StartGate = [System.IO.Path]::GetFullPath($StartGate)
$OutputPrefix = $OutputRoot.TrimEnd('\') + '\'
if (-not $OutputDirectory.StartsWith($OutputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "P1 output must be a fresh namespace below $OutputRoot"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "P1 output namespace must not already exist: $OutputDirectory"
}
if (Test-Path -LiteralPath $StartGate) {
    throw "start gate must be absent before the freeze is created: $StartGate"
}
if (Test-Path -LiteralPath $GlobalStartedPath) {
    throw "global P1 one-shot marker already exists; retry is prohibited: $GlobalStartedPath"
}
if (-not (Test-Path -LiteralPath $SourceFingerprint -PathType Leaf)) {
    throw "missing clean source fingerprint: $SourceFingerprint"
}

$Syntax = Test-HarnessSyntax
$WrapperTestPath = Join-Path $Experiment 'test_wrapper.ps1'
$WrapperTestStarted = [DateTimeOffset]::UtcNow
$WrapperTestOutput = @(& $WrapperTestPath 2>&1)
if (-not $?) {
    throw "synthetic native argv/Job Object preflight failed: $($WrapperTestOutput -join [Environment]::NewLine)"
}
$WrapperTestRecord = [ordered]@{
    path = $WrapperTestPath
    sha256 = Get-Sha256 $WrapperTestPath
    started_at = $WrapperTestStarted.ToString('o')
    seconds = ([DateTimeOffset]::UtcNow - $WrapperTestStarted).TotalSeconds
    output = $WrapperTestOutput -join [Environment]::NewLine
    passed = $true
    real_checkpoint_or_dataset_opened = $false
    heavy_work_run = $false
}
$JuliaExe = (Get-Command julia -ErrorAction Stop).Source
$Checkpoint = Join-Path $Repository '1313\mainmodel copy 3.jld2'
$Dataset = 'D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2'
$BaselineWeights = Join-Path $Repository 'artifacts\legacy_openvino\legacy_1313_weights.npz'
$AuthorizationReport = Join-Path $Repository 'reports\clean_post_f_strategy_review.md'
$Manifest = Join-Path $Repository 'Manifest.toml'
$ImmutableInputs = @(
    [ordered]@{ label = 'checkpoint'; path = $Checkpoint; expected = $ExpectedCheckpointHash },
    [ordered]@{ label = 'dataset'; path = $Dataset; expected = $ExpectedDatasetHash },
    [ordered]@{ label = 'canonical baseline weights'; path = $BaselineWeights; expected = $ExpectedBaselineWeightsHash },
    [ordered]@{ label = 'authorization report'; path = $AuthorizationReport; expected = $ExpectedAuthorizationHash }
)
foreach ($InputRecord in $ImmutableInputs) {
    if (-not (Test-Path -LiteralPath $InputRecord.path -PathType Leaf)) {
        throw "missing $($InputRecord.label): $($InputRecord.path)"
    }
    $Observed = Get-Sha256 $InputRecord.path
    if ($Observed -ne $InputRecord.expected) {
        throw "$($InputRecord.label) SHA-256 mismatch: expected $($InputRecord.expected), observed $Observed"
    }
    $InputRecord.observed = $Observed
}
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "missing Manifest.toml: $Manifest"
}
$ManifestText = Get-Content -LiteralPath $Manifest -Raw
if ($ManifestText -notmatch '(?s)\[\[deps\.Lux\]\].*?version = "1\.31\.4"') {
    throw 'Manifest does not pin Lux 1.31.4'
}
if ($ManifestText -notmatch '(?s)\[\[deps\.Zygote\]\].*?version = "0\.7\.11"') {
    throw 'Manifest does not pin Zygote 0.7.11'
}
$JuliaVersion = (& $JuliaExe --version).Trim()
if ($LASTEXITCODE -ne 0 -or $JuliaVersion -ne 'julia version 1.12.6') {
    throw "P1 requires Julia 1.12.6; observed '$JuliaVersion'"
}

$GitStatus = (& git -C $Repository status --porcelain=v1 --untracked-files=all) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'git status failed' }
if ($GitStatus.Length -ne 0) {
    throw "repository must be clean before P1 freeze:`n$GitStatus"
}
$Commit = (& git -C $Repository rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'git rev-parse HEAD failed' }
$Harness = Get-HarnessRecords
$SourceFingerprintHash = Get-Sha256 $SourceFingerprint
$SourceFingerprintValue = Get-Content -LiteralPath $SourceFingerprint -Raw | ConvertFrom-Json

[System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$FreezePath = Join-Path $OutputDirectory 'freeze.json'
$ReadyPath = Join-Path $OutputDirectory 'ready.json'
$MonitorPath = Join-Path $OutputDirectory 'monitor.json'
$WrapperResultPath = Join-Path $OutputDirectory 'wrapper_result.json'
$EligibilityPath = Join-Path $OutputDirectory 'eligibility.json'
$RowFreezePath = Join-Path $OutputDirectory 'row_freeze.json'
$TrainingNpz = Join-Path $OutputDirectory 'training.npz'
$TrainingExtractionPath = Join-Path $OutputDirectory 'training_extraction.json'
$TrainingPhasePath = Join-Path $OutputDirectory 'training_phase.json'
$CandidateWeights = Join-Path $OutputDirectory 'candidate_weights.npz'
$CandidateMerged = Join-Path $OutputDirectory 'candidate_merged.jld2'
$FinalReference = Join-Path $OutputDirectory 'final_reference.npz'
$OpenVinoGatePath = Join-Path $OutputDirectory 'openvino_gate.json'
$OfflineNpz = Join-Path $OutputDirectory 'offline.npz'
$OfflineExtractionPath = Join-Path $OutputDirectory 'offline_extraction.json'
$OfflineGatePath = Join-Path $OutputDirectory 'offline_gate.json'
$DevelopmentPath = Join-Path $OutputDirectory 'development.json'
$FinalResultPath = Join-Path $OutputDirectory 'final_result.json'

$Commands = [ordered]@{
    eligibility = @($SystemPython, (Join-Path $Experiment 'extract_dataset.py'), 'eligibility', $Dataset, $EligibilityPath)
    select_rows = @($JuliaExe, '--startup-file=no', "--project=$Repository", '--threads=20', (Join-Path $Experiment 'select_rows.jl'), $EligibilityPath, $RowFreezePath)
    extract_training = @($SystemPython, (Join-Path $Experiment 'extract_dataset.py'), 'training', $Dataset, $RowFreezePath, $TrainingNpz, $TrainingExtractionPath)
    train_partial = @($JuliaExe, '--startup-file=no', "--project=$Repository", '--threads=20', (Join-Path $Experiment 'train_partial.jl'), $OutputDirectory, $TrainingNpz, $Checkpoint, $FreezePath, $RowFreezePath)
    verify_openvino = @($OpenVinoPython, (Join-Path $Experiment 'verify_openvino.py'), $Repository, $OutputDirectory, $CandidateWeights, $FinalReference, $OpenVinoGatePath)
    extract_offline = @($SystemPython, (Join-Path $Experiment 'extract_dataset.py'), 'offline', $Dataset, $OfflineNpz, $OfflineExtractionPath)
    offline_gate = @($JuliaExe, '--startup-file=no', "--project=$Repository", '--threads=20', (Join-Path $Experiment 'offline_gate.jl'), $OfflineNpz, $CandidateWeights, $OfflineGatePath)
    evaluate_development = @($JuliaExe, '--startup-file=no', "--project=$Repository", '--threads=20', (Join-Path $Experiment 'evaluate_development.jl'), $Repository, $CandidateWeights, $BaselineWeights, $DevelopmentPath)
    finalize = @($SystemPython, (Join-Path $Experiment 'finalize_result.py'), $OutputDirectory)
}

$ProcessesBefore = @(
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessName -match '^(julia|python|pythonw)$' } |
        ForEach-Object {
            [ordered]@{
                name = $_.ProcessName
                id = $_.Id
                working_set_bytes = $_.WorkingSet64
                private_bytes = $_.PrivateMemorySize64
                path = try { $_.Path } catch { $null }
            }
        }
)
if (@($ProcessesBefore | Where-Object { $_.name -eq 'julia' }).Count -ne 0) {
    throw 'another Julia process is active; one-shot P1 execution refused before marker creation'
}
if (@($ProcessesBefore | Where-Object { $_.name -match '^python' -and $_.working_set_bytes -gt 2GB }).Count -ne 0) {
    throw 'a Python process above 2 GiB is active; one-shot P1 execution refused before marker creation'
}

$FreezeCreated = [DateTimeOffset]::UtcNow
$Freeze = [ordered]@{
    experiment = $ExperimentId
    scientific_role = 'one-shot conservative partial-tail anchored TD development pilot'
    frozen_at = $FreezeCreated.ToString('o')
    source_commit = $Commit
    repository_clean = $true
    source_fingerprint_path = $SourceFingerprint
    source_fingerprint_sha256 = $SourceFingerprintHash
    source_fingerprint = $SourceFingerprintValue
    authorization_report = $AuthorizationReport
    authorization_report_sha256 = $ExpectedAuthorizationHash
    manifest_path = $Manifest
    manifest_sha256 = Get-Sha256 $Manifest
    immutable_inputs = $ImmutableInputs
    harness_sha256 = $Harness.aggregate_sha256
    harness_files = $Harness.files
    syntax_only_preflight = $Syntax
    synthetic_native_argv_and_job_preflight = $WrapperTestRecord
    output_directory = $OutputDirectory
    global_one_shot_marker = $GlobalStartedPath
    start_gate = $StartGate
    required_start_gate_contents = $RequiredStartText
    commands = $Commands
    constants = [ordered]@{
        trainable_paths = @('board_net.resblocks.layer_29', 'board_net.resblocks.layer_31', 'board_net.conv2', 'board_net.norm2', 'score_net')
        trainable_parameter_count = 2949508
        frozen_parameter_count = 17837946
        optimizer_moment_elements = 5899016
        training_rows = @(1, 1500)
        training_seeds = @(5742, 5743, 5744, 5745, 5746, 5747)
        offline_rows = @(1501, 2000)
        offline_seeds = @(5748, 5749)
        development_seeds = @(5756, 5757)
        forbidden_validation_seeds = @(8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008)
        forbidden_test_seeds = @(91001, 91032)
        data_order_rng = 'Xoshiro(0x1313_2026)'
        updates = 300
        n_step = 3
        gamma = 0.997
        reward_scale = 600
        huber_delta = 1.0
        anchor_weight = 1.0
        loss = 'selected-action Huber(y3) + mean pointwise old-Q Huber over exact selected-action chunk'
        historical_candidate_chunk = 16
        step0_witness_source_row = 1055
        step0_witness_episode = 5
        step0_witness_episode_step = 55
        step0_witness_candidate_count = 52
        step0_witness_selected_action = 11
        step0_split_tail_max_abs_error = 0.000001
        step0_stored_old_q_max_abs_error = 0.01
        finite_difference_paths = @(
            'score_net.layer_3.bias',
            'score_net.layer_1.weight',
            'board_net.resblocks.layer_31.layer_1.weight'
        )
        finite_difference_abs_tolerance = 0.001
        finite_difference_relative_tolerance = 0.02
        learning_rate = 0.00001
        betas = @(0.9, 0.999)
        weight_decay = 0.0001
        cpu_equivalence_tolerance = 0.0001
        npu_equivalence_tolerance = 0.01
        offline_top1_minimum = 0.95
        development_pieces = 100
        development_next = 5
        development_pair_requirement = '2/2 strictly positive; mean >= +500; median > 0'
        hard_wall_seconds = $HardWallSeconds
        first_update_running_limit_seconds = $FirstUpdateSeconds
        warm_update_running_limit_seconds = $WarmUpdateSeconds
        warm_update_count = 6
        warm_median_limit_seconds = 4.5
        max_process_tree_working_set_bytes = $MaxWorkingSetBytes
    }
    machine = [ordered]@{
        computer_name = $env:COMPUTERNAME
        powershell = $PSVersionTable.PSVersion.ToString()
        julia_executable = $JuliaExe
        julia_version = $JuliaVersion
        system_python = $SystemPython
        openvino_python = $OpenVinoPython
        processes_before = $ProcessesBefore
    }
}
Write-JsonAtomic $FreezePath $Freeze 32
Write-JsonAtomic $ReadyPath ([ordered]@{
    status = 'frozen_waiting_for_explicit_start_gate'
    experiment = $ExperimentId
    wrapper_pid = $PID
    source_commit = $Commit
    freeze_path = $FreezePath
    freeze_sha256 = Get-Sha256 $FreezePath
    start_gate = $StartGate
    required_start_gate_contents = $RequiredStartText
    gate_must_be_newer_than_utc = $FreezeCreated.ToString('o')
    wait_timeout_seconds = $StartGateTimeoutSeconds
})

$WaitStarted = [DateTimeOffset]::UtcNow
while (-not (Test-Path -LiteralPath $StartGate -PathType Leaf)) {
    if (([DateTimeOffset]::UtcNow - $WaitStarted).TotalSeconds -ge $StartGateTimeoutSeconds) {
        throw "explicit start gate was not provided within $StartGateTimeoutSeconds seconds"
    }
    Start-Sleep -Milliseconds 250
}
$GateInfo = Get-Item -LiteralPath $StartGate
if ([DateTimeOffset]$GateInfo.LastWriteTimeUtc -lt $FreezeCreated) {
    throw 'start gate predates the freeze and is not valid for this P1 invocation'
}
$GateText = (Get-Content -LiteralPath $StartGate -Raw).Trim()
if ($GateText -cne $RequiredStartText) {
    throw "start gate contents must be exactly '$RequiredStartText'"
}

$GitStatusAtGate = (& git -C $Repository status --porcelain=v1 --untracked-files=all) -join "`n"
if ($LASTEXITCODE -ne 0 -or $GitStatusAtGate.Length -ne 0) {
    throw 'repository changed or became dirty between freeze and explicit start gate'
}
if ((& git -C $Repository rev-parse HEAD).Trim() -ne $Commit) {
    throw 'source commit changed between freeze and explicit start gate'
}
foreach ($InputRecord in $ImmutableInputs) {
    if ((Get-Sha256 $InputRecord.path) -ne $InputRecord.observed) {
        throw "$($InputRecord.label) changed between freeze and explicit start gate"
    }
}
$HarnessAtGate = Get-HarnessRecords
if ($HarnessAtGate.aggregate_sha256 -ne $Harness.aggregate_sha256) {
    throw 'P1 harness changed between freeze and explicit start gate'
}
if ((Get-Sha256 $SourceFingerprint) -ne $SourceFingerprintHash) {
    throw 'source fingerprint changed between freeze and explicit start gate'
}

$OneShotStarted = [DateTimeOffset]::UtcNow
Write-JsonCreateNew $GlobalStartedPath ([ordered]@{
    experiment = $ExperimentId
    started_at = $OneShotStarted.ToString('o')
    wrapper_pid = $PID
    source_commit = $Commit
    harness_sha256 = $Harness.aggregate_sha256
    freeze_sha256 = Get-Sha256 $FreezePath
    start_gate_path = $StartGate
    start_gate_sha256 = Get-Sha256 $StartGate
    output_directory = $OutputDirectory
    retry_prohibited = $true
})
Write-JsonCreateNew (Join-Path $OutputDirectory 'started.json') ([ordered]@{
    experiment = $ExperimentId
    started_at = $OneShotStarted.ToString('o')
    wrapper_pid = $PID
    source_commit = $Commit
    global_marker = $GlobalStartedPath
    one_shot = $true
})

$env:OPENBLAS_NUM_THREADS = '10'
$env:JULIA_NUM_THREADS = '20'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:P1_OUTPUT_DIRECTORY = $OutputDirectory
$env:P1_CHECKPOINT_PATH = $Checkpoint
$env:P1_DATASET_PATH = $Dataset
$env:P1_FREEZE_PATH = $FreezePath
$env:P1_ROW_FREEZE_PATH = $RowFreezePath
$env:P1_TRAINING_NPZ = $TrainingNpz
$env:P1_TRAINING_PHASE_PATH = $TrainingPhasePath
$env:P1_STAGE_PATH = $TrainingPhasePath
$env:P1_CANDIDATE_WEIGHTS_PATH = $CandidateWeights
$env:P1_CANDIDATE_MERGED_PATH = $CandidateMerged
$env:P1_FINAL_REFERENCE_PATH = $FinalReference
$env:P1_OPENVINO_GATE_PATH = $OpenVinoGatePath
$env:P1_OFFLINE_NPZ = $OfflineNpz
$env:P1_OFFLINE_GATE_PATH = $OfflineGatePath
$env:P1_DEVELOPMENT_PATH = $DevelopmentPath
$env:P1_BASELINE_WEIGHTS_PATH = $BaselineWeights
$env:P1_MONITOR_PATH = $MonitorPath
$env:P1_GLOBAL_MARKER_PATH = $GlobalStartedPath

$script:RunStarted = $OneShotStarted
$script:Deadline = $script:RunStarted.AddSeconds($HardWallSeconds)
$env:P1_HARD_DEADLINE_UNIX = [string]($script:Deadline.ToUnixTimeMilliseconds() / 1000.0)
$script:PeakTreeWorkingSet = [int64]0
$script:PeakTreePrivateBytes = [int64]0
$script:PhaseRecords = @()
$script:SkippedPhases = @()
$script:Failures = [System.Collections.Generic.List[string]]::new()

function Add-Failure([string]$Message) {
    $script:Failures.Add($Message)
}

function Get-JobMemorySnapshot([P1Native.JobObject]$Job) {
    $WorkingSet = [int64]0
    $PrivateBytes = [int64]0
    $Processes = @()
    foreach ($ProcessId in @($Job.GetProcessIds())) {
        $Child = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($null -eq $Child) { continue }
        try {
            $WorkingSet += [int64]$Child.WorkingSet64
            $PrivateBytes += [int64]$Child.PrivateMemorySize64
            $Processes += [ordered]@{
                id = $Child.Id
                name = $Child.ProcessName
                working_set_bytes = [int64]$Child.WorkingSet64
                private_bytes = [int64]$Child.PrivateMemorySize64
            }
        } catch {
            # A process can disappear between the job query and Get-Process.
        }
    }
    return [ordered]@{
        working_set_bytes = $WorkingSet
        private_bytes = $PrivateBytes
        processes = $Processes
    }
}

function Get-TrainingStageStopReason {
    if (-not (Test-Path -LiteralPath $TrainingPhasePath -PathType Leaf)) {
        return $null
    }
    $StageState = Get-Content -LiteralPath $TrainingPhasePath -Raw | ConvertFrom-Json
    if (-not ($StageState.PSObject.Properties.Name -contains 'stage')) {
        return $null
    }
    $Stage = [string]$StageState.stage
    if ($Stage -ne 'first_update_running' -and
        $Stage -notmatch '^warm_update_[1-6]_running$' -and
        $Stage -notmatch '^update_[0-9]+_running$') {
        return $null
    }
    if (-not ($StageState.PSObject.Properties.Name -contains 'stage_started_unix')) {
        return "training monitor contract failure: $Stage lacks stage_started_unix"
    }
    $StartedUnix = [double]$StageState.stage_started_unix
    $NowUnix = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $Elapsed = $NowUnix - $StartedUnix
    if (-not [double]::IsFinite($StartedUnix) -or $Elapsed -lt -2.0) {
        return "training monitor contract failure: invalid stage_started_unix for $Stage"
    }
    if ($Stage -eq 'first_update_running' -and $Elapsed -ge $FirstUpdateSeconds) {
        return "external first_update_running deadline reached $FirstUpdateSeconds seconds"
    }
    if ($Stage -match '^warm_update_[1-6]_running$' -and $Elapsed -ge $WarmUpdateSeconds) {
        return "external $Stage deadline reached $WarmUpdateSeconds seconds"
    }
    if ($Stage -match '^update_[0-9]+_running$' -and $Elapsed -ge $WarmUpdateSeconds) {
        return "external $Stage deadline reached $WarmUpdateSeconds seconds"
    }
    return $null
}

function Write-MonitorSnapshot([bool]$Complete = $false) {
    $Now = [DateTimeOffset]::UtcNow
    Write-JsonAtomic $MonitorPath ([ordered]@{
        experiment = $ExperimentId
        started_at = $script:RunStarted.ToString('o')
        hard_deadline = $script:Deadline.ToString('o')
        updated_at = $Now.ToString('o')
        ended_at = if ($Complete) { $Now.ToString('o') } else { $null }
        complete = $Complete
        wall_seconds = ($Now - $script:RunStarted).TotalSeconds
        hard_wall_seconds = $HardWallSeconds
        peak_process_tree_working_set_bytes = $script:PeakTreeWorkingSet
        peak_process_tree_private_bytes = $script:PeakTreePrivateBytes
        max_process_tree_working_set_bytes = $MaxWorkingSetBytes
        phases = $script:PhaseRecords
        skipped_phases = $script:SkippedPhases
        failures = @($script:Failures)
        stop_reason = if ($script:Failures.Count -eq 0) { 'completed' } else { [string]$script:Failures[0] }
        whole_tree_enforcement = 'Windows Job Object with KILL_ON_JOB_CLOSE; every sample sums all assigned process working sets'
        one_shot_retry_prohibited = $true
    }) 32
}

function Invoke-MonitoredPhase(
    [string]$Name,
    [string]$FilePath,
    [string[]]$ArgumentList
) {
    if ([DateTimeOffset]::UtcNow -ge $script:Deadline) {
        $Reason = "hard 35-minute wall reached before phase $Name"
        Add-Failure $Reason
        $script:SkippedPhases += [ordered]@{ phase = $Name; reason = $Reason }
        return $false
    }

    $Stdout = Join-Path $OutputDirectory "$Name.stdout.log"
    $Stderr = Join-Path $OutputDirectory "$Name.stderr.log"
    $Started = [DateTimeOffset]::UtcNow
    $NativeArguments = Join-NativeArguments $ArgumentList
    $Process = $null
    $Job = [P1Native.JobObject]::new()
    $StopReason = 'completed'
    $StageParseFailures = 0
    $PhasePeakWorkingSet = [int64]0
    $PhasePeakPrivateBytes = [int64]0
    try {
        $Process = Start-Process -FilePath $FilePath -ArgumentList $NativeArguments `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
        try {
            $Job.Assign($Process)
        } catch {
            & (Join-Path $env:SystemRoot 'System32\taskkill.exe') /PID $Process.Id /T /F *> $null
            throw "could not place phase $Name in its process-tree Job Object: $($_.Exception.Message)"
        }
        Write-JsonAtomic (Join-Path $OutputDirectory "$Name.started.json") ([ordered]@{
            phase = $Name
            pid = $Process.Id
            started_at = $Started.ToString('o')
            executable = $FilePath
            arguments = $ArgumentList
            native_argument_string = $NativeArguments
            process_tree_job_object = $true
        })

        while ($true) {
            $ProcessIds = @($Job.GetProcessIds())
            if ($ProcessIds.Count -eq 0) { break }
            $Memory = Get-JobMemorySnapshot $Job
            $PhasePeakWorkingSet = [Math]::Max($PhasePeakWorkingSet, [int64]$Memory.working_set_bytes)
            $PhasePeakPrivateBytes = [Math]::Max($PhasePeakPrivateBytes, [int64]$Memory.private_bytes)
            $script:PeakTreeWorkingSet = [Math]::Max($script:PeakTreeWorkingSet, [int64]$Memory.working_set_bytes)
            $script:PeakTreePrivateBytes = [Math]::Max($script:PeakTreePrivateBytes, [int64]$Memory.private_bytes)

            if ([int64]$Memory.working_set_bytes -gt $MaxWorkingSetBytes) {
                $StopReason = 'process-tree working set exceeded 8 GiB'
            } elseif ([DateTimeOffset]::UtcNow -ge $script:Deadline) {
                $StopReason = 'hard 35-minute wall reached'
            } elseif ($Name -eq 'train_partial') {
                try {
                    $TrainingStop = Get-TrainingStageStopReason
                    $StageParseFailures = 0
                    if ($null -ne $TrainingStop) { $StopReason = $TrainingStop }
                } catch {
                    $StageParseFailures += 1
                    if ($StageParseFailures -ge 5) {
                        $StopReason = "training stage monitor was unreadable for five consecutive samples: $($_.Exception.Message)"
                    }
                }
            }

            if ($StopReason -ne 'completed') {
                try { $Job.Terminate(222) } catch { $StopReason += "; job termination error: $($_.Exception.Message)" }
                break
            }
            Start-Sleep -Milliseconds 200
        }

        if ($StopReason -ne 'completed') {
            $KillWait = [Diagnostics.Stopwatch]::StartNew()
            while (@($Job.GetProcessIds()).Count -ne 0 -and $KillWait.Elapsed.TotalSeconds -lt 10) {
                Start-Sleep -Milliseconds 50
            }
        }
        $Process.WaitForExit()
        $ExitCode = if ($StopReason -eq 'completed') { $Process.ExitCode } else { -999 }
    } catch {
        if ($StopReason -eq 'completed') {
            $StopReason = "phase launch/monitor exception: $($_.Exception.Message)"
        }
        if ($null -ne $Process -and -not $Process.HasExited) {
            try { $Job.Terminate(223) } catch { }
        }
        $ExitCode = -998
    } finally {
        $Job.Dispose()
    }

    $Ended = [DateTimeOffset]::UtcNow
    $EffectiveReason = if ($ExitCode -eq 0) { 'completed' } elseif ($StopReason -eq 'completed') { "process exit $ExitCode" } else { $StopReason }
    $Record = [ordered]@{
        phase = $Name
        pid = if ($null -eq $Process) { $null } else { $Process.Id }
        started_at = $Started.ToString('o')
        ended_at = $Ended.ToString('o')
        seconds = ($Ended - $Started).TotalSeconds
        exit_code = $ExitCode
        stop_reason = $EffectiveReason
        peak_process_tree_working_set_bytes = $PhasePeakWorkingSet
        peak_process_tree_private_bytes = $PhasePeakPrivateBytes
        stdout = $Stdout
        stderr = $Stderr
    }
    $script:PhaseRecords += $Record
    if ($ExitCode -ne 0) {
        Add-Failure "phase $Name failed: $EffectiveReason"
    }
    Write-MonitorSnapshot
    return $ExitCode -eq 0
}

function Confirm-JsonArtifact([string]$Phase, [string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Phase did not create required artifact: $Path"
    }
    $Value = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $ExpectedStatuses = @{
        eligibility = 'training_eligibility_complete'
        select_rows = 'training_row_freeze_complete'
        extract_training = 'training_extraction_complete'
        train_partial = 'training_phase_complete'
        verify_openvino = 'openvino_gate_pass'
        extract_offline = 'offline_extraction_complete'
        offline_gate = 'offline_gate_pass'
        evaluate_development = 'P1-development-pass'
    }
    if ($ExpectedStatuses.ContainsKey($Phase)) {
        if (-not ($Value.PSObject.Properties.Name -contains 'status') -or
            [string]$Value.status -cne [string]$ExpectedStatuses[$Phase]) {
            throw "$Phase artifact status '$($Value.status)' != '$($ExpectedStatuses[$Phase])'"
        }
    }
    foreach ($BooleanName in @('success', 'passed', 'gate_passed')) {
        if ($Value.PSObject.Properties.Name -contains $BooleanName) {
            if ($Value.$BooleanName -is [bool] -and -not [bool]$Value.$BooleanName) {
                throw "$Phase artifact reports $BooleanName=false"
            }
        }
    }
    foreach ($TextName in @('status', 'decision', 'result')) {
        if ($Value.PSObject.Properties.Name -contains $TextName) {
            if ([string]$Value.$TextName -match '(?i)(fail|reject|infeasible|error)') {
                throw "$Phase artifact reports $TextName=$($Value.$TextName)"
            }
        }
    }
}

function Invoke-AndConfirm(
    [string]$Name,
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$ArtifactPath
) {
    if (-not (Invoke-MonitoredPhase $Name $FilePath $Arguments)) {
        return $false
    }
    try {
        Confirm-JsonArtifact $Name $ArtifactPath
        return $true
    } catch {
        Add-Failure $_.Exception.Message
        Write-MonitorSnapshot
        return $false
    }
}

function Add-Skipped([string[]]$Names, [string]$Reason) {
    foreach ($Name in $Names) {
        $script:SkippedPhases += [ordered]@{ phase = $Name; reason = $Reason }
    }
}

$PipelineOk = $true
try {
    $PipelineOk = Invoke-AndConfirm 'eligibility' $SystemPython @(
        (Join-Path $Experiment 'extract_dataset.py'), 'eligibility', $Dataset, $EligibilityPath
    ) $EligibilityPath
    if ($PipelineOk) {
        $PipelineOk = Invoke-AndConfirm 'select_rows' $JuliaExe @(
            '--startup-file=no', "--project=$Repository", '--threads=20',
            (Join-Path $Experiment 'select_rows.jl'), $EligibilityPath, $RowFreezePath
        ) $RowFreezePath
    }
    if ($PipelineOk) {
        $PipelineOk = Invoke-AndConfirm 'extract_training' $SystemPython @(
            (Join-Path $Experiment 'extract_dataset.py'), 'training', $Dataset,
            $RowFreezePath, $TrainingNpz, $TrainingExtractionPath
        ) $TrainingExtractionPath
    }
    if ($PipelineOk) {
        $PipelineOk = Invoke-AndConfirm 'train_partial' $JuliaExe @(
            '--startup-file=no', "--project=$Repository", '--threads=20',
            (Join-Path $Experiment 'train_partial.jl'), $OutputDirectory, $TrainingNpz,
            $Checkpoint, $FreezePath, $RowFreezePath
        ) $TrainingPhasePath
    }
    if ($PipelineOk) {
        $PipelineOk = Invoke-AndConfirm 'verify_openvino' $OpenVinoPython @(
            (Join-Path $Experiment 'verify_openvino.py'), $Repository, $OutputDirectory,
            $CandidateWeights, $FinalReference, $OpenVinoGatePath
        ) $OpenVinoGatePath
    }
    if ($PipelineOk) {
        # The offline split is intentionally not even opened until all
        # training, merge/export, and fresh CPU/NPU equivalence gates pass.
        $PipelineOk = Invoke-AndConfirm 'extract_offline' $SystemPython @(
            (Join-Path $Experiment 'extract_dataset.py'), 'offline', $Dataset,
            $OfflineNpz, $OfflineExtractionPath
        ) $OfflineExtractionPath
    }
    if ($PipelineOk) {
        $PipelineOk = Invoke-AndConfirm 'offline_gate' $JuliaExe @(
            '--startup-file=no', "--project=$Repository", '--threads=20',
            (Join-Path $Experiment 'offline_gate.jl'), $OfflineNpz,
            $CandidateWeights, $OfflineGatePath
        ) $OfflineGatePath
    }
    if ($PipelineOk) {
        $PipelineOk = Invoke-AndConfirm 'evaluate_development' $JuliaExe @(
            '--startup-file=no', "--project=$Repository", '--threads=20',
            (Join-Path $Experiment 'evaluate_development.jl'), $Repository,
            $CandidateWeights, $BaselineWeights, $DevelopmentPath
        ) $DevelopmentPath
    }
} catch {
    $PipelineOk = $false
    Add-Failure "wrapper pipeline exception: $($_.Exception.Message)"
}

if (-not $PipelineOk) {
    $Launched = @($script:PhaseRecords | ForEach-Object { $_.phase })
    $Remaining = @(
        'eligibility', 'select_rows', 'extract_training', 'train_partial',
        'verify_openvino', 'extract_offline', 'offline_gate', 'evaluate_development'
    ) | Where-Object { $_ -notin $Launched }
    Add-Skipped $Remaining 'a prior required phase or gate failed; no rescue or later data role is permitted'
}

# Give the finalizer the completed pre-finalizer phase ledger.  It is the only
# phase allowed after a gate failure, and it never trains, exports, or evaluates.
Write-MonitorSnapshot
$FinalizerOk = $false
if ([DateTimeOffset]::UtcNow -lt $script:Deadline) {
    $FinalizerOk = Invoke-MonitoredPhase 'finalize' $SystemPython @(
        (Join-Path $Experiment 'finalize_result.py'), $OutputDirectory
    )
} else {
    $Reason = 'hard 35-minute wall exhausted; finalizer was not launched'
    Add-Failure $Reason
    Add-Skipped @('finalize') $Reason
}

if ($FinalizerOk) {
    try {
        if (-not (Test-Path -LiteralPath $FinalResultPath -PathType Leaf)) {
            throw 'finalizer did not create final_result.json'
        }
        $FinalResult = Get-Content -LiteralPath $FinalResultPath -Raw | ConvertFrom-Json
        if (-not ($FinalResult.PSObject.Properties.Name -contains 'success') -or
            $FinalResult.success -isnot [bool]) {
            throw 'final_result.json lacks a Boolean success field'
        }
        if (-not [bool]$FinalResult.success) {
            $FinalFailures = @($FinalResult.failures) -join '; '
            Add-Failure "P1 finalizer rejected the fixed candidate: $FinalFailures"
        }
    } catch {
        Add-Failure "invalid finalizer result: $($_.Exception.Message)"
        $FinalizerOk = $false
    }
}

if ([DateTimeOffset]::UtcNow -gt $script:Deadline -and
    -not (@($script:Failures) -match 'hard 35-minute wall')) {
    Add-Failure 'hard wall exceeded 35 minutes'
}
if ($script:PeakTreeWorkingSet -gt $MaxWorkingSetBytes -and
    -not (@($script:Failures) -match 'working set exceeded')) {
    Add-Failure 'process-tree peak working set exceeded 8 GiB'
}
Write-MonitorSnapshot $true

$WrapperSuccess = $PipelineOk -and $FinalizerOk -and $script:Failures.Count -eq 0
Write-JsonAtomic $WrapperResultPath ([ordered]@{
    experiment = $ExperimentId
    status = if ($WrapperSuccess) { 'wrapper-complete' } else { 'wrapper-failed' }
    success = $WrapperSuccess
    failures = @($script:Failures)
    source_commit = $Commit
    freeze_path = $FreezePath
    monitor_path = $MonitorPath
    final_result_path = $FinalResultPath
    final_result_present = Test-Path -LiteralPath $FinalResultPath -PathType Leaf
    global_one_shot_marker = $GlobalStartedPath
    retry_prohibited = $true
    output_directory = $OutputDirectory
}) 16

if (-not $WrapperSuccess) { exit 2 }
exit 0
