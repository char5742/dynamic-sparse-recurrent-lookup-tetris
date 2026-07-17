[CmdletBinding()]
param(
    [string]$Repository,
    [string]$OutputDirectory,
    [string]$SourceFingerprint,
    [string]$StartGate,
    [Alias('AuthorizedCommit')]
    [string]$AuthorizedHardeningCommit,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExperimentId = 'frozen_old_additive_residual_Q1'
$GlobalStartedPath = 'D:\tetris-paper-plus\runs\frozen_old_additive_residual_Q1.started.json'
$OutputRoot = [IO.Path]::GetFullPath('D:\tetris-paper-plus\runs')
$HardWallSeconds = 720.0
$MaxProcessTreeBytes = [int64](4GB)
$SystemPython = 'C:\Program Files\Python310\python.exe'
$OpenVinoPython = 'D:\tetris-paper-plus\python-env\Scripts\python.exe'
$ExpectedInputs = [ordered]@{
    old_checkpoint = @('1313\mainmodel copy 3.jld2', '7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1')
    old_openvino = @('artifacts\legacy_openvino\legacy_1313_weights.npz', '2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba')
    initializer = @('D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2', '1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270')
    dataset = @('D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2', '4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba')
    authorization = @('reports\clean_post_p1_parse_failure_strategy.md', 'f0cd7bce2c39b353a3377dc2ebdd624ab485a2b96c5750f4bc97e7fd91a5cf00')
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Repository = Join-Path $PSScriptRoot '..\..'
}
$Repository = [IO.Path]::GetFullPath($Repository)
$Experiment = Join-Path $Repository 'experiments\frozen_old_additive_residual_q1'
. (Join-Path $Experiment 'native_arguments.ps1')
. (Join-Path $Experiment 'result_composition.ps1')

$RequiredFiles = @(
    'contract.jl', 'model.jl', 'freeze_order.jl', 'train_q1.jl',
    'extract_dataset.py', 'correction_openvino.py', 'verify_openvino.py',
    'offline_gate.py', 'validate_source_fingerprint.py', 'finalize_result.py',
    'native_arguments.ps1', 'result_composition.ps1', 'invoke_once.ps1',
    'test_contract.jl', 'test_export_gate_contract.jl',
    'test_export_gate_static.py', 'test_source_fingerprint.py',
    'test_freeze_order_production.py',
    'test_native_arguments.ps1', 'test_finalization.py',
    'test_result_composition.ps1', 'test_static.py', 'EXPERIMENT.md'
)
$JuliaEntrypoints = @(
    'freeze_order.jl', 'train_q1.jl', 'test_contract.jl',
    'test_export_gate_contract.jl'
)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Q1HarnessRecords {
    $Records = @()
    $Builder = [Text.StringBuilder]::new()
    foreach ($Name in $RequiredFiles) {
        $Path = Join-Path $Experiment $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "missing required Q1 harness file: $Path"
        }
    }
    foreach ($File in @(Get-ChildItem -LiteralPath $Experiment -File | Sort-Object Name)) {
        $Hash = Get-Sha256 $File.FullName
        $Records += [ordered]@{
            path = "experiments/frozen_old_additive_residual_q1/$($File.Name)"
            sha256 = $Hash
        }
        [void]$Builder.Append("$($File.Name)`0$Hash`n")
    }
    $Sha = [Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [Text.Encoding]::UTF8.GetBytes($Builder.ToString())
        $Aggregate = ([BitConverter]::ToString($Sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    } finally { $Sha.Dispose() }
    return [ordered]@{ aggregate_sha256 = $Aggregate; files = $Records }
}

function Test-Q1HarnessPreflight {
    foreach ($Name in $RequiredFiles) {
        $Path = Join-Path $Experiment $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "missing required Q1 harness file: $Path"
        }
    }
    $PowerShellFiles = @(Get-ChildItem -LiteralPath $Experiment -Filter '*.ps1' -File)
    foreach ($File in $PowerShellFiles) {
        $Tokens = $null; $Errors = $null
        [void][Management.Automation.Language.Parser]::ParseFile(
            $File.FullName, [ref]$Tokens, [ref]$Errors
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
    $PythonFiles = @(Get-ChildItem -LiteralPath $Experiment -Filter '*.py' -File | ForEach-Object FullName)
    $AstCode = 'import ast,pathlib,sys;[ast.parse(pathlib.Path(p).read_bytes(),filename=p) for p in sys.argv[1:]]'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $SystemPython -c $AstCode @PythonFiles | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Python AST parse failed' }

    $JuliaExe = (Get-Command julia -ErrorAction Stop).Source
    $JuliaFiles = @(Get-ChildItem -LiteralPath $Experiment -Filter '*.jl' -File | ForEach-Object FullName)
    $ParseCode = 'for path in ARGS; Base.JuliaSyntax.parseall(Base.JuliaSyntax.SyntaxNode, read(path,String); filename=path, ignore_errors=false); end'
    & $JuliaExe --startup-file=no --history-file=no -e $ParseCode @JuliaFiles | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'strict JuliaSyntax parseall failed' }

    # Prove the parser gate rejects the exact macro-precedence class that
    # previously escaped a Meta.parseall-only preflight.
    $BadJulia = Join-Path ([IO.Path]::GetTempPath()) ('q1-bad-' + [guid]::NewGuid().ToString('N') + '.jl')
    try {
        [IO.File]::WriteAllText($BadJulia, "abspath(PROGRAM_FILE) == @__FILE__ && main()`n")
        $RejectProbe = 'try; Base.JuliaSyntax.parseall(Base.JuliaSyntax.SyntaxNode, read(ARGS[1],String); filename=ARGS[1], ignore_errors=false); exit(2); catch; exit(0); end'
        & $JuliaExe --startup-file=no --history-file=no -e $RejectProbe $BadJulia | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'strict Julia parser accepted malformed macro-precedence fixture' }
    } finally { Remove-Item -LiteralPath $BadJulia -Force -ErrorAction SilentlyContinue }

    foreach ($Name in $JuliaEntrypoints) {
        $Path = Join-Path $Experiment $Name
        $Source = Get-Content -LiteralPath $Path -Raw
        if ($Source -notmatch 'if\s+abspath\(PROGRAM_FILE\)\s*==\s*abspath\(@__FILE__\)') {
            throw "Julia entrypoint lacks the exact guarded-main idiom: $Name"
        }
        & $JuliaExe --startup-file=no --history-file=no "--project=$Repository" $Path --self-check | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "fresh Julia startup self-check failed: $Name" }
    }
    foreach ($Test in @('test_source_fingerprint.py', 'test_export_gate_static.py', 'test_freeze_order_production.py', 'test_finalization.py', 'test_static.py')) {
        & $SystemPython (Join-Path $Experiment $Test) | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Python synthetic test failed: $Test" }
    }
    foreach ($Test in @('test_native_arguments.ps1', 'test_result_composition.ps1')) {
        & (Join-Path $Experiment $Test) | Out-Null
        if (-not $?) { throw "PowerShell synthetic test failed: $Test" }
    }
    return [ordered]@{
        powershell_files = $PowerShellFiles.Count
        python_files = $PythonFiles.Count
        julia_files = $JuliaFiles.Count
        fresh_julia_entrypoints = $JuliaEntrypoints.Count
        freeze_order_production_branch_executed = $true
        freeze_order_production_branch_used_synthetic_eligibility = $true
        malformed_julia_fixture_rejected = $true
        checkpoint_or_dataset_opened = $false
        model_or_openvino_loaded = $false
        game_run = $false
        marker_read_or_written = $false
    }
}

$Syntax = Test-Q1HarnessPreflight
if ($ValidateOnly) {
    [ordered]@{
        status = 'Q1-validation-only-passed'
        experiment = $ExperimentId
        syntax = $Syntax
        note = 'no real checkpoint/model/dataset/OpenVINO/game/global-marker access'
    } | ConvertTo-Json -Depth 12
    exit 0
}

foreach ($Name in @('OutputDirectory', 'SourceFingerprint', 'StartGate', 'AuthorizedHardeningCommit')) {
    if ([string]::IsNullOrWhiteSpace([string](Get-Variable -Name $Name -ValueOnly))) {
        throw "-$Name is required"
    }
}
if ($AuthorizedHardeningCommit -notmatch '^[0-9a-fA-F]{40}$') {
    throw '-AuthorizedHardeningCommit must be full 40-hex'
}
$AuthorizedHardeningCommit = $AuthorizedHardeningCommit.ToLowerInvariant()
$RequiredStartText = "START $ExperimentId $AuthorizedHardeningCommit"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$SourceFingerprint = [IO.Path]::GetFullPath($SourceFingerprint)
$StartGate = [IO.Path]::GetFullPath($StartGate)
$OutputPrefix = $OutputRoot.TrimEnd('\') + '\'
if (-not $OutputDirectory.StartsWith($OutputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Q1 output must be below $OutputRoot"
}
if (Test-Path -LiteralPath $OutputDirectory) { throw "Q1 output already exists: $OutputDirectory" }
if (Test-Path -LiteralPath $StartGate) { throw "Q1 start gate must initially be absent" }
if (Test-Path -LiteralPath $GlobalStartedPath) { throw "global Q1 marker exists; retry prohibited" }
if (-not (Test-Path -LiteralPath $SourceFingerprint -PathType Leaf)) { throw 'source fingerprint missing' }

$JuliaExe = (Get-Command julia -ErrorAction Stop).Source
$InputRecords = @()
foreach ($Name in $ExpectedInputs.Keys) {
    $Spec = $ExpectedInputs[$Name]
    $Path = if ([IO.Path]::IsPathRooted($Spec[0])) { $Spec[0] } else { Join-Path $Repository $Spec[0] }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "missing immutable input ${Name}: $Path" }
    $Observed = Get-Sha256 $Path
    if ($Observed -ne $Spec[1]) { throw "immutable input $Name SHA mismatch: $Observed" }
    $InputRecords += [ordered]@{ label=$Name; path=[IO.Path]::GetFullPath($Path); expected=$Spec[1]; observed=$Observed }
}
$Manifest = Join-Path $Repository 'Manifest.toml'
$ManifestText = Get-Content -LiteralPath $Manifest -Raw
if ($ManifestText -notmatch '(?s)\[\[deps\.Lux\]\].*?version = "1\.31\.4"' -or
    $ManifestText -notmatch '(?s)\[\[deps\.Zygote\]\].*?version = "0\.7\.11"') {
    throw 'Manifest does not pin Lux 1.31.4 and Zygote 0.7.11'
}
$JuliaVersion = (& $JuliaExe --version).Trim()
if ($JuliaVersion -ne 'julia version 1.12.6') { throw "Julia version mismatch: $JuliaVersion" }

$AuditText = @(& $SystemPython (Join-Path $Experiment 'validate_source_fingerprint.py') `
    $Repository $SourceFingerprint $AuthorizedHardeningCommit 2>&1) -join [Environment]::NewLine
try { $SourceAudit = $AuditText | ConvertFrom-Json } catch { throw "malformed source audit: $AuditText" }
if ($LASTEXITCODE -ne 0 -or -not [bool]$SourceAudit.valid) {
    throw "source binding rejected: $($SourceAudit.failures -join '; ')"
}
$Harness = Get-Q1HarnessRecords
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$EligibilityPath = Join-Path $OutputDirectory 'eligibility.json'
$OrderPath = Join-Path $OutputDirectory 'order_freeze.json'
$Dataset = ($InputRecords | Where-Object label -eq 'dataset').path
$EligibilityStdout = Join-Path $OutputDirectory 'eligibility.stdout.log'
$EligibilityStderr = Join-Path $OutputDirectory 'eligibility.stderr.log'
& $SystemPython (Join-Path $Experiment 'extract_dataset.py') eligibility $Dataset $EligibilityPath `
    1>$EligibilityStdout 2>$EligibilityStderr
if ($LASTEXITCODE -ne 0) { throw "training-only eligibility preflight failed" }
& $JuliaExe --startup-file=no --history-file=no "--project=$Repository" `
    (Join-Path $Experiment 'freeze_order.jl') $EligibilityPath $OrderPath `
    1>(Join-Path $OutputDirectory 'order_freeze.stdout.log') `
    2>(Join-Path $OutputDirectory 'order_freeze.stderr.log')
if ($LASTEXITCODE -ne 0) { throw 'Q1 order freeze failed' }
$Eligibility = Get-Content -LiteralPath $EligibilityPath -Raw | ConvertFrom-Json
$Order = Get-Content -LiteralPath $OrderPath -Raw | ConvertFrom-Json
if ($Eligibility.offline_rows_loaded -isnot [bool] -or [bool]$Eligibility.offline_rows_loaded) {
    throw 'offline rows were loaded before candidate freeze'
}
if ($Order.ordered_rows.Count -ne 8000 -or $Order.minibatches.Count -ne 2000) {
    throw 'Q1 row-order freeze cardinality mismatch'
}

$FreezePath = Join-Path $OutputDirectory 'freeze.json'
$ReadyPath = Join-Path $OutputDirectory 'ready.json'
$MonitorPath = Join-Path $OutputDirectory 'monitor.json'
$WrapperResultPath = Join-Path $OutputDirectory 'wrapper_result.json'
$FinalResultPath = Join-Path $OutputDirectory 'final_result.json'
$FreezeCreated = [DateTimeOffset]::UtcNow
$Commands = [ordered]@{
    extract_training = @($SystemPython, (Join-Path $Experiment 'extract_dataset.py'), 'training', $Dataset, (Join-Path $OutputDirectory 'training.npz'), (Join-Path $OutputDirectory 'training_extraction.json'))
    train_correction = @($JuliaExe, '--startup-file=no', '--history-file=no', "--project=$Repository", '--threads=20', (Join-Path $Experiment 'train_q1.jl'), $OutputDirectory, (Join-Path $OutputDirectory 'training.npz'), ($InputRecords | Where-Object label -eq 'initializer').path, $FreezePath, $OrderPath, ($InputRecords | Where-Object label -eq 'old_checkpoint').path)
    verify_openvino = @($OpenVinoPython, (Join-Path $Experiment 'verify_openvino.py'), $OutputDirectory, (Join-Path $OutputDirectory 'correction_weights.npz'), (Join-Path $OutputDirectory 'combined_reference.npz'), (Join-Path $OutputDirectory 'openvino_gate.json'))
    extract_offline = @($SystemPython, (Join-Path $Experiment 'extract_dataset.py'), 'offline', $Dataset, (Join-Path $OutputDirectory 'offline.npz'), (Join-Path $OutputDirectory 'offline_extraction.json'))
    offline_gate = @($OpenVinoPython, (Join-Path $Experiment 'offline_gate.py'), (Join-Path $OutputDirectory 'offline.npz'), $OutputDirectory, (Join-Path $OutputDirectory 'correction_weights.npz'), (Join-Path $OutputDirectory 'correction_dynamic.xml'), (Join-Path $OutputDirectory 'offline_gate.json'))
    finalize_assessment = @($SystemPython, (Join-Path $Experiment 'finalize_result.py'), $OutputDirectory)
}
$Freeze = [ordered]@{
    experiment=$ExperimentId
    scientific_role='one-shot frozen-old additive residual offline safety probe'
    frozen_at=$FreezeCreated.ToString('o')
    source_commit=$AuthorizedHardeningCommit
    authorized_hardening_commit=$AuthorizedHardeningCommit
    authorized_base_commit='8d784985f300598d2a05ed4402902ae86dfb4908'
    actual_parent_commit=$SourceAudit.repository_binding.parent
    repository_clean=$true
    source_fingerprint_path=$SourceFingerprint
    source_fingerprint_sha256=Get-Sha256 $SourceFingerprint
    source_fingerprint_audit=$SourceAudit
    source_fingerprint=$SourceAudit.fingerprint
    manifest_path=$Manifest
    manifest_sha256=Get-Sha256 $Manifest
    immutable_inputs=$InputRecords
    harness_sha256=$Harness.aggregate_sha256
    harness_files=$Harness.files
    strict_syntax_and_fresh_startup_preflight=$Syntax
    eligibility_path=$EligibilityPath
    eligibility_sha256=Get-Sha256 $EligibilityPath
    eligibility=$Eligibility
    order_freeze_path=$OrderPath
    order_freeze_sha256=Get-Sha256 $OrderPath
    ordered_rows_sha256=$Order.ordered_rows_sha256
    constants=[ordered]@{
        actions=74; batch=4; updates=2000; rng='Xoshiro(0x5131_2026)'
        n_step=3; parameter_count=165051; hard_wall_seconds=720
        max_process_tree_bytes=$MaxProcessTreeBytes
    }
    updates=2000
    state_batch=4
    candidate_actions=74
    offline_loaded_before_candidate_freeze=$false
    initializer_exposed_to_offline_rows=$true
    offline_role='reused_development_guard'
    output_directory=$OutputDirectory
    global_one_shot_marker=$GlobalStartedPath
    start_gate=$StartGate
    required_start_gate_contents=$RequiredStartText
    commands=$Commands
    limits=[ordered]@{ hard_wall_seconds=720; max_process_tree_bytes=$MaxProcessTreeBytes; first_update_seconds=60; warm_update_seconds=1; warm_median_seconds=0.25 }
    validation_seed_used=$false
    sealed_test_seed_used=$false
    game_run=$false
}
Write-Q1JsonDurableAtomic $FreezePath $Freeze
Write-Q1JsonDurableAtomic $ReadyPath ([ordered]@{
    status='frozen_waiting_for_explicit_start_gate'; experiment=$ExperimentId
    wrapper_pid=$PID; source_commit=$AuthorizedHardeningCommit
    authorized_hardening_commit=$AuthorizedHardeningCommit
    authorized_base_commit='8d784985f300598d2a05ed4402902ae86dfb4908'
    actual_parent_commit=$SourceAudit.repository_binding.parent; freeze_path=$FreezePath
    freeze_sha256=Get-Sha256 $FreezePath; required_start_gate_contents=$RequiredStartText
    offline_rows_loaded=$false; one_shot_marker_created=$false
})

$GateDeadline = [DateTimeOffset]::UtcNow.AddMinutes(10)
while (-not (Test-Path -LiteralPath $StartGate -PathType Leaf)) {
    if ([DateTimeOffset]::UtcNow -gt $GateDeadline) { throw 'Q1 start gate timeout before marker' }
    Start-Sleep -Milliseconds 200
}
if ((Get-Content -LiteralPath $StartGate -Raw).Trim() -cne $RequiredStartText) {
    throw 'Q1 start gate content mismatch'
}
if ((Get-Item -LiteralPath $StartGate).LastWriteTimeUtc -lt $FreezeCreated.UtcDateTime) {
    throw 'Q1 start gate predates freeze'
}
$GateHarness = Get-Q1HarnessRecords
if ($GateHarness.aggregate_sha256 -ne $Harness.aggregate_sha256) { throw 'Q1 harness changed while waiting' }
foreach ($Record in $InputRecords) {
    if ((Get-Sha256 $Record.path) -ne $Record.expected) { throw "immutable input changed at gate: $($Record.label)" }
}
$GateAuditText = @(& $SystemPython (Join-Path $Experiment 'validate_source_fingerprint.py') `
    $Repository $SourceFingerprint $AuthorizedHardeningCommit 2>&1) -join [Environment]::NewLine
try { $GateAudit = $GateAuditText | ConvertFrom-Json } catch { throw 'malformed gate source audit' }
if ($LASTEXITCODE -ne 0 -or -not [bool]$GateAudit.valid) { throw 'source binding changed at gate' }
if (Test-Path -LiteralPath $GlobalStartedPath) { throw 'global Q1 marker appeared concurrently' }

$Marker = [ordered]@{
    experiment=$ExperimentId; started_at=[DateTimeOffset]::UtcNow.ToString('o')
    source_commit=$AuthorizedHardeningCommit
    authorized_hardening_commit=$AuthorizedHardeningCommit
    authorized_base_commit='8d784985f300598d2a05ed4402902ae86dfb4908'
    actual_parent_commit=$GateAudit.repository_binding.parent
    output_directory=$OutputDirectory
    freeze_sha256=Get-Sha256 $FreezePath; order_freeze_sha256=Get-Sha256 $OrderPath
    retry_prohibited=$true; rescue_prohibited=$true
}
Write-Q1JsonDurableAtomic $GlobalStartedPath $Marker
Write-Q1JsonDurableAtomic (Join-Path $OutputDirectory 'started.json') $Marker

function Get-Q1JobMemory($Job) {
    $Working = [int64]0; $Private = [int64]0
    foreach ($Id in @($Job.GetProcessIds())) {
        try {
            $Process = Get-Process -Id $Id -ErrorAction Stop
            $Working += [int64]$Process.WorkingSet64
            $Private += [int64]$Process.PrivateMemorySize64
        } catch { }
    }
    return [ordered]@{ working=$Working; private=$Private }
}

function Invoke-Q1Phase([string]$Name, [string[]]$Command, [datetimeoffset]$RunStarted) {
    $CurrentHarness = Get-Q1HarnessRecords
    if ($CurrentHarness.aggregate_sha256 -ne $Harness.aggregate_sha256) {
        throw "harness changed before phase $Name"
    }
    $Stdout = Join-Path $OutputDirectory "$Name.stdout.log"
    $Stderr = Join-Path $OutputDirectory "$Name.stderr.log"
    Write-Q1JsonDurableAtomic (Join-Path $OutputDirectory "$Name.started.json") ([ordered]@{
        phase=$Name; started_at=[DateTimeOffset]::UtcNow.ToString('o')
        command=$Command; harness_sha256=$CurrentHarness.aggregate_sha256
    })
    $Job = [Q1Native.JobObject]::new()
    $Started = [DateTimeOffset]::UtcNow
    $PeakWorking = [int64]0; $PeakPrivate = [int64]0
    $StopReason = 'completed'
    try {
        $Arguments = if ($Command.Count -gt 1) { Join-NativeArguments $Command[1..($Command.Count - 1)] } else { '' }
        $Process = Start-Process -FilePath $Command[0] -ArgumentList $Arguments `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
        $Job.Assign($Process)
        while (-not $Process.HasExited) {
            $Memory = Get-Q1JobMemory $Job
            $PeakWorking = [Math]::Max($PeakWorking, [int64]$Memory.working)
            $PeakPrivate = [Math]::Max($PeakPrivate, [int64]$Memory.private)
            if ($Memory.working -gt $MaxProcessTreeBytes -or $Memory.private -gt $MaxProcessTreeBytes) {
                $StopReason = 'process-tree memory exceeded 4 GiB'; $Job.Terminate(137); break
            }
            if (([DateTimeOffset]::UtcNow - $RunStarted).TotalSeconds -gt $HardWallSeconds) {
                $StopReason = '12-minute hard wall exceeded'; $Job.Terminate(124); break
            }
            Start-Sleep -Milliseconds 200
            $Process.Refresh()
        }
        $Process.WaitForExit(); $Process.Refresh()
        $ExitCode = if ($StopReason -eq 'completed') { [int]$Process.ExitCode } else { 1 }
        if ($ExitCode -ne 0 -and $StopReason -eq 'completed') { $StopReason = "process exit $ExitCode" }
        return [ordered]@{
            phase=$Name; exit_code=$ExitCode; stop_reason=$StopReason
            seconds=([DateTimeOffset]::UtcNow - $Started).TotalSeconds
            peak_process_tree_working_set_bytes=$PeakWorking
            peak_process_tree_private_bytes=$PeakPrivate
            stdout=$Stdout; stderr=$Stderr
        }
    } finally { $Job.Dispose() }
}

$RunStarted = [DateTimeOffset]::UtcNow
$Phases = @(); $Skipped = @(); $Failures = @(); $PipelineOpen = $true
$PeakWorking = [int64]0; $PeakPrivate = [int64]0
foreach ($Name in $script:Q1ExpectedPhaseOrder) {
    if ($Name -ne 'finalize_assessment' -and -not $PipelineOpen) {
        $Skipped += [ordered]@{ phase=$Name; reason='prior required phase failed; no rescue' }
        continue
    }
    try {
        $Record = Invoke-Q1Phase $Name $Commands[$Name] $RunStarted
        $Phases += $Record
        $PeakWorking = [Math]::Max($PeakWorking, [int64]$Record.peak_process_tree_working_set_bytes)
        $PeakPrivate = [Math]::Max($PeakPrivate, [int64]$Record.peak_process_tree_private_bytes)
        if ($Record.exit_code -ne 0) {
            $Failures += "phase $Name failed: $($Record.stop_reason)"
            $PipelineOpen = $false
        }
    } catch {
        $Failures += "phase $Name launch/setup failed: $($_.Exception.Message)"
        $PipelineOpen = $false
        $Phases += [ordered]@{
            phase=$Name; exit_code=1; stop_reason='wrapper launch/setup failure'; seconds=0.0
            peak_process_tree_working_set_bytes=0; peak_process_tree_private_bytes=0
        }
    }
}
$AssessmentPath = Join-Path $OutputDirectory 'assessment.json'
$Assessment = if (Test-Path -LiteralPath $AssessmentPath) {
    Get-Content -LiteralPath $AssessmentPath -Raw | ConvertFrom-Json
} else {
    [pscustomobject]@{ status='assessment-fail'; success=$false; failures=@('assessment artifact missing'); promotion='none'; scope='offline reused-development safety evidence only'; game_strength_evidence=$false; model_beat_claim=$false; game_evaluation_authorized=$false; validation_seed_used=$false; sealed_test_seed_used=$false; sealed_test_authorized=$false }
}
$Monitor = [ordered]@{
    complete=$true; ended_at=[DateTimeOffset]::UtcNow.ToString('o')
    wall_seconds=([DateTimeOffset]::UtcNow - $RunStarted).TotalSeconds
    peak_process_tree_working_set_bytes=$PeakWorking
    peak_process_tree_private_bytes=$PeakPrivate
    stop_reason=if ($Failures.Count -eq 0) { 'completed' } else { 'failed' }
    failures=@($Failures); phases=@($Phases); skipped_phases=@($Skipped)
}
$Provisional = New-Q1FinalResult $Assessment $Monitor
$Wrapper = [ordered]@{
    experiment=$ExperimentId; status=if ($Provisional.success) { 'wrapper-complete' } else { 'wrapper-failed' }
    success=[bool]$Provisional.success; source_commit=$AuthorizedHardeningCommit
    authorized_hardening_commit=$AuthorizedHardeningCommit
    authorized_base_commit='8d784985f300598d2a05ed4402902ae86dfb4908'
    actual_parent_commit=$GateAudit.repository_binding.parent
    global_one_shot_marker=$GlobalStartedPath; retry_prohibited=$true
    final_result_is_authoritative_terminal_artifact=$true
}
$Published = Write-Q1TerminalArtifacts $MonitorPath $WrapperResultPath $FinalResultPath `
    $Assessment $Monitor $Wrapper
if ($Published.success) { exit 0 } else { exit 2 }
