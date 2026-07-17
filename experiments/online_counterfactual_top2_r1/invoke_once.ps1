[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [string]$AuthorizedCommit = '',
    [string]$OutputDirectory = '',
    [string]$StartGatePath = 'D:\tetris-paper-plus\runs\online_counterfactual_top2_R1.start.txt'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'wrapper_runtime.ps1')
. (Join-Path $PSScriptRoot 'result_composition.ps1')

$R1OriginalPycachePrefix = [Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process')
$R1PycachePrefix = Join-Path 'D:\tetris-paper-plus\python-pycache' (
    "online_counterfactual_top2_r1_${PID}_$([guid]::NewGuid().ToString('N'))"
)
[IO.Directory]::CreateDirectory($R1PycachePrefix) | Out-Null
$env:PYTHONPYCACHEPREFIX = $R1PycachePrefix

function Get-R1ImmutableFileRecord([string]$Path) {
    $Full = [IO.Path]::GetFullPath($Path)
    if (-not [IO.File]::Exists($Full)) { throw "missing immutable R1 input: $Full" }
    $Item = Get-Item -LiteralPath $Full -Force
    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "immutable R1 input is a reparse point: $Full"
    }
    return [ordered]@{
        path=$Full
        bytes=[int64]$Item.Length
        sha256=(Get-FileHash -LiteralPath $Full -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Assert-R1ImmutableFileRecords($Records) {
    foreach ($Record in @($Records)) {
        $Current = Get-R1ImmutableFileRecord ([string]$Record.path)
        if ([int64]$Current.bytes -ne [int64]$Record.bytes -or
            [string]$Current.sha256 -cne [string]$Record.sha256) {
            throw "R1 immutable input changed after freeze: $($Record.path)"
        }
    }
}

function Invoke-R1ExternalChecked([string[]]$Command, [string]$Label) {
    & $Command[0] $Command[1..($Command.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit $LASTEXITCODE" }
}

function Invoke-R1ValidateOnly {
    $Repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    $Julia = Resolve-R1ConcreteJulia
    $Python = Resolve-R1FrozenPython
    $Manifest = Assert-R1FrozenManifest $Repository
    $PwshRecord = Resolve-R1ConcretePwsh (Get-Process -Id $PID).Path
    $Pwsh = $PwshRecord.path
    Invoke-R1ExternalChecked @(
        $Python.venv_launcher_path, (Join-Path $PSScriptRoot 'syntax_preflight.py'),
        $PSScriptRoot, '--julia', $Julia.path,
        '--entrypoint', 'freeze_design.jl',
        '--entrypoint', 'collect_online.jl'
    ) 'strict syntax preflight'
    Invoke-R1ExternalChecked @(
        $Python.venv_launcher_path, '-m', 'unittest', 'discover',
        '-s', $PSScriptRoot, '-p', 'test_*.py'
    ) 'Python synthetic suite'
    Invoke-R1ExternalChecked @(
        $Python.venv_launcher_path, (Join-Path $PSScriptRoot 'test_python_ridge_fit.py'),
        '--python', $Python.venv_launcher_path, '--julia', $Julia.path
    ) 'Python/Julia ridge fit conformance and handoff suite'
    foreach ($Test in @(
        'test_collector.jl','test_engine_adapter_resolution.jl',
        'test_ridge_gate.jl','test_calibration_gate.jl'
    )) {
        Invoke-R1ExternalChecked @(
            $Julia.path, '--startup-file=no', '--history-file=no', "--project=$Repository",
            (Join-Path $PSScriptRoot $Test)
        ) $Test
    }
    foreach ($Test in @('test_wrapper_runtime.ps1','test_result_composition.ps1','test_production_argv.ps1')) {
        Invoke-R1ExternalChecked @(
            $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-File', (Join-Path $PSScriptRoot $Test)
        ) $Test
    }
    [ordered]@{
        status='R1-validate-only-pass'
        julia=$Julia
        python=$Python
        pwsh=$PwshRecord
        manifest=$Manifest
        strict_syntax=$true
        all_five_production_argv_synthetic=$true
        producer_completion_sha_handoff_tamper_test=$true
        real_model_or_checkpoint_loaded=$false
        game_run=$false
        reserved_seed_loaded=$false
        global_marker_read_or_written=$false
    } | ConvertTo-Json -Depth 10
}

try {
if ($ValidateOnly) {
    if (-not [string]::IsNullOrWhiteSpace($AuthorizedCommit) -or
        -not [string]::IsNullOrWhiteSpace($OutputDirectory)) {
        throw '-ValidateOnly does not accept production commit/output arguments'
    }
    Invoke-R1ValidateOnly
    return
}

if ($AuthorizedCommit -notmatch '^[0-9a-f]{40}$') {
    throw 'production R1 requires -AuthorizedCommit with exact lower-case 40-hex HEAD'
}

$Repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$Julia = Resolve-R1ConcreteJulia
$Python = Resolve-R1FrozenPython
$Pwsh = Resolve-R1ConcretePwsh (Get-Process -Id $PID).Path
$Manifest = Assert-R1FrozenManifest $Repository
$RunRoot = 'D:\tetris-paper-plus\runs'
$GlobalMarker = Join-Path $RunRoot 'online_counterfactual_top2_R1.started.json'
if ([IO.File]::Exists($GlobalMarker)) { throw "R1 global one-shot marker is already consumed: $GlobalMarker" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $Stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $OutputDirectory = Join-Path $RunRoot "online_counterfactual_top2_R1_$($AuthorizedCommit.Substring(0,8))_$Stamp"
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $OutputDirectory.StartsWith(([IO.Path]::GetFullPath($RunRoot) + '\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'R1 output directory must be a fresh child of D:\tetris-paper-plus\runs'
}
if ([IO.Directory]::Exists($OutputDirectory) -or [IO.File]::Exists($OutputDirectory)) {
    throw "R1 output path is not fresh: $OutputDirectory"
}
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null

$Eligibility = Join-Path $OutputDirectory 'eligibility.json'
$DesignFreeze = Join-Path $OutputDirectory 'design_freeze.json'
$SourceFingerprint = Join-Path $OutputDirectory 'source_fingerprint.json'
$FreezePlan = Join-Path $OutputDirectory 'one_shot_freeze.json'
$ReadyPath = Join-Path $OutputDirectory 'ready_for_start_gate.json'
$TrainingTable = Join-Path $OutputDirectory 'training_table.json'
$TrainingManifest = Join-Path $OutputDirectory 'training_manifest.json'
$RidgeArtifact = Join-Path $OutputDirectory 'ridge_artifact.json'
$CalibrationTable = Join-Path $OutputDirectory 'calibration_table.json'
$CalibrationManifest = Join-Path $OutputDirectory 'calibration_manifest.json'
$CalibrationAssessment = Join-Path $OutputDirectory 'calibration_assessment.json'
$AssessmentPath = Join-Path $OutputDirectory 'assessment.json'
$MonitorPath = Join-Path $OutputDirectory 'monitor.json'
$WrapperPath = Join-Path $OutputDirectory 'wrapper_result.json'
$FinalPath = Join-Path $OutputDirectory 'final_result.json'

$StartGateNonceBytes = [byte[]]::new(32)
[Security.Cryptography.RandomNumberGenerator]::Fill($StartGateNonceBytes)
$StartGateNonce = [Convert]::ToHexString($StartGateNonceBytes).ToLowerInvariant()
$StartGateNonceSha256 = [Convert]::ToHexString(
    [Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($StartGateNonce))
).ToLowerInvariant()
$ExactStartGateText = "START online_counterfactual_top2_R1 $AuthorizedCommit $StartGateNonce"

Invoke-R1ExternalChecked @(
    $Python.venv_launcher_path, (Join-Path $PSScriptRoot 'make_eligibility.py'), $Eligibility
) 'eligibility freeze'
Invoke-R1ExternalChecked @(
    $Julia.path, '--startup-file=no', '--history-file=no', "--project=$Repository",
    (Join-Path $PSScriptRoot 'freeze_design.jl'), $Eligibility, $DesignFreeze
) 'design freeze'
Invoke-R1ExternalChecked @(
    $Python.venv_launcher_path, (Join-Path $PSScriptRoot 'validate_source_fingerprint.py'),
    '--generate', $Repository, $SourceFingerprint
) 'source fingerprint generation'
$AuditText = @(& $Python.venv_launcher_path `
    (Join-Path $PSScriptRoot 'validate_source_fingerprint.py') `
    $Repository $SourceFingerprint $AuthorizedCommit 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "source/commit audit failed before readiness: $AuditText" }
$SourceAudit = $AuditText | ConvertFrom-Json
if (-not $SourceAudit.valid) { throw 'source/commit audit was not exactly valid' }

$ImmutableRecords = @(
    Get-R1ImmutableFileRecord (Join-Path $Repository '1313\mainmodel copy 3.jld2')
    Get-R1ImmutableFileRecord (Join-Path $Repository 'artifacts\legacy_openvino\legacy_1313_weights.npz')
    Get-R1ImmutableFileRecord (Join-Path $Repository 'reports\clean_post_q1_memory_failure_strategy.md')
    Get-R1ImmutableFileRecord $Julia.path
    Get-R1ImmutableFileRecord $Python.path
    Get-R1ImmutableFileRecord $Python.venv_launcher_path
    Get-R1ImmutableFileRecord $Pwsh.path
    Get-R1ImmutableFileRecord $Manifest.path
    Get-R1ImmutableFileRecord (Join-Path $PSScriptRoot 'contract.json')
)
$RuntimeSourceRecordMap = [ordered]@{}
$AuditedSourceEntries = @($SourceAudit.fingerprint.files) + `
    @($SourceAudit.fingerprint.runtime_closure.files)
foreach ($AuditedSource in $AuditedSourceEntries) {
    $AuditedPath = [IO.Path]::GetFullPath((Join-Path $Repository ([string]$AuditedSource.path)))
    $CapturedSource = Get-R1ImmutableFileRecord $AuditedPath
    if ([int64]$CapturedSource.bytes -ne [int64]$AuditedSource.bytes -or
        [string]$CapturedSource.sha256 -cne [string]$AuditedSource.sha256) {
        throw "source-audit file changed before immutable closure capture: $AuditedPath"
    }
    $RuntimeSourceRecordMap[$AuditedPath.ToLowerInvariant()] = $CapturedSource
}
foreach ($UpstreamRelative in @(
    'upstream\TetrisAI\src\core\components\node.jl',
    'upstream\TetrisAI\src\core\analyzer.jl'
)) {
    $UpstreamPath = [IO.Path]::GetFullPath((Join-Path $Repository $UpstreamRelative))
    $RuntimeSourceRecordMap[$UpstreamPath.ToLowerInvariant()] = Get-R1ImmutableFileRecord $UpstreamPath
}
$RuntimeSourceRecords = @($RuntimeSourceRecordMap.Values)
if ($RuntimeSourceRecords.Count -lt @($SourceAudit.fingerprint.files).Count) {
    throw 'immutable runtime/source closure lost audited source entries'
}
if ($ImmutableRecords[0].sha256 -cne '7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1' -or
    $ImmutableRecords[1].sha256 -cne '2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba') {
    throw 'canonical checkpoint/OpenVINO weight digest differs from the frozen contract'
}

$Commands = [ordered]@{
    collect_training=@(
        $Julia.path, '--startup-file=no', '--history-file=no', "--project=$Repository",
        (Join-Path $PSScriptRoot 'collect_online.jl'), 'train', $TrainingTable, $TrainingManifest,
        (Join-Path $OutputDirectory 'collect_training.child_milestones.jsonl')
    )
    fit_ridge=@(
        $Python.path, (Join-Path $PSScriptRoot 'fit_ridge.py'), $TrainingTable, $TrainingManifest,
        $DesignFreeze, $RidgeArtifact, (Join-Path $OutputDirectory 'fit_ridge.child_milestones.jsonl')
    )
    collect_calibration=@(
        $Julia.path, '--startup-file=no', '--history-file=no', "--project=$Repository",
        (Join-Path $PSScriptRoot 'collect_online.jl'), 'calibration', $CalibrationTable, $CalibrationManifest,
        (Join-Path $OutputDirectory 'collect_calibration.child_milestones.jsonl')
    )
    calibration_gate=@(
        $Python.path, (Join-Path $PSScriptRoot 'calibration_gate.py'), $CalibrationTable,
        $CalibrationManifest, $RidgeArtifact, $DesignFreeze, $CalibrationAssessment,
        (Join-Path $OutputDirectory 'calibration_gate.child_milestones.jsonl')
    )
    finalize_assessment=@(
        $Python.path, (Join-Path $PSScriptRoot 'finalize_assessment.py'), $TrainingTable,
        $TrainingManifest, $RidgeArtifact, $CalibrationTable, $CalibrationManifest,
        $CalibrationAssessment, $DesignFreeze, $AssessmentPath,
        (Join-Path $OutputDirectory 'finalize_assessment.child_milestones.jsonl')
    )
}
$PhaseArtifacts = [ordered]@{
    collect_training=@($TrainingTable,$TrainingManifest)
    fit_ridge=@($RidgeArtifact)
    collect_calibration=@($CalibrationTable,$CalibrationManifest)
    calibration_gate=@($CalibrationAssessment)
    finalize_assessment=@($AssessmentPath)
}
$DesignFreezeSha = (Get-FileHash -LiteralPath $DesignFreeze -Algorithm SHA256).Hash.ToLowerInvariant()
function New-R1ProducedHandoff([string]$Phase, [string]$Path) {
    return [ordered]@{ source_phase=$Phase; path=[IO.Path]::GetFullPath($Path) }
}
function New-R1FrozenHandoff([string]$Path, [string]$Sha256) {
    return [ordered]@{ frozen_path=[IO.Path]::GetFullPath($Path); frozen_sha256=$Sha256 }
}
$PhaseEnvironment = [ordered]@{
    fit_ridge=[ordered]@{
        R1_EXPECTED_TRAINING_TABLE_SHA256=New-R1ProducedHandoff 'collect_training' $TrainingTable
        R1_EXPECTED_TRAINING_MANIFEST_SHA256=New-R1ProducedHandoff 'collect_training' $TrainingManifest
        R1_EXPECTED_DESIGN_FREEZE_SHA256=New-R1FrozenHandoff $DesignFreeze $DesignFreezeSha
    }
    calibration_gate=[ordered]@{
        R1_EXPECTED_CALIBRATION_TABLE_SHA256=New-R1ProducedHandoff 'collect_calibration' $CalibrationTable
        R1_EXPECTED_CALIBRATION_MANIFEST_SHA256=New-R1ProducedHandoff 'collect_calibration' $CalibrationManifest
        R1_EXPECTED_RIDGE_ARTIFACT_SHA256=New-R1ProducedHandoff 'fit_ridge' $RidgeArtifact
        R1_EXPECTED_DESIGN_FREEZE_SHA256=New-R1FrozenHandoff $DesignFreeze $DesignFreezeSha
    }
    finalize_assessment=[ordered]@{
        R1_EXPECTED_TRAINING_TABLE_SHA256=New-R1ProducedHandoff 'collect_training' $TrainingTable
        R1_EXPECTED_TRAINING_MANIFEST_SHA256=New-R1ProducedHandoff 'collect_training' $TrainingManifest
        R1_EXPECTED_RIDGE_ARTIFACT_SHA256=New-R1ProducedHandoff 'fit_ridge' $RidgeArtifact
        R1_EXPECTED_CALIBRATION_TABLE_SHA256=New-R1ProducedHandoff 'collect_calibration' $CalibrationTable
        R1_EXPECTED_CALIBRATION_MANIFEST_SHA256=New-R1ProducedHandoff 'collect_calibration' $CalibrationManifest
        R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256=New-R1ProducedHandoff 'calibration_gate' $CalibrationAssessment
        R1_EXPECTED_DESIGN_FREEZE_SHA256=New-R1FrozenHandoff $DesignFreeze $DesignFreezeSha
    }
}

$FreezeDocument = [ordered]@{
    schema='r1-one-shot-freeze-v1'
    experiment='online_counterfactual_top2_R1'
    authorized_commit=$AuthorizedCommit
    repository=$Repository
    output_directory=$OutputDirectory
    global_marker=$GlobalMarker
    start_gate=[IO.Path]::GetFullPath($StartGatePath)
    start_gate_nonce_sha256=$StartGateNonceSha256
    exact_start_gate_format="START online_counterfactual_top2_R1 $AuthorizedCommit <64-lower-hex-nonce>"
    source_audit=$SourceAudit
    source_fingerprint_sha256=(Get-FileHash -LiteralPath $SourceFingerprint -Algorithm SHA256).Hash.ToLowerInvariant()
    eligibility_sha256=(Get-FileHash -LiteralPath $Eligibility -Algorithm SHA256).Hash.ToLowerInvariant()
    design_freeze_sha256=$DesignFreezeSha
    immutable_files=$ImmutableRecords
    runtime_source_files=$RuntimeSourceRecords
    fit=[ordered]@{
        quantile_method='linear_type7_position_1_plus_n_minus_1_p'
    }
    openvino=[ordered]@{
        exact_full_build=$Python.openvino_version
    }
    python_pycache_prefix=$R1PycachePrefix
    commands=$Commands
    phase_artifacts=$PhaseArtifacts
    phase_environment_bindings=$PhaseEnvironment
    phase_order=@($script:R1ExpectedPhaseOrder)
    total_hard_wall_seconds=3900
    telemetry_interval_ms=200
    development_enabled=$false
    validation_seed_used=$false
    sealed_test_seed_used=$false
    retry_prohibited=$true
    rescue_prohibited=$true
}
Write-R1JsonDurableAtomic $FreezePlan $FreezeDocument
if ([IO.File]::Exists($StartGatePath)) {
    throw 'R1 start gate must be absent at readiness and created only after the freeze'
}
Write-R1JsonDurableAtomic $ReadyPath ([ordered]@{
    status='ready_for_external_start_gate'
    freeze_path=$FreezePlan
    freeze_sha256=(Get-FileHash -LiteralPath $FreezePlan -Algorithm SHA256).Hash.ToLowerInvariant()
    start_gate_nonce=$StartGateNonce
    expected_gate=$ExactStartGateText
    marker_absent=$true
    model_or_checkpoint_loaded=$false
    game_run=$false
    reserved_seed_loaded=$false
})
$ReadyItem = Get-Item -LiteralPath $ReadyPath
$GateDeadline = [DateTimeOffset]::UtcNow.AddMinutes(10)
$LastAudit = [DateTimeOffset]::MinValue
while ($true) {
    if ([IO.File]::Exists($GlobalMarker)) { throw 'R1 marker appeared before this wrapper atomically consumed it' }
    Assert-R1ImmutableFileRecords $ImmutableRecords
    Assert-R1ImmutableFileRecords $RuntimeSourceRecords
    if (([DateTimeOffset]::UtcNow - $LastAudit).TotalSeconds -ge 30) {
        $AuditText = @(& $Python.venv_launcher_path `
            (Join-Path $PSScriptRoot 'validate_source_fingerprint.py') `
            $Repository $SourceFingerprint $AuthorizedCommit 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "source/commit changed while waiting: $AuditText" }
        $LastAudit = [DateTimeOffset]::UtcNow
    }
    if ([IO.File]::Exists($StartGatePath)) {
        $GateItem = Get-Item -LiteralPath $StartGatePath -Force
        $GateObservedAtUtc = [DateTime]::UtcNow
        if (($GateItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'start gate must be a regular non-linked file'
        }
        if ($GateItem.CreationTimeUtc -le $ReadyItem.CreationTimeUtc -or
            $GateItem.LastWriteTimeUtc -le $ReadyItem.LastWriteTimeUtc) {
            throw 'start gate timestamps do not postdate readiness publication'
        }
        if ($GateItem.CreationTimeUtc -gt $GateObservedAtUtc -or
            $GateItem.LastWriteTimeUtc -gt $GateObservedAtUtc) {
            throw 'start gate has a future creation or modification timestamp'
        }
        $GateSnapshotBytes = [IO.File]::ReadAllBytes($StartGatePath)
        $GateSnapshotSha256 = [Convert]::ToHexString(
            [Security.Cryptography.SHA256]::HashData($GateSnapshotBytes)
        ).ToLowerInvariant()
        $GateText = [Text.Encoding]::UTF8.GetString($GateSnapshotBytes).TrimEnd("`r","`n")
        if ($GateText -cne $ExactStartGateText) {
            throw 'start gate contents differ from the exact frozen nonce-bound authorization line'
        }
        break
    }
    if ([DateTimeOffset]::UtcNow -ge $GateDeadline) { throw 'timed out waiting 10 minutes for exact R1 start gate' }
    Start-Sleep -Seconds 2
}
Assert-R1ImmutableFileRecords $ImmutableRecords
Assert-R1ImmutableFileRecords $RuntimeSourceRecords
$AuditText = @(& $Python.venv_launcher_path `
    (Join-Path $PSScriptRoot 'validate_source_fingerprint.py') `
    $Repository $SourceFingerprint $AuthorizedCommit 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "final pre-marker source/commit audit failed: $AuditText" }
$GateRecord = Get-R1ImmutableFileRecord $StartGatePath
if ([int64]$GateRecord.bytes -ne [int64]$GateSnapshotBytes.LongLength -or
    [string]$GateRecord.sha256 -cne $GateSnapshotSha256) {
    throw 'start gate changed between exact snapshot validation and immutable capture'
}
$GateRecord['creation_time_utc'] = (Get-Item -LiteralPath $StartGatePath -Force).CreationTimeUtc.ToString('o')
$GateRecord['nonce'] = $StartGateNonce
$EligibilityRecord = Get-R1ImmutableFileRecord $Eligibility
$DesignFreezeRecord = Get-R1ImmutableFileRecord $DesignFreeze
$SourceFingerprintRecord = Get-R1ImmutableFileRecord $SourceFingerprint
$FreezePlanRecord = Get-R1ImmutableFileRecord $FreezePlan
$ReadyRecord = Get-R1ImmutableFileRecord $ReadyPath
$PreMarkerPersistentRecords = @(
    $EligibilityRecord
    $DesignFreezeRecord
    $SourceFingerprintRecord
    $FreezePlanRecord
    $ReadyRecord
    $GateRecord
)
Assert-R1ImmutableFileRecords $PreMarkerPersistentRecords
$TerminalFailurePath = Join-Path $OutputDirectory 'terminal_failure.json'
$Monitor = $null
$Assessment = $null
try {
Write-R1JsonDurableAtomic $GlobalMarker ([ordered]@{
    experiment='online_counterfactual_top2_R1'
    started_at=[DateTimeOffset]::UtcNow.ToString('o')
    authorized_commit=$AuthorizedCommit
    output_directory=$OutputDirectory
    freeze_sha256=[string]$FreezePlanRecord.sha256
    ready_sha256=[string]$ReadyRecord.sha256
    start_gate=$GateRecord
    retry_prohibited=$true
    rescue_prohibited=$true
})
$MarkerRecord = Get-R1ImmutableFileRecord $GlobalMarker
$PersistentRunRecords = @($PreMarkerPersistentRecords) + @($MarkerRecord)
Assert-R1ImmutableFileRecords $PersistentRunRecords

$BoundaryIntegrityCheck = {
    param([string]$Boundary)
    Assert-R1ImmutableFileRecords $ImmutableRecords
    Assert-R1ImmutableFileRecords $RuntimeSourceRecords
    Assert-R1ImmutableFileRecords $PersistentRunRecords
    $BoundaryAuditText = @(& $Python.venv_launcher_path `
        (Join-Path $PSScriptRoot 'validate_source_fingerprint.py') `
        $Repository $SourceFingerprint $AuthorizedCommit 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "source/HEAD/fingerprint audit failed at $Boundary boundary: $BoundaryAuditText"
    }
}
    $Monitor = Invoke-R1PhasePipeline -Commands $Commands -WorkingDirectory $Repository `
        -OutputDirectory $OutputDirectory -PhaseArtifacts $PhaseArtifacts `
        -PhaseEnvironment $PhaseEnvironment -BoundaryIntegrityCheck $BoundaryIntegrityCheck `
        -TotalHardWallSeconds 3900
    & $BoundaryIntegrityCheck 'after_pipeline'

    $AssessmentSnapshotSha = $null
    if ($null -ne $Monitor.authoritative_failure_assessment) {
        $Assessment = $Monitor.authoritative_failure_assessment
    } elseif ([IO.File]::Exists($AssessmentPath)) {
        $FinalizerPhase = @($Monitor.phases | Where-Object { $_.phase -ceq 'finalize_assessment' })
        if ($FinalizerPhase.Count -ne 1 -or @($FinalizerPhase[0].phase_products).Count -ne 1) {
            throw 'assessment exists without one finalizer phase-product binding'
        }
        $AssessmentBytes = [IO.File]::ReadAllBytes($AssessmentPath)
        $AssessmentSnapshotSha = [Convert]::ToHexString(
            [Security.Cryptography.SHA256]::HashData($AssessmentBytes)
        ).ToLowerInvariant()
        if ($AssessmentSnapshotSha -cne [string]$FinalizerPhase[0].phase_products[0].sha256) {
            throw 'assessment changed after finalizer producer completion'
        }
        $Assessment = [Text.Encoding]::UTF8.GetString($AssessmentBytes) | ConvertFrom-Json
    } else {
        throw 'finalizer produced neither an assessment nor an authoritative wrapper failure assessment'
    }
    $Monitor['assessment_snapshot_sha256'] = $AssessmentSnapshotSha
    & $BoundaryIntegrityCheck 'after_assessment_parse'

    $MarkerCaptured = @($PersistentRunRecords | Where-Object {
        ([string]$_.path).Equals([IO.Path]::GetFullPath($GlobalMarker), [StringComparison]::OrdinalIgnoreCase)
    })
    $FreezeCaptured = @($PersistentRunRecords | Where-Object {
        ([string]$_.path).Equals([IO.Path]::GetFullPath($FreezePlan), [StringComparison]::OrdinalIgnoreCase)
    })
    if ($MarkerCaptured.Count -ne 1 -or $FreezeCaptured.Count -ne 1) {
        throw 'captured marker/freeze immutable records are missing or ambiguous'
    }
    $Wrapper = [ordered]@{
        experiment='online_counterfactual_top2_R1'
        authorized_commit=$AuthorizedCommit
        global_marker=$GlobalMarker
        global_marker_sha256=[string]$MarkerCaptured[0].sha256
        freeze_path=$FreezePlan
        freeze_sha256=[string]$FreezeCaptured[0].sha256
        persistent_run_records=$PersistentRunRecords
        runtime_source_records=$RuntimeSourceRecords
        durable_publication=(Get-R1DurablePublicationEvidence)
        output_directory=$OutputDirectory
        retry_prohibited=$true
        rescue_prohibited=$true
        optional_development_enabled=$false
        validation_seed_used=$false
        sealed_test_seed_used=$false
        game_run=$false
    }
    & $BoundaryIntegrityCheck 'before_composition'
    $Final = Write-R1TerminalArtifacts -MonitorPath $MonitorPath -WrapperPath $WrapperPath `
        -FinalPath $FinalPath -FailurePath $TerminalFailurePath `
        -Assessment $Assessment -Monitor $Monitor -Wrapper $Wrapper
    & $BoundaryIntegrityCheck 'after_terminal_publication'
    $Final | ConvertTo-Json -Depth 32
} catch {
    if (-not [IO.File]::Exists($GlobalMarker)) { throw }
    $PostMarkerFailure = "post-marker authoritative failure: $($_.Exception.Message)"
    if ($null -eq $Monitor) {
        $Monitor = [ordered]@{
            complete=$true
            ended_at=[DateTimeOffset]::UtcNow.ToString('o')
            wall_seconds=0.0
            stop_reason='failed'
            failures=@($PostMarkerFailure)
            phases=@()
            skipped_phases=@($script:R1ExpectedPhaseOrder | ForEach-Object {
                [ordered]@{ phase=$_; reason=$PostMarkerFailure }
            })
            producer_completion_ledger=@()
            assessment_snapshot_sha256=$null
            finalizer_disposition='outer_post_marker_failure'
            authoritative_failure_assessment=(New-R1AuthoritativeFailureAssessment `
                $PostMarkerFailure 'outer_post_marker')
            optional_development_enabled=$false
            development_seed_used=$false
            validation_seed_used=$false
            sealed_test_seed_used=$false
            game_run=$false
        }
    } else {
        $Monitor['stop_reason'] = 'failed'
        $Monitor['failures'] = @($Monitor.failures) + @($PostMarkerFailure)
        $Monitor['authoritative_failure_assessment'] = New-R1AuthoritativeFailureAssessment `
            $PostMarkerFailure 'outer_post_marker'
    }
    $Failure = Write-R1TerminalFailureVeto -FinalPath $FinalPath `
        -FailurePath $TerminalFailurePath -Reason $PostMarkerFailure -Monitor $Monitor
    $Failure | ConvertTo-Json -Depth 32
    throw
}
} finally {
    [Environment]::SetEnvironmentVariable(
        'PYTHONPYCACHEPREFIX', $R1OriginalPycachePrefix, 'Process'
    )
}
