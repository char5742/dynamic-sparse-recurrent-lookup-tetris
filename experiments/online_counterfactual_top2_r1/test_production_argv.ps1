[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'wrapper_runtime.ps1')
$OriginalPycachePrefix = [Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process')
$env:PYTHONPYCACHEPREFIX = Join-Path 'D:\tetris-paper-plus\python-pycache' (
    "r1_test_argv_${PID}_$([guid]::NewGuid().ToString('N'))"
)
[IO.Directory]::CreateDirectory($env:PYTHONPYCACHEPREFIX) | Out-Null

$Repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$Julia = Resolve-R1ConcreteJulia
$Python = Resolve-R1FrozenPython
$Pwsh = Resolve-R1ConcretePwsh (Get-Process -Id $PID).Path
$Manifest = Assert-R1FrozenManifest $Repository
$Root = Join-Path ([IO.Path]::GetTempPath()) ('r1-production-argv-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($Root) | Out-Null
try {
    $Eligibility = Join-Path $Root 'eligibility.json'
    $Design = Join-Path $Root 'design_freeze.json'
    & $Python.path (Join-Path $PSScriptRoot 'make_eligibility.py') $Eligibility | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'synthetic eligibility generation failed' }
    & $Julia.path --startup-file=no --history-file=no "--project=$Repository" `
        (Join-Path $PSScriptRoot 'freeze_design.jl') $Eligibility $Design | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'synthetic design freeze failed' }

    $RunStarted = [DateTimeOffset]::UtcNow
    $TrainingTable = Join-Path $Root 'synthetic_collector_training_table.json'
    $TrainingManifest = Join-Path $Root 'synthetic_collector_training_manifest.json'
    $TrainingMilestones = Join-Path $Root 'collect_training.child_milestones.jsonl'
    $Train = Invoke-R1MonitoredPhase -Phase collect_training -Command @(
        $Julia.path, '--startup-file=no', '--history-file=no', "--project=$Repository",
        (Join-Path $PSScriptRoot 'collect_online.jl'), 'train',
        $TrainingTable, $TrainingManifest,
        $TrainingMilestones, '--synthetic'
    ) -WorkingDirectory $Repository -OutputDirectory $Root -RunStarted $RunStarted `
        -TotalHardWallSeconds 120 -MaxPrivateCommittedBytes ([int64](4GB)) `
        -MaxWorkingSetResidentBytes ([int64](2GB)) -RequireChildMilestones `
        -RequiredPhaseArtifacts @($TrainingTable,$TrainingManifest)
    if ($Train.exit_code -ne 0) { throw "training collection production argv failed: $($Train.stop_reason)" }

    $FitTrainingTable = Join-Path $Root 'training_table.json'
    $FitTrainingManifest = Join-Path $Root 'training_manifest.json'
    $FittedRidgeArtifact = Join-Path $Root 'fitted_ridge_artifact.json'
    $FitMilestones = Join-Path $Root 'fit_ridge.child_milestones.jsonl'
    $Fit = Invoke-R1MonitoredPhase -Phase fit_ridge -Command @(
        $Python.path, (Join-Path $PSScriptRoot 'fit_ridge.py'),
        $FitTrainingTable, $FitTrainingManifest,
        $Design, $FittedRidgeArtifact,
        $FitMilestones, '--synthetic'
    ) -WorkingDirectory $Repository -OutputDirectory $Root -RunStarted $RunStarted `
        -TotalHardWallSeconds 120 -MaxPrivateCommittedBytes ([int64](1GB)) `
        -MaxWorkingSetResidentBytes ([int64]::MaxValue) -RequireChildMilestones `
        -RequiredPhaseArtifacts @($FitTrainingTable,$FitTrainingManifest,$FittedRidgeArtifact)
    if ($Fit.exit_code -ne 0) {
        $FitError = if (Test-Path -LiteralPath $Fit.stderr) { Get-Content -LiteralPath $Fit.stderr -Raw } else { '' }
        throw "Python ridge production argv failed: $($Fit.stop_reason); stderr=$FitError"
    }

    $CalibrationSmokeTable = Join-Path $Root 'synthetic_collector_calibration_table.json'
    $CalibrationSmokeManifest = Join-Path $Root 'synthetic_collector_calibration_manifest.json'
    $CalibrationMilestones = Join-Path $Root 'collect_calibration.child_milestones.jsonl'
    $Calibration = Invoke-R1MonitoredPhase -Phase collect_calibration -Command @(
        $Julia.path, '--startup-file=no', '--history-file=no', "--project=$Repository",
        (Join-Path $PSScriptRoot 'collect_online.jl'), 'calibration',
        $CalibrationSmokeTable, $CalibrationSmokeManifest,
        $CalibrationMilestones, '--synthetic'
    ) -WorkingDirectory $Repository -OutputDirectory $Root -RunStarted $RunStarted `
        -TotalHardWallSeconds 120 -MaxPrivateCommittedBytes ([int64](4GB)) `
        -MaxWorkingSetResidentBytes ([int64](2GB)) -RequireChildMilestones `
        -RequiredPhaseArtifacts @($CalibrationSmokeTable,$CalibrationSmokeManifest)
    if ($Calibration.exit_code -ne 0) { throw "calibration collection production argv failed: $($Calibration.stop_reason)" }

    $CalibrationTable = Join-Path $Root 'calibration_table.json'
    $CalibrationManifest = Join-Path $Root 'calibration_manifest.json'
    $RidgeArtifact = Join-Path $Root 'synthetic_fixture_ridge_artifact.json'
    & $Python.venv_launcher_path `
        (Join-Path $PSScriptRoot 'make_synthetic_calibration_fixture.py') `
        $FitTrainingTable $FittedRidgeArtifact $RidgeArtifact `
        $CalibrationTable $CalibrationManifest | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'synthetic calibration fixture generation failed' }

    $CalibrationGateOutput = Join-Path $Root 'calibration_gate.json'
    $CalibrationGateMilestones = Join-Path $Root 'calibration_gate.child_milestones.jsonl'
    $CalibrationGate = Invoke-R1MonitoredPhase -Phase calibration_gate -Command @(
        $Python.path, (Join-Path $PSScriptRoot 'calibration_gate.py'),
        $CalibrationTable, $CalibrationManifest, $RidgeArtifact, $Design,
        $CalibrationGateOutput, $CalibrationGateMilestones, '--synthetic'
    ) -WorkingDirectory $Repository -OutputDirectory $Root -RunStarted $RunStarted `
        -TotalHardWallSeconds 120 -MaxPrivateCommittedBytes ([int64](4GB)) `
        -MaxWorkingSetResidentBytes ([int64]::MaxValue) -RequireChildMilestones `
        -RequiredPhaseArtifacts @($CalibrationGateOutput) -EnvironmentOverrides ([ordered]@{
            R1_EXPECTED_CALIBRATION_TABLE_SHA256=(Get-FileHash -LiteralPath $CalibrationTable -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_CALIBRATION_MANIFEST_SHA256=(Get-FileHash -LiteralPath $CalibrationManifest -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_RIDGE_ARTIFACT_SHA256=(Get-FileHash -LiteralPath $RidgeArtifact -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_DESIGN_FREEZE_SHA256=(Get-FileHash -LiteralPath $Design -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    if ($CalibrationGate.exit_code -ne 0) {
        $GateError = if (Test-Path -LiteralPath $CalibrationGate.stderr) { Get-Content -LiteralPath $CalibrationGate.stderr -Raw } else { '' }
        throw "calibration gate production argv failed: $($CalibrationGate.stop_reason); stderr=$GateError"
    }

    $AssessmentPath = Join-Path $Root 'assessment.json'
    $FinalizeMilestones = Join-Path $Root 'finalize_assessment.child_milestones.jsonl'
    $Finalize = Invoke-R1MonitoredPhase -Phase finalize_assessment -Command @(
        $Python.path, (Join-Path $PSScriptRoot 'finalize_assessment.py'),
        $FitTrainingTable, $FitTrainingManifest, $RidgeArtifact,
        $CalibrationTable, $CalibrationManifest, $CalibrationGateOutput,
        $Design, $AssessmentPath, $FinalizeMilestones, '--synthetic'
    ) -WorkingDirectory $Repository -OutputDirectory $Root -RunStarted $RunStarted `
        -TotalHardWallSeconds 120 -MaxPrivateCommittedBytes ([int64](4GB)) `
        -MaxWorkingSetResidentBytes ([int64]::MaxValue) -RequireChildMilestones `
        -RequiredPhaseArtifacts @($AssessmentPath) -EnvironmentOverrides ([ordered]@{
            R1_EXPECTED_TRAINING_TABLE_SHA256=(Get-FileHash -LiteralPath $FitTrainingTable -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_TRAINING_MANIFEST_SHA256=(Get-FileHash -LiteralPath $FitTrainingManifest -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_CALIBRATION_TABLE_SHA256=(Get-FileHash -LiteralPath $CalibrationTable -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_CALIBRATION_MANIFEST_SHA256=(Get-FileHash -LiteralPath $CalibrationManifest -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_RIDGE_ARTIFACT_SHA256=(Get-FileHash -LiteralPath $RidgeArtifact -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_DESIGN_FREEZE_SHA256=(Get-FileHash -LiteralPath $Design -Algorithm SHA256).Hash.ToLowerInvariant()
            R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256=(Get-FileHash -LiteralPath $CalibrationGateOutput -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    if ($Finalize.exit_code -ne 0) {
        $FinalizeError = if (Test-Path -LiteralPath $Finalize.stderr) { Get-Content -LiteralPath $Finalize.stderr -Raw } else { '' }
        throw "finalizer production argv failed: $($Finalize.stop_reason); stderr=$FinalizeError"
    }
    $Assessment = Get-Content -LiteralPath $AssessmentPath -Raw | ConvertFrom-Json
    if ($Assessment.status -cne 'assessment-pass' -or -not $Assessment.success) {
        throw "synthetic finalizer did not pass: $($Assessment.reasons -join '; ')"
    }

    [ordered]@{
        status='r1_all_five_production_argv_synthetic_checks_passed'
        concrete_julia=$Julia; frozen_python=$Python; frozen_pwsh=$Pwsh; manifest=$Manifest
        phases=@($Train.phase,$Fit.phase,$Calibration.phase,$CalibrationGate.phase,$Finalize.phase)
        python_fit_succeeded_under_1gib=$true
        python_fit_peak_private_committed_bytes=$Fit.peak_process_tree_private_committed_bytes
        python_fit_peak_working_set_resident_bytes=$Fit.peak_process_tree_working_set_resident_bytes
        all_used_suspended_job_runner=$true
        all_required_child_milestones_verified=$true
        checkpoint_or_model_loaded=$false
        reserved_seed_loaded=$false
        game_run=$false
        marker_read_or_written=$false
    } | ConvertTo-Json -Depth 10
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPYCACHEPREFIX', $OriginalPycachePrefix, 'Process')
    if ([IO.Directory]::Exists($Root)) { Remove-Item -LiteralPath $Root -Recurse -Force }
}
if ([Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process') -cne $OriginalPycachePrefix) {
    throw 'production-argv test did not restore PYTHONPYCACHEPREFIX'
}
