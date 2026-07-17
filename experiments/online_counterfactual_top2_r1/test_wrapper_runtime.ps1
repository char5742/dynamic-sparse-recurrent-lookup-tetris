[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'wrapper_runtime.ps1')
$OriginalPycachePrefix = [Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process')
$env:PYTHONPYCACHEPREFIX = Join-Path 'D:\tetris-paper-plus\python-pycache' (
    "r1_test_wrapper_${PID}_$([guid]::NewGuid().ToString('N'))"
)
[IO.Directory]::CreateDirectory($env:PYTHONPYCACHEPREFIX) | Out-Null

$Julia = Resolve-R1ConcreteJulia
if ($Julia.path -match '(?i)\\Microsoft\\WindowsApps\\' -or $Julia.version -cne '1.12.6') {
    throw 'concrete Julia resolver returned a launcher or wrong version'
}
$OriginalPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$OriginalPythonHome = [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process')
$OriginalPythonNoUserSite = [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process')
try {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\ambient-pythonpath-poison', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONHOME', 'C:\ambient-pythonhome-poison', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', '0', 'Process')
    $Python = Resolve-R1FrozenPython
    if (-not $Python.pythonpath_cleared -or -not $Python.pythonhome_cleared -or
        $Python.python_no_user_site -cne '1' -or
        $Python.python_executable -cne 'D:\tetris-paper-plus\python-env\Scripts\python.exe' -or
        $Python.python_base_prefix -cne 'C:\Users\fshuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python' -or
        $Python.venv_prefix -cne 'D:\tetris-paper-plus\python-env' -or
        $Python.openvino_module_file -cne 'D:\tetris-paper-plus\python-env\Lib\site-packages\openvino\__init__.py' -or
        $Python.openvino_package_tree_sha256 -cne 'c292b25245f36e937f21b105023737be80491e70de5f24b1704ad1ced8547e43' -or
        $Python.openvino_loaded_native_sha256 -cne '929dd49859750bfa59c850234c8eeb872c84db05c1b60510e9a9db8b7d756a74' -or
        @($Python.openvino_loaded_native_modules).Count -ne 5) {
        throw 'frozen Python/OpenVINO origin proof is incomplete'
    }
    if ([Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process') -cne 'C:\ambient-pythonpath-poison' -or
        [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process') -cne 'C:\ambient-pythonhome-poison' -or
        [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process') -cne '0') {
        throw 'Python origin probe did not restore the parent import environment'
    }
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $OriginalPythonPath, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONHOME', $OriginalPythonHome, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', $OriginalPythonNoUserSite, 'Process')
}
$PwshRecord = Resolve-R1ConcretePwsh (Get-Process -Id $PID).Path
$WindowsPowerShellRejected = $false
try {
    [void](Resolve-R1ConcretePwsh "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe")
} catch { $WindowsPowerShellRejected = $true }
if (-not $WindowsPowerShellRejected -or $PwshRecord.version -cne '7.4.17') {
    throw 'exact pwsh 7.4.17 enforcement or Windows PowerShell 5.1 rejection failed'
}

$Durability = Get-R1DurablePublicationEvidence
if ($Durability.publication_flag -cne 'MOVEFILE_WRITE_THROUGH (0x00000008)' -or
    $Durability.overwrite_allowed) {
    throw 'Windows durable atomic publication evidence is incomplete'
}

$Root = Join-Path ([IO.Path]::GetTempPath()) ('r1-wrapper-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($Root) | Out-Null
try {
    $Pwsh = $PwshRecord.path
    $SuccessRoot = Join-Path $Root 'success'
    $RunStarted = [DateTimeOffset]::UtcNow
    $Command = @(
        $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
        '[IO.File]::WriteAllText($env:R1_PRE_RESUME_SENTINEL_PATH,"executed"); $child = Start-Process -FilePath (Join-Path $PSHOME "pwsh.exe") -ArgumentList "-NoLogo","-NoProfile","-NonInteractive","-Command","Start-Sleep -Milliseconds 300" -WindowStyle Hidden -PassThru; $child.WaitForExit(); Start-Sleep -Milliseconds 250; [Console]::Out.WriteLine("synthetic-ok"); exit 0'
    )
    $Sentinel = Join-Path $Root 'executed.sentinel'
    $Record = Invoke-R1MonitoredPhase -Phase collect_training -Command $Command `
        -WorkingDirectory $Root -OutputDirectory $SuccessRoot -RunStarted $RunStarted `
        -TotalHardWallSeconds 10 -MaxPrivateCommittedBytes ([int64](1GB)) `
        -MaxWorkingSetResidentBytes ([int64](1GB)) -PreResumeSentinelPath $Sentinel
    if ($Record.exit_code -ne 0 -or $Record.stop_reason -cne 'completed') {
        throw "synthetic monitored process failed: $($Record.stop_reason)"
    }
    if (-not $Record.assignment_proof.create_process_flag_suspended -or
        -not $Record.assignment_proof.job_assignment_succeeded_before_resume -or
        -not $Record.assignment_proof.job_membership_verified_before_resume -or
        -not $Record.assignment_proof.root_image_matches_requested_executable -or
        -not $Record.assignment_proof.resume_succeeded_after_assignment) {
        throw 'suspend/assign/resume proof is incomplete'
    }
    if ($Record.telemetry_samples -lt 2) { throw '200-ms telemetry did not collect multiple samples' }
    if (-not [IO.File]::Exists($Sentinel) -or
        -not $Record.pre_resume_sentinel_absent_before_resume) {
        throw 'pre-resume execution sentinel proof failed'
    }
    $Lines = @(Get-Content -LiteralPath $Record.telemetry_path | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Lines.Count -ne $Record.telemetry_samples -or
        @($Lines | Where-Object polling_interval_target_ms -ne 200).Count -ne 0) {
        throw 'telemetry JSONL schema/sample count mismatch'
    }
    if (@($Lines | Where-Object { $_.terminology.private_committed_bytes -notmatch 'not resident RAM' }).Count -ne 0) {
        throw 'private commit terminology disclosure missing'
    }
    if ($Record.per_pid_peak_private_committed_bytes.Keys.Count -lt 1) {
        throw 'per-PID private commit peaks missing'
    }
    if ($Record.maximum_observed_process_count -lt 2 -or
        $Record.per_pid_peak_private_committed_bytes.Keys.Count -lt 2) {
        throw 'grandchild process was not observed inside the Job telemetry'
    }
    if ($Record.windows_job_peak_memory_used_bytes -le 0 -or
        -not $Record.os_job_private_commit_cap_enabled) {
        throw 'OS Job peak/cap evidence missing'
    }
    if ((Get-Content -LiteralPath $Record.stdout -Raw).Trim() -cne 'synthetic-ok') {
        throw 'redirected stdout mismatch'
    }
    $Lifecycle = @(Get-Content -LiteralPath $Record.wrapper_milestones_path | ForEach-Object { ($_ | ConvertFrom-Json).milestone })
    $ExpectedLifecycle = @(
        'launch_intent','root_created_suspended',
        'job_assigned_and_verified_before_resume','root_resumed_after_job_assignment',
        'root_exit_observed','job_tree_empty','process_monitor_closed','phase_result_ready'
    )
    if (($Lifecycle -join "`n") -cne ($ExpectedLifecycle -join "`n")) {
        throw "wrapper lifecycle order mismatch: $($Lifecycle -join ',')"
    }

    $OldPhasePythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    $OldPhasePythonHome = [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process')
    $OldPhaseNoUserSite = [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\phase-path-poison', 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONHOME', 'C:\phase-home-poison', 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', '0', 'Process')
        $EnvironmentRoot = Join-Path $Root 'python-environment-isolation'
        $EnvironmentRecord = Invoke-R1MonitoredPhase -Phase fit_ridge -Command @(
            $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
            '[Console]::Out.Write((@{pythonpath=$env:PYTHONPATH;pythonhome=$env:PYTHONHOME;nousersite=$env:PYTHONNOUSERSITE}|ConvertTo-Json -Compress))'
        ) -WorkingDirectory $Root -OutputDirectory $EnvironmentRoot `
            -RunStarted ([DateTimeOffset]::UtcNow) -TotalHardWallSeconds 10 `
            -MaxPrivateCommittedBytes ([int64](1GB)) -MaxWorkingSetResidentBytes ([int64](1GB))
        $EnvironmentFacts = Get-Content -LiteralPath $EnvironmentRecord.stdout -Raw | ConvertFrom-Json
        if ($EnvironmentRecord.exit_code -ne 0 -or $null -ne $EnvironmentFacts.pythonpath -or
            $null -ne $EnvironmentFacts.pythonhome -or $EnvironmentFacts.nousersite -cne '1') {
            throw 'monitored child inherited an ambient Python import environment'
        }
        if ([Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process') -cne 'C:\phase-path-poison' -or
            [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process') -cne 'C:\phase-home-poison' -or
            [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process') -cne '0') {
            throw 'monitored phase did not restore parent Python environment'
        }
    } finally {
        [Environment]::SetEnvironmentVariable('PYTHONPATH', $OldPhasePythonPath, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONHOME', $OldPhasePythonHome, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', $OldPhaseNoUserSite, 'Process')
    }

    $MemoryRoot = Join-Path $Root 'memory-stop'
    $Killed = Invoke-R1MonitoredPhase -Phase fit_ridge -Command @(
        $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
        'try { $memory = [byte[]]::new(180MB) } catch { [Environment]::Exit(42) }; if ($null -eq $memory) { [Environment]::Exit(43) }; Start-Sleep -Seconds 5'
    ) -WorkingDirectory $Root -OutputDirectory $MemoryRoot `
        -RunStarted ([DateTimeOffset]::UtcNow) -TotalHardWallSeconds 10 `
        -MaxPrivateCommittedBytes ([int64](128MB)) -MaxWorkingSetResidentBytes ([int64]::MaxValue)
    if ($Killed.exit_code -eq 0 -or $Killed.stop_reason -ceq 'completed' -or
        $Killed.windows_job_peak_memory_used_bytes -gt [uint64](128MB)) {
        throw 'OS Job private committed-memory cap did not fail closed'
    }

    $HangRoot = Join-Path $Root 'hang-stop'
    $Hung = Invoke-R1MonitoredPhase -Phase calibration_gate -Command @(
        $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
        'Start-Sleep -Seconds 5'
    ) -WorkingDirectory $Root -OutputDirectory $HangRoot `
        -RunStarted ([DateTimeOffset]::UtcNow) -TotalHardWallSeconds 0.4 `
        -MaxPrivateCommittedBytes ([int64](1GB)) -MaxWorkingSetResidentBytes ([int64]::MaxValue)
    if ($Hung.exit_code -eq 0 -or $Hung.stop_reason -cne 'R1 total hard wall exceeded' -or
        -not $Hung.job_tree_empty_verified) {
        throw 'hung process was not terminated with an empty Job tree'
    }

    $BadPlan = [ordered]@{
        collect_training=$Command
        fit_ridge=$Command
        collect_calibration=$Command
        calibration_gate=$Command
        development=$Command
        finalize_assessment=$Command
    }
    $Rejected = $false
    try {
        [void](Invoke-R1PhasePipeline -Commands $BadPlan -PhaseArtifacts $BadPlan -WorkingDirectory $Root `
            -OutputDirectory (Join-Path $Root 'bad-plan') -TotalHardWallSeconds 10)
    } catch { $Rejected = $true }
    if (-not $Rejected) { throw 'optional development phase was not rejected' }

    $LedgerProduct = Join-Path $Root 'producer-ledger-product.json'
    [IO.File]::WriteAllText($LedgerProduct, '{"version":1}')
    $LedgerDigest = (Get-FileHash -LiteralPath $LedgerProduct -Algorithm SHA256).Hash.ToLowerInvariant()
    $ProducerLedger = [ordered]@{}
    $ProducerLedger[[IO.Path]::GetFullPath($LedgerProduct)] = [ordered]@{
        phase='collect_training'; sha256=$LedgerDigest
    }
    $HandoffPlan = [ordered]@{
        R1_EXPECTED_TRAINING_TABLE_SHA256=[ordered]@{
            source_phase='collect_training'; path=$LedgerProduct
        }
    }
    $Bound = Resolve-R1PhaseEnvironmentHandoffs -Phase fit_ridge `
        -Plan $HandoffPlan -ProducerLedger $ProducerLedger
    if ($Bound.R1_EXPECTED_TRAINING_TABLE_SHA256 -cne $LedgerDigest) {
        throw 'producer-completion SHA ledger did not bind the original artifact'
    }
    [IO.File]::WriteAllText($LedgerProduct, '{"version":2}')
    $MutationRejected = $false
    try {
        [void](Resolve-R1PhaseEnvironmentHandoffs -Phase fit_ridge `
            -Plan $HandoffPlan -ProducerLedger $ProducerLedger)
    } catch { $MutationRejected = $true }
    if (-not $MutationRejected) {
        throw 'artifact mutation between producer completion and consumer was not rejected'
    }

    $StaleName = 'R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256'
    $OldStale = [Environment]::GetEnvironmentVariable($StaleName, 'Process')
    [Environment]::SetEnvironmentVariable($StaleName, ('f' * 64), 'Process')
    try {
        $IsolationRoot = Join-Path $Root 'environment-isolation'
        $Isolation = Invoke-R1MonitoredPhase -Phase finalize_assessment -Command @(
            $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
            '[Console]::Out.Write($(if ($null -eq $env:R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256) { "absent" } else { "leaked" }))'
        ) -WorkingDirectory $Root -OutputDirectory $IsolationRoot `
            -RunStarted ([DateTimeOffset]::UtcNow) -TotalHardWallSeconds 10 `
            -MaxPrivateCommittedBytes ([int64](1GB)) -MaxWorkingSetResidentBytes ([int64]::MaxValue)
        if ($Isolation.exit_code -ne 0 -or (Get-Content -LiteralPath $Isolation.stdout -Raw) -cne 'absent') {
            throw 'stale external R1_EXPECTED SHA leaked into a phase without that binding'
        }
        if ([Environment]::GetEnvironmentVariable($StaleName, 'Process') -cne ('f' * 64)) {
            throw 'wrapper did not restore the preexisting parent environment after child creation'
        }
    } finally {
        [Environment]::SetEnvironmentVariable($StaleName, $OldStale, 'Process')
    }

    $PipelineRoot = Join-Path $Root 'pipeline-finalizer-after-wrapper-failure'
    [IO.Directory]::CreateDirectory($PipelineRoot) | Out-Null
    $SyntheticChild = Join-Path $PipelineRoot 'synthetic_phase_child.ps1'
    [IO.File]::WriteAllText($SyntheticChild, @'
param([string]$Phase,[string]$Product,[string]$Milestones)
$writer = [IO.StreamWriter]::new($Milestones,$false,[Text.UTF8Encoding]::new($false))
try {
    foreach($stage in @('script_enter','imports_begin','phase_complete')) {
        $writer.WriteLine(([ordered]@{phase=$Phase;stage=$stage;pid=$PID} | ConvertTo-Json -Compress))
        $writer.Flush(); $writer.BaseStream.Flush($true)
    }
} finally { $writer.Dispose() }
[IO.File]::WriteAllText($Product,"{`"phase`":`"$Phase`"}")
'@)
    $PipelineCommands = [ordered]@{}
    $PipelineArtifacts = [ordered]@{}
    foreach ($PhaseName in $script:R1ExpectedPhaseOrder) {
        $Product = Join-Path $PipelineRoot "$PhaseName.product.json"
        $ChildMilestone = Join-Path $PipelineRoot "$PhaseName.child_milestones.jsonl"
        $PipelineCommands[$PhaseName] = @(
            $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
            $SyntheticChild, $PhaseName, $Product, $ChildMilestone
        )
        $PipelineArtifacts[$PhaseName] = @($Product)
    }
    $BoundaryFailure = {
        param([string]$PhaseName)
        if ($PhaseName -ceq 'fit_ridge') { throw 'synthetic prelaunch integrity failure' }
    }
    $FailedPipeline = Invoke-R1PhasePipeline -Commands $PipelineCommands `
        -PhaseArtifacts $PipelineArtifacts -WorkingDirectory $Root `
        -OutputDirectory $PipelineRoot -TotalHardWallSeconds 30 `
        -BoundaryIntegrityCheck $BoundaryFailure
    if (@($FailedPipeline.phases | Where-Object phase -ceq 'finalize_assessment').Count -ne 1 -or
        @($FailedPipeline.skipped_phases | Where-Object phase -ceq 'fit_ridge').Count -ne 1 -or
        -not [IO.File]::Exists($PipelineArtifacts.finalize_assessment[0])) {
        throw 'strict finalizer was not invoked after a wrapper-side prelaunch failure'
    }

    $PostprocessRoot = Join-Path $Root 'pipeline-finalizer-after-wrapper-postprocess-failure'
    [IO.Directory]::CreateDirectory($PostprocessRoot) | Out-Null
    $PostprocessCommands = [ordered]@{}
    $PostprocessArtifacts = [ordered]@{}
    foreach ($PhaseName in $script:R1ExpectedPhaseOrder) {
        $Product = Join-Path $PostprocessRoot "$PhaseName.product.json"
        $ChildMilestone = Join-Path $PostprocessRoot "$PhaseName.child_milestones.jsonl"
        $PostprocessCommands[$PhaseName] = @(
            $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
            $SyntheticChild, $PhaseName, $Product, $ChildMilestone
        )
        $PostprocessArtifacts[$PhaseName] = @($Product)
    }
    # A duplicate required artifact yields two closed product records for the
    # same path.  The child and phase record complete successfully; only the
    # producer-ledger postprocess fails.  The mandatory finalizer must still run.
    $PostprocessArtifacts.fit_ridge = @(
        $PostprocessArtifacts.fit_ridge[0], $PostprocessArtifacts.fit_ridge[0]
    )
    $PostprocessFailedPipeline = Invoke-R1PhasePipeline -Commands $PostprocessCommands `
        -PhaseArtifacts $PostprocessArtifacts -WorkingDirectory $Root `
        -OutputDirectory $PostprocessRoot -TotalHardWallSeconds 30
    if (@($PostprocessFailedPipeline.phases | Where-Object phase -ceq 'finalize_assessment').Count -ne 1 -or
        @($PostprocessFailedPipeline.failures | Where-Object { $_ -like '*wrapper postprocess failed*' }).Count -ne 1 -or
        -not [IO.File]::Exists($PostprocessArtifacts.finalize_assessment[0])) {
        throw 'strict finalizer was not invoked after a wrapper-side postprocess failure'
    }

    function New-R1SyntheticPipelinePlan([string]$PlanRoot) {
        [IO.Directory]::CreateDirectory($PlanRoot) | Out-Null
        $PlanCommands = [ordered]@{}
        $PlanArtifacts = [ordered]@{}
        foreach ($PhaseName in $script:R1ExpectedPhaseOrder) {
            $Product = Join-Path $PlanRoot "$PhaseName.product.json"
            $ChildMilestone = Join-Path $PlanRoot "$PhaseName.child_milestones.jsonl"
            $PlanCommands[$PhaseName] = @(
                $Pwsh, '-NoLogo', '-NoProfile', '-NonInteractive', '-File',
                $SyntheticChild, $PhaseName, $Product, $ChildMilestone
            )
            $PlanArtifacts[$PhaseName] = @($Product)
        }
        return [ordered]@{ commands=$PlanCommands; artifacts=$PlanArtifacts }
    }

    $FinalizerBoundaryRoot = Join-Path $Root 'finalizer-boundary-authoritative-failure'
    $FinalizerBoundaryPlan = New-R1SyntheticPipelinePlan $FinalizerBoundaryRoot
    $FinalizerBoundaryCheck = {
        param([string]$Boundary)
        if ($Boundary -ceq 'finalize_assessment') { throw 'synthetic finalizer boundary failure' }
    }
    $FinalizerBoundaryResult = Invoke-R1PhasePipeline `
        -Commands $FinalizerBoundaryPlan.commands -PhaseArtifacts $FinalizerBoundaryPlan.artifacts `
        -WorkingDirectory $Root -OutputDirectory $FinalizerBoundaryRoot `
        -BoundaryIntegrityCheck $FinalizerBoundaryCheck -TotalHardWallSeconds 30
    if ($FinalizerBoundaryResult.finalizer_disposition -cne 'blocked_by_boundary_integrity_failure' -or
        $FinalizerBoundaryResult.authoritative_failure_assessment.status -cne 'assessment-fail' -or
        $FinalizerBoundaryResult.authoritative_failure_assessment.success -or
        @($FinalizerBoundaryResult.phases | Where-Object phase -ceq 'finalize_assessment').Count -ne 0) {
        throw 'finalizer boundary failure did not produce authoritative fail assessment semantics'
    }

    $FinalizerInvokeRoot = Join-Path $Root 'finalizer-invocation-authoritative-failure'
    $FinalizerInvokePlan = New-R1SyntheticPipelinePlan $FinalizerInvokeRoot
    $FinalizerInvokePlan.commands.finalize_assessment = @(
        (Join-Path $FinalizerInvokeRoot 'missing-finalizer.exe'),
        (Join-Path $FinalizerInvokeRoot 'missing-finalizer.child_milestones.jsonl')
    )
    $FinalizerInvokeResult = Invoke-R1PhasePipeline `
        -Commands $FinalizerInvokePlan.commands -PhaseArtifacts $FinalizerInvokePlan.artifacts `
        -WorkingDirectory $Root -OutputDirectory $FinalizerInvokeRoot -TotalHardWallSeconds 30
    if ($FinalizerInvokeResult.finalizer_disposition -cne 'invocation_failure' -or
        $FinalizerInvokeResult.authoritative_failure_assessment.status -cne 'assessment-fail' -or
        $FinalizerInvokeResult.authoritative_failure_assessment.success) {
        throw 'finalizer invocation failure did not become authoritative failure'
    }

    $TerminalBoundaryRoot = Join-Path $Root 'terminal-boundary-authoritative-failure'
    $TerminalBoundaryPlan = New-R1SyntheticPipelinePlan $TerminalBoundaryRoot
    $TerminalBoundaryCheck = {
        param([string]$Boundary)
        if ($Boundary -ceq 'terminal') { throw 'synthetic during-finalizer/terminal integrity failure' }
    }
    $TerminalBoundaryResult = Invoke-R1PhasePipeline `
        -Commands $TerminalBoundaryPlan.commands -PhaseArtifacts $TerminalBoundaryPlan.artifacts `
        -WorkingDirectory $Root -OutputDirectory $TerminalBoundaryRoot `
        -BoundaryIntegrityCheck $TerminalBoundaryCheck -TotalHardWallSeconds 30
    if ($TerminalBoundaryResult.finalizer_disposition -cne 'terminal_integrity_failure' -or
        $TerminalBoundaryResult.authoritative_failure_assessment.failure_stage -cne 'after_finalize_assessment' -or
        $TerminalBoundaryResult.authoritative_failure_assessment.success) {
        throw 'during-finalizer/terminal integrity failure did not override the completed assessment'
    }

    [ordered]@{
        status='r1_wrapper_runtime_synthetic_checks_passed'
        concrete_julia=$Julia
        monitored_success_samples=$Record.telemetry_samples
        per_pid_telemetry_verified=$true
        suspended_assign_resume_verified=$true
        pre_resume_sentinel_verified=$true
        os_job_private_commit_cap_verified=$true
        hang_termination_tree_empty_verified=$true
        development_disabled_verified=$true
        producer_completion_sha_handoff_verified=$true
        post_producer_artifact_mutation_rejected=$true
        stale_handoff_environment_isolated_and_restored=$true
        finalizer_after_wrapper_prelaunch_failure_verified=$true
        finalizer_after_wrapper_postprocess_failure_verified=$true
        finalizer_boundary_failure_authoritative=$true
        finalizer_invocation_failure_authoritative=$true
        during_finalizer_terminal_integrity_failure_authoritative=$true
        exact_pwsh_7_4_17_verified=$true
        windows_powershell_5_1_rejected=$true
        frozen_python_openvino_origin_verified=$true
        child_python_environment_isolated_and_restored=$true
        movefile_write_through_publication_verified=$true
        checkpoint_or_model_loaded=$false
        game_run=$false
        seed_loaded=$false
        marker_read_or_written=$false
    } | ConvertTo-Json -Depth 8
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPYCACHEPREFIX', $OriginalPycachePrefix, 'Process')
    if ([IO.Directory]::Exists($Root)) {
        Remove-Item -LiteralPath $Root -Recurse -Force
    }
}
if ([Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process') -cne $OriginalPycachePrefix) {
    throw 'wrapper test did not restore PYTHONPYCACHEPREFIX'
}
