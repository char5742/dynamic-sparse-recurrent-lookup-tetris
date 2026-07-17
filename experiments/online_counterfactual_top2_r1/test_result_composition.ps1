[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'result_composition.ps1')
$OriginalPycachePrefix = [Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process')
$env:PYTHONPYCACHEPREFIX = Join-Path 'D:\tetris-paper-plus\python-pycache' (
    "r1_test_result_${PID}_$([guid]::NewGuid().ToString('N'))"
)
[IO.Directory]::CreateDirectory($env:PYTHONPYCACHEPREFIX) | Out-Null

$ProductRoot = Join-Path ([IO.Path]::GetTempPath()) ('r1-result-products-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($ProductRoot) | Out-Null

function New-R1SyntheticPhase([string]$Name) {
    $ProductCount = if ($Name -in @('collect_training','collect_calibration')) { 2 } else { 1 }
    $Products = @()
    foreach ($Index in 1..$ProductCount) {
        $ProductPath = Join-Path $ProductRoot "$Name.product.$Index.json"
        [IO.File]::WriteAllText($ProductPath, "{`"phase`":`"$Name`",`"index`":$Index}")
        $Products += [ordered]@{
            path=$ProductPath; bytes=[int64](Get-Item -LiteralPath $ProductPath).Length
            sha256=(Get-FileHash -LiteralPath $ProductPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    return [ordered]@{
        phase=$Name; exit_code=0; stop_reason='completed'; seconds=0.1
        assignment_proof=[ordered]@{
            create_process_flag_suspended=$true
            job_assignment_succeeded_before_resume=$true
            job_membership_verified_before_resume=$true
            root_image_matches_requested_executable=$true
            resume_succeeded_after_assignment=$true
        }
        job_tree_empty_verified=$true
        child_milestones_required=$true
        child_milestone_count=4
        child_milestone_binding='exact argv path'
        phase_result_path='synthetic.phase.json'
        phase_result_sha256=('c' * 64)
        closed_artifacts=@(1..5 | ForEach-Object { [ordered]@{
            path='synthetic'; bytes=1
            sha256=('a' * 64)
        } })
        required_phase_product_count=$ProductCount
        phase_products=$Products
        root_executable_unchanged=$true
        root_executable_sha256_before=('b' * 64)
        root_executable_sha256_after=('b' * 64)
        argv_source_files=@([ordered]@{
            path=(Join-Path $PSScriptRoot 'result_composition.ps1')
            bytes=[int64](Get-Item -LiteralPath (Join-Path $PSScriptRoot 'result_composition.ps1')).Length
            sha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'result_composition.ps1') -Algorithm SHA256).Hash.ToLowerInvariant()
        })
        max_private_committed_bytes=if ($Name -eq 'fit_ridge') { [int64](1GB) } else { [int64](4GB) }
        max_working_set_resident_bytes=if ($Name -in @('collect_training','collect_calibration')) { [int64](2GB) } else { [int64]::MaxValue }
        peak_process_tree_private_committed_bytes=[int64](100MB)
        peak_process_tree_working_set_resident_bytes=[int64](80MB)
        os_job_private_commit_cap_enabled=$true
        telemetry_interval_target_ms=200
        telemetry_samples=4
        validation_seed_used=$false
        sealed_test_seed_used=$false
        game_run=$false
    }
}

$Assessment = [ordered]@{
    status='assessment-pass'; success=$true; failures=@()
    promotion='R1-calibration-promoted'
    game_strength_evidence=$false; model_beat_claim=$false
    game_evaluation_authorized=$false; development_authorized=$false
    validation_seed_used=$false; sealed_test_seed_used=$false
    sealed_test_authorized=$false; game_run=$false
}
$SyntheticPhases = @($script:R1ExpectedPhaseOrder | ForEach-Object { New-R1SyntheticPhase $_ })
$SyntheticLedger = @($SyntheticPhases | ForEach-Object {
    $PhaseRecord = $_
    @($PhaseRecord.phase_products | ForEach-Object { [ordered]@{
        phase=$PhaseRecord.phase; path=$_.path; sha256=$_.sha256
        captured_after_tree_empty=$true; phase_result_sha256=$PhaseRecord.phase_result_sha256
    } })
})
$FinalizerProduct = @($SyntheticPhases | Where-Object phase -ceq 'finalize_assessment')[0].phase_products[0]
$Monitor = [ordered]@{
    complete=$true; ended_at='2026-07-18T00:00:00Z'; wall_seconds=10.0
    stop_reason='completed'; failures=@()
    phases=$SyntheticPhases
    skipped_phases=@()
    producer_completion_ledger=$SyntheticLedger
    assessment_snapshot_sha256=$FinalizerProduct.sha256
    finalizer_disposition='launched_completed'
    authoritative_failure_assessment=$null
    optional_development_enabled=$false; development_seed_used=$false
    validation_seed_used=$false; sealed_test_seed_used=$false; game_run=$false
}

$Pass = New-R1FinalResult $Assessment $Monitor
if (-not $Pass.success -or $Pass.status -cne 'R1-calibration-promoted' -or
    $Pass.development_authorized -or $Pass.game_strength_evidence) {
    throw "valid calibration-only terminal record failed: $($Pass.failures -join '; ')"
}

$NegativeCases = @()
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.optional_development_enabled = $true
$NegativeCases += ,@($Assessment,$Bad,'development enabled')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.phases[0].assignment_proof.job_membership_verified_before_resume = $false
$NegativeCases += ,@($Assessment,$Bad,'membership proof missing')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.phases[0].job_tree_empty_verified = $false
$NegativeCases += ,@($Assessment,$Bad,'tree not empty')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.phases[0].closed_artifacts[0].sha256 = 'bad'
$NegativeCases += ,@($Assessment,$Bad,'artifact hash missing')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.phases = @($Bad.phases | Where-Object phase -ne 'fit_ridge')
$NegativeCases += ,@($Assessment,$Bad,'phase omitted')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.producer_completion_ledger[0].sha256 = ('0' * 64)
$NegativeCases += ,@($Assessment,$Bad,'producer ledger changed')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.assessment_snapshot_sha256 = ('0' * 64)
$NegativeCases += ,@($Assessment,$Bad,'assessment snapshot unbound')
$BadAssessment = $Assessment | ConvertTo-Json -Depth 10 | ConvertFrom-Json
$BadAssessment.development_authorized = $true
$NegativeCases += ,@($BadAssessment,$Monitor,'development authorized')
$Bad = $Monitor | ConvertTo-Json -Depth 20 | ConvertFrom-Json
$Bad.stop_reason = 'failed'
$Bad.finalizer_disposition = 'terminal_integrity_failure'
$Bad.authoritative_failure_assessment = [ordered]@{
    status='assessment-fail'; success=$false; authority='wrapper-fail-closed'
    failure_stage='after_finalize_assessment'; failures=@('synthetic boundary failure')
}
$NegativeCases += ,@($Assessment,$Bad,'authoritative wrapper failure overrides passing assessment')
foreach ($Case in $NegativeCases) {
    if ((New-R1FinalResult $Case[0] $Case[1]).success) {
        throw "negative terminal case passed: $($Case[2])"
    }
}

$Root = Join-Path ([IO.Path]::GetTempPath()) ('r1-terminal-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($Root) | Out-Null
try {
    $Written = Write-R1TerminalArtifacts `
        (Join-Path $Root 'monitor.json') `
        (Join-Path $Root 'wrapper_result.json') `
        (Join-Path $Root 'final_result.json') `
        $Assessment $Monitor ([ordered]@{ status='wrapper-complete'; success=$true })
    if (-not $Written.success -or
        -not [IO.File]::Exists((Join-Path $Root 'final_result.json'))) {
        throw 'durable terminal publication failed'
    }
    foreach ($Point in @('before_monitor','after_monitor','before_wrapper','after_wrapper','before_final','after_final')) {
        $CaseRoot = Join-Path $Root $Point
        [IO.Directory]::CreateDirectory($CaseRoot) | Out-Null
        $Failed = $false
        try {
            [void](Write-R1TerminalArtifacts `
                (Join-Path $CaseRoot 'monitor.json') `
                (Join-Path $CaseRoot 'wrapper_result.json') `
                (Join-Path $CaseRoot 'final_result.json') `
                $Assessment $Monitor ([ordered]@{ status='wrapper-complete'; success=$true }) $Point)
        } catch { $Failed = $true }
        $CaseFinalPath = Join-Path $CaseRoot 'final_result.json'
        $CaseFailurePath = Join-Path $CaseRoot 'terminal_failure.json'
        if (-not $Failed -or -not [IO.File]::Exists($CaseFinalPath)) {
            throw "terminal failure injection did not retain a terminal record: $Point"
        }
        $CaseFinal = Get-Content -LiteralPath $CaseFinalPath -Raw | ConvertFrom-Json
        if ($Point -ceq 'after_final') {
            if (-not [IO.File]::Exists($CaseFailurePath)) {
                throw 'post-success publication failure lacks the higher-authority veto'
            }
            $CaseVeto = Get-Content -LiteralPath $CaseFailurePath -Raw | ConvertFrom-Json
            if (-not $CaseVeto.authoritative_terminal_failure -or $CaseVeto.success -or
                -not $CaseVeto.supersedes_final_result) {
                throw 'post-success terminal failure veto is not authoritative'
            }
        } elseif (-not $CaseFinal.authoritative_terminal_failure -or $CaseFinal.success) {
            throw "publication failure did not leave an authoritative rejection: $Point"
        }
        $AuthoritativeCase = Read-R1AuthoritativeTerminalResult `
            -FinalPath $CaseFinalPath -FailurePath $CaseFailurePath
        if ($AuthoritativeCase.success -or -not $AuthoritativeCase.authoritative_terminal_failure) {
            throw "terminal authority reader did not prefer failure: $Point"
        }
    }

    $DurablePath = Join-Path $Root 'durable-no-replace.json'
    Write-R1JsonDurableAtomic $DurablePath ([ordered]@{ generation=1 })
    $FirstDurableSha = (Get-FileHash -LiteralPath $DurablePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $OverwriteRejected = $false
    try { Write-R1JsonDurableAtomic $DurablePath ([ordered]@{ generation=2 }) }
    catch { $OverwriteRejected = $true }
    if (-not $OverwriteRejected -or
        (Get-FileHash -LiteralPath $DurablePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $FirstDurableSha) {
        throw 'MoveFileEx write-through publication overwrote an existing artifact'
    }
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPYCACHEPREFIX', $OriginalPycachePrefix, 'Process')
    if ([IO.Directory]::Exists($Root)) { Remove-Item -LiteralPath $Root -Recurse -Force }
}
if ([Environment]::GetEnvironmentVariable('PYTHONPYCACHEPREFIX', 'Process') -cne $OriginalPycachePrefix) {
    throw 'result-composition test did not restore PYTHONPYCACHEPREFIX'
}

[ordered]@{
    status='r1_result_composition_synthetic_checks_passed'
    positive_cases=2
    negative_cases=$NegativeCases.Count
    terminal_failure_injections=6
    append_only_terminal_failure_veto_verified=$true
    movefile_write_through_no_replace_verified=$true
    game_run=$false
    seed_loaded=$false
    marker_read_or_written=$false
} | ConvertTo-Json

if ([IO.Directory]::Exists($ProductRoot)) {
    Remove-Item -LiteralPath $ProductRoot -Recurse -Force
}
