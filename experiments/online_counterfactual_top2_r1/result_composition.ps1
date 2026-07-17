Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'wrapper_runtime.ps1')

function Test-R1Property($Value, [string]$Name) {
    if ($Value -is [Collections.IDictionary]) { return $Value.Contains($Name) }
    return $null -ne $Value -and $Value.PSObject.Properties.Name -contains $Name
}

function Get-R1Property($Value, [string]$Name) {
    if (-not (Test-R1Property $Value $Name)) { return $null }
    $Result = if ($Value -is [Collections.IDictionary]) { $Value[$Name] } else { $Value.$Name }
    return $Result
}

function Get-R1ArrayProperty($Value, [string]$Name) {
    if (-not (Test-R1Property $Value $Name)) { return }
    $Raw = if ($Value -is [Collections.IDictionary]) { $Value[$Name] } else { $Value.$Name }
    if ($null -eq $Raw) { return }
    if ($Raw -is [Array]) {
        foreach ($Item in $Raw) { Write-Output -NoEnumerate $Item }
    } else {
        Write-Output -NoEnumerate $Raw
    }
}

function Test-R1False($Value, [string]$Name) {
    $Observed = Get-R1Property $Value $Name
    return (Test-R1Property $Value $Name) -and $Observed -is [bool] -and -not $Observed
}

function New-R1FinalResult($Assessment, $Monitor) {
    $Failures = [Collections.Generic.List[string]]::new()
    if ($null -eq $Monitor) {
        $Failures.Add('missing external monitor')
    } else {
        foreach ($Field in @(
            'complete','ended_at','wall_seconds','stop_reason','failures',
            'phases','skipped_phases','producer_completion_ledger','assessment_snapshot_sha256','optional_development_enabled',
            'development_seed_used','validation_seed_used','sealed_test_seed_used','game_run'
        )) {
            if (-not (Test-R1Property $Monitor $Field)) { $Failures.Add("monitor lacks $Field") }
        }
        if ((Get-R1Property $Monitor 'complete') -isnot [bool] -or -not (Get-R1Property $Monitor 'complete')) {
            $Failures.Add('monitor is incomplete')
        }
        $Wall = Get-R1Property $Monitor 'wall_seconds'
        if ($Wall -isnot [ValueType] -or [double]$Wall -lt 0 -or [double]$Wall -gt 3900) {
            $Failures.Add('monitor wall is invalid or exceeds 65 minutes')
        }
        foreach ($Field in @('optional_development_enabled','development_seed_used','validation_seed_used','sealed_test_seed_used','game_run')) {
            if (-not (Test-R1False $Monitor $Field)) { $Failures.Add("monitor scope flag $Field is not exactly false") }
        }
        if ((Get-R1Property $Monitor 'stop_reason') -cne 'completed') {
            $Failures.Add("monitor stop reason: $(Get-R1Property $Monitor 'stop_reason')")
        }
        foreach ($Failure in @((Get-R1Property $Monitor 'failures'))) {
            if (-not [string]::IsNullOrWhiteSpace([string]$Failure)) { $Failures.Add("monitor: $Failure") }
        }
        $Phases = @($(if ($Monitor -is [Collections.IDictionary]) { $Monitor['phases'] } else { $Monitor.phases }))
        $ProducerLedger = @($(if ($Monitor -is [Collections.IDictionary]) { $Monitor['producer_completion_ledger'] } else { $Monitor.producer_completion_ledger }))
        $Skipped = @($(if ($Monitor -is [Collections.IDictionary]) { $Monitor['skipped_phases'] } else { $Monitor.skipped_phases }))
        $LaunchedNames = @($Phases | ForEach-Object { [string](Get-R1Property $_ 'phase') })
        $SkippedNames = @($Skipped | ForEach-Object { [string](Get-R1Property $_ 'phase') })
        $Combined = @($LaunchedNames + $SkippedNames)
        if ($Combined.Count -ne $script:R1ExpectedPhaseOrder.Count -or
            @($Combined | Select-Object -Unique).Count -ne $Combined.Count) {
            $Failures.Add('phase ledger has omissions or duplicates')
        }
        foreach ($Expected in $script:R1ExpectedPhaseOrder) {
            if ($Expected -notin $Combined) { $Failures.Add("phase ledger omits $Expected") }
        }
        if ($LaunchedNames.Count -lt 1 -or $LaunchedNames[-1] -cne 'finalize_assessment') {
            $Failures.Add('finalize_assessment was not the last launched phase')
        }
        $ExpectedLaunched = @($script:R1ExpectedPhaseOrder | Where-Object { $_ -in $LaunchedNames })
        if (($ExpectedLaunched -join "`n") -cne ($LaunchedNames -join "`n")) {
            $Failures.Add('launched phases are out of preregistered order')
        }
        foreach ($Phase in $Phases) {
            $Name = [string](Get-R1Property $Phase 'phase')
            if ((Get-R1Property $Phase 'exit_code') -ne 0 -or
                (Get-R1Property $Phase 'stop_reason') -cne 'completed') {
                $Failures.Add("phase $Name did not complete")
            }
            $Proof = Get-R1Property $Phase 'assignment_proof'
            foreach ($Field in @(
                'create_process_flag_suspended','job_assignment_succeeded_before_resume',
                'job_membership_verified_before_resume','root_image_matches_requested_executable',
                'resume_succeeded_after_assignment'
            )) {
                if ((Get-R1Property $Proof $Field) -isnot [bool] -or -not (Get-R1Property $Proof $Field)) {
                    $Failures.Add("phase $Name lacks $Field proof")
                }
            }
            if ((Get-R1Property $Phase 'job_tree_empty_verified') -isnot [bool] -or
                -not (Get-R1Property $Phase 'job_tree_empty_verified')) {
                $Failures.Add("phase $Name Job tree was not verified empty")
            }
            if ((Get-R1Property $Phase 'root_executable_unchanged') -isnot [bool] -or
                -not (Get-R1Property $Phase 'root_executable_unchanged') -or
                [string](Get-R1Property $Phase 'root_executable_sha256_before') -notmatch '^[0-9a-f]{64}$' -or
                (Get-R1Property $Phase 'root_executable_sha256_before') -cne
                    (Get-R1Property $Phase 'root_executable_sha256_after')) {
                $Failures.Add("phase $Name executable binding changed or is missing")
            }
            $ArgvSources = @($(if ($Phase -is [Collections.IDictionary]) { $Phase['argv_source_files'] } else { $Phase.argv_source_files }))
            if ($ArgvSources.Count -lt 1) { $Failures.Add("phase $Name lacks argv source-file binding") }
            foreach ($Source in $ArgvSources) {
                $SourcePath = [string](Get-R1Property $Source 'path')
                $SourceSha = [string](Get-R1Property $Source 'sha256')
                if ($SourceSha -notmatch '^[0-9a-f]{64}$' -or -not [IO.File]::Exists($SourcePath) -or
                    (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $SourceSha) {
                    $Failures.Add("phase $Name argv source-file binding is invalid or changed")
                }
            }
            if ((Get-R1Property $Phase 'child_milestones_required') -isnot [bool] -or
                -not (Get-R1Property $Phase 'child_milestones_required') -or
                [int](Get-R1Property $Phase 'child_milestone_count') -lt 3) {
                $Failures.Add("phase $Name child milestones are incomplete")
            }
            if ((Get-R1Property $Phase 'child_milestone_binding') -cne 'exact argv path') {
                $Failures.Add("phase $Name child milestone ownership is missing")
            }
            if ([string](Get-R1Property $Phase 'phase_result_sha256') -notmatch '^[0-9a-f]{64}$' -or
                [string]::IsNullOrWhiteSpace([string](Get-R1Property $Phase 'phase_result_path'))) {
                $Failures.Add("phase $Name result artifact hash is missing")
            }
            $Artifacts = @($(if ($Phase -is [Collections.IDictionary]) { $Phase['closed_artifacts'] } else { $Phase.closed_artifacts }))
            if ($Artifacts.Count -lt 5) { $Failures.Add("phase $Name lacks closed log artifacts") }
            foreach ($Artifact in $Artifacts) {
                if ([string](Get-R1Property $Artifact 'sha256') -notmatch '^[0-9a-f]{64}$') {
                    $Failures.Add("phase $Name has an unhashed closed artifact")
                }
            }
            $RawProducts = if ($Phase -is [Collections.IDictionary]) { $Phase['phase_products'] } else { $Phase.phase_products }
            $Products = [Collections.Generic.List[object]]::new()
            if ($RawProducts -is [Array]) { foreach ($Item in $RawProducts) { $Products.Add($Item) } }
            elseif ($null -ne $RawProducts) { $Products.Add($RawProducts) }
            $RequiredProductCount = Get-R1Property $Phase 'required_phase_product_count'
            if ($RequiredProductCount -isnot [ValueType] -or [int]$RequiredProductCount -lt 1 -or
                $Products.Count -ne [int]$RequiredProductCount) {
                $Failures.Add("phase $Name product inventory is incomplete")
            }
            foreach ($Product in $Products) {
                if ([string](Get-R1Property $Product 'sha256') -notmatch '^[0-9a-f]{64}$' -or
                    [string]::IsNullOrWhiteSpace([string](Get-R1Property $Product 'path'))) {
                    $Failures.Add("phase $Name has an invalid product hash")
                }
            }
            foreach ($Field in @('validation_seed_used','sealed_test_seed_used','game_run')) {
                if (-not (Test-R1False $Phase $Field)) { $Failures.Add("phase $Name scope flag $Field is not false") }
            }
            $PeakPrivate = Get-R1Property $Phase 'peak_process_tree_private_committed_bytes'
            $PeakWorking = Get-R1Property $Phase 'peak_process_tree_working_set_resident_bytes'
            $LimitPrivate = Get-R1Property $Phase 'max_private_committed_bytes'
            $LimitWorking = Get-R1Property $Phase 'max_working_set_resident_bytes'
            foreach ($Pair in @(
                @($PeakPrivate,$LimitPrivate,'private committed'),
                @($PeakWorking,$LimitWorking,'resident working set')
            )) {
                if ($Pair[0] -isnot [ValueType] -or $Pair[1] -isnot [ValueType] -or
                    [int64]$Pair[0] -lt 0 -or [int64]$Pair[0] -gt [int64]$Pair[1]) {
                    $Failures.Add("phase $Name exceeded or lacks $($Pair[2]) accounting")
                }
            }
            $ExpectedPrivate = if ($Name -eq 'fit_ridge') { [int64](1GB) } else { [int64](4GB) }
            $ExpectedWorking = if ($Name -in @('collect_training','collect_calibration')) { [int64](2GB) } else { [int64]::MaxValue }
            if ([int64]$LimitPrivate -ne $ExpectedPrivate -or [int64]$LimitWorking -ne $ExpectedWorking) {
                $Failures.Add("phase $Name resource limit mismatch")
            }
            if ((Get-R1Property $Phase 'os_job_private_commit_cap_enabled') -isnot [bool] -or
                -not (Get-R1Property $Phase 'os_job_private_commit_cap_enabled') -or
                [int](Get-R1Property $Phase 'telemetry_interval_target_ms') -ne 200 -or
                [int](Get-R1Property $Phase 'telemetry_samples') -lt 1) {
                $Failures.Add("phase $Name OS cap/telemetry evidence missing")
            }
        }
        $ExpectedLedgerEntries = 0
        foreach ($SuccessfulPhase in @($Phases | Where-Object {
            (Get-R1Property $_ 'exit_code') -eq 0 -and (Get-R1Property $_ 'stop_reason') -ceq 'completed'
        })) {
            $RawSuccessfulProducts = if ($SuccessfulPhase -is [Collections.IDictionary]) { $SuccessfulPhase['phase_products'] } else { $SuccessfulPhase.phase_products }
            $ExpectedLedgerEntries += if ($RawSuccessfulProducts -is [Array]) { $RawSuccessfulProducts.Count } elseif ($null -eq $RawSuccessfulProducts) { 0 } else { 1 }
        }
        if ($ProducerLedger.Count -ne $ExpectedLedgerEntries) {
            $Failures.Add('producer-completion ledger cardinality differs from successful phase products')
        }
        foreach ($Entry in $ProducerLedger) {
            $LedgerPath = [string](Get-R1Property $Entry 'path')
            $LedgerSha = [string](Get-R1Property $Entry 'sha256')
            $LedgerPhase = [string](Get-R1Property $Entry 'phase')
            $MatchingPhase = @($Phases | Where-Object { (Get-R1Property $_ 'phase') -ceq $LedgerPhase })
            $MatchingProductCount = 0
            if ($MatchingPhase.Count -eq 1) {
                $RawPhaseProducts = if ($MatchingPhase[0] -is [Collections.IDictionary]) { $MatchingPhase[0]['phase_products'] } else { $MatchingPhase[0].phase_products }
                $PhaseProducts = [Collections.Generic.List[object]]::new()
                if ($RawPhaseProducts -is [Array]) { foreach ($Item in $RawPhaseProducts) { $PhaseProducts.Add($Item) } }
                elseif ($null -ne $RawPhaseProducts) { $PhaseProducts.Add($RawPhaseProducts) }
                foreach ($Candidate in $PhaseProducts) {
                    $CandidatePath = [IO.Path]::GetFullPath([string](Get-R1Property $Candidate 'path'))
                    if ($CandidatePath.Equals([IO.Path]::GetFullPath($LedgerPath), [StringComparison]::OrdinalIgnoreCase) -and
                        [string](Get-R1Property $Candidate 'sha256') -ceq $LedgerSha) {
                        $MatchingProductCount += 1
                    }
                }
            }
            if ($LedgerSha -notmatch '^[0-9a-f]{64}$' -or $MatchingProductCount -ne 1 -or
                -not [IO.File]::Exists($LedgerPath) -or
                (Get-FileHash -LiteralPath $LedgerPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $LedgerSha) {
                $Failures.Add("invalid, unmatched, or changed producer-completion ledger entry: $LedgerPath (phase_matches=$($MatchingPhase.Count), product_matches=$MatchingProductCount)")
            }
        }
        $Finalizer = @($Phases | Where-Object { (Get-R1Property $_ 'phase') -ceq 'finalize_assessment' })
        $FinalizerProducts = [Collections.Generic.List[object]]::new()
        if ($Finalizer.Count -eq 1) {
            $RawFinalizerProducts = if ($Finalizer[0] -is [Collections.IDictionary]) { $Finalizer[0]['phase_products'] } else { $Finalizer[0].phase_products }
            if ($RawFinalizerProducts -is [Array]) { foreach ($Item in $RawFinalizerProducts) { $FinalizerProducts.Add($Item) } }
            elseif ($null -ne $RawFinalizerProducts) { $FinalizerProducts.Add($RawFinalizerProducts) }
        }
        $AssessmentSnapshotSha = [string](Get-R1Property $Monitor 'assessment_snapshot_sha256')
        if ($AssessmentSnapshotSha -notmatch '^[0-9a-f]{64}$' -or $FinalizerProducts.Count -ne 1 -or
            [string](Get-R1Property $FinalizerProducts[0] 'sha256') -cne $AssessmentSnapshotSha) {
            $Failures.Add("assessment snapshot is not bound to the finalizer producer-completion product (products=$($FinalizerProducts.Count), snapshot=$AssessmentSnapshotSha)")
        }
    }

    if ($null -eq $Assessment) {
        $Failures.Add('missing assessment')
    } else {
        foreach ($Pair in @(
            @('status','assessment-pass'),
            @('promotion','R1-calibration-promoted')
        )) {
            if ((Get-R1Property $Assessment $Pair[0]) -cne $Pair[1]) {
                $Failures.Add("assessment $($Pair[0]) mismatch")
            }
        }
        if ((Get-R1Property $Assessment 'success') -isnot [bool] -or
            -not (Get-R1Property $Assessment 'success')) {
            $Failures.Add('assessment did not pass')
        }
        foreach ($Field in @(
            'game_strength_evidence','model_beat_claim','game_evaluation_authorized',
            'development_authorized','validation_seed_used','sealed_test_seed_used',
            'sealed_test_authorized','game_run'
        )) {
            if (-not (Test-R1False $Assessment $Field)) { $Failures.Add("assessment scope flag $Field is not false") }
        }
    }
    $Unique = @($Failures | Select-Object -Unique)
    $Success = $Unique.Count -eq 0
    return [ordered]@{
        experiment='R1 online counterfactual top-2 safety gate'
        status=if ($Success) { 'R1-calibration-promoted' } else { 'R1-calibration-rejected' }
        success=$Success
        failures=$Unique
        assessment=$Assessment
        monitor=$Monitor
        scope='calibration promotion only; development requires a separate future freeze review'
        calibration_candidate_only=$Success
        game_system_candidate=$false
        model_improvement_claim=$false
        game_strength_evidence=$false
        development_authorized=$false
        validation_seed_used=$false
        sealed_test_seed_used=$false
        sealed_test_authorized=$false
        game_run=$false
        retry_prohibited=$true
        rescue_prohibited=$true
    }
}

function Write-R1TerminalArtifacts(
    [string]$MonitorPath,
    [string]$WrapperPath,
    [string]$FinalPath,
    $Assessment,
    $Monitor,
    $Wrapper,
    [string]$FailurePoint = ''
) {
    foreach ($Path in @($MonitorPath,$WrapperPath,$FinalPath)) {
        if ([IO.File]::Exists($Path)) { throw "terminal artifact exists: $Path" }
    }
    $Final = New-R1FinalResult $Assessment $Monitor
    $WrapperToWrite = [ordered]@{}
    if ($Wrapper -is [Collections.IDictionary]) {
        foreach ($Key in $Wrapper.Keys) { $WrapperToWrite[[string]$Key] = $Wrapper[$Key] }
    } else {
        foreach ($Property in $Wrapper.PSObject.Properties) {
            $WrapperToWrite[$Property.Name] = $Property.Value
        }
    }
    try {
        if ($FailurePoint -ceq 'before_monitor') { throw 'injected before_monitor failure' }
        Write-R1JsonDurableAtomic $MonitorPath $Monitor
        $MonitorSha = (Get-FileHash -LiteralPath $MonitorPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $WrapperToWrite['monitor_path'] = [IO.Path]::GetFullPath($MonitorPath)
        $WrapperToWrite['monitor_sha256'] = $MonitorSha
        if ($FailurePoint -ceq 'after_monitor') { throw 'injected after_monitor failure' }
        if ($FailurePoint -ceq 'before_wrapper') { throw 'injected before_wrapper failure' }
        Write-R1JsonDurableAtomic $WrapperPath $WrapperToWrite
        $WrapperSha = (Get-FileHash -LiteralPath $WrapperPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $Final['monitor_path'] = [IO.Path]::GetFullPath($MonitorPath)
        $Final['monitor_sha256'] = $MonitorSha
        $Final['wrapper_result_path'] = [IO.Path]::GetFullPath($WrapperPath)
        $Final['wrapper_result_sha256'] = $WrapperSha
        $Final['final_result_is_last_fallible_publication'] = $true
        if ($FailurePoint -ceq 'after_wrapper') { throw 'injected after_wrapper failure' }
        if ($FailurePoint -ceq 'before_final') { throw 'injected before_final failure' }
        Write-R1JsonDurableAtomic $FinalPath $Final
        if ($FailurePoint -ceq 'after_final') { throw 'injected after_final failure' }
    } catch {
        if ([IO.File]::Exists($FinalPath)) { [IO.File]::Delete($FinalPath) }
        throw
    }
    return $Final
}
