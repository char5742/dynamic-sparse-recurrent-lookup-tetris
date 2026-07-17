Set-StrictMode -Version Latest

$script:P1ExpectedPhaseOrder = @(
    'eligibility',
    'select_rows',
    'extract_training',
    'train_partial',
    'verify_openvino',
    'extract_offline',
    'offline_gate',
    'evaluate_development',
    'finalize_assessment'
)

function Test-P1HasProperty($Value, [string]$Name) {
    if ($Value -is [System.Collections.IDictionary]) {
        return $Value.Contains($Name)
    }
    return $null -ne $Value -and $Value.PSObject.Properties.Name -contains $Name
}

function Get-P1Value($Value, [string]$Name) {
    if ($null -eq $Value) { return $null }
    if ($Value -is [System.Collections.IDictionary]) {
        $Result = $Value[$Name]
    } else {
        $Property = $Value.PSObject.Properties[$Name]
        if ($null -eq $Property) { return $null }
        $Result = $Property.Value
    }
    if ($Result -is [System.Array]) {
        return ,$Result
    }
    return $Result
}

function Test-P1FiniteNonnegativeNumber($Value) {
    if ($Value -isnot [byte] -and $Value -isnot [int16] -and
        $Value -isnot [int32] -and $Value -isnot [int64] -and
        $Value -isnot [single] -and $Value -isnot [double] -and
        $Value -isnot [decimal]) {
        return $false
    }
    $Number = [double]$Value
    return -not [double]::IsNaN($Number) -and
           -not [double]::IsInfinity($Number) -and $Number -ge 0
}

function Test-P1Integer($Value) {
    return $Value -is [byte] -or $Value -is [int16] -or
           $Value -is [int32] -or $Value -is [int64]
}

function Test-P1EndedAt($Value) {
    if ($Value -is [string]) {
        return -not [string]::IsNullOrWhiteSpace($Value)
    }
    return $Value -is [datetime] -or $Value -is [datetimeoffset]
}

function New-P1FinalResult($Assessment, $Monitor) {
    $Failures = [System.Collections.Generic.List[string]]::new()
    $AssessmentSuccess = $false
    if ($null -eq $Assessment) {
        $Failures.Add('missing post-pipeline assessment')
    } else {
        $AssessmentStatus = Get-P1Value $Assessment 'status'
        $AssessmentSuccessValue = Get-P1Value $Assessment 'success'
        $AssessmentFailures = Get-P1Value $Assessment 'failures'
        if (-not (Test-P1HasProperty $Assessment 'status') -or
            $AssessmentStatus -isnot [string] -or
            [string]$AssessmentStatus -cne 'assessment-pass') {
            $Failures.Add('assessment status is not assessment-pass')
        }
        if (-not (Test-P1HasProperty $Assessment 'success') -or
            $AssessmentSuccessValue -isnot [bool]) {
            $Failures.Add('assessment lacks a Boolean success field')
        } elseif (-not (Test-P1HasProperty $Assessment 'failures') -or
            $AssessmentFailures -isnot [System.Array]) {
            $Failures.Add('assessment lacks failures field')
        } elseif ([bool]$AssessmentSuccessValue -and @($AssessmentFailures).Count -ne 0) {
            $Failures.Add('passing assessment contains failures')
        } elseif (-not [bool]$AssessmentSuccessValue) {
            foreach ($Failure in @($AssessmentFailures)) {
                if (-not [string]::IsNullOrWhiteSpace([string]$Failure)) {
                    $Failures.Add("assessment: $Failure")
                }
            }
            if (@($AssessmentFailures).Count -eq 0) {
                $Failures.Add('assessment rejected without a recorded reason')
            }
        }
        $AssessmentSuccess = $AssessmentSuccessValue -is [bool] -and
            [bool]$AssessmentSuccessValue
    }

    if ($null -eq $Monitor) {
        $Failures.Add('missing completed external monitor')
    } else {
        $MonitorComplete = Get-P1Value $Monitor 'complete'
        $MonitorEndedAt = Get-P1Value $Monitor 'ended_at'
        $MonitorWallSeconds = Get-P1Value $Monitor 'wall_seconds'
        $MonitorPeakWorkingSet = Get-P1Value $Monitor 'peak_process_tree_working_set_bytes'
        $MonitorPeakPrivateBytes = Get-P1Value $Monitor 'peak_process_tree_private_bytes'
        $MonitorStopReason = Get-P1Value $Monitor 'stop_reason'
        foreach ($Required in @(
            'complete', 'ended_at', 'wall_seconds',
            'peak_process_tree_working_set_bytes',
            'peak_process_tree_private_bytes', 'stop_reason', 'failures',
            'phases', 'skipped_phases'
        )) {
            if (-not (Test-P1HasProperty $Monitor $Required)) {
                $Failures.Add("external monitor lacks required field $Required")
            }
        }
        foreach ($ArrayField in @('failures', 'phases', 'skipped_phases')) {
            if (Test-P1HasProperty $Monitor $ArrayField) {
                if ((Get-P1Value $Monitor $ArrayField) -isnot [System.Array]) {
                    $Failures.Add("external monitor $ArrayField is not an array")
                }
            }
        }
        if (-not (Test-P1HasProperty $Monitor 'complete') -or
            $MonitorComplete -isnot [bool] -or -not [bool]$MonitorComplete) {
            $Failures.Add('external monitor is not complete')
        }
        if (-not (Test-P1HasProperty $Monitor 'ended_at') -or
            -not (Test-P1EndedAt $MonitorEndedAt)) {
            $Failures.Add('external monitor lacks ended_at')
        }
        if (-not (Test-P1HasProperty $Monitor 'wall_seconds') -or
            -not (Test-P1FiniteNonnegativeNumber $MonitorWallSeconds)) {
            $Failures.Add('external monitor wall_seconds is missing or invalid')
        } elseif ([double]$MonitorWallSeconds -gt 2100.0) {
            $Failures.Add('external monitor exceeded 35-minute wall')
        }
        if (-not (Test-P1HasProperty $Monitor 'peak_process_tree_working_set_bytes') -or
            -not (Test-P1Integer $MonitorPeakWorkingSet) -or
            -not (Test-P1FiniteNonnegativeNumber $MonitorPeakWorkingSet)) {
            $Failures.Add('external monitor peak working set is missing or invalid')
        } elseif ([int64]$MonitorPeakWorkingSet -gt [int64](8GB)) {
            $Failures.Add('external monitor exceeded 8-GiB process-tree working set')
        }
        if (-not (Test-P1HasProperty $Monitor 'peak_process_tree_private_bytes') -or
            -not (Test-P1Integer $MonitorPeakPrivateBytes) -or
            -not (Test-P1FiniteNonnegativeNumber $MonitorPeakPrivateBytes)) {
            $Failures.Add('external monitor peak private bytes is missing or invalid')
        }
        if (-not (Test-P1HasProperty $Monitor 'stop_reason') -or
            $MonitorStopReason -isnot [string] -or
            [string]$MonitorStopReason -cne 'completed') {
            $Failures.Add("external monitor stop reason: $MonitorStopReason")
        }
        $MonitorFailures = if (Test-P1HasProperty $Monitor 'failures') {
            @((Get-P1Value $Monitor 'failures'))
        } else {
            @()
        }
        foreach ($Failure in $MonitorFailures) {
            if (-not [string]::IsNullOrWhiteSpace([string]$Failure)) {
                $Failures.Add("monitor: $Failure")
            }
        }

        $PhaseValues = if (Test-P1HasProperty $Monitor 'phases') { @((Get-P1Value $Monitor 'phases')) } else { @() }
        $SkippedValues = if (Test-P1HasProperty $Monitor 'skipped_phases') { @((Get-P1Value $Monitor 'skipped_phases')) } else { @() }
        $Launched = @($PhaseValues | ForEach-Object { if (Test-P1HasProperty $_ 'phase') { [string](Get-P1Value $_ 'phase') } else { '' } })
        $Skipped = @($SkippedValues | ForEach-Object { if (Test-P1HasProperty $_ 'phase') { [string](Get-P1Value $_ 'phase') } else { '' } })
        $Combined = @($Launched + $Skipped)
        if (@($Combined | Select-Object -Unique).Count -ne $Combined.Count) {
            $Failures.Add('phase ledger contains duplicate launched/skipped entries')
        }
        foreach ($Expected in $script:P1ExpectedPhaseOrder) {
            if ($Expected -notin $Combined) {
                $Failures.Add("phase ledger omits $Expected")
            }
        }
        foreach ($Observed in $Combined) {
            if ($Observed -notin $script:P1ExpectedPhaseOrder) {
                $Failures.Add("phase ledger contains unexpected phase $Observed")
            }
        }
        $LaunchedExpectedOrder = @(
            $script:P1ExpectedPhaseOrder | Where-Object { $_ -in $Launched }
        )
        if (($Launched -join "`n") -cne ($LaunchedExpectedOrder -join "`n")) {
            $Failures.Add('launched phases are not in preregistered order')
        }
        foreach ($Phase in $PhaseValues) {
            $PhaseName = Get-P1Value $Phase 'phase'
            $PhaseExitCode = Get-P1Value $Phase 'exit_code'
            $PhaseStopReason = Get-P1Value $Phase 'stop_reason'
            $PhaseSeconds = Get-P1Value $Phase 'seconds'
            $PhasePeakWorkingSet = Get-P1Value $Phase 'peak_process_tree_working_set_bytes'
            $PhasePeakPrivateBytes = Get-P1Value $Phase 'peak_process_tree_private_bytes'
            foreach ($Required in @(
                'phase', 'exit_code', 'stop_reason', 'seconds',
                'peak_process_tree_working_set_bytes',
                'peak_process_tree_private_bytes'
            )) {
                if (-not (Test-P1HasProperty $Phase $Required)) {
                    $Failures.Add("phase ledger entry lacks $Required")
                }
            }
            if (-not (Test-P1HasProperty $Phase 'phase') -or
                $PhaseName -isnot [string] -or
                [string]::IsNullOrWhiteSpace([string]$PhaseName)) {
                $Failures.Add('phase ledger entry has invalid phase name')
            }
            if (-not (Test-P1HasProperty $Phase 'exit_code') -or
                -not (Test-P1Integer $PhaseExitCode) -or
                [int]$PhaseExitCode -ne 0 -or
                -not (Test-P1HasProperty $Phase 'stop_reason') -or
                $PhaseStopReason -isnot [string] -or
                [string]$PhaseStopReason -cne 'completed') {
                $Failures.Add("phase $PhaseName did not complete cleanly")
            }
            if (-not (Test-P1HasProperty $Phase 'seconds') -or
                -not (Test-P1FiniteNonnegativeNumber $PhaseSeconds) -or
                -not (Test-P1HasProperty $Phase 'peak_process_tree_working_set_bytes') -or
                -not (Test-P1Integer $PhasePeakWorkingSet) -or
                -not (Test-P1FiniteNonnegativeNumber $PhasePeakWorkingSet) -or
                -not (Test-P1HasProperty $Phase 'peak_process_tree_private_bytes') -or
                -not (Test-P1Integer $PhasePeakPrivateBytes) -or
                -not (Test-P1FiniteNonnegativeNumber $PhasePeakPrivateBytes)) {
                $Failures.Add("phase $PhaseName has invalid accounting")
            }
        }
        if ($AssessmentSuccess) {
            if ($Skipped.Count -ne 0 -or
                ($Launched -join "`n") -cne ($script:P1ExpectedPhaseOrder -join "`n")) {
                $Failures.Add('passing assessment lacks the complete successful phase ledger')
            }
        }
    }

    $UniqueFailures = @($Failures | Select-Object -Unique)
    $Passed = $UniqueFailures.Count -eq 0
    return [ordered]@{
        experiment = 'P1 conservative legacy partial-tail anchored TD'
        status = if ($Passed) { 'P1-development-pass' } else { 'P1-development-fail' }
        success = $Passed
        failures = $UniqueFailures
        assessment = $Assessment
        monitor = $Monitor
        scope = 'development evidence only; not statistical model-beat evidence'
        checkpoint_frozen_for_sealed_review = $false
        sealed_test_authorized = $false
        validation_seed_used = $false
        sealed_test_seed_used = $false
        rescue_run_authorized = $false
        original_checkpoint_overwritten = $false
        existing_weight_artifact_overwritten = $false
    }
}

function Write-P1JsonDurableAtomic(
    [string]$Path,
    $Value,
    [int]$Depth = 64,
    [switch]$ReplaceExisting,
    [switch]$InjectAfterFlushBeforeRename
) {
    $Json = $Value | ConvertTo-Json -Depth $Depth
    $Bytes = [System.Text.UTF8Encoding]::new($false).GetBytes(
        $Json + [Environment]::NewLine
    )
    $Temporary = "$Path.tmp"
    $Stream = [System.IO.File]::Open(
        $Temporary,
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
    try {
        if ($InjectAfterFlushBeforeRename) {
            throw 'injected failure after durable flush and before atomic rename'
        }
        if ($ReplaceExisting -and [System.IO.File]::Exists($Path)) {
            $Backup = "$Path.replace-backup"
            if ([System.IO.File]::Exists($Backup)) {
                throw "atomic replacement backup already exists: $Backup"
            }
            [System.IO.File]::Replace($Temporary, $Path, $Backup, $true)
            [System.IO.File]::Delete($Backup)
        } else {
            [System.IO.File]::Move($Temporary, $Path)
        }
    } catch {
        if ([System.IO.File]::Exists($Temporary)) {
            [System.IO.File]::Delete($Temporary)
        }
        throw
    }
}

function Write-P1TerminalArtifacts(
    [string]$MonitorPath,
    [string]$WrapperResultPath,
    [string]$FinalResultPath,
    $Assessment,
    $Monitor,
    $WrapperResult,
    [ValidateSet(
        '', 'before_monitor', 'monitor_after_flush', 'after_monitor',
        'before_wrapper', 'wrapper_after_flush', 'after_wrapper',
        'before_final', 'final_after_flush'
    )]
    [string]$FailureInjection = ''
) {
    $FinalResult = New-P1FinalResult $Assessment $Monitor
    $WrapperSuccess = Get-P1Value $WrapperResult 'success'
    $FinalSuccess = Get-P1Value $FinalResult 'success'
    if (-not (Test-P1HasProperty $WrapperResult 'success') -or
        $WrapperSuccess -isnot [bool] -or
        [bool]$WrapperSuccess -ne [bool]$FinalSuccess) {
        throw 'wrapper ledger success does not match composed final result'
    }
    if ($FailureInjection -ceq 'before_monitor') { throw 'injected before monitor write' }
    Write-P1JsonDurableAtomic $MonitorPath $Monitor -ReplaceExisting `
        -InjectAfterFlushBeforeRename:($FailureInjection -ceq 'monitor_after_flush')
    if ($FailureInjection -ceq 'after_monitor') { throw 'injected after monitor write' }
    if ($FailureInjection -ceq 'before_wrapper') { throw 'injected before wrapper write' }
    Write-P1JsonDurableAtomic $WrapperResultPath $WrapperResult `
        -InjectAfterFlushBeforeRename:($FailureInjection -ceq 'wrapper_after_flush')
    if ($FailureInjection -ceq 'after_wrapper') { throw 'injected after wrapper write' }
    if ($FailureInjection -ceq 'before_final') { throw 'injected before final write' }
    Write-P1JsonDurableAtomic $FinalResultPath $FinalResult `
        -InjectAfterFlushBeforeRename:($FailureInjection -ceq 'final_after_flush')
    return $FinalResult
}
