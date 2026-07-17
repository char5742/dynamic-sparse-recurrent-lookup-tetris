Set-StrictMode -Version Latest

$script:Q1ExpectedPhaseOrder = @(
    'extract_training',
    'train_correction',
    'verify_openvino',
    'extract_offline',
    'offline_gate',
    'finalize_assessment'
)

function Test-Q1HasProperty($Value, [string]$Name) {
    if ($Value -is [System.Collections.IDictionary]) {
        return $Value.Contains($Name)
    }
    return $null -ne $Value -and $Value.PSObject.Properties.Name -contains $Name
}

function Get-Q1Value($Value, [string]$Name) {
    if ($null -eq $Value) { return $null }
    if ($Value -is [System.Collections.IDictionary]) {
        $Result = $Value[$Name]
    } else {
        $Property = $Value.PSObject.Properties[$Name]
        if ($null -eq $Property) { return $null }
        $Result = $Property.Value
    }
    # Prevent PowerShell from unrolling actual JSON/list arrays while preserving
    # scalar dictionary values.  The caller can still enumerate with @(...).
    if ($Result -is [System.Array]) { return ,$Result }
    return $Result
}

function Test-Q1FiniteNonnegativeNumber($Value) {
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

function Test-Q1Integer($Value) {
    return $Value -is [byte] -or $Value -is [int16] -or
           $Value -is [int32] -or $Value -is [int64]
}

function Test-Q1EndedAt($Value) {
    if ($Value -is [string]) {
        return -not [string]::IsNullOrWhiteSpace($Value)
    }
    return $Value -is [datetime] -or $Value -is [datetimeoffset]
}

function New-Q1FinalResult($Assessment, $Monitor) {
    $Failures = [System.Collections.Generic.List[string]]::new()
    $AssessmentSuccess = $false
    if ($null -eq $Assessment) {
        $Failures.Add('missing post-pipeline assessment')
    } else {
        $Status = Get-Q1Value $Assessment 'status'
        $Success = Get-Q1Value $Assessment 'success'
        $AssessmentFailures = Get-Q1Value $Assessment 'failures'
        if (-not (Test-Q1HasProperty $Assessment 'status') -or
            $Status -isnot [string] -or $Status -cne 'assessment-pass') {
            $Failures.Add('assessment status is not assessment-pass')
        }
        if (-not (Test-Q1HasProperty $Assessment 'success') -or
            $Success -isnot [bool]) {
            $Failures.Add('assessment lacks a Boolean success field')
        }
        if (-not (Test-Q1HasProperty $Assessment 'failures') -or
            $AssessmentFailures -isnot [System.Array]) {
            $Failures.Add('assessment lacks an array failures field')
        } elseif ($Success -is [bool] -and [bool]$Success -and
            @($AssessmentFailures).Count -ne 0) {
            $Failures.Add('passing assessment contains failures')
        } elseif ($Success -is [bool] -and -not [bool]$Success) {
            foreach ($Failure in @($AssessmentFailures)) {
                if (-not [string]::IsNullOrWhiteSpace([string]$Failure)) {
                    $Failures.Add("assessment: $Failure")
                }
            }
            if (@($AssessmentFailures).Count -eq 0) {
                $Failures.Add('assessment rejected without a recorded reason')
            }
        }
        foreach ($Pair in @(
            @('promotion', 'Q1-offline-promoted'),
            @('scope', 'offline reused-development safety evidence only')
        )) {
            $Value = Get-Q1Value $Assessment $Pair[0]
            if (-not (Test-Q1HasProperty $Assessment $Pair[0]) -or
                $Value -isnot [string] -or $Value -cne $Pair[1]) {
                $Failures.Add("assessment $($Pair[0]) mismatch")
            }
        }
        foreach ($Field in @(
            'game_strength_evidence', 'model_beat_claim',
            'game_evaluation_authorized', 'validation_seed_used',
            'sealed_test_seed_used', 'sealed_test_authorized'
        )) {
            $Value = Get-Q1Value $Assessment $Field
            if (-not (Test-Q1HasProperty $Assessment $Field) -or
                $Value -isnot [bool] -or [bool]$Value) {
                $Failures.Add("assessment scope field $Field is not exactly false")
            }
        }
        $AssessmentSuccess = $Success -is [bool] -and [bool]$Success
    }

    if ($null -eq $Monitor) {
        $Failures.Add('missing completed external monitor')
    } else {
        foreach ($Required in @(
            'complete', 'ended_at', 'wall_seconds',
            'peak_process_tree_working_set_bytes',
            'peak_process_tree_private_bytes', 'stop_reason', 'failures',
            'phases', 'skipped_phases'
        )) {
            if (-not (Test-Q1HasProperty $Monitor $Required)) {
                $Failures.Add("external monitor lacks required field $Required")
            }
        }
        foreach ($ArrayField in @('failures', 'phases', 'skipped_phases')) {
            if (Test-Q1HasProperty $Monitor $ArrayField) {
                if ((Get-Q1Value $Monitor $ArrayField) -isnot [System.Array]) {
                    $Failures.Add("external monitor $ArrayField is not an array")
                }
            }
        }
        $Complete = Get-Q1Value $Monitor 'complete'
        if (-not (Test-Q1HasProperty $Monitor 'complete') -or
            $Complete -isnot [bool] -or -not [bool]$Complete) {
            $Failures.Add('external monitor is not complete')
        }
        $EndedAt = Get-Q1Value $Monitor 'ended_at'
        if (-not (Test-Q1HasProperty $Monitor 'ended_at') -or
            -not (Test-Q1EndedAt $EndedAt)) {
            $Failures.Add('external monitor lacks ended_at')
        }
        $Wall = Get-Q1Value $Monitor 'wall_seconds'
        if (-not (Test-Q1HasProperty $Monitor 'wall_seconds') -or
            -not (Test-Q1FiniteNonnegativeNumber $Wall)) {
            $Failures.Add('external monitor wall_seconds is missing or invalid')
        } elseif ([double]$Wall -gt 720.0) {
            $Failures.Add('external monitor exceeded 12-minute wall')
        }
        foreach ($Spec in @(
            @('peak_process_tree_working_set_bytes', 'working set'),
            @('peak_process_tree_private_bytes', 'private bytes')
        )) {
            $Value = Get-Q1Value $Monitor $Spec[0]
            if (-not (Test-Q1HasProperty $Monitor $Spec[0]) -or
                -not (Test-Q1Integer $Value) -or
                -not (Test-Q1FiniteNonnegativeNumber $Value)) {
                $Failures.Add("external monitor peak $($Spec[1]) is missing or invalid")
            } elseif ([int64]$Value -gt [int64](4GB)) {
                $Failures.Add("external monitor exceeded 4-GiB process-tree $($Spec[1])")
            }
        }
        $StopReason = Get-Q1Value $Monitor 'stop_reason'
        if (-not (Test-Q1HasProperty $Monitor 'stop_reason') -or
            $StopReason -isnot [string] -or $StopReason -cne 'completed') {
            $Failures.Add("external monitor stop reason: $StopReason")
        }
        if (Test-Q1HasProperty $Monitor 'failures') {
            foreach ($Failure in @((Get-Q1Value $Monitor 'failures'))) {
                if (-not [string]::IsNullOrWhiteSpace([string]$Failure)) {
                    $Failures.Add("monitor: $Failure")
                }
            }
        }

        $PhaseValues = if (Test-Q1HasProperty $Monitor 'phases') {
            @((Get-Q1Value $Monitor 'phases'))
        } else { @() }
        $SkippedValues = if (Test-Q1HasProperty $Monitor 'skipped_phases') {
            @((Get-Q1Value $Monitor 'skipped_phases'))
        } else { @() }
        $Launched = @($PhaseValues | ForEach-Object {
            if (Test-Q1HasProperty $_ 'phase') { [string](Get-Q1Value $_ 'phase') } else { '' }
        })
        $Skipped = @($SkippedValues | ForEach-Object {
            if (Test-Q1HasProperty $_ 'phase') { [string](Get-Q1Value $_ 'phase') } else { '' }
        })
        $Combined = @($Launched + $Skipped)
        if (@($Combined | Select-Object -Unique).Count -ne $Combined.Count) {
            $Failures.Add('phase ledger contains duplicate launched/skipped entries')
        }
        foreach ($Expected in $script:Q1ExpectedPhaseOrder) {
            if ($Expected -notin $Combined) { $Failures.Add("phase ledger omits $Expected") }
        }
        foreach ($Observed in $Combined) {
            if ($Observed -notin $script:Q1ExpectedPhaseOrder) {
                $Failures.Add("phase ledger contains unexpected phase $Observed")
            }
        }
        $ExpectedLaunchedOrder = @($script:Q1ExpectedPhaseOrder | Where-Object { $_ -in $Launched })
        if (($Launched -join "`n") -cne ($ExpectedLaunchedOrder -join "`n")) {
            $Failures.Add('launched phases are not in preregistered order')
        }
        foreach ($Phase in $PhaseValues) {
            $Name = Get-Q1Value $Phase 'phase'
            foreach ($Required in @(
                'phase', 'exit_code', 'stop_reason', 'seconds',
                'peak_process_tree_working_set_bytes',
                'peak_process_tree_private_bytes'
            )) {
                if (-not (Test-Q1HasProperty $Phase $Required)) {
                    $Failures.Add("phase ledger entry lacks $Required")
                }
            }
            $ExitCode = Get-Q1Value $Phase 'exit_code'
            $PhaseStop = Get-Q1Value $Phase 'stop_reason'
            if ($Name -isnot [string] -or [string]::IsNullOrWhiteSpace([string]$Name)) {
                $Failures.Add('phase ledger entry has invalid phase name')
            }
            if (-not (Test-Q1Integer $ExitCode) -or [int]$ExitCode -ne 0 -or
                $PhaseStop -isnot [string] -or $PhaseStop -cne 'completed') {
                $Failures.Add("phase $Name did not complete cleanly")
            }
            foreach ($Field in @(
                'seconds', 'peak_process_tree_working_set_bytes',
                'peak_process_tree_private_bytes'
            )) {
                $Value = Get-Q1Value $Phase $Field
                if (-not (Test-Q1FiniteNonnegativeNumber $Value) -or
                    ($Field -ne 'seconds' -and -not (Test-Q1Integer $Value))) {
                    $Failures.Add("phase $Name has invalid accounting")
                }
            }
        }
        if ($AssessmentSuccess -and
            ($Skipped.Count -ne 0 -or
             ($Launched -join "`n") -cne ($script:Q1ExpectedPhaseOrder -join "`n"))) {
            $Failures.Add('passing assessment lacks the complete successful phase ledger')
        }
    }

    $UniqueFailures = @($Failures | Select-Object -Unique)
    $Passed = $UniqueFailures.Count -eq 0
    return [ordered]@{
        experiment = 'Q1 frozen-old additive residual critic'
        status = if ($Passed) { 'Q1-offline-promoted' } else { 'Q1-offline-rejected' }
        success = $Passed
        failures = $UniqueFailures
        assessment = $Assessment
        monitor = $Monitor
        scope = 'offline reused-development safety evidence only; no game-strength evidence'
        game_strength_evidence = $false
        model_beat_claim = $false
        game_evaluation_authorized = $false
        checkpoint_frozen_for_later_review = $false
        validation_seed_used = $false
        sealed_test_seed_used = $false
        sealed_test_authorized = $false
        retry_authorized = $false
        original_checkpoint_overwritten = $false
        existing_weight_artifact_overwritten = $false
    }
}

function Write-Q1JsonDurableAtomic(
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

function Write-Q1TerminalArtifacts(
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
    $FinalResult = New-Q1FinalResult $Assessment $Monitor
    $WrapperSuccess = Get-Q1Value $WrapperResult 'success'
    $FinalSuccess = Get-Q1Value $FinalResult 'success'
    if (-not (Test-Q1HasProperty $WrapperResult 'success') -or
        $WrapperSuccess -isnot [bool] -or
        [bool]$WrapperSuccess -ne [bool]$FinalSuccess) {
        throw 'wrapper ledger success does not match composed final result'
    }
    if ($FailureInjection -ceq 'before_monitor') { throw 'injected before monitor write' }
    Write-Q1JsonDurableAtomic $MonitorPath $Monitor -ReplaceExisting `
        -InjectAfterFlushBeforeRename:($FailureInjection -ceq 'monitor_after_flush')
    if ($FailureInjection -ceq 'after_monitor') { throw 'injected after monitor write' }
    if ($FailureInjection -ceq 'before_wrapper') { throw 'injected before wrapper write' }
    Write-Q1JsonDurableAtomic $WrapperResultPath $WrapperResult `
        -InjectAfterFlushBeforeRename:($FailureInjection -ceq 'wrapper_after_flush')
    if ($FailureInjection -ceq 'after_wrapper') { throw 'injected after wrapper write' }
    if ($FailureInjection -ceq 'before_final') { throw 'injected before final write' }
    Write-Q1JsonDurableAtomic $FinalResultPath $FinalResult `
        -InjectAfterFlushBeforeRename:($FailureInjection -ceq 'final_after_flush')
    return $FinalResult
}
