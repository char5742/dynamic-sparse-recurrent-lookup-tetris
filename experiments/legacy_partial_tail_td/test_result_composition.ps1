[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'result_composition.ps1')

function New-Phase([string]$Name) {
    return [ordered]@{
        phase = $Name
        exit_code = 0
        stop_reason = 'completed'
        seconds = 0.01
        peak_process_tree_working_set_bytes = 1024
        peak_process_tree_private_bytes = 1024
    }
}

$Assessment = [pscustomobject]@{ status = 'assessment-pass'; success = $true; failures = @() }
$CompleteMonitor = [ordered]@{
    complete = $true
    ended_at = '2026-07-18T00:00:00Z'
    wall_seconds = 100.0
    peak_process_tree_working_set_bytes = 1024
    peak_process_tree_private_bytes = 768
    stop_reason = 'completed'
    failures = @()
    phases = @($script:P1ExpectedPhaseOrder | ForEach-Object { New-Phase $_ })
    skipped_phases = @()
}

$OrderedComplete = Get-P1Value $CompleteMonitor 'complete'
$OrderedWall = Get-P1Value $CompleteMonitor 'wall_seconds'
$OrderedStopReason = Get-P1Value $CompleteMonitor 'stop_reason'
$OrderedPhases = Get-P1Value $CompleteMonitor 'phases'
if ($OrderedComplete -isnot [bool]) {
    throw "ordered complete accessor returned $($OrderedComplete.GetType().FullName), not Boolean"
}
if (-not (Test-P1FiniteNonnegativeNumber $OrderedWall)) {
    throw 'ordered wall_seconds accessor did not return a numeric scalar'
}
if ($OrderedStopReason -isnot [string]) {
    throw 'ordered stop_reason accessor did not return a string scalar'
}
if ($OrderedPhases -isnot [System.Array] -or
    @($OrderedPhases).Count -ne $script:P1ExpectedPhaseOrder.Count) {
    throw 'ordered phases accessor did not preserve the enumerable phase array'
}

$Pass = New-P1FinalResult $Assessment $CompleteMonitor
if (-not $Pass.success) { throw "complete synthetic ledger did not pass: $($Pass.failures -join '; ')" }
$RoundTripMonitor = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$RoundTripAssessment = $Assessment | ConvertTo-Json -Depth 4 | ConvertFrom-Json
$RoundTripMonitor.ended_at = [datetime]::Parse(
    '2026-07-18T00:00:00Z',
    [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::RoundtripKind
)
$RoundTripPass = New-P1FinalResult $RoundTripAssessment $RoundTripMonitor
if (-not $RoundTripPass.success) {
    throw "JSON-round-trip complete ledger did not pass: $($RoundTripPass.failures -join '; ')"
}

$Incomplete = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$Incomplete.complete = $false
if ((New-P1FinalResult $Assessment $Incomplete).success) {
    throw 'incomplete monitor incorrectly passed'
}

$MissingFinalizer = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$MissingFinalizer.phases = @($MissingFinalizer.phases | Where-Object { $_.phase -ne 'finalize_assessment' })
if ((New-P1FinalResult $Assessment $MissingFinalizer).success) {
    throw 'missing finalizer phase incorrectly passed'
}

$RejectedAssessment = [pscustomobject]@{ status = 'assessment-fail'; success = $false; failures = @('stub') }
if ((New-P1FinalResult $RejectedAssessment $CompleteMonitor).success) {
    throw 'rejected assessment incorrectly passed'
}

$BadAccounting = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$BadAccounting.phases = @($BadAccounting.phases)
$BadAccounting.phases[0] = New-Phase 'eligibility'
$BadAccounting.phases[0].seconds = -1
if ((New-P1FinalResult $Assessment $BadAccounting).success) {
    throw 'negative phase accounting incorrectly passed'
}

$MissingMonitorFields = @(
    'wall_seconds',
    'peak_process_tree_working_set_bytes',
    'peak_process_tree_private_bytes',
    'phases',
    'skipped_phases',
    'failures'
)
foreach ($Field in $MissingMonitorFields) {
    $Value = $RoundTripMonitor | Select-Object * -ExcludeProperty $Field
    if ((New-P1FinalResult $Assessment $Value).success) {
        throw "monitor missing $Field incorrectly passed"
    }
}

$MissingPhaseAccounting = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$MissingPhaseAccounting.phases = @($MissingPhaseAccounting.phases)
$MissingPhaseAccounting.phases[0] = $MissingPhaseAccounting.phases[0] |
    Select-Object * -ExcludeProperty peak_process_tree_private_bytes
if ((New-P1FinalResult $Assessment $MissingPhaseAccounting).success) {
    throw 'phase missing private-byte accounting incorrectly passed'
}

$TerminalRoot = Join-Path ([IO.Path]::GetTempPath()) (
    'p1-terminal-' + [guid]::NewGuid().ToString('N')
)
[IO.Directory]::CreateDirectory($TerminalRoot) | Out-Null
try {
    $PositiveRoot = Join-Path $TerminalRoot 'positive'
    [IO.Directory]::CreateDirectory($PositiveRoot) | Out-Null
    $PositiveMonitor = Join-Path $PositiveRoot 'monitor.json'
    [IO.File]::WriteAllText($PositiveMonitor, "{}`n")
    $PositiveWrapper = Join-Path $PositiveRoot 'wrapper_result.json'
    $PositiveFinal = Join-Path $PositiveRoot 'final_result.json'
    $WrapperLedger = [ordered]@{ success = $true; status = 'wrapper-complete' }
    $Published = Write-P1TerminalArtifacts $PositiveMonitor $PositiveWrapper `
        $PositiveFinal $Assessment $CompleteMonitor $WrapperLedger
    if (-not $Published.success -or
        -not [IO.File]::Exists($PositiveMonitor) -or
        -not [IO.File]::Exists($PositiveWrapper) -or
        -not [IO.File]::Exists($PositiveFinal)) {
        throw 'durable terminal publication positive case failed'
    }

    $InjectionCases = @(
        'before_monitor', 'monitor_after_flush', 'after_monitor',
        'before_wrapper', 'wrapper_after_flush', 'after_wrapper',
        'before_final', 'final_after_flush'
    )
    foreach ($Injection in $InjectionCases) {
        $CaseRoot = Join-Path $TerminalRoot $Injection
        [IO.Directory]::CreateDirectory($CaseRoot) | Out-Null
        $CaseMonitor = Join-Path $CaseRoot 'monitor.json'
        [IO.File]::WriteAllText($CaseMonitor, "{}`n")
        $CaseWrapper = Join-Path $CaseRoot 'wrapper_result.json'
        $CaseFinal = Join-Path $CaseRoot 'final_result.json'
        $Failed = $false
        try {
            [void](Write-P1TerminalArtifacts $CaseMonitor $CaseWrapper $CaseFinal `
                $Assessment $CompleteMonitor $WrapperLedger $Injection)
        } catch {
            $Failed = $true
        }
        if (-not $Failed) { throw "failure injection $Injection did not fail" }
        if ([IO.File]::Exists($CaseFinal)) {
            $Retained = Get-Content -LiteralPath $CaseFinal -Raw | ConvertFrom-Json
            if ([bool]$Retained.success) {
                throw "failure injection $Injection retained a passing final result"
            }
            throw "failure injection $Injection unexpectedly retained final_result.json"
        }
    }
} finally {
    if ([IO.Directory]::Exists($TerminalRoot)) {
        Remove-Item -LiteralPath $TerminalRoot -Recurse -Force
    }
}

[ordered]@{
    status = 'result_composition_negative_checks_passed'
    positive_cases = 3
    negative_cases = 19
    injected_terminal_failures = 8
    checkpoint_or_dataset_opened = $false
    heavy_work_run = $false
} | ConvertTo-Json
