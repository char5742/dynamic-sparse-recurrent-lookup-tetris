[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'result_composition.ps1')

function New-Q1SyntheticPhase([string]$Name) {
    return [ordered]@{
        phase = $Name
        exit_code = 0
        stop_reason = 'completed'
        seconds = 0.01
        peak_process_tree_working_set_bytes = [int64]1024
        peak_process_tree_private_bytes = [int64]768
    }
}

$Assessment = [pscustomobject]@{
    status = 'assessment-pass'
    success = $true
    failures = @()
    promotion = 'Q1-offline-promoted'
    scope = 'offline reused-development safety evidence only'
    game_strength_evidence = $false
    model_beat_claim = $false
    game_evaluation_authorized = $false
    validation_seed_used = $false
    sealed_test_seed_used = $false
    sealed_test_authorized = $false
}
$CompleteMonitor = [ordered]@{
    complete = $true
    ended_at = '2026-07-18T00:00:00Z'
    wall_seconds = 100.0
    peak_process_tree_working_set_bytes = [int64]1024
    peak_process_tree_private_bytes = [int64]768
    stop_reason = 'completed'
    failures = @()
    phases = @($script:Q1ExpectedPhaseOrder | ForEach-Object { New-Q1SyntheticPhase $_ })
    skipped_phases = @()
}

$OrderedComplete = Get-Q1Value $CompleteMonitor 'complete'
$OrderedWall = Get-Q1Value $CompleteMonitor 'wall_seconds'
$OrderedPhases = Get-Q1Value $CompleteMonitor 'phases'
if ($OrderedComplete -isnot [bool]) { throw 'ordered complete accessor is not Boolean' }
if (-not (Test-Q1FiniteNonnegativeNumber $OrderedWall)) { throw 'ordered wall accessor is not numeric' }
if ($OrderedPhases -isnot [System.Array] -or @($OrderedPhases).Count -ne 6) {
    throw 'ordered phases accessor did not preserve the array'
}

$Pass = New-Q1FinalResult $Assessment $CompleteMonitor
if (-not $Pass.success -or $Pass.status -cne 'Q1-offline-promoted') {
    throw "complete Q1 synthetic ledger did not pass: $($Pass.failures -join '; ')"
}
if ($Pass.game_strength_evidence -or $Pass.model_beat_claim -or
    $Pass.game_evaluation_authorized) {
    throw 'offline promotion was incorrectly elevated to game/model evidence'
}

$RoundTripMonitor = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$RoundTripAssessment = $Assessment | ConvertTo-Json -Depth 6 | ConvertFrom-Json
$RoundTripMonitor.ended_at = [datetime]::Parse(
    '2026-07-18T00:00:00Z',
    [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::RoundtripKind
)
if (-not (New-Q1FinalResult $RoundTripAssessment $RoundTripMonitor).success) {
    throw 'JSON round-trip Q1 ledger did not pass'
}

$NegativeCases = @()
$Value = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$Value.complete = $false
$NegativeCases += ,@($Assessment, $Value, 'incomplete monitor')
$Value = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$Value.wall_seconds = 720.001
$NegativeCases += ,@($Assessment, $Value, 'wall overrun')
$Value = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$Value.peak_process_tree_working_set_bytes = [int64](4GB + 1)
$NegativeCases += ,@($Assessment, $Value, 'working-set overrun')
$Value = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$Value.peak_process_tree_private_bytes = [int64](4GB + 1)
$NegativeCases += ,@($Assessment, $Value, 'private-byte overrun')
$Value = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$Value.phases = @($Value.phases | Where-Object { $_.phase -ne 'offline_gate' })
$NegativeCases += ,@($Assessment, $Value, 'missing phase')
$Value = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
[array]::Reverse($Value.phases)
$NegativeCases += ,@($Assessment, $Value, 'phase order')
$BadAssessment = $Assessment | ConvertTo-Json -Depth 6 | ConvertFrom-Json
$BadAssessment.game_strength_evidence = $true
$NegativeCases += ,@($BadAssessment, $CompleteMonitor, 'strength overclaim')
$BadAssessment = $Assessment | ConvertTo-Json -Depth 6 | ConvertFrom-Json
$BadAssessment.PSObject.Properties.Remove('sealed_test_authorized')
$NegativeCases += ,@($BadAssessment, $CompleteMonitor, 'missing scope flag')

foreach ($Case in $NegativeCases) {
    if ((New-Q1FinalResult $Case[0] $Case[1]).success) {
        throw "$($Case[2]) incorrectly passed"
    }
}

$MissingPhaseAccounting = $CompleteMonitor | ConvertTo-Json -Depth 12 | ConvertFrom-Json
$MissingPhaseAccounting.phases[0] = $MissingPhaseAccounting.phases[0] |
    Select-Object * -ExcludeProperty peak_process_tree_private_bytes
if ((New-Q1FinalResult $Assessment $MissingPhaseAccounting).success) {
    throw 'phase missing private-byte accounting incorrectly passed'
}

$TerminalRoot = Join-Path ([IO.Path]::GetTempPath()) (
    'q1-terminal-' + [guid]::NewGuid().ToString('N')
)
[IO.Directory]::CreateDirectory($TerminalRoot) | Out-Null
try {
    $PositiveRoot = Join-Path $TerminalRoot 'positive'
    [IO.Directory]::CreateDirectory($PositiveRoot) | Out-Null
    $MonitorPath = Join-Path $PositiveRoot 'monitor.json'
    [IO.File]::WriteAllText($MonitorPath, "{}`n")
    $WrapperPath = Join-Path $PositiveRoot 'wrapper_result.json'
    $FinalPath = Join-Path $PositiveRoot 'final_result.json'
    $Wrapper = [ordered]@{ success = $true; status = 'wrapper-complete' }
    $Published = Write-Q1TerminalArtifacts $MonitorPath $WrapperPath $FinalPath `
        $Assessment $CompleteMonitor $Wrapper
    if (-not $Published.success -or -not [IO.File]::Exists($MonitorPath) -or
        -not [IO.File]::Exists($WrapperPath) -or -not [IO.File]::Exists($FinalPath)) {
        throw 'durable terminal publication positive case failed'
    }

    $Injections = @(
        'before_monitor', 'monitor_after_flush', 'after_monitor',
        'before_wrapper', 'wrapper_after_flush', 'after_wrapper',
        'before_final', 'final_after_flush'
    )
    foreach ($Injection in $Injections) {
        $CaseRoot = Join-Path $TerminalRoot $Injection
        [IO.Directory]::CreateDirectory($CaseRoot) | Out-Null
        $CaseMonitor = Join-Path $CaseRoot 'monitor.json'
        [IO.File]::WriteAllText($CaseMonitor, "{}`n")
        $CaseWrapper = Join-Path $CaseRoot 'wrapper_result.json'
        $CaseFinal = Join-Path $CaseRoot 'final_result.json'
        $Failed = $false
        try {
            [void](Write-Q1TerminalArtifacts $CaseMonitor $CaseWrapper $CaseFinal `
                $Assessment $CompleteMonitor $Wrapper $Injection)
        } catch {
            $Failed = $true
        }
        if (-not $Failed) { throw "failure injection $Injection did not fail" }
        if ([IO.File]::Exists($CaseFinal)) {
            throw "failure injection $Injection unexpectedly retained final_result.json"
        }
    }
} finally {
    if ([IO.Directory]::Exists($TerminalRoot)) {
        Remove-Item -LiteralPath $TerminalRoot -Recurse -Force
    }
}

[ordered]@{
    status = 'q1_result_composition_synthetic_checks_passed'
    positive_cases = 3
    negative_cases = 9
    injected_terminal_failures = 8
    checkpoint_or_dataset_opened = $false
    heavy_work_run = $false
} | ConvertTo-Json
