param(
    [switch]$SyntheticNoMarker,
    [string]$SyntheticEntryPoint = ""
)

# N1 has exactly one production launch.  This deliberately is a small,
# single-process runner rather than a reusable experiment framework.
$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\fshuu\Documents\tetris"
$NamespaceRelative = "experiments/neural_counterfactual_top2_n1"
$NamespaceDirectory = Join-Path $RepoRoot "experiments\neural_counterfactual_top2_n1"
$ProductionEntryPoint = Join-Path $NamespaceDirectory "calibration_once.jl"
$ProductionResultName = "calibration_result.json"
$ProductionOutputRoot = "D:\tetris-paper-plus\runs"
$MarkerPath = Join-Path $RepoRoot ".experiment_state\n1_calibration_started.marker"
$ReadinessPath = Join-Path $RepoRoot ".experiment_state\n1_calibration_readiness.json"
$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$Manifest = Join-Path $RepoRoot "Manifest.toml"
$C13Checkpoint = "D:\tetris-paper-plus\checkpoints\learning\C13_round1_preregistered500_warm_c11b_best.jld2"
$OldCheckpoint = Join-Path $RepoRoot "1313\mainmodel copy 3.jld2"
$OldWeights = Join-Path $RepoRoot "artifacts\legacy_openvino\legacy_1313_weights.npz"

$ExpectedJuliaSha256 = "4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043"
$ExpectedJuliaSize = [int64]170952
$ExpectedManifestSha256 = "2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09"
$ExpectedC13Sha256 = "1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270"
$ExpectedOldCheckpointSha256 = "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
$ExpectedOldWeightsSha256 = "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"

$PrivateLimit = [int64](4GB)
$WorkingSetLimit = [int64](2GB)
$TimeoutSeconds = 4500.0
$SampleIntervalMilliseconds = 200
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Get-LowerSha256([string]$Path) {
    if (-not [System.IO.File]::Exists($Path)) {
        throw "required file is missing: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Require-ExactFile(
    [string]$Description,
    [string]$Path,
    [string]$ExpectedSha256,
    [Nullable[int64]]$ExpectedSize
) {
    $Item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($Item.PSIsContainer -or ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "$Description must be a regular, non-reparse file: $Path"
    }
    if ($null -ne $ExpectedSize -and [int64]$Item.Length -ne [int64]$ExpectedSize) {
        throw "$Description size mismatch: observed $($Item.Length)"
    }
    $Observed = Get-LowerSha256 $Path
    if ($Observed -cne $ExpectedSha256) {
        throw "$Description SHA-256 mismatch: observed $Observed"
    }
    return $Observed
}

function Get-GitOutput([string[]]$Arguments) {
    $Output = @(& git -C $RepoRoot @Arguments 2>&1)
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "git failed ($Code): $($Output -join ' ')"
    }
    return @($Output | ForEach-Object { [string]$_ })
}

function Get-N1SourceTreeSha256 {
    $Lines = @(Get-GitOutput @(
        "ls-tree", "-r", "--full-tree", "HEAD", "--", $NamespaceRelative
    ))
    if ($Lines.Count -eq 0) {
        throw "the committed N1 source tree is empty"
    }
    # Frozen scheme: UTF-8 of exact git-ls-tree lines, LF joined, final LF.
    $Canonical = ($Lines -join "`n") + "`n"
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString(
            $Hasher.ComputeHash($Utf8NoBom.GetBytes($Canonical))
        ) -replace "-", "").ToLowerInvariant()
    } finally {
        $Hasher.Dispose()
    }
}

function Require-ReadinessAndCleanSource {
    $Status = @(Get-GitOutput @(
        "status", "--porcelain=v1", "--untracked-files=all"
    ))
    # The post-commit readiness document is deliberately out-of-band because
    # it binds HEAD and the audit of that HEAD.  It is the sole permitted
    # status entry; all tracked or untracked source/data changes fail closed.
    $ReadinessRelative = ".experiment_state/n1_calibration_readiness.json"
    $AllowedReadinessStatus = "?? $ReadinessRelative"
    $Dirty = @($Status | Where-Object { $_ -cne $AllowedReadinessStatus })
    if ($Dirty.Count -ne 0) {
        throw "production launch refuses a dirty repository: $($Dirty -join '; ')"
    }
    if (-not ($Status -contains $AllowedReadinessStatus)) {
        throw "production launch requires the sole out-of-band readiness status entry"
    }

    $HeadLines = @(Get-GitOutput @("rev-parse", "--verify", "HEAD"))
    if ($HeadLines.Count -ne 1 -or $HeadLines[0] -notmatch "^[0-9a-f]{40}$") {
        throw "could not resolve one exact source HEAD"
    }
    $Head = $HeadLines[0]
    $TreeSha256 = Get-N1SourceTreeSha256
    $EntrySha256 = Get-LowerSha256 $ProductionEntryPoint
    $ReadinessSha256 = Get-LowerSha256 $ReadinessPath
    try {
        $Ready = Get-Content -LiteralPath $ReadinessPath -Raw -Encoding UTF8 |
            ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw "readiness is not valid JSON: $($_.Exception.Message)"
    }

    $RequiredText = @(
        "status", "source_commit", "n1_source_tree_sha256",
        "clean_audit_status", "audit_report_path", "audit_report_sha256",
        "entrypoint_sha256", "manifest_sha256", "julia_sha256",
        "c13_checkpoint_sha256", "old_checkpoint_sha256", "old_weights_sha256"
    )
    foreach ($Name in $RequiredText) {
        if ($null -eq $Ready.PSObject.Properties[$Name] -or
            [string]::IsNullOrWhiteSpace([string]$Ready.$Name)) {
            throw "readiness is missing nonempty field '$Name'"
        }
    }
    if ([string]$Ready.status -cne "N1-calibration-ready") {
        throw "readiness status is not N1-calibration-ready"
    }
    if ([string]$Ready.clean_audit_status -cne "GO") {
        throw "readiness clean_audit_status is not GO"
    }

    $ExactBindings = [ordered]@{
        source_commit = $Head
        n1_source_tree_sha256 = $TreeSha256
        entrypoint_sha256 = $EntrySha256
        manifest_sha256 = $ExpectedManifestSha256
        julia_sha256 = $ExpectedJuliaSha256
        c13_checkpoint_sha256 = $ExpectedC13Sha256
        old_checkpoint_sha256 = $ExpectedOldCheckpointSha256
        old_weights_sha256 = $ExpectedOldWeightsSha256
    }
    foreach ($Pair in $ExactBindings.GetEnumerator()) {
        if ([string]$Ready.($Pair.Key) -cne [string]$Pair.Value) {
            throw "readiness binding mismatch for $($Pair.Key)"
        }
    }
    $AuditReport = [System.IO.Path]::GetFullPath([string]$Ready.audit_report_path)
    $AuditSha256 = Get-LowerSha256 $AuditReport
    if ($AuditSha256 -cne [string]$Ready.audit_report_sha256) {
        throw "clean audit report SHA-256 does not match readiness"
    }

    return [ordered]@{
        source_commit = $Head
        n1_source_tree_sha256 = $TreeSha256
        n1_source_tree_hash_scheme = "sha256(UTF-8(git ls-tree -r --full-tree HEAD -- namespace, LF joined, final LF))"
        entrypoint_sha256 = $EntrySha256
        readiness_path = $ReadinessPath
        readiness_sha256 = $ReadinessSha256
        clean_audit_status = "GO"
        audit_report_path = $AuditReport
        audit_report_sha256 = $AuditSha256
        manifest_sha256 = $ExpectedManifestSha256
        julia_sha256 = $ExpectedJuliaSha256
        c13_checkpoint_sha256 = $ExpectedC13Sha256
        old_checkpoint_sha256 = $ExpectedOldCheckpointSha256
        old_weights_sha256 = $ExpectedOldWeightsSha256
    }
}

function Quote-NativeArgument([string]$Value) {
    if ($Value.Contains('"')) {
        throw "N1 runner refuses a quote in a native argument"
    }
    return '"' + $Value + '"'
}

function Write-AtomicExclusiveMarker([Collections.IDictionary]$Payload) {
    $Parent = Split-Path -Parent $MarkerPath
    [void][System.IO.Directory]::CreateDirectory($Parent)
    $Bytes = $Utf8NoBom.GetBytes(($Payload | ConvertTo-Json -Depth 8))
    $Stream = $null
    try {
        $Stream = [System.IO.File]::Open(
            $MarkerPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } catch [System.IO.IOException] {
        throw "N1 production marker already exists or could not be created exclusively: $MarkerPath"
    } finally {
        if ($null -ne $Stream) { $Stream.Dispose() }
    }
}

function Get-TreeProcessIds([int]$RootId) {
    $Rows = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    $Known = [Collections.Generic.HashSet[int]]::new()
    [void]$Known.Add($RootId)
    $Changed = $true
    while ($Changed) {
        $Changed = $false
        foreach ($Row in $Rows) {
            if ($Known.Contains([int]$Row.ParentProcessId) -and
                -not $Known.Contains([int]$Row.ProcessId)) {
                [void]$Known.Add([int]$Row.ProcessId)
                $Changed = $true
            }
        }
    }
    return @($Known)
}

function Stop-N1Process([System.Diagnostics.Process]$RootProcess) {
    if ($null -eq $RootProcess -or $RootProcess.HasExited) { return }
    foreach ($TreeId in @((Get-TreeProcessIds $RootProcess.Id) | Sort-Object -Descending)) {
        if ($TreeId -ne $RootProcess.Id) {
            Stop-Process -Id $TreeId -Force -ErrorAction SilentlyContinue
        }
    }
    Stop-Process -Id $RootProcess.Id -Force -ErrorAction SilentlyContinue
    try { $RootProcess.WaitForExit() } catch { }
}

# Synthetic mode cannot point at the repository, cannot write the production
# marker, and writes only below the current user's temporary directory.
$Mode = "production"
$EntryPoint = $ProductionEntryPoint
$OutputRoot = $ProductionOutputRoot
$Bindings = $null
[void](Require-ExactFile "Julia executable" $Julia $ExpectedJuliaSha256 $ExpectedJuliaSize)
[void](Require-ExactFile "Manifest" $Manifest $ExpectedManifestSha256 $null)
if ($SyntheticNoMarker) {
    $Mode = "synthetic-no-marker"
    if ([string]::IsNullOrWhiteSpace($SyntheticEntryPoint)) {
        throw "-SyntheticNoMarker requires -SyntheticEntryPoint"
    }
    $TempRoot = [System.IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) "n1_runner_synthetic"))
    $EntryPoint = [System.IO.Path]::GetFullPath($SyntheticEntryPoint)
    $TempPrefix = $TempRoot.TrimEnd('\') + '\'
    if (-not $EntryPoint.StartsWith($TempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "synthetic entrypoint must be below $TempRoot"
    }
    $RepoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    if ($EntryPoint.StartsWith($RepoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "synthetic entrypoint cannot be repository source"
    }
    $SyntheticText = Get-Content -LiteralPath $EntryPoint -Raw -Encoding UTF8
    if (-not $SyntheticText.StartsWith("# N1_SYNTHETIC_RUNNER_TEST_ONLY`n")) {
        throw "synthetic entrypoint lacks the exact test-only sentinel"
    }
    $OutputRoot = Join-Path $TempRoot "runs"
    $Bindings = [ordered]@{
        synthetic_entrypoint_sha256 = Get-LowerSha256 $EntryPoint
        marker_access = "structurally disabled"
    }
} else {
    if (-not [string]::IsNullOrEmpty($SyntheticEntryPoint)) {
        throw "production launch refuses -SyntheticEntryPoint"
    }
    [void](Require-ExactFile "C13 checkpoint" $C13Checkpoint $ExpectedC13Sha256 $null)
    [void](Require-ExactFile "old Lux checkpoint" $OldCheckpoint $ExpectedOldCheckpointSha256 $null)
    [void](Require-ExactFile "old OpenVINO weights" $OldWeights $ExpectedOldWeightsSha256 $null)
    $Bindings = Require-ReadinessAndCleanSource
}

[void][System.IO.Directory]::CreateDirectory($OutputRoot)
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$Nonce = [Guid]::NewGuid().ToString("N").Substring(0, 12)
$RunName = if ($Mode -ceq "production") {
    "n1_calibration_${Timestamp}_${Nonce}"
} else {
    "n1_runner_synthetic_${Timestamp}_${Nonce}"
}
$OutputDirectory = Join-Path $OutputRoot $RunName
[void][System.IO.Directory]::CreateDirectory($OutputDirectory)

$Stdout = Join-Path $OutputDirectory "stdout.log"
$Stderr = Join-Path $OutputDirectory "stderr.log"
$Telemetry = Join-Path $OutputDirectory "resource_samples.jsonl"
$Monitor = Join-Path $OutputDirectory "monitor_result.json"
$LaunchManifest = Join-Path $OutputDirectory "launch_manifest.json"
$FinalResult = Join-Path $OutputDirectory $ProductionResultName
$Arguments = @(
    "--startup-file=no", "--history-file=no", "--threads=1",
    "--project=$RepoRoot", $EntryPoint, $OutputDirectory
)
$Launch = [ordered]@{
    schema = "n1-single-calibration-launch-v1"
    mode = $Mode
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    output_directory = $OutputDirectory
    executable = $Julia
    argv = $Arguments
    working_directory = $RepoRoot
    required_result_name = $ProductionResultName
    sample_interval_milliseconds = $SampleIntervalMilliseconds
    private_limit_bytes = $PrivateLimit
    working_set_limit_bytes = $WorkingSetLimit
    timeout_seconds = $TimeoutSeconds
    bindings = $Bindings
}
[System.IO.File]::WriteAllText(
    $LaunchManifest, ($Launch | ConvertTo-Json -Depth 10), $Utf8NoBom
)
$LaunchManifestSha256 = Get-LowerSha256 $LaunchManifest

$MarkerSha256AtLaunch = $null
if ($Mode -ceq "production") {
    $Marker = [ordered]@{
        schema = "n1-calibration-started-marker-v1"
        status = "N1-calibration-started"
        created_utc = (Get-Date).ToUniversalTime().ToString("o")
        output_directory = $OutputDirectory
        launch_manifest_sha256 = $LaunchManifestSha256
        source_commit = $Bindings.source_commit
        n1_source_tree_sha256 = $Bindings.n1_source_tree_sha256
        entrypoint_sha256 = $Bindings.entrypoint_sha256
        readiness_sha256 = $Bindings.readiness_sha256
        clean_audit_status = $Bindings.clean_audit_status
        audit_report_sha256 = $Bindings.audit_report_sha256
        manifest_sha256 = $Bindings.manifest_sha256
        julia_sha256 = $Bindings.julia_sha256
        c13_checkpoint_sha256 = $Bindings.c13_checkpoint_sha256
        old_checkpoint_sha256 = $Bindings.old_checkpoint_sha256
        old_weights_sha256 = $Bindings.old_weights_sha256
    }
    Write-AtomicExclusiveMarker $Marker
    $MarkerSha256AtLaunch = Get-LowerSha256 $MarkerPath
}

$StartInfo = New-Object System.Diagnostics.ProcessStartInfo
$StartInfo.FileName = $Julia
$StartInfo.Arguments = (($Arguments | ForEach-Object { Quote-NativeArgument $_ }) -join " ")
$StartInfo.WorkingDirectory = $RepoRoot
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
$StartInfo.EnvironmentVariables["JULIA_NUM_THREADS"] = "1"
$StartInfo.EnvironmentVariables["N1_RUN_MODE"] = $Mode
$StartInfo.EnvironmentVariables["N1_OUTPUT_DIRECTORY"] = $OutputDirectory
$StartInfo.EnvironmentVariables["N1_LAUNCH_MANIFEST_SHA256"] = $LaunchManifestSha256
if ($Mode -ceq "production") {
    $StartInfo.EnvironmentVariables["N1_SOURCE_COMMIT"] = $Bindings.source_commit
    $StartInfo.EnvironmentVariables["N1_SOURCE_TREE_SHA256"] = $Bindings.n1_source_tree_sha256
    $StartInfo.EnvironmentVariables["N1_READINESS_SHA256"] = $Bindings.readiness_sha256
    $StartInfo.EnvironmentVariables["N1_CLEAN_AUDIT_SHA256"] = $Bindings.audit_report_sha256
    $StartInfo.EnvironmentVariables["N1_MARKER_SHA256"] = $MarkerSha256AtLaunch
}

$Child = New-Object System.Diagnostics.Process
$Child.StartInfo = $StartInfo
$StdoutTask = $null
$StderrTask = $null
$TelemetryWriter = $null
$Started = Get-Date
$StartedProcess = $false
$ChildExitCode = $null
$RunnerFailure = $null
$LimitReason = $null
$PeakPrivate = [int64]0
$PeakWorkingSet = [int64]0
$SampleCount = 0

try {
    $TelemetryWriter = [System.IO.StreamWriter]::new($Telemetry, $false, $Utf8NoBom)
    $TelemetryWriter.AutoFlush = $true
    [void]$Child.Start()
    $StartedProcess = $true
    $StdoutTask = $Child.StandardOutput.ReadToEndAsync()
    $StderrTask = $Child.StandardError.ReadToEndAsync()

    $ObservedImage = (Get-Process -Id $Child.Id -ErrorAction Stop).Path
    if (-not [string]::Equals(
        [IO.Path]::GetFullPath($ObservedImage), [IO.Path]::GetFullPath($Julia),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        $LimitReason = "root process image is not the frozen Julia executable"
    }

    while (-not $Child.HasExited -and $null -eq $LimitReason) {
        $Now = Get-Date
        $Elapsed = ($Now - $Started).TotalSeconds
        $Child.Refresh()
        $TreeIds = @(Get-TreeProcessIds $Child.Id)
        $Private = [int64]0
        $WorkingSet = [int64]0
        $ObservedProcesses = @()
        foreach ($TreeId in $TreeIds) {
            try {
                $ObservedProcess = Get-Process -Id $TreeId -ErrorAction Stop
                $ObservedPrivate = [int64]$ObservedProcess.PrivateMemorySize64
                $ObservedWorkingSet = [int64]$ObservedProcess.WorkingSet64
                $Private += $ObservedPrivate
                $WorkingSet += $ObservedWorkingSet
                $ObservedProcesses += [ordered]@{
                    pid = [int]$TreeId
                    private_bytes = $ObservedPrivate
                    working_set_bytes = $ObservedWorkingSet
                }
            } catch {
                # A short-lived descendant can exit between snapshots.
            }
        }
        $PeakPrivate = [Math]::Max($PeakPrivate, $Private)
        $PeakWorkingSet = [Math]::Max($PeakWorkingSet, $WorkingSet)
        $SampleCount += 1
        $Sample = [ordered]@{
            utc = $Now.ToUniversalTime().ToString("o")
            elapsed_seconds = $Elapsed
            pid = [int]$Child.Id
            private_bytes = $Private
            working_set_bytes = $WorkingSet
            process_count = $ObservedProcesses.Count
            processes = $ObservedProcesses
        }
        $TelemetryWriter.WriteLine(($Sample | ConvertTo-Json -Compress -Depth 4))

        if ($Private -gt $PrivateLimit) {
            $LimitReason = "private committed memory exceeded 4 GiB"
        } elseif ($WorkingSet -gt $WorkingSetLimit) {
            $LimitReason = "working set exceeded 2 GiB"
        } elseif ($Elapsed -gt $TimeoutSeconds) {
            $LimitReason = "wall time exceeded 4500 seconds"
        }
        if ($null -eq $LimitReason) {
            Start-Sleep -Milliseconds $SampleIntervalMilliseconds
            $Child.Refresh()
        }
    }
    if ($null -ne $LimitReason) {
        Stop-N1Process $Child
    }
    $Child.WaitForExit()
    $ChildExitCode = [int]$Child.ExitCode
} catch {
    $RunnerFailure = $_.Exception.ToString()
    Stop-N1Process $Child
    if ($StartedProcess -and $Child.HasExited) {
        $ChildExitCode = [int]$Child.ExitCode
    }
} finally {
    if ($null -ne $TelemetryWriter) { $TelemetryWriter.Dispose() }
    if ($StartedProcess -and -not $Child.HasExited) { Stop-N1Process $Child }
    $StdoutText = ""
    $StderrText = ""
    if ($null -ne $StdoutTask) {
        try { $StdoutText = [string]$StdoutTask.Result } catch { $RunnerFailure = $_.Exception.ToString() }
    }
    if ($null -ne $StderrTask) {
        try { $StderrText = [string]$StderrTask.Result } catch { $RunnerFailure = $_.Exception.ToString() }
    }
    [System.IO.File]::WriteAllText($Stdout, $StdoutText, $Utf8NoBom)
    [System.IO.File]::WriteAllText($Stderr, $StderrText, $Utf8NoBom)
}

# The marker binds the pre-launch manifest; neither may change while the child
# runs.  Convert such tampering into a retained fail-closed monitor result.
try {
    if ((Get-LowerSha256 $LaunchManifest) -cne $LaunchManifestSha256) {
        throw "launch_manifest.json changed after its bound pre-launch hash"
    }
    if ($Mode -ceq "production" -and
        (Get-LowerSha256 $MarkerPath) -cne $MarkerSha256AtLaunch) {
        throw "the exclusive production marker changed during execution"
    }
} catch {
    if ($null -eq $RunnerFailure) { $RunnerFailure = $_.Exception.ToString() }
}

$ResultPresent = [System.IO.File]::Exists($FinalResult)
$ResultValidJson = $false
if ($ResultPresent) {
    try {
        $ParsedResult = Get-Content -LiteralPath $FinalResult -Raw -Encoding UTF8 |
            ConvertFrom-Json -ErrorAction Stop
        $ResultValidJson = $null -ne $ParsedResult
    } catch {
        $ResultValidJson = $false
        if ($null -eq $RunnerFailure) {
            $RunnerFailure = "calibration_result.json is not valid JSON: $($_.Exception.Message)"
        }
    }
}
$Success = (
    $StartedProcess -and $null -eq $RunnerFailure -and $null -eq $LimitReason -and
    $null -ne $ChildExitCode -and [int]$ChildExitCode -eq 0 -and
    $ResultPresent -and $ResultValidJson
)
$WallSeconds = ((Get-Date) - $Started).TotalSeconds
$ArtifactHashes = [ordered]@{}
foreach ($Artifact in @($LaunchManifest, $Stdout, $Stderr, $Telemetry, $FinalResult)) {
    if ([System.IO.File]::Exists($Artifact)) {
        $ArtifactHashes[[IO.Path]::GetFileName($Artifact)] = Get-LowerSha256 $Artifact
    }
}
$MonitorPayload = [ordered]@{
    schema = "n1-single-calibration-monitor-v1"
    status = if ($Success) { "N1-calibration-runner-pass" } else { "N1-calibration-runner-fail" }
    mode = $Mode
    output_directory = $OutputDirectory
    root_pid = if ($StartedProcess) { [int]$Child.Id } else { $null }
    child_exit_code = if ($null -ne $ChildExitCode) { [int]$ChildExitCode } else { $null }
    child_exit_code_dotnet_type = if ($null -ne $ChildExitCode) { $ChildExitCode.GetType().FullName } else { $null }
    wall_seconds = $WallSeconds
    sample_interval_milliseconds = $SampleIntervalMilliseconds
    sample_count = $SampleCount
    peak_process_tree_private_bytes = $PeakPrivate
    peak_process_tree_working_set_bytes = $PeakWorkingSet
    private_limit_bytes = $PrivateLimit
    working_set_limit_bytes = $WorkingSetLimit
    timeout_seconds = $TimeoutSeconds
    limit_reason = $LimitReason
    runner_failure = $RunnerFailure
    required_result_name = $ProductionResultName
    result_present = $ResultPresent
    result_valid_json = $ResultValidJson
    launch_manifest_sha256 = $LaunchManifestSha256
    artifact_sha256 = $ArtifactHashes
}
[System.IO.File]::WriteAllText(
    $Monitor, ($MonitorPayload | ConvertTo-Json -Depth 10), $Utf8NoBom
)

Write-Output $OutputDirectory
if (-not $Success) { exit 1 }
exit 0
