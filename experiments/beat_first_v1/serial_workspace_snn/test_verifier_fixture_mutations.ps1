[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,

    [Parameter(Mandatory = $true)]
    [string]$LaunchManifestPath,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2)]
    [int]$ExpectedUpdates,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^contract_fixture_[A-Za-z0-9_.-]+$")]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [ValidateSet("scratch", "resume", "finalize-only")]
    [string]$StartMode,

    [string]$ParentCheckpoint = "",
    [string]$ParentSha256 = "",
    [int]$ParentUpdate = -1,
    [string]$JuliaExecutable = "",
    [string]$ProjectPath = "",
    [switch]$AllowSacrificialMutation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Contract {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw "CONTRACT ASSERTION FAILED: $Message"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            return (
                [BitConverter]::ToString($sha.ComputeHash($stream))
            ).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )
    [IO.File]::WriteAllText(
        $Path,
        $Value,
        [Text.UTF8Encoding]::new($false)
    )
}

function Test-IsUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $resolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar
    )
    $resolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar
    )
    return (
        $resolvedPath.Equals(
            $resolvedRoot,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        $resolvedPath.StartsWith(
            $resolvedRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    )
}

function Assert-NoReparseComponent {
    param([Parameter(Mandatory = $true)][string]$Path)
    $current = Get-Item -LiteralPath $Path -Force
    while ($null -ne $current) {
        Assert-Contract `
            -Condition (
                ($current.Attributes -band
                    [IO.FileAttributes]::ReparsePoint) -eq 0
            ) `
            -Message "sacrificial path contains a reparse point: $($current.FullName)"
        $parent = Split-Path -Parent $current.FullName
        if (
            [string]::IsNullOrWhiteSpace($parent) -or
            $parent -eq $current.FullName
        ) {
            break
        }
        $current = Get-Item -LiteralPath $parent -Force
    }
}

function Assert-SingleHardLink {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fsutil = Join-Path $env:SystemRoot "System32\fsutil.exe"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $fsutil -PathType Leaf) `
        -Message "fsutil.exe is required for fail-closed hardlink checks"
    $links = @(& $fsutil hardlink list $Path)
    Assert-Contract `
        -Condition ($LASTEXITCODE -eq 0) `
        -Message "could not query hardlinks for sacrificial file: $Path"
    $linkPaths = @(
        $links | Where-Object {
            -not [string]::IsNullOrWhiteSpace([string]$_)
        }
    )
    Assert-Contract `
        -Condition ($linkPaths.Count -eq 1) `
        -Message (
            "sacrificial mutation rejects multiply-linked file: " +
            "$Path links=$($linkPaths.Count)"
        )
}

function Invoke-VerifierCase {
    param(
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][bool]$ExpectSuccess,
        [string[]]$RuntimeArguments = @(
            "--startup-file=no",
            "--history-file=no"
        ),
        [int]$DefaultThreads = 1,
        [int]$InteractiveThreads = 0,
        [string]$EffectiveProjectPath = "",
        [string]$DependencyFallbackProject = "",
        [string]$ExpectedErrorPattern = "",
        [switch]$RequireFailureArtifact
    )

    if ([string]::IsNullOrWhiteSpace($EffectiveProjectPath)) {
        $EffectiveProjectPath = $script:ResolvedProject
    }
    $manifestSha256 = Get-Sha256 -Path $ManifestPath
    $outputPath = Join-Path $script:CaseRoot (
        $CaseName + ".verification.json"
    )
    $arguments = @(
        $RuntimeArguments
        "--project=$EffectiveProjectPath"
        "--threads=$DefaultThreads,$InteractiveThreads"
        $script:VerifierPath
        "--run-dir=$script:ResolvedRunDirectory"
        "--expected-updates=$ExpectedUpdates"
        "--expected-run-id=$RunId"
        "--expected-start-mode=$StartMode"
        "--launch-manifest=$ManifestPath"
        "--launch-manifest-sha256=$manifestSha256"
        "--verification-output=$outputPath"
    )
    if ($StartMode -ne "scratch") {
        $arguments += @(
            "--parent-checkpoint=$script:ResolvedParentCheckpoint"
            "--parent-sha256=$ParentSha256"
            "--parent-update=$ParentUpdate"
        )
    }
    $savedLoadPath = $env:JULIA_LOAD_PATH
    try {
        if (
            -not [string]::IsNullOrWhiteSpace(
                $DependencyFallbackProject
            )
        ) {
            $env:JULIA_LOAD_PATH = (
                "@;" + $DependencyFallbackProject + ";@stdlib"
            )
        }
        & $script:ResolvedJulia @arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        if ($null -eq $savedLoadPath) {
            Remove-Item Env:JULIA_LOAD_PATH -ErrorAction SilentlyContinue
        }
        else {
            $env:JULIA_LOAD_PATH = $savedLoadPath
        }
    }
    if ($ExpectSuccess) {
        Assert-Contract `
            -Condition ($exitCode -eq 0) `
            -Message "$CaseName verifier exit must be 0, got $exitCode"
        Assert-Contract `
            -Condition (
                Test-Path -LiteralPath $outputPath -PathType Leaf
            ) `
            -Message "$CaseName did not write verification output"
        $report = Get-Content -Raw -LiteralPath $outputPath |
            ConvertFrom-Json
        Assert-Contract `
            -Condition (
                $report.verified -eq $true -and
                [string]$report.status -eq "verified_complete" -and
                $report.metrics_verified -eq $true
            ) `
            -Message "$CaseName did not report verified_complete"
        $checkpointReports = @($report.checkpoints)
        Assert-Contract `
            -Condition ($checkpointReports.Count -ge 1) `
            -Message "$CaseName verification checkpoints are empty"
        foreach ($checkpointReport in $checkpointReports) {
            Assert-Contract `
                -Condition (
                    $null -ne $checkpointReport.PSObject.Properties[
                        "checkpoint_kind"
                    ]
                ) `
                -Message (
                    "$CaseName checkpoint is missing authoritative " +
                    "checkpoint_kind"
                )
            Assert-Contract `
                -Condition (
                    $null -eq $checkpointReport.PSObject.Properties["kind"]
                ) `
                -Message (
                    "$CaseName checkpoint contains the forbidden kind alias"
                )
            Assert-Contract `
                -Condition (
                    [string]$checkpointReport.checkpoint_kind -eq
                        "training"
                ) `
                -Message (
                    "$CaseName training checkpoint kind differs: " +
                    [string]$checkpointReport.checkpoint_kind
                )
        }
        if ($StartMode -eq "scratch") {
            $updateZeroTraining = @(
                $checkpointReports | Where-Object {
                    [int64]$_.update -eq 0 -and
                    [string]$_.checkpoint_kind -eq "training"
                }
            )
            Assert-Contract `
                -Condition ($updateZeroTraining.Count -eq 1) `
                -Message (
                    "$CaseName must expose exactly one U0 training checkpoint"
                )
        }
    }
    else {
        Assert-Contract `
            -Condition ($exitCode -ne 0) `
            -Message "$CaseName unexpectedly passed verification"
        if ($RequireFailureArtifact) {
            Assert-Contract `
                -Condition (
                    Test-Path -LiteralPath $outputPath -PathType Leaf
                ) `
                -Message "$CaseName did not write a failure artifact"
        }
        if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
            $failure = Get-Content -Raw -LiteralPath $outputPath |
                ConvertFrom-Json
            Assert-Contract `
                -Condition (
                    $failure.verified -ne $true -and
                    [string]$failure.status -eq "verification_failed"
                ) `
                -Message "$CaseName failure artifact is not fail-closed"
            if (-not [string]::IsNullOrWhiteSpace($ExpectedErrorPattern)) {
                Assert-Contract `
                    -Condition (
                        [string]$failure.error -match
                            $ExpectedErrorPattern
                    ) `
                    -Message (
                        "$CaseName failed for an unrelated invariant: " +
                        [string]$failure.error
                    )
            }
        }
    }
    Write-Host (
        "PASS verifier case $CaseName " +
        "(expected_success=$ExpectSuccess exit=$exitCode)"
    )
}

function Invoke-LaunchMutation {
    param(
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][scriptblock]$Mutation,
        [Parameter(Mandatory = $true)][string]$ExpectedErrorPattern
    )
    $launch = Get-Content -Raw -LiteralPath $script:ResolvedLaunchManifest |
        ConvertFrom-Json
    & $Mutation $launch
    $caseManifest = Join-Path $script:CaseRoot (
        $CaseName + ".launch_manifest.json"
    )
    Write-Utf8NoBom `
        -Path $caseManifest `
        -Value ($launch | ConvertTo-Json -Depth 100)
    Invoke-VerifierCase `
        -CaseName $CaseName `
        -ManifestPath $caseManifest `
        -ExpectSuccess $false `
        -RequireFailureArtifact `
        -ExpectedErrorPattern $ExpectedErrorPattern
}

function Invoke-RestoredMutation {
    param(
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][scriptblock]$Mutation,
        [Parameter(Mandatory = $true)][string]$ExpectedErrorPattern
    )
    Assert-NoReparseComponent -Path $Path
    Assert-SingleHardLink -Path $Path
    $beforeSha256 = Get-Sha256 -Path $Path
    $backupPath = Join-Path $script:CaseRoot (
        $CaseName + "." + [IO.Path]::GetFileName($Path) + ".backup"
    )
    Copy-Item -LiteralPath $Path -Destination $backupPath
    try {
        & $Mutation $Path
        Invoke-VerifierCase `
            -CaseName $CaseName `
            -ManifestPath $script:ResolvedLaunchManifest `
            -ExpectSuccess $false `
            -RequireFailureArtifact `
            -ExpectedErrorPattern $ExpectedErrorPattern
    }
    finally {
        Copy-Item -LiteralPath $backupPath -Destination $Path -Force
        $afterSha256 = Get-Sha256 -Path $Path
        Assert-Contract `
            -Condition ($afterSha256 -eq $beforeSha256) `
            -Message "$CaseName did not byte-exactly restore $Path"
    }
}

Assert-Contract `
    -Condition $AllowSacrificialMutation `
    -Message (
        "pass -AllowSacrificialMutation; this suite temporarily mutates " +
        "and byte-exactly restores a disposable U2 fixture"
    )

$script:ResolvedRunDirectory =
    (Resolve-Path -LiteralPath $RunDirectory).Path
$script:ResolvedLaunchManifest =
    (Resolve-Path -LiteralPath $LaunchManifestPath).Path
Assert-NoReparseComponent -Path $script:ResolvedRunDirectory
Assert-NoReparseComponent -Path $script:ResolvedLaunchManifest
Assert-Contract `
    -Condition (
        [IO.Path]::GetFileName($script:ResolvedRunDirectory) -eq $RunId
    ) `
    -Message "RunDirectory basename must equal RunId"

$allowedRoots = @(
    [IO.Path]::GetTempPath(),
    "C:\tmp"
)
Assert-Contract `
    -Condition (
        @(
            $allowedRoots | Where-Object {
                Test-IsUnderRoot `
                    -Path $script:ResolvedRunDirectory `
                    -Root $_
            }
        ).Count -gt 0
    ) `
    -Message "sacrificial fixture must be under the OS temp root or C:\tmp"

if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
    $ProjectPath = Join-Path $PSScriptRoot ".."
}
$script:ResolvedProject = (Resolve-Path -LiteralPath $ProjectPath).Path
$script:VerifierPath = (
    Resolve-Path -LiteralPath (
        Join-Path $PSScriptRoot "verify_arena_run.jl"
    )
).Path
if ([string]::IsNullOrWhiteSpace($JuliaExecutable)) {
    if (
        -not [string]::IsNullOrWhiteSpace(
            $env:SWSNN_JULIA_EXECUTABLE
        )
    ) {
        $JuliaExecutable = $env:SWSNN_JULIA_EXECUTABLE
    }
    else {
        $JuliaExecutable = "julia"
    }
}
$script:ResolvedJulia = if (
    [IO.Path]::IsPathRooted($JuliaExecutable)
) {
    (Resolve-Path -LiteralPath $JuliaExecutable).Path
}
else {
    (Get-Command -Name $JuliaExecutable -CommandType Application).Source
}

if ($StartMode -eq "scratch") {
    Assert-Contract `
        -Condition (
            [string]::IsNullOrWhiteSpace($ParentCheckpoint) -and
            [string]::IsNullOrWhiteSpace($ParentSha256) -and
            $ParentUpdate -eq -1
        ) `
        -Message "scratch fixture cannot accept parent arguments"
    $script:ResolvedParentCheckpoint = $null
}
else {
    Assert-Contract `
        -Condition (
            -not [string]::IsNullOrWhiteSpace($ParentCheckpoint) -and
            $ParentSha256 -match "^[0-9A-Fa-f]{64}$" -and
            $ParentUpdate -ge 0
        ) `
        -Message "non-scratch fixture requires the complete parent triple"
    $script:ResolvedParentCheckpoint =
        (Resolve-Path -LiteralPath $ParentCheckpoint).Path
    Assert-NoReparseComponent -Path $script:ResolvedParentCheckpoint
    Assert-Contract `
        -Condition (
            @(
                $allowedRoots | Where-Object {
                    Test-IsUnderRoot `
                        -Path $script:ResolvedParentCheckpoint `
                        -Root $_
                }
            ).Count -gt 0
        ) `
        -Message "parent checkpoint must also be a disposable temp fixture"
    Assert-Contract `
        -Condition (
            (Get-Sha256 -Path $script:ResolvedParentCheckpoint) -eq
                $ParentSha256.ToLowerInvariant()
        ) `
        -Message "parent checkpoint SHA-256 argument differs before testing"
}

$launch = Get-Content -Raw -LiteralPath $script:ResolvedLaunchManifest |
    ConvertFrom-Json
Assert-Contract `
    -Condition ([string]$launch.run_id -eq $RunId) `
    -Message "launch manifest run ID differs"
Assert-Contract `
    -Condition ([int]$launch.expected_updates -eq $ExpectedUpdates) `
    -Message "launch manifest expected update count differs"
Assert-Contract `
    -Condition ([string]$launch.start_mode -eq $StartMode) `
    -Message "launch manifest start mode differs"
$markerPath = Join-Path (
    $script:ResolvedRunDirectory
) ".swsnn_disposable_fixture.json"
Assert-Contract `
    -Condition (Test-Path -LiteralPath $markerPath -PathType Leaf) `
    -Message (
        "disposable marker is missing; follow the documented marker " +
        "creation step before allowing in-place mutations"
    )
Assert-NoReparseComponent -Path $markerPath
$marker = Get-Content -Raw -LiteralPath $markerPath | ConvertFrom-Json
Assert-Contract `
    -Condition (
        [string]$marker.format -eq
            "serial-workspace-snn-disposable-contract-fixture" -and
        [int]$marker.version -eq 1 -and
        [string]$marker.run_id -eq $RunId -and
        [int]$marker.expected_updates -eq $ExpectedUpdates -and
        $marker.disposable -eq $true -and
        [string]$marker.launch_manifest_sha256 -eq
            (Get-Sha256 -Path $script:ResolvedLaunchManifest)
    ) `
    -Message "disposable marker does not bind this exact U2 launch"

$script:CaseRoot = Join-Path (
    [IO.Path]::GetTempPath()
) (
    "swsnn_verifier_mutations_" +
    (Get-Date -Format "yyyyMMdd_HHmmss") +
    "_" +
    [guid]::NewGuid().ToString("N")
)
[IO.Directory]::CreateDirectory($script:CaseRoot) | Out-Null

Invoke-VerifierCase `
    -CaseName "baseline_exact" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $true

$roundTripLaunch = Get-Content `
    -Raw `
    -LiteralPath $script:ResolvedLaunchManifest |
    ConvertFrom-Json
$roundTripLaunchPath = Join-Path (
    $script:CaseRoot
) "launch_roundtrip_control.launch_manifest.json"
Write-Utf8NoBom `
    -Path $roundTripLaunchPath `
    -Value ($roundTripLaunch | ConvertTo-Json -Depth 100)
Invoke-VerifierCase `
    -CaseName "launch_foreign_same_identity_copy" `
    -ManifestPath $roundTripLaunchPath `
    -ExpectSuccess $false `
    -RequireFailureArtifact `
    -ExpectedErrorPattern "run launch binding path differs"

Invoke-VerifierCase `
    -CaseName "runtime_missing_startup_flag" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $false `
    -RuntimeArguments @("--history-file=no")
Invoke-VerifierCase `
    -CaseName "runtime_missing_history_flag" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $false `
    -RuntimeArguments @("--startup-file=no")
Invoke-VerifierCase `
    -CaseName "runtime_wrong_default_threads" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $false `
    -DefaultThreads 2
Invoke-VerifierCase `
    -CaseName "runtime_wrong_interactive_threads" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $false `
    -InteractiveThreads 1

$wrongProject = Join-Path $script:CaseRoot "wrong_project"
[IO.Directory]::CreateDirectory($wrongProject) | Out-Null
Write-Utf8NoBom `
    -Path (Join-Path $wrongProject "Project.toml") `
    -Value (
        "name = `"SwsnnWrongVerifierProject`"" +
        [Environment]::NewLine +
        "uuid = `"8d326647-d38c-43eb-9cc4-f11d5ec22962`"" +
        [Environment]::NewLine +
        "version = `"0.1.0`"" +
        [Environment]::NewLine
    )
Invoke-VerifierCase `
    -CaseName "runtime_noncanonical_active_project" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $false `
    -EffectiveProjectPath $wrongProject `
    -DependencyFallbackProject $script:ResolvedProject

Invoke-LaunchMutation `
    -CaseName "launch_runtime_argv_reordered" `
    -ExpectedErrorPattern "launch manifest Julia runtime arguments differs" `
    -Mutation {
        param($document)
        $document.julia_runtime_arguments = @(
            "--history-file=no",
            "--startup-file=no"
        )
    }
Invoke-LaunchMutation `
    -CaseName "launch_project_noncanonical" `
    -ExpectedErrorPattern "launch manifest canonical project path differs" `
    -Mutation {
        param($document)
        $document.project_path = $script:CaseRoot
    }
Invoke-LaunchMutation `
    -CaseName "launch_training_threads_tampered" `
    -ExpectedErrorPattern "launch Julia thread count differs" `
    -Mutation {
        param($document)
        $document.julia_threads = [int]$document.julia_threads + 1
    }
Invoke-LaunchMutation `
    -CaseName "launch_startup_contract_tampered" `
    -ExpectedErrorPattern "launch expected startup-file flag differs" `
    -Mutation {
        param($document)
        $document.expected_contract.startup_file = $true
    }

Invoke-RestoredMutation `
    -CaseName "launch_binding_live_sha_tamper" `
    -Path $script:ResolvedLaunchManifest `
    -ExpectedErrorPattern "run launch binding live SHA-256 differs" `
    -Mutation {
        param($path)
        $document = Get-Content -Raw -LiteralPath $path |
            ConvertFrom-Json
        $document.created_at = "foreign-same-contract-copy"
        Write-Utf8NoBom `
            -Path $path `
            -Value ($document | ConvertTo-Json -Depth 100)
    }

$checkpointManifestPath = if ($StartMode -eq "finalize-only") {
    Join-Path (
        Split-Path -Parent (
            Split-Path -Parent $script:ResolvedParentCheckpoint
        )
    ) "checkpoint_manifest.jsonl"
}
else {
    Join-Path $script:ResolvedRunDirectory "checkpoint_manifest.jsonl"
}
Assert-Contract `
    -Condition (
        Test-Path -LiteralPath $checkpointManifestPath -PathType Leaf
    ) `
    -Message "fixture checkpoint manifest is missing"
Invoke-RestoredMutation `
    -CaseName "checkpoint_manifest_blank_line" `
    -Path $checkpointManifestPath `
    -ExpectedErrorPattern "checkpoint manifest contains a blank line" `
    -Mutation {
        param($path)
        [IO.File]::AppendAllText(
            $path,
            [Environment]::NewLine + [Environment]::NewLine,
            [Text.UTF8Encoding]::new($false)
        )
    }

$extraCheckpointPath = Join-Path (
    Join-Path $script:ResolvedRunDirectory "checkpoints"
) "checkpoint_latest.jld2"
Assert-Contract `
    -Condition (-not (Test-Path -LiteralPath $extraCheckpointPath)) `
    -Message "fixture already contains checkpoint_latest.jld2"
try {
    Write-Utf8NoBom `
        -Path $extraCheckpointPath `
        -Value "contract-test-only unexpected alias"
    Invoke-VerifierCase `
        -CaseName "checkpoint_extra_alias" `
        -ManifestPath $script:ResolvedLaunchManifest `
        -ExpectSuccess $false `
        -RequireFailureArtifact `
        -ExpectedErrorPattern "unexpected regular file"
}
finally {
    if (Test-Path -LiteralPath $extraCheckpointPath -PathType Leaf) {
        Remove-Item -LiteralPath $extraCheckpointPath -Force
    }
}

$resultsPath = Join-Path $script:ResolvedRunDirectory "results.json"
Assert-Contract `
    -Condition (Test-Path -LiteralPath $resultsPath -PathType Leaf) `
    -Message "fixture results.json is missing"
Invoke-RestoredMutation `
    -CaseName "results_material_metric_tamper" `
    -Path $resultsPath `
    -ExpectedErrorPattern "finalization results artifact (byte size|SHA-256) differs" `
    -Mutation {
        param($path)
        $results = Get-Content -Raw -LiteralPath $path |
            ConvertFrom-Json
        $results.final.composite_loss =
            [double]$results.final.composite_loss + 0.0001
        Write-Utf8NoBom `
            -Path $path `
            -Value ($results | ConvertTo-Json -Depth 100)
    }

$finalizationManifestPath = Join-Path (
    $script:ResolvedRunDirectory
) "finalization_manifest.json"
Assert-Contract `
    -Condition (
        Test-Path -LiteralPath $finalizationManifestPath -PathType Leaf
    ) `
    -Message "fixture finalization_manifest.json is missing"
Invoke-RestoredMutation `
    -CaseName "finalization_training_reference_tamper" `
    -Path $finalizationManifestPath `
    -ExpectedErrorPattern "finalization training checkpoint reference differs" `
    -Mutation {
        param($path)
        $manifest = Get-Content -Raw -LiteralPath $path |
            ConvertFrom-Json
        $manifest.training_checkpoint.sha256 = "0" * 64
        Write-Utf8NoBom `
            -Path $path `
            -Value ($manifest | ConvertTo-Json -Depth 100)
    }

if ($StartMode -eq "finalize-only") {
    Invoke-RestoredMutation `
        -CaseName "finalize_parent_manifest_live_path_tamper" `
        -Path $checkpointManifestPath `
        -ExpectedErrorPattern "(manifest|checkpoint).*path differs" `
        -Mutation {
            param($path)
            $output = New-Object Collections.Generic.List[string]
            foreach ($line in [IO.File]::ReadAllLines($path)) {
                $record = $line | ConvertFrom-Json
                if ([int]$record.update -eq $ParentUpdate) {
                    $record.path = [string]$record.path + ".shadow"
                }
                $output.Add(($record | ConvertTo-Json -Compress -Depth 20))
            }
            Write-Utf8NoBom `
                -Path $path `
                -Value (($output -join [Environment]::NewLine) +
                    [Environment]::NewLine)
        }

    Invoke-RestoredMutation `
        -CaseName "finalize_parent_live_checkpoint_tamper" `
        -Path $script:ResolvedParentCheckpoint `
        -ExpectedErrorPattern "parent checkpoint SHA-256 differs" `
        -Mutation {
            param($path)
            $stream = [IO.File]::Open(
                $path,
                [IO.FileMode]::Append,
                [IO.FileAccess]::Write,
                [IO.FileShare]::None
            )
            try {
                $stream.WriteByte(0x7f)
                $stream.Flush()
            }
            finally {
                $stream.Dispose()
            }
        }
}

Invoke-VerifierCase `
    -CaseName "restored_baseline" `
    -ManifestPath $script:ResolvedLaunchManifest `
    -ExpectSuccess $true

Write-Host (
    "PASS sacrificial verifier mutation suite; all source artifacts " +
    "restored byte-exactly; case reports retained at $script:CaseRoot"
)
