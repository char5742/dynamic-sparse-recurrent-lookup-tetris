[CmdletBinding()]
param(
    [string]$JuliaExecutable =
        "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe",
    [string]$ProjectPath = "",
    [string]$DatasetPath =
        "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [ValidateRange(2, 256)]
    [int]$CoreJuliaThreads = 20,
    [string]$LogDirectory = "",
    [switch]$SkipBoundsLanes,
    [switch]$SkipDatasetTests,
    [switch]$SkipWatchdogProbes,
    [switch]$AllowExistingJuliaProcesses,
    [string]$FixtureRunDirectory = "",
    [string]$FixtureLaunchManifestPath = "",
    [switch]$AllowSacrificialFixtureMutation,
    [switch]$ListOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CanonicalPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [ValidateSet("Leaf", "Container")]
        [string]$PathType
    )

    if (-not (Test-Path -LiteralPath $Path -PathType $PathType)) {
        throw "Required $PathType path does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Test-SamePath {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right
    )

    return [string]::Equals(
        [IO.Path]::GetFullPath($Left).TrimEnd("\"),
        [IO.Path]::GetFullPath($Right).TrimEnd("\"),
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Format-CommandArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Format-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][object[]]$Arguments
    )

    $parts = @((Format-CommandArgument -Value $Executable))
    foreach ($argument in $Arguments) {
        $parts += Format-CommandArgument -Value ([string]$argument)
    }
    return $parts -join " "
}

function New-SuiteCase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Category,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][object[]]$Arguments,
        [hashtable]$Environment = @{}
    )

    return [pscustomobject][ordered]@{
        name = $Name
        category = $Category
        executable = $Executable
        arguments = @($Arguments)
        environment = $Environment
        command = Format-Command `
            -Executable $Executable `
            -Arguments @($Arguments)
    }
}

function New-JuliaCase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Category,
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $true)][int]$Threads,
        [ValidateSet("auto", "yes")]
        [string]$CheckBounds = "auto",
        [hashtable]$Environment = @{}
    )

    $arguments = @(
        "--startup-file=no",
        "--history-file=no",
        "--project=$script:CanonicalProject",
        "--threads=$Threads,0",
        "--check-bounds=$CheckBounds",
        (Join-Path $script:SuiteDirectory $ScriptName)
    )
    $juliaEnvironment = @{
        OPENBLAS_NUM_THREADS = "1"
        MKL_NUM_THREADS = "1"
    }
    foreach ($key in $Environment.Keys) {
        $juliaEnvironment[[string]$key] = [string]$Environment[$key]
    }
    return New-SuiteCase `
        -Name $Name `
        -Category $Category `
        -Executable $script:CanonicalJulia `
        -Arguments $arguments `
        -Environment $juliaEnvironment
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )

    [IO.File]::WriteAllText(
        $Path,
        $Text,
        [Text.UTF8Encoding]::new($false)
    )
}

function Append-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )

    [IO.File]::AppendAllText(
        $Path,
        $Text,
        [Text.UTF8Encoding]::new($false)
    )
}

function Write-SuiteSummary {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [AllowNull()][string]$Failure
    )

    $value = [ordered]@{
        format = "serial-workspace-snn-full-test-suite"
        version = 1
        status = $Status
        started_at = $script:SuiteStartedAt
        updated_at = [DateTime]::UtcNow.ToString("o")
        project_path = $script:CanonicalProject
        julia_executable = $script:CanonicalJulia
        julia_runtime_arguments = @(
            "--startup-file=no",
            "--history-file=no",
            "--project=$script:CanonicalProject"
        )
        core_julia_threads = $CoreJuliaThreads
        dataset_path = if ($SkipDatasetTests) {
            $null
        }
        else {
            $script:CanonicalDataset
        }
        fail_fast = $true
        planned_cases = $script:Cases.Count
        plan = @(
            $script:Cases | ForEach-Object {
                [ordered]@{
                    name = $_.name
                    category = $_.category
                    executable = $_.executable
                    arguments = @($_.arguments)
                    command = $_.command
                    environment = $_.environment
                }
            }
        )
        completed_cases = $script:Results.Count
        skipped = @($script:Skipped)
        results = @($script:Results)
        failure = $Failure
    }
    Write-Utf8NoBom `
        -Path $script:SummaryPath `
        -Text (
            ($value | ConvertTo-Json -Depth 12) +
            [Environment]::NewLine
        )
}

function Invoke-SuiteCase {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Case,
        [Parameter(Mandatory = $true)][int]$Sequence
    )

    $safeName = $Case.name -replace '[^A-Za-z0-9_.-]', "_"
    $logName = "{0:D2}_{1}.log" -f $Sequence, $safeName
    $logPath = Join-Path $script:ResolvedLogDirectory $logName
    $startedAt = [DateTime]::UtcNow
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $exitCode = -1
    $exceptionText = $null
    $previousEnvironment = @{}
    $missingEnvironment = @()
    $previousLocation = Get-Location
    $previousErrorActionPreference = $ErrorActionPreference

    $header = @(
        "name=$($Case.name)"
        "category=$($Case.category)"
        "started_at=$($startedAt.ToString("o"))"
        "working_directory=$script:RepositoryRoot"
        "command=$($Case.command)"
        "environment=$((
            $Case.environment |
                ConvertTo-Json -Compress -Depth 4
        ))"
        ""
    ) -join [Environment]::NewLine
    Write-Utf8NoBom -Path $logPath -Text $header

    Write-Host ""
    Write-Host ("[{0}/{1}] START {2}" -f
        $Sequence, $script:Cases.Count, $Case.name)
    Write-Host $Case.command

    try {
        foreach ($keyObject in $Case.environment.Keys) {
            $key = [string]$keyObject
            $environmentPath = "Env:$key"
            if (Test-Path -LiteralPath $environmentPath) {
                $previousEnvironment[$key] =
                    [Environment]::GetEnvironmentVariable(
                        $key,
                        "Process"
                    )
            }
            else {
                $missingEnvironment += $key
            }
            [Environment]::SetEnvironmentVariable(
                $key,
                [string]$Case.environment[$keyObject],
                "Process"
            )
        }

        Set-Location -LiteralPath $script:RepositoryRoot
        $ErrorActionPreference = "Continue"
        $LASTEXITCODE = 0
        $logWriter = [IO.StreamWriter]::new(
            $logPath,
            $true,
            [Text.UTF8Encoding]::new($false)
        )
        try {
            $logWriter.AutoFlush = $true
            & $Case.executable @($Case.arguments) 2>&1 |
                ForEach-Object {
                    $line = [string]$_
                    $logWriter.WriteLine($line)
                    Write-Host $line
                }
            $exitCode = [int]$LASTEXITCODE
        }
        finally {
            $logWriter.Dispose()
        }
    }
    catch {
        $exceptionText = $_ | Out-String
        Append-Utf8NoBom `
            -Path $logPath `
            -Text (
                [Environment]::NewLine +
                "runner_exception=" +
                $exceptionText +
                [Environment]::NewLine
            )
        $exitCode = -1
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Set-Location -LiteralPath $previousLocation
        foreach ($key in $previousEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable(
                [string]$key,
                [string]$previousEnvironment[$key],
                "Process"
            )
        }
        foreach ($key in $missingEnvironment) {
            [Environment]::SetEnvironmentVariable(
                [string]$key,
                $null,
                "Process"
            )
        }
        $stopwatch.Stop()
    }

    $status = if ($exitCode -eq 0) { "passed" } else { "failed" }
    $finishedAt = [DateTime]::UtcNow
    $result = [pscustomobject][ordered]@{
        sequence = $Sequence
        name = $Case.name
        category = $Case.category
        status = $status
        exit_code = $exitCode
        started_at = $startedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        duration_seconds = [Math]::Round(
            $stopwatch.Elapsed.TotalSeconds,
            6
        )
        executable = $Case.executable
        arguments = @($Case.arguments)
        command = $Case.command
        environment = $Case.environment
        log_path = $logPath
        runner_exception = $exceptionText
    }
    $footer = @(
        ""
        "status=$status"
        "exit_code=$exitCode"
        "finished_at=$($finishedAt.ToString("o"))"
        "duration_seconds=$($result.duration_seconds)"
        ""
    ) -join [Environment]::NewLine
    Append-Utf8NoBom -Path $logPath -Text $footer

    $event = $result | ConvertTo-Json -Compress -Depth 6
    Append-Utf8NoBom `
        -Path $script:EventsPath `
        -Text ($event + [Environment]::NewLine)

    if ($exitCode -eq 0) {
        Write-Host (
            "[{0}/{1}] PASS {2} ({3:N3}s)" -f
            $Sequence,
            $script:Cases.Count,
            $Case.name,
            $stopwatch.Elapsed.TotalSeconds
        )
    }
    else {
        Write-Host (
            "[{0}/{1}] FAIL {2} exit={3} ({4:N3}s)" -f
            $Sequence,
            $script:Cases.Count,
            $Case.name,
            $exitCode,
            $stopwatch.Elapsed.TotalSeconds
        ) -ForegroundColor Red
    }
    return $result
}

$script:SuiteDirectory = Get-CanonicalPath `
    -Path $PSScriptRoot `
    -PathType Container
$script:RepositoryRoot = Get-CanonicalPath `
    -Path (Join-Path $script:SuiteDirectory "..\..\..") `
    -PathType Container
$canonicalProjectFromLayout = Get-CanonicalPath `
    -Path (Join-Path $script:SuiteDirectory "..") `
    -PathType Container

if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
    $ProjectPath = $canonicalProjectFromLayout
}
$script:CanonicalProject = Get-CanonicalPath `
    -Path $ProjectPath `
    -PathType Container
if (-not (Test-SamePath `
    -Left $script:CanonicalProject `
    -Right $canonicalProjectFromLayout)) {
    throw (
        "ProjectPath must be the canonical beat_first_v1 project: " +
        $canonicalProjectFromLayout
    )
}
$projectToml = Join-Path $script:CanonicalProject "Project.toml"
$manifestToml = Join-Path $script:CanonicalProject "Manifest.toml"
$null = Get-CanonicalPath -Path $projectToml -PathType Leaf
$null = Get-CanonicalPath -Path $manifestToml -PathType Leaf

$script:CanonicalJulia = Get-CanonicalPath `
    -Path $JuliaExecutable `
    -PathType Leaf

if ($SkipDatasetTests) {
    $script:CanonicalDataset = $null
}
else {
    $script:CanonicalDataset = Get-CanonicalPath `
        -Path $DatasetPath `
        -PathType Container
}

$windowsPowerShell = Join-Path (
    [Environment]::GetFolderPath("System")
) "WindowsPowerShell\v1.0\powershell.exe"
$canonicalPowerShell = Get-CanonicalPath `
    -Path $windowsPowerShell `
    -PathType Leaf
$controllerContractPath = Join-Path `
    $script:SuiteDirectory `
    "test_controller_contract.ps1"
$mutationContractPath = Join-Path `
    $script:SuiteDirectory `
    "test_verifier_fixture_mutations.ps1"
$null = Get-CanonicalPath -Path $controllerContractPath -PathType Leaf
$null = Get-CanonicalPath -Path $mutationContractPath -PathType Leaf

$fixtureWasRequested =
    -not [string]::IsNullOrWhiteSpace($FixtureRunDirectory) -or
    -not [string]::IsNullOrWhiteSpace($FixtureLaunchManifestPath) -or
    $AllowSacrificialFixtureMutation
if ($fixtureWasRequested) {
    if (
        [string]::IsNullOrWhiteSpace($FixtureRunDirectory) -or
        [string]::IsNullOrWhiteSpace($FixtureLaunchManifestPath) -or
        -not $AllowSacrificialFixtureMutation
    ) {
        throw (
            "Fixture mutation requires -FixtureRunDirectory, " +
            "-FixtureLaunchManifestPath, and " +
            "-AllowSacrificialFixtureMutation together"
        )
    }
}

$script:Cases = @()
$script:Skipped = @()

$script:Cases += New-SuiteCase `
    -Name "controller_parse" `
    -Category "powershell_static" `
    -Executable $canonicalPowerShell `
    -Arguments @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $controllerContractPath,
        "-ParseOnly"
    )
$script:Cases += New-SuiteCase `
    -Name "controller_static_contract" `
    -Category "powershell_static" `
    -Executable $canonicalPowerShell `
    -Arguments @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $controllerContractPath
    )

$normalJuliaDefinitions = @(
    [pscustomobject]@{
        script = "runtests.jl"
        threads = 1
        category = "model_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_listnet_diagnostics.jl"
        threads = 1
        category = "learning_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_workspace_routing_invariants.jl"
        threads = 1
        category = "routing_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_routing_regularizer.jl"
        threads = $CoreJuliaThreads
        category = "routing_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_structural_utility.jl"
        threads = $CoreJuliaThreads
        category = "structural_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_arena_training.jl"
        threads = $CoreJuliaThreads
        category = "training_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_eprop_shadow.jl"
        threads = $CoreJuliaThreads
        category = "learning_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_barrierless_training.jl"
        threads = $CoreJuliaThreads
        category = "parallel_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_arena_checkpoint_resume.jl"
        threads = $CoreJuliaThreads
        category = "checkpoint_core"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_train_production_contract.jl"
        threads = 1
        category = "production_contract"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_verification_contract_units.jl"
        threads = 1
        category = "production_contract"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_analyzer_contract.jl"
        threads = 1
        category = "analysis_contract"
        dataset = $false
    }
    [pscustomobject]@{
        script = "test_arena_real_batch.jl"
        threads = $CoreJuliaThreads
        category = "dataset_integration"
        dataset = $true
    }
)

$registeredJuliaTests = @(
    $normalJuliaDefinitions |
        ForEach-Object { [string]$_.script } |
        Where-Object { $_ -ne "runtests.jl" } |
        Sort-Object -Unique
)
$discoveredJuliaTests = @(
    Get-ChildItem `
        -LiteralPath $script:SuiteDirectory `
        -File `
        -Filter "test_*.jl" |
        ForEach-Object { $_.Name } |
        Sort-Object -Unique
)
$juliaInventoryDifference = @(
    Compare-Object `
        -ReferenceObject $discoveredJuliaTests `
        -DifferenceObject $registeredJuliaTests
)
if ($juliaInventoryDifference.Count -ne 0) {
    throw (
        "Julia test inventory differs from the full-suite registry: " +
        (($juliaInventoryDifference | Out-String).Trim())
    )
}

foreach ($definition in $normalJuliaDefinitions) {
    if ([bool]$definition.dataset -and $SkipDatasetTests) {
        $script:Skipped += [pscustomobject][ordered]@{
            name = "julia_normal_$($definition.script)"
            reason = "SkipDatasetTests"
        }
        continue
    }
    $environment = @{}
    if ([bool]$definition.dataset) {
        $environment["SWSNN_DATASET"] = $script:CanonicalDataset
    }
    $stem = [IO.Path]::GetFileNameWithoutExtension(
        [string]$definition.script
    )
    $script:Cases += New-JuliaCase `
        -Name "julia_normal_$stem" `
        -Category ([string]$definition.category) `
        -ScriptName ([string]$definition.script) `
        -Threads ([int]$definition.threads) `
        -CheckBounds "auto" `
        -Environment $environment
}

$boundsScripts = @(
    "runtests.jl",
    "test_listnet_diagnostics.jl",
    "test_workspace_routing_invariants.jl",
    "test_routing_regularizer.jl",
    "test_structural_utility.jl",
    "test_arena_training.jl",
    "test_eprop_shadow.jl",
    "test_barrierless_training.jl",
    "test_arena_checkpoint_resume.jl",
    "test_arena_real_batch.jl"
)
if ($SkipBoundsLanes) {
    foreach ($scriptName in $boundsScripts) {
        $script:Skipped += [pscustomobject][ordered]@{
            name = (
                "julia_bounds_" +
                [IO.Path]::GetFileNameWithoutExtension($scriptName)
            )
            reason = "SkipBoundsLanes"
        }
    }
}
else {
    foreach ($scriptName in $boundsScripts) {
        $definition = @(
            $normalJuliaDefinitions |
                Where-Object { $_.script -eq $scriptName }
        )
        if ($definition.Count -ne 1) {
            throw "Bounds test has no unique normal definition: $scriptName"
        }
        if ([bool]$definition[0].dataset -and $SkipDatasetTests) {
            $script:Skipped += [pscustomobject][ordered]@{
                name = (
                    "julia_bounds_" +
                    [IO.Path]::GetFileNameWithoutExtension($scriptName)
                )
                reason = "SkipDatasetTests"
            }
            continue
        }
        $environment = @{}
        if ([bool]$definition[0].dataset) {
            $environment["SWSNN_DATASET"] = $script:CanonicalDataset
        }
        $stem = [IO.Path]::GetFileNameWithoutExtension($scriptName)
        $script:Cases += New-JuliaCase `
            -Name "julia_bounds_$stem" `
            -Category "bounds_safety" `
            -ScriptName $scriptName `
            -Threads ([int]$definition[0].threads) `
            -CheckBounds "yes" `
            -Environment $environment
    }
}

if ($SkipWatchdogProbes) {
    $script:Skipped += [pscustomobject][ordered]@{
        name = "controller_watchdog_recovery_orphan"
        reason = "SkipWatchdogProbes"
    }
}
else {
    $script:Cases += New-SuiteCase `
        -Name "controller_watchdog_recovery_orphan" `
        -Category "powershell_runtime_contract" `
        -Executable $canonicalPowerShell `
        -Arguments @(
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $controllerContractPath,
            "-RunWatchdogProbes"
        )
}

if ($fixtureWasRequested) {
    $canonicalFixtureRun = Get-CanonicalPath `
        -Path $FixtureRunDirectory `
        -PathType Container
    $canonicalFixtureLaunch = Get-CanonicalPath `
        -Path $FixtureLaunchManifestPath `
        -PathType Leaf
    $fixtureLaunch = Get-Content `
        -Raw `
        -LiteralPath $canonicalFixtureLaunch |
        ConvertFrom-Json
    foreach ($requiredProperty in @(
        "run_id",
        "run_directory",
        "expected_updates",
        "start_mode",
        "parent_checkpoint"
    )) {
        if (
            -not (
                $fixtureLaunch.PSObject.Properties.Name -contains
                    $requiredProperty
            )
        ) {
            throw (
                "Fixture launch manifest is missing property: " +
                $requiredProperty
            )
        }
    }
    if (-not (Test-SamePath `
        -Left $canonicalFixtureRun `
        -Right ([string]$fixtureLaunch.run_directory))) {
        throw (
            "FixtureRunDirectory differs from launch manifest " +
            "run_directory"
        )
    }
    $fixtureRunId = [string]$fixtureLaunch.run_id
    $fixtureExpectedUpdates = [int]$fixtureLaunch.expected_updates
    $fixtureStartMode = [string]$fixtureLaunch.start_mode
    if ($fixtureExpectedUpdates -lt 1 -or $fixtureExpectedUpdates -gt 2) {
        throw "Fixture expected_updates must be 1 or 2"
    }
    if (
        @("scratch", "resume", "finalize-only") -notcontains
            $fixtureStartMode
    ) {
        throw "Fixture launch has an unsupported start_mode"
    }

    $mutationArguments = @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $mutationContractPath,
        "-RunDirectory",
        $canonicalFixtureRun,
        "-LaunchManifestPath",
        $canonicalFixtureLaunch,
        "-ExpectedUpdates",
        [string]$fixtureExpectedUpdates,
        "-RunId",
        $fixtureRunId,
        "-StartMode",
        $fixtureStartMode,
        "-JuliaExecutable",
        $script:CanonicalJulia,
        "-ProjectPath",
        $script:CanonicalProject,
        "-AllowSacrificialMutation"
    )
    if ($fixtureStartMode -eq "scratch") {
        if ($null -ne $fixtureLaunch.parent_checkpoint) {
            throw "Scratch fixture launch must not have a parent checkpoint"
        }
    }
    else {
        if ($null -eq $fixtureLaunch.parent_checkpoint) {
            throw "Non-scratch fixture launch is missing its parent triple"
        }
        foreach ($requiredParentProperty in @(
            "path",
            "sha256",
            "update"
        )) {
            if (
                -not (
                    $fixtureLaunch.parent_checkpoint.PSObject.
                        Properties.Name -contains
                        $requiredParentProperty
                )
            ) {
                throw (
                    "Fixture parent is missing property: " +
                    $requiredParentProperty
                )
            }
        }
        $mutationArguments += @(
            "-ParentCheckpoint",
            [string]$fixtureLaunch.parent_checkpoint.path,
            "-ParentSha256",
            [string]$fixtureLaunch.parent_checkpoint.sha256,
            "-ParentUpdate",
            [string]$fixtureLaunch.parent_checkpoint.update
        )
    }
    $script:Cases += New-SuiteCase `
        -Name "verifier_disposable_fixture_mutations" `
        -Category "fixture_mutation_contract" `
        -Executable $canonicalPowerShell `
        -Arguments $mutationArguments
}
else {
    $script:Skipped += [pscustomobject][ordered]@{
        name = "verifier_disposable_fixture_mutations"
        reason = "no explicit disposable fixture paths supplied"
    }
}

$registeredPowerShellTests = @(
    "test_controller_contract.ps1",
    "test_verifier_fixture_mutations.ps1"
) | Sort-Object -Unique
$discoveredPowerShellTests = @(
    Get-ChildItem `
        -LiteralPath $script:SuiteDirectory `
        -File `
        -Filter "test_*.ps1" |
        ForEach-Object { $_.Name } |
        Sort-Object -Unique
)
$powerShellInventoryDifference = @(
    Compare-Object `
        -ReferenceObject $discoveredPowerShellTests `
        -DifferenceObject $registeredPowerShellTests
)
if ($powerShellInventoryDifference.Count -ne 0) {
    throw (
        "PowerShell test inventory differs from the full-suite registry: " +
        (($powerShellInventoryDifference | Out-String).Trim())
    )
}

if ($ListOnly) {
    Write-Host (
        "SerialWorkspaceSNN full suite: {0} runnable cases, {1} skipped" -f
        $script:Cases.Count,
        $script:Skipped.Count
    )
    $sequence = 0
    foreach ($case in $script:Cases) {
        $sequence += 1
        Write-Host (
            "[{0:D2}] {1} [{2}]" -f
            $sequence,
            $case.name,
            $case.category
        )
        Write-Host ("     " + $case.command)
    }
    foreach ($skippedCase in $script:Skipped) {
        Write-Host (
            "[SKIP] {0}: {1}" -f
            $skippedCase.name,
            $skippedCase.reason
        )
    }
    return
}

if (-not $AllowExistingJuliaProcesses) {
    $existingJulia = @(
        Get-Process -Name "julia" -ErrorAction SilentlyContinue
    )
    if ($existingJulia.Count -ne 0) {
        $juliaPids = (
            $existingJulia |
                ForEach-Object { [string]$_.Id }
        ) -join ","
        throw (
            "Refusing to overlap the full suite with existing Julia " +
            "processes (PIDs $juliaPids). Stop them first or use the " +
            "explicit -AllowExistingJuliaProcesses override."
        )
    }
}

if ([string]::IsNullOrWhiteSpace($LogDirectory)) {
    $LogDirectory = Join-Path (
        [IO.Path]::GetTempPath()
    ) (
        "swsnn_full_suite_" +
        (Get-Date -Format "yyyyMMdd_HHmmss") +
        "_" +
        [guid]::NewGuid().ToString("N")
    )
}
$resolvedLogCandidate = [IO.Path]::GetFullPath($LogDirectory)
if (Test-Path -LiteralPath $resolvedLogCandidate) {
    throw (
        "LogDirectory already exists; refusing to overwrite prior evidence: " +
        $resolvedLogCandidate
    )
}
[IO.Directory]::CreateDirectory($resolvedLogCandidate) | Out-Null
$script:ResolvedLogDirectory = Get-CanonicalPath `
    -Path $resolvedLogCandidate `
    -PathType Container
$script:SummaryPath = Join-Path `
    $script:ResolvedLogDirectory `
    "suite_summary.json"
$script:EventsPath = Join-Path `
    $script:ResolvedLogDirectory `
    "suite_events.jsonl"
$script:SuiteStartedAt = [DateTime]::UtcNow.ToString("o")
$script:Results = @()
$failureMessage = $null
Write-Utf8NoBom -Path $script:EventsPath -Text ""
Write-SuiteSummary -Status "running" -Failure $null

try {
    $sequence = 0
    foreach ($case in $script:Cases) {
        $sequence += 1
        $result = Invoke-SuiteCase `
            -Case $case `
            -Sequence $sequence
        $script:Results += $result
        if ($result.exit_code -ne 0) {
            $failureMessage = (
                "Fail-fast stop after {0}: exit={1}; log={2}" -f
                $result.name,
                $result.exit_code,
                $result.log_path
            )
            Write-SuiteSummary `
                -Status "failed" `
                -Failure $failureMessage
            throw $failureMessage
        }
        Write-SuiteSummary -Status "running" -Failure $null
    }
    Write-SuiteSummary -Status "passed" -Failure $null
}
catch {
    if ([string]::IsNullOrWhiteSpace($failureMessage)) {
        $failureMessage = $_ | Out-String
        Write-SuiteSummary `
            -Status "failed" `
            -Failure $failureMessage
    }
    Write-Host ""
    Write-Host $failureMessage -ForegroundColor Red
    Write-Host ("Suite evidence: " + $script:ResolvedLogDirectory)
    throw
}

Write-Host ""
Write-Host (
    "PASS SerialWorkspaceSNN full suite: {0} cases" -f
    $script:Results.Count
)
Write-Host ("Suite evidence: " + $script:ResolvedLogDirectory)
