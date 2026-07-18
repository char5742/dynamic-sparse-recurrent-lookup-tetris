param(
    [int]$GeneratorProcessId = 8040,
    [string]$RepositoryRoot = 'C:\Users\fshuu\Documents\tetris',
    [string]$DatasetRoot = 'D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3',
    [string]$GeneratorRunRoot = 'D:\tetris-paper-plus\runs\beat_first_v1\dataset_generation_f4a43d3_20260718T141116',
    [string]$GateRunRoot = 'D:\tetris-paper-plus\runs\beat_first_v1\post_teacher_v3_gate_20260718',
    [switch]$DatasetAuditOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-AtomicJson {
    param([string]$Path, [object]$Value)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Test-NoReparsePoint {
    param(
        [string]$Root,
        [string]$Target
    )
    $trimChars = [char[]]@('\', '/')
    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd($trimChars)
    $targetFull = [System.IO.Path]::GetFullPath($Target)
    $prefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $targetFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    $rootItem = Get-Item -LiteralPath $rootFull -ErrorAction Stop
    if (($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        return $false
    }
    $current = $rootFull
    $suffix = $targetFull.Substring($prefix.Length)
    foreach ($segment in @($suffix -split '[\\/]' | Where-Object { $_.Length -gt 0 })) {
        $current = Join-Path $current $segment
        $item = Get-Item -LiteralPath $current -ErrorAction Stop
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            return $false
        }
    }
    return $true
}

function Invoke-LoggedProcess {
    param(
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments
    )
    $stdout = Join-Path $GateRunRoot "$Name.stdout.log"
    $stderr = Join-Path $GateRunRoot "$Name.stderr.log"
    $started = Get-Date
    Push-Location $RepositoryRoot
    try {
        & $Executable @Arguments 1> $stdout 2> $stderr
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $record = [ordered]@{
        name = $Name
        started_at = $started.ToString('o')
        finished_at = (Get-Date).ToString('o')
        exit_code = $exitCode
        stdout = $stdout
        stderr = $stderr
    }
    Write-AtomicJson -Path (Join-Path $GateRunRoot "$Name.json") -Value $record
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
    return $record
}

trap {
    try {
        Write-AtomicJson -Path $statusPath -Value ([ordered]@{
            status = 'failed'
            generated_at = (Get-Date).ToString('o')
            error = $_.Exception.Message
        })
    } catch {
        # Preserve the original failure when even status publication fails.
    }
    exit 1
}

New-Item -ItemType Directory -Path $GateRunRoot -Force | Out-Null
$statusPath = Join-Path $GateRunRoot 'status.json'
Write-AtomicJson -Path $statusPath -Value ([ordered]@{
    status = 'waiting_for_generator'
    generated_at = (Get-Date).ToString('o')
    generator_process_id = $GeneratorProcessId
    dataset_root = $DatasetRoot
})

$generator = Get-Process -Id $GeneratorProcessId -ErrorAction SilentlyContinue
if ($null -ne $generator) {
    Wait-Process -Id $GeneratorProcessId
}

$manifestPath = Join-Path $DatasetRoot 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "teacher manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$parts = @($manifest.parts)
$validation = @($parts | Where-Object split -eq 'validation')
$train = @($parts | Where-Object split -eq 'train')
$validationStates = ($validation | Measure-Object row_count -Sum).Sum
$trainStates = ($train | Measure-Object row_count -Sum).Sum
$allStates = ($parts | Measure-Object row_count -Sum).Sum
$allCandidates = ($parts | Measure-Object candidate_total -Sum).Sum
$validationSeeds = @($validation | ForEach-Object { [int]$_.seed })
$trainSeeds = @($train | ForEach-Object { [int]$_.seed })
$expectedValidationSeeds = @((120001..120024) + (121001..121024))
$reservedSeeds = @((5742..5757) + (8001..8008) + (91001..91032))
$stderrPath = Join-Path $GeneratorRunRoot 'stderr.log'
$stderrBytes = if (Test-Path -LiteralPath $stderrPath) {
    (Get-Item -LiteralPath $stderrPath).Length
} else {
    -1
}
$stderrLines = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
    @(Get-Content -LiteralPath $stderrPath -Encoding UTF8)
} else {
    @()
}
# Julia's Logging stdlib writes normal @info records to stderr.  Treat the
# structured Info blocks as the generator's audit log, but fail closed on any
# warning, error, stacktrace, native crash text, or otherwise unexpected line.
$unexpectedStderrLines = @($stderrLines | Where-Object {
    $_.Length -gt 0 -and $_ -notmatch '^(\u250C Info:|\u2502|\u2514)'
})
$generatorInfoLogOnly = $stderrBytes -ge 0 -and $unexpectedStderrLines.Count -eq 0
$generatorTargetReachedLogged = @($stderrLines | Where-Object {
    $_ -match '^\u250C Info: Teacher dataset target reached$'
}).Count -eq 1
$generatorCleanStopLogged = @($stderrLines | Where-Object {
    $_ -match '^\u250C Info: Streaming teacher generation stopped cleanly$'
}).Count -eq 1
$datasetPrefix = [System.IO.Path]::GetFullPath($DatasetRoot).TrimEnd('\') + '\'
$partChecks = @($parts | ForEach-Object {
    $relativePath = [string]$_.relative_path
    $relativeIsRelative = -not [System.IO.Path]::IsPathRooted($relativePath)
    $absolutePath = [System.IO.Path]::GetFullPath((Join-Path $DatasetRoot $relativePath))
    $insideDataset = $absolutePath.StartsWith(
        $datasetPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    $lexicallyPresent = $relativeIsRelative -and $insideDataset -and
        (Test-Path -LiteralPath $absolutePath -PathType Leaf)
    $noReparsePoint = $lexicallyPresent -and
        (Test-NoReparsePoint -Root $DatasetRoot -Target $absolutePath)
    $exists = $lexicallyPresent -and $noReparsePoint
    $actualBytes = if ($exists) { (Get-Item -LiteralPath $absolutePath).Length } else { -1 }
    $expectedBytes = [int64]$_.bytes
    $actualHash = if ($exists) {
        (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else {
        ''
    }
    [pscustomobject]@{
        relative_path = $relativePath
        relative_is_relative = $relativeIsRelative
        canonical_path = $absolutePath.ToLowerInvariant()
        inside_dataset = $insideDataset
        no_reparse_point = $noReparsePoint
        exists = $exists
        expected_bytes = $expectedBytes
        actual_bytes = $actualBytes
        bytes_match = $exists -and ($actualBytes -eq $expectedBytes)
        expected_sha256 = ([string]$_.sha256).ToLowerInvariant()
        actual_sha256 = $actualHash
        sha256_match = $exists -and ($actualHash -eq ([string]$_.sha256).ToLowerInvariant())
    }
})
$canonicalPathSet = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$canonicalPathsUnique = $true
foreach ($check in $partChecks) {
    if (-not $canonicalPathSet.Add([string]$check.canonical_path)) {
        $canonicalPathsUnique = $false
    }
}
$partFailures = @($partChecks | Where-Object {
    -not $_.relative_is_relative -or -not $_.inside_dataset -or
    -not $_.no_reparse_point -or -not $_.exists -or
    -not $_.bytes_match -or -not $_.sha256_match
})
$audit = [ordered]@{
    status = 'audited'
    generated_at = (Get-Date).ToString('o')
    format_version = [int]$manifest.format_version
    manifest_total_states = [int]$manifest.counts.'states.total'
    summed_total_states = [int]$allStates
    manifest_total_candidates = [int64]$manifest.counts.'candidates.total'
    summed_total_candidates = [int64]$allCandidates
    manifest_total_episodes = [int]$manifest.counts.'episodes.total'
    summed_total_episodes = $parts.Count
    manifest_train_states = [int]$manifest.counts.'states.train'
    summed_train_states = [int]$trainStates
    manifest_validation_states = [int]$manifest.counts.'states.validation'
    summed_validation_states = [int]$validationStates
    validation_episodes = $validation.Count
    validation_seed_set_exact = (@(Compare-Object ($validationSeeds | Sort-Object) ($expectedValidationSeeds | Sort-Object)).Count -eq 0)
    validation_seeds_unique = (@($validationSeeds | Select-Object -Unique).Count -eq $validationSeeds.Count)
    train_seeds_unique = (@($trainSeeds | Select-Object -Unique).Count -eq $trainSeeds.Count)
    split_seed_overlap = @($validationSeeds | Where-Object { $trainSeeds -contains $_ }).Count
    reserved_seed_hits = @(($validationSeeds + $trainSeeds) | Where-Object { $reservedSeeds -contains $_ })
    part_paths_unique = (@($parts.relative_path | Select-Object -Unique).Count -eq $parts.Count)
    canonical_part_paths_unique = $canonicalPathsUnique
    episode_keys_unique = (@($parts.episode_key | Select-Object -Unique).Count -eq $parts.Count)
    manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    part_files_inside_dataset = (@($partChecks | Where-Object { -not $_.inside_dataset }).Count -eq 0)
    part_paths_all_relative = (@($partChecks | Where-Object { -not $_.relative_is_relative }).Count -eq 0)
    part_files_no_reparse_point = (@($partChecks | Where-Object { -not $_.no_reparse_point }).Count -eq 0)
    part_files_present = (@($partChecks | Where-Object { -not $_.exists }).Count -eq 0)
    part_bytes_match = (@($partChecks | Where-Object { -not $_.bytes_match }).Count -eq 0)
    part_sha256_match = (@($partChecks | Where-Object { -not $_.sha256_match }).Count -eq 0)
    total_part_bytes = [int64](($partChecks | Measure-Object actual_bytes -Sum).Sum)
    part_integrity_failures = @($partFailures | ForEach-Object { $_.relative_path })
    generator_stderr_bytes = $stderrBytes
    generator_info_log_only = $generatorInfoLogOnly
    generator_unexpected_stderr_line_count = $unexpectedStderrLines.Count
    generator_unexpected_stderr_lines = @($unexpectedStderrLines | Select-Object -First 20)
    generator_target_reached_logged = $generatorTargetReachedLogged
    generator_clean_stop_logged = $generatorCleanStopLogged
}
Write-AtomicJson -Path (Join-Path $GateRunRoot 'dataset_audit.json') -Value $audit

$passed = $audit.format_version -eq 3 -and
    $audit.summed_train_states -ge 100000 -and
    $audit.manifest_train_states -eq $audit.summed_train_states -and
    $audit.manifest_validation_states -eq $audit.summed_validation_states -and
    $audit.manifest_total_states -eq $audit.summed_total_states -and
    $audit.manifest_total_candidates -eq $audit.summed_total_candidates -and
    $audit.manifest_total_episodes -eq $audit.summed_total_episodes -and
    $audit.validation_episodes -eq 48 -and
    $audit.validation_seed_set_exact -and
    $audit.validation_seeds_unique -and
    $audit.train_seeds_unique -and
    $audit.split_seed_overlap -eq 0 -and
    $audit.reserved_seed_hits.Count -eq 0 -and
    $audit.part_paths_unique -and
    $audit.canonical_part_paths_unique -and
    $audit.episode_keys_unique -and
    $audit.part_files_inside_dataset -and
    $audit.part_paths_all_relative -and
    $audit.part_files_no_reparse_point -and
    $audit.part_files_present -and
    $audit.part_bytes_match -and
    $audit.part_sha256_match -and
    $audit.part_integrity_failures.Count -eq 0 -and
    $audit.generator_info_log_only -and
    $audit.generator_unexpected_stderr_line_count -eq 0 -and
    $audit.generator_target_reached_logged -and
    $audit.generator_clean_stop_logged
if (-not $passed) {
    Write-AtomicJson -Path $statusPath -Value ([ordered]@{
        status = 'dataset_gate_failed'
        generated_at = (Get-Date).ToString('o')
        audit = (Join-Path $GateRunRoot 'dataset_audit.json')
    })
    throw 'completed teacher dataset failed the post-generation gate'
}

if ($DatasetAuditOnly) {
    Write-AtomicJson -Path $statusPath -Value ([ordered]@{
        status = 'dataset_audit_complete'
        generated_at = (Get-Date).ToString('o')
        dataset_audit = (Join-Path $GateRunRoot 'dataset_audit.json')
        heavy_process_started = $false
    })
    return
}

$julia = (Get-Command julia -ErrorAction Stop).Source
$project = Join-Path $RepositoryRoot 'experiments\beat_first_v1'
Invoke-LoggedProcess -Name 'resolve' -Executable $julia -Arguments @(
    '--startup-file=no', "--project=$project", '-e',
    'using Pkg; Pkg.resolve(); Pkg.instantiate()'
) | Out-Null
Invoke-LoggedProcess -Name 'tests' -Executable $julia -Arguments @(
    '--startup-file=no', "--project=$project",
    (Join-Path $project 'training\test_training.jl')
) | Out-Null
Invoke-LoggedProcess -Name 'trainer_load' -Executable $julia -Arguments @(
    '--startup-file=no', "--project=$project", '-e',
    'include(raw"experiments/beat_first_v1/training/train_supervised.jl"); include(raw"experiments/beat_first_v1/training/train_rl.jl"); println("trainer-load-ok")'
) | Out-Null

$env:BEAT_TEACHER_DATASET = $DatasetRoot
$env:BEAT_SMOKE_VARIANTS = 'gravity_film_medium'
$env:BEAT_SMOKE_STATE_BATCHES = '4'
$env:BEAT_SMOKE_UPDATES = '10'
$env:BEAT_SMOKE_OUTPUT = Join-Path $GateRunRoot 'gravity_medium_v3_smoke.json'
Invoke-LoggedProcess -Name 'gravity_medium_v3_smoke' -Executable $julia -Arguments @(
    '--startup-file=no', "--project=$project",
    (Join-Path $project 'training\post_n1_smoke.jl')
) | Out-Null

Write-AtomicJson -Path $statusPath -Value ([ordered]@{
    status = 'complete'
    generated_at = (Get-Date).ToString('o')
    dataset_audit = (Join-Path $GateRunRoot 'dataset_audit.json')
    gravity_medium_v3_smoke = $env:BEAT_SMOKE_OUTPUT
})
