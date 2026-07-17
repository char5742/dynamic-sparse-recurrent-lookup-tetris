param(
    [int]$Threads = 20,
    [int]$Steps = 1000,
    [int]$NativeTimeoutSeconds = 300,
    [int]$ReactantTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$Project = $PSScriptRoot
$Artifacts = Join-Path $Project "artifacts"
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

function Invoke-JuliaCase {
    param(
        [string]$Name,
        [string]$Script,
        [int]$Batch,
        [int]$TimeoutSeconds,
        [hashtable]$ExtraEnvironment = @{}
    )
    $stdout = Join-Path $Artifacts "$Name.log"
    $stderr = Join-Path $Artifacts "$Name.stderr.log"
    $output = Join-Path $Artifacts "$Name.json"
    $environment = @{
        AD_CLEAN_BATCH = "$Batch"
        AD_CLEAN_STEPS = "$Steps"
        AD_CLEAN_BLAS_THREADS = "$Threads"
        AD_CLEAN_OUTPUT = $output
    }
    foreach ($key in $ExtraEnvironment.Keys) { $environment[$key] = $ExtraEnvironment[$key] }
    $saved = @{}
    foreach ($key in $environment.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $environment[$key], "Process")
    }
    try {
        $process = Start-Process -FilePath "julia" -ArgumentList @("--threads=$Threads", "--project=$Project", (Join-Path $Project $Script)) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force
            throw "$Name exceeded ${TimeoutSeconds}s; see $stdout and $stderr"
        }
        if ($process.ExitCode -ne 0) { throw "$Name exited $($process.ExitCode); see $stdout and $stderr" }
    }
    finally {
        foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], "Process") }
    }
}

foreach ($batch in 16, 32, 64, 128) {
    Invoke-JuliaCase -Name "zygote_b${batch}_n${Steps}" -Script "benchmark_native.jl" -Batch $batch -TimeoutSeconds $NativeTimeoutSeconds -ExtraEnvironment @{ AD_CLEAN_BACKEND = "zygote" }
    Invoke-JuliaCase -Name "enzyme_static_b${batch}_n${Steps}" -Script "benchmark_native.jl" -Batch $batch -TimeoutSeconds $NativeTimeoutSeconds -ExtraEnvironment @{ AD_CLEAN_BACKEND = "enzyme_static" }
    Invoke-JuliaCase -Name "reactant_b${batch}_n${Steps}" -Script "benchmark_reactant.jl" -Batch $batch -TimeoutSeconds $ReactantTimeoutSeconds
}
