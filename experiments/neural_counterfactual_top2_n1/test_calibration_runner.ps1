$ErrorActionPreference = "Stop"
$RepoRoot = "C:\Users\fshuu\Documents\tetris"
$Runner = Join-Path $RepoRoot "experiments\neural_counterfactual_top2_n1\run_calibration_once.ps1"
$Marker = Join-Path $RepoRoot ".experiment_state\n1_calibration_started.marker"
$SyntheticRoot = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) "n1_runner_synthetic"))
$SourceRoot = Join-Path $SyntheticRoot "sources"
$Utf8NoBom = [Text.UTF8Encoding]::new($false)
[void][IO.Directory]::CreateDirectory($SourceRoot)

function Get-MarkerIdentity {
    if (-not [IO.File]::Exists($Marker)) {
        return [ordered]@{ exists = $false; sha256 = $null; length = $null; last_write_utc = $null }
    }
    $Item = Get-Item -LiteralPath $Marker -Force
    return [ordered]@{
        exists = $true
        sha256 = (Get-FileHash -LiteralPath $Marker -Algorithm SHA256).Hash.ToLowerInvariant()
        length = [int64]$Item.Length
        last_write_utc = $Item.LastWriteTimeUtc.ToString("o")
    }
}

function Invoke-SyntheticRunner([string]$Script, [int]$ExpectedExit) {
    $PowerShell = (Get-Process -Id $PID).Path
    $Info = New-Object Diagnostics.ProcessStartInfo
    $Info.FileName = $PowerShell
    $Info.Arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $Runner + '"'),
        "-SyntheticNoMarker", "-SyntheticEntryPoint", ('"' + $Script + '"')
    ) -join " "
    $Info.WorkingDirectory = $RepoRoot
    $Info.UseShellExecute = $false
    $Info.CreateNoWindow = $true
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    $Process = New-Object Diagnostics.Process
    $Process.StartInfo = $Info
    [void]$Process.Start()
    $OutTask = $Process.StandardOutput.ReadToEndAsync()
    $ErrTask = $Process.StandardError.ReadToEndAsync()
    if (-not $Process.WaitForExit(120000)) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw "synthetic runner test timed out"
    }
    $Exit = [int]$Process.ExitCode
    $Out = [string]$OutTask.Result
    $Err = [string]$ErrTask.Result
    if ($Exit -ne $ExpectedExit) {
        throw "synthetic runner exit mismatch: expected $ExpectedExit observed $Exit stdout=$Out stderr=$Err"
    }
    $Lines = @($Out -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($Lines.Count -ne 1) {
        throw "runner did not return exactly one output directory: $Out"
    }
    $Directory = [IO.Path]::GetFullPath($Lines[0])
    $RequiredPrefix = $SyntheticRoot.TrimEnd('\') + '\'
    if (-not $Directory.StartsWith($RequiredPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "synthetic output escaped its temporary root: $Directory"
    }
    if (-not [IO.Directory]::Exists($Directory)) {
        throw "synthetic output directory was not retained"
    }
    return $Directory
}

function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

$BeforeMarker = Get-MarkerIdentity
$Nonce = [Guid]::NewGuid().ToString("N")
$PassScript = Join-Path $SourceRoot ("pass_" + $Nonce + ".jl")
$FailScript = Join-Path $SourceRoot ("fail_" + $Nonce + ".jl")
$PassSource = @'
# N1_SYNTHETIC_RUNNER_TEST_ONLY
using JSON3
output_directory = only(ARGS)
sleep(0.55)
open(joinpath(output_directory, "calibration_result.json"), "w") do io
    JSON3.write(io, (; status="synthetic-pass", payload="safe-synthetic"))
end
println("synthetic-safe-stdout")
println(stderr, "synthetic-safe-stderr")
'@
$FailSource = @'
# N1_SYNTHETIC_RUNNER_TEST_ONLY
sleep(0.25)
println("synthetic-intentional-exit")
exit(7)
'@
[IO.File]::WriteAllText($PassScript, $PassSource.Replace("`r`n", "`n"), $Utf8NoBom)
[IO.File]::WriteAllText($FailScript, $FailSource.Replace("`r`n", "`n"), $Utf8NoBom)

try {
    $PassDirectory = Invoke-SyntheticRunner $PassScript 0
    $PassMonitorPath = Join-Path $PassDirectory "monitor_result.json"
    $PassMonitor = Get-Content -LiteralPath $PassMonitorPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Require ($PassMonitor.status -ceq "N1-calibration-runner-pass") "pass monitor status mismatch"
    Require ($PassMonitor.mode -ceq "synthetic-no-marker") "pass mode mismatch"
    Require ($PassMonitor.child_exit_code -is [int]) "pass exit code is not a JSON integer"
    Require ([int]$PassMonitor.child_exit_code -eq 0) "pass exit code is not zero"
    Require ($PassMonitor.child_exit_code_dotnet_type -ceq "System.Int32") "pass exit type is not Int32"
    Require ([int]$PassMonitor.sample_interval_milliseconds -eq 200) "sample interval is not 200 ms"
    Require ([int]$PassMonitor.sample_count -ge 2) "pass run has too few 200 ms samples"
    Require ([bool]$PassMonitor.result_present) "pass result is absent"
    Require ([bool]$PassMonitor.result_valid_json) "pass result is not valid JSON"
    foreach ($Name in @(
        "launch_manifest.json", "stdout.log", "stderr.log",
        "resource_samples.jsonl", "calibration_result.json", "monitor_result.json"
    )) {
        Require (Test-Path -LiteralPath (Join-Path $PassDirectory $Name)) "pass artifact missing: $Name"
    }
    $Samples = @(Get-Content -LiteralPath (Join-Path $PassDirectory "resource_samples.jsonl"))
    Require ($Samples.Count -ge 2) "telemetry JSONL has too few rows"
    foreach ($Line in $Samples) {
        $Sample = $Line | ConvertFrom-Json
        Require ($Sample.pid -is [int]) "telemetry pid is not an integer"
        Require ([int64]$Sample.private_bytes -le [int64](4GB)) "telemetry private cap exceeded"
        Require ([int64]$Sample.working_set_bytes -le [int64](2GB)) "telemetry working-set cap exceeded"
    }

    $FailDirectory = Invoke-SyntheticRunner $FailScript 1
    $FailMonitor = Get-Content -LiteralPath (Join-Path $FailDirectory "monitor_result.json") -Raw -Encoding UTF8 |
        ConvertFrom-Json
    Require ($FailMonitor.status -ceq "N1-calibration-runner-fail") "fail monitor status mismatch"
    Require ($FailMonitor.child_exit_code -is [int]) "failure exit code is not a JSON integer"
    Require ([int]$FailMonitor.child_exit_code -eq 7) "child exit code 7 was not retained"
    Require (-not [bool]$FailMonitor.result_present) "failure unexpectedly produced a result"
    foreach ($Name in @("launch_manifest.json", "stdout.log", "stderr.log", "resource_samples.jsonl", "monitor_result.json")) {
        Require (Test-Path -LiteralPath (Join-Path $FailDirectory $Name)) "failure artifact missing: $Name"
    }

    $AfterMarker = Get-MarkerIdentity
    Require ($BeforeMarker.exists -eq $AfterMarker.exists) "synthetic tests changed marker existence"
    Require ($BeforeMarker.sha256 -eq $AfterMarker.sha256) "synthetic tests changed marker content"
    Require ($BeforeMarker.length -eq $AfterMarker.length) "synthetic tests changed marker length"
    Require ($BeforeMarker.last_write_utc -eq $AfterMarker.last_write_utc) "synthetic tests changed marker timestamp"

    [ordered]@{
        status = "N1-calibration-runner-synthetic-tests-pass"
        pass_output = $PassDirectory
        fail_output = $FailDirectory
        production_marker_unchanged = $true
    } | ConvertTo-Json -Depth 4
} finally {
    Remove-Item -LiteralPath $PassScript -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $FailScript -Force -ErrorAction SilentlyContinue
}
