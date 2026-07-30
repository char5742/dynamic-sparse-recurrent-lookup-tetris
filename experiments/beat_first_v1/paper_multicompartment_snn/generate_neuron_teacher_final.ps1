[CmdletBinding()]
param(
    [ValidateSet("tiny", "smoke", "production")]
    [string]$Preset = "smoke",
    [string]$OutputDirectory = "",
    [string]$ModelDbRoot = "C:\tmp\hay_modeldb_139653",
    [string]$Distro = "Ubuntu",
    [string]$Venv = "/opt/hd_swsnn_twinprop_neuron",
    [int]$TrainTrials = -1,
    [int]$ValidationTrialsFromTrain = -1,
    [int]$TestTrials = -1,
    [int]$DurationMs = -1,
    [int]$ShardSize = -1,
    [int]$Workers = 1,
    [int]$DiagnosticSegments = -1,
    [int]$DiagnosticStrideBins = -1,
    [switch]$NoDenseAxonEvents,
    [switch]$Acknowledge8000ContactInterpretation,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "neuron_hay_teacher_final.py"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot "artifacts\neuron_teacher_final_$Preset"
}

function Convert-ToWslPath([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLowerInvariant()
        $tail = $Matches[2].Replace("\", "/")
        return "/mnt/$drive/$tail"
    }
    throw "Only absolute Windows drive paths are supported: $resolved"
}

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'`"`"'`"`'") + "'"
}

$scriptWsl = Convert-ToWslPath $scriptPath
$outputWsl = Convert-ToWslPath $OutputDirectory
$modelDbWsl = Convert-ToWslPath $ModelDbRoot
$arguments = @(
    "source $(Quote-Bash "$Venv/bin/activate")",
    "python $(Quote-Bash $scriptWsl)",
    "--modeldb-root $(Quote-Bash $modelDbWsl)",
    "--output $(Quote-Bash $outputWsl)",
    "--preset $(Quote-Bash $Preset)",
    "--workers $Workers"
)

$optional = @(
    @($TrainTrials, "--train-trials"),
    @($ValidationTrialsFromTrain, "--validation-trials-from-train"),
    @($TestTrials, "--test-trials"),
    @($DurationMs, "--duration-ms"),
    @($ShardSize, "--shard-size"),
    @($DiagnosticSegments, "--diagnostic-segments"),
    @($DiagnosticStrideBins, "--diagnostic-stride-bins")
)
foreach ($entry in $optional) {
    if ([int]$entry[0] -ge 0) {
        $arguments += "$($entry[1]) $($entry[0])"
    }
}
if ($NoDenseAxonEvents) {
    $arguments += "--no-store-dense-axon-events"
}
if ($Acknowledge8000ContactInterpretation) {
    $arguments += "--acknowledge-8000-contact-interpretation"
}
if ($NoResume) {
    $arguments += "--no-resume"
}

$command = $arguments[0] + " && " + ($arguments[1..($arguments.Count - 1)] -join " ")
& wsl.exe -d $Distro -- bash -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "Final NEURON teacher generation failed with exit code $LASTEXITCODE"
}
