[CmdletBinding()]
param(
    [ValidateSet("tiny", "smoke", "production")]
    [string]$Preset = "smoke",
    [string]$OutputDirectory = "",
    [string]$ModelDbRoot = "C:\tmp\hay_modeldb_139653",
    [string]$Distro = "Ubuntu",
    [string]$Venv = "/opt/hd_swsnn_twinprop_neuron",
    [int]$TrainTrials = -1,
    [int]$ValidationTrials = -1,
    [int]$TestTrials = -1,
    [int]$DurationMs = -1,
    [int]$ShardSize = -1,
    [switch]$NoDenseEvents
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "neuron_hay_teacher.py"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot "artifacts\neuron_teacher_$Preset"
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
    "--preset $(Quote-Bash $Preset)"
)

$optional = @(
    @("TrainTrials", $TrainTrials, "--train-trials"),
    @("ValidationTrials", $ValidationTrials, "--validation-trials"),
    @("TestTrials", $TestTrials, "--test-trials"),
    @("DurationMs", $DurationMs, "--duration-ms"),
    @("ShardSize", $ShardSize, "--shard-size")
)
foreach ($entry in $optional) {
    if ([int]$entry[1] -ge 0) {
        $arguments += "$($entry[2]) $($entry[1])"
    }
}
if ($NoDenseEvents) {
    $arguments += "--no-store-dense-events"
}

$command = $arguments[0] + " && " + ($arguments[1..($arguments.Count - 1)] -join " ")
& wsl.exe -d $Distro -- bash -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "NEURON teacher generation failed with exit code $LASTEXITCODE"
}
