param(
    [string]$Output = (Join-Path $PSScriptRoot "artifacts\environment.json")
)

$ErrorActionPreference = "Stop"
$cpu = Get-CimInstance Win32_Processor
$computer = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$juliaVersion = (& julia --version) -join "`n"
$status = (& julia --project=$PSScriptRoot -e 'using Pkg; Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)') -join "`n"
$document = [ordered]@{
    collected_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    cpu = $cpu.Name
    physical_cores = $cpu.NumberOfCores
    logical_processors = $cpu.NumberOfLogicalProcessors
    max_clock_mhz = $cpu.MaxClockSpeed
    physical_memory_bytes = $computer.TotalPhysicalMemory
    os = $os.Caption
    os_version = $os.Version
    os_build = $os.BuildNumber
    powershell = $PSVersionTable.PSVersion.ToString()
    julia_version_command = $juliaVersion
    julia_executable = (Get-Command julia).Source
    manifest_status = $status
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
$document | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $Output
Write-Output $Output
