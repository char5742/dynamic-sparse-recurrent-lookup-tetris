[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $OutputFile,
    [Parameter(Mandatory = $true)] [string] $CpuSetTopologyPath,
    [Parameter(Mandatory = $true)] [string] $CpuSetTopologySha256,
    [Parameter(Mandatory = $true)] [string] $SystemContractPath,
    [Parameter(Mandatory = $true)] [string] $SystemContractSha256,
    [Parameter(Mandatory = $true)] [string] $BenchmarkContractPath,
    [Parameter(Mandatory = $true)] [string] $BenchmarkContractSha256,
    [Parameter(Mandatory = $true)] [string] $RunnerSourcePath,
    [Parameter(Mandatory = $true)] [string] $RunnerSourceSha256,
    [Parameter(Mandatory = $true)] [string] $ProviderSourcePath,
    [Parameter(Mandatory = $true)] [string] $ProviderSourceSha256,
    [Parameter(Mandatory = $true)] [string] $SubstrateSourcePath,
    [Parameter(Mandatory = $true)] [string] $SubstrateSourceSha256,
    [Parameter(Mandatory = $true)] [string] $SourceManifestPath,
    [Parameter(Mandatory = $true)] [string] $SourceManifestSha256,
    [Parameter(Mandatory = $true)] [string] $RunnerResultPath,
    [Parameter(Mandatory = $true)] [string] $RunnerResultSha256,
    [Parameter(Mandatory = $true)] [string] $RawQpcArtifactPath,
    [Parameter(Mandatory = $true)] [string] $RawQpcArtifactSha256,
    [Parameter(Mandatory = $true)] [string] $EvidenceContractPath,
    [Parameter(Mandatory = $true)] [string] $EvidenceContractSha256,
    [Parameter(Mandatory = $true)] [ValidateSet(
        "H0_cpu_hashstem_cpu_sparse", "H1_npu_hashstem_cpu_sparse"
    )] [string] $MatrixCellId,
    [Parameter(Mandatory = $true)] [string] $XperfPath,
    [Parameter(Mandatory = $true)] [string] $EtwRawQpcExtractorPath,
    [Parameter(Mandatory = $true)] [string] $PcmMemoryPath,
    [Parameter(Mandatory = $true)] [string] $PcmWindowsDriverPath,
    [Parameter(Mandatory = $true)] [string] $PcmRawImcAdapterPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ImplementationStatus = "UNEXECUTED_STATIC_ONLY"
$EtwCapability = "RAW_QPC_CSWITCH_BOUNDARY_RECONSTRUCTION_V1"
$ImcCapability = "INTEL_PCM_RAW_IMC_RD_WR_COUNTERS_V1"

function Assert-CanonicalHash {
    param([string] $Value, [string] $Label)
    if ($Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label must be a lowercase SHA-256 digest"
    }
}

function Resolve-ExplicitLeaf {
    param([string] $Path, [string] $Label)
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) {
        throw "$Label must be an explicit absolute path; PATH lookup is forbidden"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is unavailable at the explicit path"
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $current = Get-Item -LiteralPath $resolved -Force
    while ($null -ne $current) {
        if (($current.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label traverses a reparse point"
        }
        $parent = if ($current -is [System.IO.FileInfo]) {
            $current.Directory
        }
        else {
            $current.Parent
        }
        if ($null -eq $parent) { break }
        $current = $parent
    }
    return $resolved
}

function Get-VerifiedBinding {
    param(
        [string] $Name,
        [string] $Path,
        [string] $ExpectedSha256
    )
    Assert-CanonicalHash -Value $ExpectedSha256 -Label "$Name expected SHA-256"
    $resolved = Resolve-ExplicitLeaf -Path $Path -Label $Name
    $observed = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -cne $ExpectedSha256) {
        throw "$Name SHA-256 mismatch"
    }
    return [ordered]@{ path = $resolved; sha256 = $observed }
}

function Get-ToolRecord {
    param([string] $Path, [string] $Label)
    $resolved = Resolve-ExplicitLeaf -Path $Path -Label $Label
    $item = Get-Item -LiteralPath $resolved -Force
    return [ordered]@{
        label = $Label
        path = $resolved
        file_name = $item.Name
        sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
        file_version = $item.VersionInfo.FileVersion
        product_name = $item.VersionInfo.ProductName
        product_version = $item.VersionInfo.ProductVersion
    }
}

function Invoke-JsonCapabilityProbe {
    param(
        [string] $Executable,
        [string] $ExpectedSchema,
        [string] $ExpectedCapability,
        [string[]] $AdditionalArguments
    )
    $probeOutput = Join-Path ([System.IO.Path]::GetTempPath()) (
        "tetris-evidence-probe-" + [Guid]::NewGuid().ToString("N") + ".json"
    )
    try {
        $arguments = @("--probe-json", $probeOutput) + $AdditionalArguments
        & $Executable @arguments | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "capability probe exited $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $probeOutput -PathType Leaf)) {
            throw "capability probe did not create its requested JSON"
        }
        $probe = Get-Content -LiteralPath $probeOutput -Raw | ConvertFrom-Json
        if ($probe.schema -cne $ExpectedSchema) { throw "capability probe schema mismatch" }
        if ($probe.capability -cne $ExpectedCapability) { throw "capability mismatch" }
        if ($probe.available -ne $true) { throw "capability reports unavailable" }
        if ($probe.fallback_used -ne $false) { throw "capability probe used a fallback" }
        return $probe
    }
    finally {
        if (Test-Path -LiteralPath $probeOutput) {
            Remove-Item -LiteralPath $probeOutput -Force
        }
    }
}

function Write-FreshJson {
    param([string] $Path, [object] $Value)
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) {
        throw "OutputFile must be an absolute path"
    }
    if (Test-Path -LiteralPath $Path) {
        throw "refusing to overwrite OutputFile"
    }
    $directory = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -ItemType Directory -Path $directory | Out-Null
    }
    $temporary = $Path + ".tmp." + [Guid]::NewGuid().ToString("N")
    try {
        $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
        Move-Item -LiteralPath $temporary -Destination $Path
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

$bindings = [ordered]@{
    cpu_set_topology = Get-VerifiedBinding -Name "CPU Set topology" `
        -Path $CpuSetTopologyPath -ExpectedSha256 $CpuSetTopologySha256
    system_contract = Get-VerifiedBinding -Name "system contract" `
        -Path $SystemContractPath -ExpectedSha256 $SystemContractSha256
    benchmark_contract = Get-VerifiedBinding -Name "benchmark contract" `
        -Path $BenchmarkContractPath -ExpectedSha256 $BenchmarkContractSha256
    runner_source = Get-VerifiedBinding -Name "runner source" `
        -Path $RunnerSourcePath -ExpectedSha256 $RunnerSourceSha256
    provider_source = Get-VerifiedBinding -Name "provider source" `
        -Path $ProviderSourcePath -ExpectedSha256 $ProviderSourceSha256
    substrate_source = Get-VerifiedBinding -Name "substrate source" `
        -Path $SubstrateSourcePath -ExpectedSha256 $SubstrateSourceSha256
    source_manifest = Get-VerifiedBinding -Name "source manifest" `
        -Path $SourceManifestPath -ExpectedSha256 $SourceManifestSha256
    runner_result = Get-VerifiedBinding -Name "runner result" `
        -Path $RunnerResultPath -ExpectedSha256 $RunnerResultSha256
    raw_qpc_artifact = Get-VerifiedBinding -Name "raw QPC artifact" `
        -Path $RawQpcArtifactPath -ExpectedSha256 $RawQpcArtifactSha256
    evidence_contract = Get-VerifiedBinding -Name "evidence contract" `
        -Path $EvidenceContractPath -ExpectedSha256 $EvidenceContractSha256
}

$scriptPath = Resolve-ExplicitLeaf -Path $PSCommandPath -Label "probe source"
$producerSha256 = (Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$evidenceBindings = [ordered]@{
    cpu_set_topology_sha256 = $bindings.cpu_set_topology.sha256
    system_contract_sha256 = $bindings.system_contract.sha256
    benchmark_contract_sha256 = $bindings.benchmark_contract.sha256
    runner_source_sha256 = $bindings.runner_source.sha256
    provider_source_sha256 = $bindings.provider_source.sha256
    substrate_source_sha256 = $bindings.substrate_source.sha256
    source_manifest_sha256 = $bindings.source_manifest.sha256
    runner_result_sha256 = $bindings.runner_result.sha256
    raw_qpc_artifact_sha256 = $bindings.raw_qpc_artifact.sha256
    evidence_contract_sha256 = $bindings.evidence_contract.sha256
    producer_source_sha256 = $producerSha256
}

$inventory = [ordered]@{}
$failures = [System.Collections.Generic.List[string]]::new()

try {
    $inventory.xperf = Get-ToolRecord -Path $XperfPath -Label "Microsoft WPT xperf.exe"
    if ($inventory.xperf.file_name -cne "xperf.exe") { throw "xperf leaf name mismatch" }
    $kernelFlags = (& $inventory.xperf.path -providers kf 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "xperf -providers kf exited $LASTEXITCODE" }
    if ($kernelFlags -notmatch '(?m)\bPROC_THREAD\b') { throw "xperf lacks PROC_THREAD" }
    if ($kernelFlags -notmatch '(?m)\bCSWITCH\b') { throw "xperf lacks CSWITCH" }
    $inventory.xperf.kernel_keyword_probe = "PASS"
}
catch {
    $failures.Add("ETW_XPERF_PROVIDER_PROBE: " + $_.Exception.Message)
}

try {
    $inventory.etw_raw_qpc_extractor = Get-ToolRecord `
        -Path $EtwRawQpcExtractorPath -Label "raw-QPC ETW extractor"
    $etwProbe = Invoke-JsonCapabilityProbe `
        -Executable $inventory.etw_raw_qpc_extractor.path `
        -ExpectedSchema "tetris-etw-raw-qpc-extractor-probe-v1" `
        -ExpectedCapability $EtwCapability `
        -AdditionalArguments @()
    foreach ($field in @(
        "supports_process_thread_lifetimes",
        "supports_cswitch_boundary_state",
        "supports_processor_groups",
        "reports_lost_events",
        "reports_unknown_event_versions"
    )) {
        if ($etwProbe.$field -ne $true) { throw "extractor lacks $field" }
    }
    $inventory.etw_raw_qpc_extractor.capability_probe = "PASS"
}
catch {
    $failures.Add("ETW_RAW_QPC_EXTRACTOR_PROBE: " + $_.Exception.Message)
}

try {
    $inventory.pcm_memory = Get-ToolRecord -Path $PcmMemoryPath -Label "Intel PCM pcm-memory.exe"
    if ($inventory.pcm_memory.file_name -cne "pcm-memory.exe") {
        throw "Intel PCM tool leaf name mismatch"
    }
}
catch {
    $failures.Add("INTEL_PCM_TOOL_PROBE: " + $_.Exception.Message)
}

try {
    $inventory.pcm_windows_driver = Get-ToolRecord `
        -Path $PcmWindowsDriverPath -Label "Intel PCM Windows driver"
}
catch {
    $failures.Add("INTEL_PCM_DRIVER_PROBE: " + $_.Exception.Message)
}

try {
    $inventory.pcm_raw_imc_adapter = Get-ToolRecord `
        -Path $PcmRawImcAdapterPath -Label "Intel PCM raw IMC adapter"
    if ((-not $inventory.Contains("pcm_memory")) -or
        (-not $inventory.Contains("pcm_windows_driver"))) {
        throw "raw adapter probe requires the explicit PCM tool and driver"
    }
    $imcProbe = Invoke-JsonCapabilityProbe `
        -Executable $inventory.pcm_raw_imc_adapter.path `
        -ExpectedSchema "tetris-intel-pcm-raw-imc-adapter-probe-v1" `
        -ExpectedCapability $ImcCapability `
        -AdditionalArguments @(
            "--pcm-memory", $inventory.pcm_memory.path,
            "--windows-driver", $inventory.pcm_windows_driver.path
        )
    foreach ($field in @(
        "driver_loaded",
        "supports_raw_before_after",
        "supports_counter_width",
        "supports_overflow_status",
        "supports_multiplex_status",
        "supports_qpc_window",
        "supports_read_write_imc",
        "supports_all_populated_channel_inventory"
    )) {
        if ($imcProbe.$field -ne $true) { throw "raw IMC adapter lacks $field" }
    }
    $inventory.pcm_raw_imc_adapter.capability_probe = "PASS"
}
catch {
    $failures.Add("INTEL_PCM_RAW_ADAPTER_PROBE: " + $_.Exception.Message)
}

$available = $failures.Count -eq 0
$artifact = [ordered]@{
    schema = "heterogeneous-265k-residency-ram-tool-probe-v1"
    implementation_status = $ImplementationStatus
    status = $(if ($available) { "AVAILABLE_PENDING_EXCLUSIVE_CAPTURE" } else { "UNAVAILABLE_FAIL_CLOSED" })
    decision = $(if ($available) { "NO_ADOPTION_BEFORE_CAPTURE_AND_VALIDATION" } else { "INCONCLUSIVE_FAIL_CLOSED" })
    adoption_allowed = $false
    matrix_cell_id = $MatrixCellId
    providers = [ordered]@{
        etw = [ordered]@{
            name = "NT Kernel Logger/SystemTraceProvider"
            guid = "{9e814aad-3204-11d2-9a82-006008a86939}"
            capture_tool = "Microsoft Windows Performance Toolkit xperf.exe"
            extractor_capability = $EtwCapability
        }
        imc = [ordered]@{
            name = "Intel Processor Counter Monitor (Intel PCM)"
            project = "https://github.com/intel/pcm"
            tool = "pcm-memory.exe"
            adapter_capability = $ImcCapability
        }
    }
    bindings = $evidenceBindings
    tool_inventory = $inventory
    failures = @($failures)
    stock_pcm_csv_is_sufficient = $false
    fallback_used = $false
    prohibited_substitutions = @(
        "ETW point samples",
        "process I/O bytes",
        "theoretical DIMM bandwidth",
        "pcm-memory CSV without raw counter state"
    )
    probe_source_path = $scriptPath
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    exit_code = $(if ($available) { 0 } else { 2 })
}

Write-FreshJson -Path $OutputFile -Value $artifact
if ($available) { exit 0 }
exit 2
