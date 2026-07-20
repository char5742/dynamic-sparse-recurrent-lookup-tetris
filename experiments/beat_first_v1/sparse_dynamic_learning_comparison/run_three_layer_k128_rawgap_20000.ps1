[CmdletBinding()]
param(
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1"
$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$Reservation = "D:\tetris-paper-plus\runs\beat_first_v1\.sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1.reservation.json"
$Consumed = "D:\tetris-paper-plus\runs\beat_first_v1\.sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1.consumed.json"

$contract = Join-Path $PSScriptRoot "three_layer_cpf_k128_rawgap_20000_update_contract.toml"
$controller = Join-Path $PSScriptRoot "run_three_layer_cpf_2500_signal.ps1"
$manifest = Join-Path $Dataset "manifest.json"
$expectedContractBytes = 5125
$expectedContractSha256 = "5a766947e9e34b1de4822b6622880db0892ed6fb4326cbff10c415ba4c393263"
$expectedManifestSha256 = "1f63172f33f8cee17b7ada88d4f35cdfa94b8d7dd5751c8e8244008caa526ded"

foreach ($required in @($contract, $controller, $manifest, $Julia)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Raw-gap one-shot prerequisite is missing: $required"
    }
}
$contractItem = Get-Item -LiteralPath $contract
if ([int64]$contractItem.Length -ne $expectedContractBytes) {
    throw "Raw-gap preregistration contract byte count changed"
}
$contractSha256 = (Get-FileHash -LiteralPath $contract -Algorithm SHA256).Hash.ToLowerInvariant()
if ($contractSha256 -cne $expectedContractSha256) {
    throw "Raw-gap preregistration contract SHA-256 changed"
}
$manifestSha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestSha256 -cne $expectedManifestSha256) {
    throw "Raw-gap teacher manifest SHA-256 changed"
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Raw-gap one-shot output root already exists; retry/resume is forbidden: $resolvedOutput"
}
if ((Test-Path -LiteralPath $Reservation) -or (Test-Path -LiteralPath $Consumed)) {
    throw "Raw-gap one-shot reservation or consumed sentinel already exists; retry is forbidden"
}

if ($ValidateOnly) {
    [pscustomobject]@{
        validated = $true
        mutation = $false
        launched = $false
        variant = "k128"
        maximum_updates = 20000
        routing_mode = "fixed_wta"
        objective_profile = "rawgap100"
        objective_mode = "raw_teacher_top_gap_huber"
        configured_margin_weight = 1.0
        effective_listnet_weight = 0.0
        effective_raw_top_gap_weight = 1.0
        contract_bytes = $expectedContractBytes
        contract_sha256 = $contractSha256
        dataset_manifest_sha256 = $manifestSha256
        output_root = $resolvedOutput
        reservation_path = $Reservation
        consumed_path = $Consumed
    }
    return
}

# This wrapper exposes no variant, update, routing, objective, resume, retry, or
# tuning switch.  Atomically reserve the only fixed invocation before handing
# its lowercase nonce to the shared controller.  The controller atomically
# moves this file to the consumed sentinel before creating the output root.
$nonce = [Guid]::NewGuid().ToString("N")
$reservationDocument = [ordered]@{
    format_version = 1
    experiment_id = "sparse_3l_k128_rawgap_teacher_signal_cpf_20000_v1"
    status = "reserved_for_atomic_consume"
    nonce = $nonce
    dataset_root = [IO.Path]::GetFullPath($Dataset)
    output_root = $resolvedOutput
    julia_path = [IO.Path]::GetFullPath($Julia)
    reservation_path = [IO.Path]::GetFullPath($Reservation)
    consumed_path = [IO.Path]::GetFullPath($Consumed)
    contract_sha256 = $contractSha256
    dataset_manifest_sha256 = $manifestSha256
} | ConvertTo-Json -Compress
$reservationBytes = [Text.UTF8Encoding]::new($false).GetBytes(
    $reservationDocument + [Environment]::NewLine
)
$reservationStream = [IO.FileStream]::new(
    $Reservation,
    [IO.FileMode]::CreateNew,
    [IO.FileAccess]::Write,
    [IO.FileShare]::None
)
try {
    $reservationStream.Write($reservationBytes, 0, $reservationBytes.Length)
    $reservationStream.Flush($true)
} finally {
    $reservationStream.Dispose()
}

& $controller `
    -Dataset $Dataset `
    -OutputRoot $resolvedOutput `
    -MaximumUpdates 20000 `
    -Variant k128 `
    -ObjectiveProfile rawgap100 `
    -Julia $Julia `
    -RawGapOneShotNonce $nonce `
    -RawGapOneShotReservation $Reservation
