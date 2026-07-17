Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native_windows.ps1')

$script:R1ExpectedPhaseOrder = @(
    'collect_training',
    'fit_ridge',
    'collect_calibration',
    'calibration_gate',
    'finalize_assessment'
)
$script:R1ExpectedHandoffEnvironmentNames = @(
    'R1_EXPECTED_TRAINING_TABLE_SHA256',
    'R1_EXPECTED_TRAINING_MANIFEST_SHA256',
    'R1_EXPECTED_RIDGE_ARTIFACT_SHA256',
    'R1_EXPECTED_CALIBRATION_TABLE_SHA256',
    'R1_EXPECTED_CALIBRATION_MANIFEST_SHA256',
    'R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256',
    'R1_EXPECTED_DESIGN_FREEZE_SHA256'
)

if ($null -eq ('R1.WindowsDurablePublication' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace R1 {
    public static class WindowsDurablePublication {
        private const uint MOVEFILE_WRITE_THROUGH = 0x00000008;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool MoveFileExW(string existingName, string newName, uint flags);

        public static void MoveNoReplaceWriteThrough(string source, string destination) {
            // MOVEFILE_REPLACE_EXISTING is deliberately absent.  The call is
            // both create-if-absent and, on Windows, does not return until the
            // rename has been carried out to disk.
            if (!MoveFileExW(source, destination, MOVEFILE_WRITE_THROUGH)) {
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                    "MoveFileExW(MOVEFILE_WRITE_THROUGH) failed");
            }
        }
    }
}
'@
}

function Get-R1DurablePublicationEvidence {
    return [ordered]@{
        platform='Windows'
        temporary_file='FileStream WriteThrough plus Flush(true)'
        publication='MoveFileExW without REPLACE_EXISTING'
        publication_flag='MOVEFILE_WRITE_THROUGH (0x00000008)'
        parent_directory_durability='MoveFileExW WRITE_THROUGH is the strongest documented Windows rename/metadata durability equivalent; Windows does not expose POSIX fsync(directory)'
        overwrite_allowed=$false
    }
}

function Write-R1JsonDurableAtomic([string]$Path, $Value) {
    $Full = [IO.Path]::GetFullPath($Path)
    $Parent = Split-Path -Parent $Full
    [IO.Directory]::CreateDirectory($Parent) | Out-Null
    if ([IO.File]::Exists($Full)) { throw "refusing to overwrite R1 artifact: $Full" }
    $Temporary = "$Full.tmp.$([guid]::NewGuid().ToString('N'))"
    $Bytes = [Text.UTF8Encoding]::new($false).GetBytes(
        (($Value | ConvertTo-Json -Depth 32) + "`n")
    )
    try {
        $Stream = [IO.FileStream]::new(
            $Temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write,
            [IO.FileShare]::None, 4096, [IO.FileOptions]::WriteThrough
        )
        try { $Stream.Write($Bytes, 0, $Bytes.Length); $Stream.Flush($true) }
        finally { $Stream.Dispose() }
        [R1.WindowsDurablePublication]::MoveNoReplaceWriteThrough($Temporary, $Full)
    } finally {
        if ([IO.File]::Exists($Temporary)) { [IO.File]::Delete($Temporary) }
    }
}

function Resolve-R1ConcretePwsh([string]$Requested = '') {
    $FrozenPath = 'C:\Program Files\PowerShell\7\pwsh.exe'
    $FrozenVersion = '7.4.17'
    $FrozenSize = [int64]290616
    $FrozenSha256 = 'fedb2341c0b3837dfedc147ec582dec7f245a7b45faa97d23c4d4a1ee316ee2a'
    $Path = if ([string]::IsNullOrWhiteSpace($Requested)) { $FrozenPath } else { $Requested }
    $Path = [IO.Path]::GetFullPath($Path)
    if (-not $Path.Equals($FrozenPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "PowerShell path differs from frozen pwsh 7.4.17: $Path"
    }
    if (-not [IO.File]::Exists($Path)) { throw "frozen pwsh is missing: $Path" }
    $Item = Get-Item -LiteralPath $Path -Force
    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        [int64]$Item.Length -ne $FrozenSize) {
        throw 'frozen pwsh file identity changed'
    }
    $ObservedSha = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ObservedSha -cne $FrozenSha256) { throw 'frozen pwsh SHA-256 changed' }
    $ObservedVersion = @(& $Path -NoLogo -NoProfile -NonInteractive -Command '$PSVersionTable.PSVersion.ToString()' 2>&1) -join ''
    if ($LASTEXITCODE -ne 0 -or $ObservedVersion.Trim() -cne $FrozenVersion) {
        throw "exact pwsh $FrozenVersion is required; observed $ObservedVersion"
    }
    return [ordered]@{
        path=$Path; version=$FrozenVersion; size_bytes=$FrozenSize
        sha256=$FrozenSha256; windows_powershell_5_1_supported=$false
        reparse_point=$false
    }
}

function Add-R1JsonLine([IO.StreamWriter]$Writer, $Value) {
    $Writer.WriteLine(($Value | ConvertTo-Json -Depth 16 -Compress))
    $Writer.Flush()
    # Lifecycle and telemetry must survive an abrupt wrapper/process stop.
    $Writer.BaseStream.Flush($true)
}

function Resolve-R1ConcreteJulia([string]$Requested = '') {
    $FrozenPath = 'C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe'
    $FrozenSize = [int64]170952
    $FrozenSha256 = '4b1984610b12c9ac119340261bee08d93a0032989b0c35d20ffddaadba241043'
    $Path = if ([string]::IsNullOrWhiteSpace($Requested)) { $FrozenPath } else { $Requested }
    $Path = [IO.Path]::GetFullPath($Path)
    if (-not $Path.Equals($FrozenPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Julia path differs from the frozen concrete binary: $Path"
    }
    if (-not [IO.File]::Exists($Path)) { throw "concrete Julia binary missing: $Path" }
    if ($Path -match '(?i)\\Microsoft\\WindowsApps\\') {
        throw 'juliaup/WindowsApps launcher is forbidden as the monitored Julia binary'
    }
    $Item = Get-Item -LiteralPath $Path -Force
    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'concrete Julia binary must not be a reparse point'
    }
    if ([int64]$Item.Length -ne $FrozenSize) { throw 'concrete Julia binary size changed' }
    $ObservedSha = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ObservedSha -cne $FrozenSha256) { throw 'concrete Julia binary SHA-256 changed' }
    $Version = @(& $Path --startup-file=no --history-file=no -e 'print(VERSION)' 2>&1) -join ''
    if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '1.12.6') {
        throw "concrete Julia must be 1.12.6, observed: $Version"
    }
    return [ordered]@{
        path = $Path
        version = $Version.Trim()
        sha256 = $ObservedSha
        size_bytes = [int64]$Item.Length
        windowsapps_launcher_used = $false
        reparse_point = $false
    }
}

function Get-R1PythonOriginProbeScript {
    return @'
import ctypes
import ctypes.wintypes
import hashlib
import json
import os
import pathlib
import sys

import numpy
import openvino

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

module_file = pathlib.Path(openvino.__file__).resolve()
package_root = module_file.parent
package_records = []
for path in sorted(
    (candidate for candidate in package_root.rglob("*") if candidate.is_file()),
    key=lambda candidate: candidate.relative_to(package_root).as_posix(),
):
    if path.is_symlink():
        raise RuntimeError(f"OpenVINO package contains a symlink: {path}")
    package_records.append({
        "path": path.relative_to(package_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    })

def record_aggregate(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()

psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
HMODULE = ctypes.wintypes.HMODULE
kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
psapi.EnumProcessModules.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.POINTER(HMODULE),
    ctypes.wintypes.DWORD,
    ctypes.POINTER(ctypes.wintypes.DWORD),
]
psapi.EnumProcessModules.restype = ctypes.wintypes.BOOL
psapi.GetModuleFileNameExW.argtypes = [
    ctypes.wintypes.HANDLE,
    HMODULE,
    ctypes.wintypes.LPWSTR,
    ctypes.wintypes.DWORD,
]
psapi.GetModuleFileNameExW.restype = ctypes.wintypes.DWORD
modules = (HMODULE * 4096)()
needed = ctypes.wintypes.DWORD()
process = kernel32.GetCurrentProcess()
if not psapi.EnumProcessModules(
    process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
):
    raise ctypes.WinError(ctypes.get_last_error())
if needed.value > ctypes.sizeof(modules):
    raise RuntimeError("loaded-module buffer was too small")
native_by_path = {}
for index in range(needed.value // ctypes.sizeof(HMODULE)):
    buffer = ctypes.create_unicode_buffer(32768)
    if not psapi.GetModuleFileNameExW(process, modules[index], buffer, len(buffer)):
        continue
    path = pathlib.Path(buffer.value).resolve()
    try:
        relative = path.relative_to(package_root).as_posix()
    except ValueError:
        continue
    if path.is_file() and path.suffix.lower() in (".dll", ".pyd"):
        native_by_path[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
native_records = [native_by_path[path] for path in sorted(native_by_path)]

facts = {
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "openvino": openvino.__version__,
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
    "prefix": str(pathlib.Path(sys.prefix).resolve()),
    "pythonpath_cleared": os.environ.get("PYTHONPATH") is None,
    "pythonhome_cleared": os.environ.get("PYTHONHOME") is None,
    "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
    "openvino_module_file": str(module_file),
    "openvino_package_root": str(package_root),
    "openvino_package_file_count": len(package_records),
    "openvino_package_bytes": sum(record["bytes"] for record in package_records),
    "openvino_package_tree_sha256": record_aggregate(package_records),
    "openvino_loaded_native_modules": native_records,
    "openvino_loaded_native_sha256": record_aggregate(native_records),
}
print(json.dumps(facts, separators=(",", ":")))
'@
}

function Resolve-R1FrozenPython([string]$Requested = '') {
    $VenvLauncher = 'D:\tetris-paper-plus\python-env\Scripts\python.exe'
    $VenvLauncherSize = [int64]262144
    $VenvLauncherSha256 = '5912d0884b23c0343983a864c6064242391e2265536f50b88624857e353882c9'
    # CreateProcess on the venv launcher creates another Python process.  R1
    # instead launches this frozen concrete image directly while supplying the
    # standard __PYVENV_LAUNCHER__ binding, preserving the venv prefix/packages
    # and a one-to-one root PID/image/milestone identity.
    $FrozenPath = 'C:\Users\fshuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $FrozenSize = [int64]91648
    $FrozenSha256 = '3c6a206b7d93cca823934a83732220dcffd413fd1036d9fb82eebb64599cf7f3'
    $ExpectedOpenVino = '2026.2.1-21919-ede283a88e3-releases/2026/2'
    $ExpectedBasePrefix = 'C:\Users\fshuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python'
    $ExpectedPrefix = 'D:\tetris-paper-plus\python-env'
    $ExpectedOpenVinoModule = 'D:\tetris-paper-plus\python-env\Lib\site-packages\openvino\__init__.py'
    $ExpectedOpenVinoRoot = 'D:\tetris-paper-plus\python-env\Lib\site-packages\openvino'
    $ExpectedPackageTreeSha256 = 'c292b25245f36e937f21b105023737be80491e70de5f24b1704ad1ced8547e43'
    $ExpectedPackageFileCount = 1102
    $ExpectedPackageBytes = [int64]234324334
    $ExpectedNativeSha256 = '929dd49859750bfa59c850234c8eeb872c84db05c1b60510e9a9db8b7d756a74'
    $ExpectedNative = @(
        [ordered]@{ path='_pyopenvino.cp312-win_amd64.pyd'; bytes=[int64]3981304; sha256='602468f4ca37ba9859a6b22220e46267170abd0f744a72a839bb5e0874f2ad56' },
        [ordered]@{ path='frontend/tensorflow/py_tensorflow_frontend.cp312-win_amd64.pyd'; bytes=[int64]477176; sha256='859f76d40fb77074bf94de2bed3317aeb835debb2e98bd4b78ce24f0b1077c28' },
        [ordered]@{ path='libs/openvino.dll'; bytes=[int64]15888376; sha256='dc29dfd84048bed1b687426f770ee355d6c2933bbb3c6aa861f6dd1154e48908' },
        [ordered]@{ path='libs/tbb12.dll'; bytes=[int64]225272; sha256='5cdc63525bc1b7f9916b329a1304955a360803b3d5780f5b23e62fd4027622be' },
        [ordered]@{ path='libs/tbbmalloc.dll'; bytes=[int64]120824; sha256='11ac3ef8f213cc647b63f92dd115c2c005e489fd050fd41c3e69cf3cd4949921' }
    )
    $Path = if ([string]::IsNullOrWhiteSpace($Requested)) { $FrozenPath } else { $Requested }
    $Path = [IO.Path]::GetFullPath($Path)
    if (-not $Path.Equals($FrozenPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Python path differs from the frozen R1 environment: $Path"
    }
    if (-not [IO.File]::Exists($Path)) { throw "frozen Python missing: $Path" }
    $Item = Get-Item -LiteralPath $Path -Force
    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'frozen Python binary must not be a reparse point'
    }
    if ([int64]$Item.Length -ne $FrozenSize) { throw 'frozen Python size changed' }
    $ObservedSha = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ObservedSha -cne $FrozenSha256) { throw 'frozen Python SHA-256 changed' }
    $LauncherItem = Get-Item -LiteralPath $VenvLauncher -Force
    if (($LauncherItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        [int64]$LauncherItem.Length -ne $VenvLauncherSize -or
        (Get-FileHash -LiteralPath $VenvLauncher -Algorithm SHA256).Hash.ToLowerInvariant() -cne $VenvLauncherSha256) {
        throw 'frozen Python venv launcher binding changed'
    }
    $OldLauncher = [Environment]::GetEnvironmentVariable('__PYVENV_LAUNCHER__', 'Process')
    $OldPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    $OldPythonHome = [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process')
    $OldPythonNoUserSite = [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process')
    [Environment]::SetEnvironmentVariable('__PYVENV_LAUNCHER__', $VenvLauncher, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $null, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONHOME', $null, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', '1', 'Process')
    try {
        $FactsText = @(& $Path -c (Get-R1PythonOriginProbeScript) 2>&1) -join ''
    } finally {
        [Environment]::SetEnvironmentVariable('__PYVENV_LAUNCHER__', $OldLauncher, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONPATH', $OldPythonPath, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONHOME', $OldPythonHome, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', $OldPythonNoUserSite, 'Process')
    }
    if ($LASTEXITCODE -ne 0) { throw "frozen Python/OpenVINO probe failed: $FactsText" }
    $Facts = $FactsText | ConvertFrom-Json
    if ($Facts.python -cne '3.12.13' -or $Facts.numpy -cne '2.4.6' -or $Facts.openvino -cne $ExpectedOpenVino -or
        -not [IO.Path]::GetFullPath([string]$Facts.executable).Equals($VenvLauncher, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$Facts.base_prefix).Equals($ExpectedBasePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$Facts.prefix).Equals($ExpectedPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not [bool]$Facts.pythonpath_cleared -or -not [bool]$Facts.pythonhome_cleared -or
        [string]$Facts.python_no_user_site -cne '1' -or
        -not [IO.Path]::GetFullPath([string]$Facts.openvino_module_file).Equals($ExpectedOpenVinoModule, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetFullPath([string]$Facts.openvino_package_root).Equals($ExpectedOpenVinoRoot, [StringComparison]::OrdinalIgnoreCase) -or
        [int]$Facts.openvino_package_file_count -ne $ExpectedPackageFileCount -or
        [int64]$Facts.openvino_package_bytes -ne $ExpectedPackageBytes -or
        [string]$Facts.openvino_package_tree_sha256 -cne $ExpectedPackageTreeSha256 -or
        [string]$Facts.openvino_loaded_native_sha256 -cne $ExpectedNativeSha256) {
        throw 'frozen Python/OpenVINO versions changed'
    }
    $ObservedNative = @($Facts.openvino_loaded_native_modules)
    if ($ObservedNative.Count -ne $ExpectedNative.Count) {
        throw 'frozen OpenVINO loaded native-module set changed'
    }
    for ($Index = 0; $Index -lt $ExpectedNative.Count; $Index += 1) {
        if ([string]$ObservedNative[$Index].path -cne [string]$ExpectedNative[$Index].path -or
            [int64]$ObservedNative[$Index].bytes -ne [int64]$ExpectedNative[$Index].bytes -or
            [string]$ObservedNative[$Index].sha256 -cne [string]$ExpectedNative[$Index].sha256) {
            throw "frozen OpenVINO loaded native module changed at index $Index"
        }
    }
    return [ordered]@{
        path=$Path; size_bytes=[int64]$Item.Length; sha256=$ObservedSha
        python_version=[string]$Facts.python; numpy_version=[string]$Facts.numpy; openvino_version=[string]$Facts.openvino
        venv_launcher_path=$VenvLauncher; venv_launcher_size_bytes=$VenvLauncherSize
        venv_launcher_sha256=$VenvLauncherSha256; python_executable=[string]$Facts.executable
        python_base_prefix=[string]$Facts.base_prefix; venv_prefix=[string]$Facts.prefix
        pythonpath_cleared=[bool]$Facts.pythonpath_cleared; pythonhome_cleared=[bool]$Facts.pythonhome_cleared
        python_no_user_site=[string]$Facts.python_no_user_site
        openvino_module_file=[string]$Facts.openvino_module_file
        openvino_package_root=[string]$Facts.openvino_package_root
        openvino_package_file_count=[int]$Facts.openvino_package_file_count
        openvino_package_bytes=[int64]$Facts.openvino_package_bytes
        openvino_package_tree_sha256=[string]$Facts.openvino_package_tree_sha256
        openvino_package_tree_aggregate='sorted relative-path NUL decimal-bytes NUL lowercase-sha256 newline'
        openvino_loaded_native_modules=@($ObservedNative)
        openvino_loaded_native_sha256=[string]$Facts.openvino_loaded_native_sha256
        reparse_point=$false
    }
}

function Assert-R1FrozenManifest([string]$Repository) {
    $Manifest = [IO.Path]::GetFullPath((Join-Path $Repository 'Manifest.toml'))
    $Expected = '2cfe650387ed772ec41bd9c3f6bba18f8d954b882d2fa3bfcc8cdbe6840c7b09'
    if (-not [IO.File]::Exists($Manifest)) { throw 'R1 Manifest.toml missing' }
    $Observed = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Observed -cne $Expected) { throw "R1 Manifest SHA-256 changed: $Observed" }
    return [ordered]@{ path=$Manifest; sha256=$Observed }
}

function Resolve-R1PhaseEnvironmentHandoffs {
    param(
        [Parameter(Mandatory)][string]$Phase,
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][System.Collections.IDictionary]$ProducerLedger
    )
    $Resolved = [ordered]@{}
    foreach ($NameValue in @($Plan.Keys)) {
        $Name = [string]$NameValue
        $Binding = $Plan[$NameValue]
        if ($Binding -isnot [System.Collections.IDictionary]) {
            throw "phase $Phase SHA handoff $Name lacks a structured immutable binding"
        }
        if ($Binding.Contains('source_phase') -and $Binding.Contains('path')) {
            $SourcePhase = [string]$Binding['source_phase']
            $Path = [IO.Path]::GetFullPath([string]$Binding['path'])
            if (-not $ProducerLedger.Contains($Path)) {
                throw "phase $Phase SHA handoff has no producer-completion ledger entry: $Path"
            }
            $Entry = $ProducerLedger[$Path]
            if ([string]$Entry.phase -cne $SourcePhase) {
                throw "phase $Phase SHA handoff producer mismatch for $Path"
            }
            if (-not [IO.File]::Exists($Path)) {
                throw "phase $Phase SHA handoff source disappeared: $Path"
            }
            $Current = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($Current -cne [string]$Entry.sha256) {
                throw "phase $Phase SHA handoff source changed after producer completion: $Path"
            }
            $Resolved[$Name] = [string]$Entry.sha256
        } elseif ($Binding.Contains('frozen_path') -and $Binding.Contains('frozen_sha256')) {
            $Path = [IO.Path]::GetFullPath([string]$Binding['frozen_path'])
            $Expected = [string]$Binding['frozen_sha256']
            if ($Expected -notmatch '^[0-9a-f]{64}$' -or -not [IO.File]::Exists($Path)) {
                throw "phase $Phase frozen SHA handoff is invalid or missing: $Path"
            }
            $Current = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($Current -cne $Expected) {
                throw "phase $Phase frozen SHA handoff changed: $Path"
            }
            $Resolved[$Name] = $Expected
        } else {
            throw "phase $Phase SHA handoff $Name has an unknown binding shape"
        }
    }
    return $Resolved
}

function Get-R1ProcessTreeSample($NativeProcess, [int]$Sequence, [datetimeoffset]$PhaseStarted) {
    $JobIds = @($NativeProcess.GetProcessIds() | Sort-Object)
    $ProcessByPid = [ordered]@{}
    $SamplingErrors = @()
    $RemainingJobIds = $JobIds
    # Process trees can grow while they are sampled. Retry the membership query
    # and sample every newly observed live PID; never silently drop a member.
    for ($Pass = 0; $Pass -lt 32; $Pass += 1) {
        foreach ($Id in @($RemainingJobIds | Where-Object { -not $ProcessByPid.Contains([string]$_) })) {
            try {
                $Process = Get-Process -Id $Id -ErrorAction Stop
                $ProcessByPid[[string]$Id] = [ordered]@{
                    pid = [int]$Id
                    process_name = [string]$Process.ProcessName
                    image_path = $(try { [string]$Process.Path } catch { $null })
                    start_time_utc = $(try { $Process.StartTime.ToUniversalTime().ToString('o') } catch { $null })
                    working_set_resident_bytes = [int64]$Process.WorkingSet64
                    private_committed_bytes = [int64]$Process.PrivateMemorySize64
                    cpu_seconds = [double]$Process.TotalProcessorTime.TotalSeconds
                }
            } catch {
                $StillLive = [int]$Id -in @($NativeProcess.GetProcessIds())
                if ($StillLive) {
                    $SamplingErrors += [ordered]@{ pid=[int]$Id; error=$_.Exception.Message }
                }
            }
        }
        $RemainingJobIds = @($NativeProcess.GetProcessIds() | Sort-Object)
        $Missing = @($RemainingJobIds | Where-Object { -not $ProcessByPid.Contains([string]$_) })
        if ($Missing.Count -eq 0) { break }
        Start-Sleep -Milliseconds 2
    }
    $Processes = @($ProcessByPid.Values | Sort-Object pid)
    $SampledIds = @($Processes | ForEach-Object pid)
    $UnsampledLiveIds = @($RemainingJobIds | Where-Object { $_ -notin $SampledIds })
    $TotalWorking = [int64]0
    $TotalPrivate = [int64]0
    foreach ($Entry in $Processes) {
        $TotalWorking += [int64]$Entry.working_set_resident_bytes
        $TotalPrivate += [int64]$Entry.private_committed_bytes
    }
    return [ordered]@{
        schema = 'r1-process-tree-telemetry-v1'
        sequence = $Sequence
        sampled_at = [DateTimeOffset]::UtcNow.ToString('o')
        elapsed_seconds = ([DateTimeOffset]::UtcNow - $PhaseStarted).TotalSeconds
        polling_interval_target_ms = 200
        root_pid = [int]$NativeProcess.ProcessId
        process_count = $Processes.Count
        job_process_ids_at_sample_start = @($JobIds)
        process_ids = @($SampledIds)
        job_process_ids_after_sample = @($RemainingJobIds)
        unsampled_live_process_ids = @($UnsampledLiveIds)
        accounting_complete = $UnsampledLiveIds.Count -eq 0
        sampling_errors = @($SamplingErrors)
        processes = @($Processes)
        process_tree_working_set_resident_bytes = $TotalWorking
        process_tree_private_committed_bytes = $TotalPrivate
        windows_job_peak_memory_used_bytes = [uint64]$NativeProcess.GetPeakJobMemoryUsed()
        terminology = [ordered]@{
            working_set_resident_bytes = 'resident working set; not committed virtual memory'
            private_committed_bytes = 'private committed virtual memory; not resident RAM'
        }
    }
}

function Assert-R1Command([string]$Phase, [string[]]$Command) {
    if ($Phase -notin $script:R1ExpectedPhaseOrder) { throw "unexpected R1 phase: $Phase" }
    if ($null -eq $Command -or $Command.Count -lt 1) { throw "empty command for phase $Phase" }
    if (-not [IO.Path]::IsPathRooted($Command[0])) { throw "phase executable must be absolute: $Phase" }
    if (-not [IO.File]::Exists($Command[0])) { throw "phase executable missing: $($Command[0])" }
    if ([IO.Path]::GetFileName($Command[0]).Equals('julia.exe', [StringComparison]::OrdinalIgnoreCase)) {
        [void](Resolve-R1ConcreteJulia $Command[0])
    }
    if ([IO.Path]::GetFullPath($Command[0]).Equals(
            'C:\Users\fshuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
            [StringComparison]::OrdinalIgnoreCase)) {
        [void](Resolve-R1FrozenPython $Command[0])
    }
}

function Sync-R1ExistingFile([string]$Path) {
    if (-not [IO.File]::Exists($Path)) { return $false }
    try {
        $Stream = [IO.FileStream]::new(
            $Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
            [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
        )
        try { $Stream.Flush($true) } finally { $Stream.Dispose() }
        return $true
    } catch {
        return $false
    }
}

function Invoke-R1MonitoredPhase {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Phase,
        [Parameter(Mandatory)][string[]]$Command,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$OutputDirectory,
        [Parameter(Mandatory)][datetimeoffset]$RunStarted,
        [Parameter(Mandatory)][double]$TotalHardWallSeconds,
        [Parameter(Mandatory)][int64]$MaxPrivateCommittedBytes,
        [Parameter(Mandatory)][int64]$MaxWorkingSetResidentBytes,
        [switch]$RequireChildMilestones,
        [string]$PreResumeSentinelPath = '',
        [string[]]$RequiredPhaseArtifacts = @(),
        [System.Collections.IDictionary]$EnvironmentOverrides = ([ordered]@{})
    )
    Assert-R1Command $Phase $Command
    $ExecutableItem = Get-Item -LiteralPath $Command[0] -Force
    if (($ExecutableItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "phase executable is a reparse point: $($Command[0])"
    }
    $ExecutableShaBefore = (Get-FileHash -LiteralPath $Command[0] -Algorithm SHA256).Hash.ToLowerInvariant()
    $ExecutableSizeBefore = [int64]$ExecutableItem.Length
    $ArgvSourceFiles = @()
    foreach ($Argument in @($Command | Select-Object -Skip 1)) {
        if ([string]$Argument -match '(?i)\.(py|jl|ps1)$' -and [IO.File]::Exists([string]$Argument)) {
            $SourcePath = [IO.Path]::GetFullPath([string]$Argument)
            $SourceItem = Get-Item -LiteralPath $SourcePath -Force
            if (($SourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "phase argv source is a reparse point: $SourcePath"
            }
            $ArgvSourceFiles += [ordered]@{
                path=$SourcePath; bytes=[int64]$SourceItem.Length
                sha256=(Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
    [IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
    $Stdout = Join-Path $OutputDirectory "$Phase.stdout.log"
    $Stderr = Join-Path $OutputDirectory "$Phase.stderr.log"
    $TelemetryPath = Join-Path $OutputDirectory "$Phase.telemetry.jsonl"
    $MilestonePath = Join-Path $OutputDirectory "$Phase.wrapper_milestones.jsonl"
    $ChildMilestonePath = Join-Path $OutputDirectory "$Phase.child_milestones.jsonl"
    foreach ($Path in @($Stdout, $Stderr, $TelemetryPath, $MilestonePath, $ChildMilestonePath, (Join-Path $OutputDirectory "$Phase.phase.json"))) {
        if ([IO.File]::Exists($Path)) { throw "R1 phase output already exists: $Path" }
    }
    $RequiredPhaseArtifacts = @($RequiredPhaseArtifacts | ForEach-Object { [IO.Path]::GetFullPath($_) })
    foreach ($Path in $RequiredPhaseArtifacts) {
        if ([IO.File]::Exists($Path)) { throw "required phase product must start absent: $Path" }
    }
    $NormalizedEnvironmentOverrides = [ordered]@{}
    foreach ($NameValue in @($EnvironmentOverrides.Keys)) {
        $Name = [string]$NameValue
        $Value = [string]$EnvironmentOverrides[$NameValue]
        if ($Name -in @('R1_CHILD_MILESTONE_PATH','R1_PRE_RESUME_SENTINEL_PATH','__PYVENV_LAUNCHER__')) {
            throw "phase environment override attempts to replace wrapper-owned binding: $Name"
        }
        if ($Name -notin $script:R1ExpectedHandoffEnvironmentNames -or $Value -notmatch '^[0-9a-f]{64}$') {
            throw "phase environment override is not an exact lower-case SHA-256 handoff: $Name"
        }
        $NormalizedEnvironmentOverrides[$Name] = $Value
    }

    $Arguments = if ($Command.Count -gt 1) { Join-R1NativeArguments $Command[1..($Command.Count - 1)] } else { '' }
    $CommandLine = (Join-R1NativeArguments @($Command[0])) + $(if ($Arguments.Length -gt 0) { " $Arguments" } else { '' })
    $Started = [DateTimeOffset]::UtcNow
    $Native = $null
    $TelemetryStream = $null
    $TelemetryWriter = $null
    $MilestoneStream = $null
    $MilestoneWriter = $null
    $PeakWorking = [int64]0
    $PeakPrivate = [int64]0
    $PeakJobMemory = [uint64]0
    $MaxProcessCount = 0
    $PeakPidWorking = [ordered]@{}
    $PeakPidPrivate = [ordered]@{}
    $Sequence = 0
    $StopReason = 'completed'
    $ExitCode = 1
    $AssignmentProof = [ordered]@{
        create_process_flag_suspended = $false
        job_assignment_succeeded_before_resume = $false
        resume_succeeded_after_assignment = $false
    }
    $RootPid = $null
    $TreeEmptyVerified = $false
    $OldMilestoneEnvironment = [Environment]::GetEnvironmentVariable('R1_CHILD_MILESTONE_PATH', 'Process')
    $OldSentinelEnvironment = [Environment]::GetEnvironmentVariable('R1_PRE_RESUME_SENTINEL_PATH', 'Process')
    $OldPythonLauncherEnvironment = [Environment]::GetEnvironmentVariable('__PYVENV_LAUNCHER__', 'Process')
    $OldPythonPathEnvironment = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    $OldPythonHomeEnvironment = [Environment]::GetEnvironmentVariable('PYTHONHOME', 'Process')
    $OldPythonNoUserSiteEnvironment = [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process')
    $OldOverrideEnvironment = [ordered]@{}
    foreach ($Name in $script:R1ExpectedHandoffEnvironmentNames) {
        $OldOverrideEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
    }
    try {
        $TelemetryStream = [IO.FileStream]::new($TelemetryPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        $TelemetryWriter = [IO.StreamWriter]::new($TelemetryStream, [Text.UTF8Encoding]::new($false))
        $TelemetryWriter.AutoFlush = $true
        $MilestoneStream = [IO.FileStream]::new($MilestonePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        $MilestoneWriter = [IO.StreamWriter]::new($MilestoneStream, [Text.UTF8Encoding]::new($false))
        $MilestoneWriter.AutoFlush = $true

        Add-R1JsonLine $MilestoneWriter ([ordered]@{
            milestone='launch_intent'; at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; executable=$Command[0]; command=@($Command)
        })

        [Environment]::SetEnvironmentVariable('R1_CHILD_MILESTONE_PATH', $ChildMilestonePath, 'Process')
        foreach ($Name in $script:R1ExpectedHandoffEnvironmentNames) {
            $PhaseValue = if ($NormalizedEnvironmentOverrides.Contains($Name)) {
                $NormalizedEnvironmentOverrides[$Name]
            } else { $null }
            [Environment]::SetEnvironmentVariable($Name, $PhaseValue, 'Process')
        }
        # Both direct Python phases and Julia/PythonCall phases inherit this
        # closed import environment.  Ambient user/site injection is never a
        # permitted part of the R1 runtime origin.
        [Environment]::SetEnvironmentVariable('PYTHONPATH', $null, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONHOME', $null, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', '1', 'Process')
        if ([IO.Path]::GetFullPath($Command[0]).Equals(
                'C:\Users\fshuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
                [StringComparison]::OrdinalIgnoreCase)) {
            [Environment]::SetEnvironmentVariable(
                '__PYVENV_LAUNCHER__',
                'D:\tetris-paper-plus\python-env\Scripts\python.exe',
                'Process'
            )
        }
        if (-not [string]::IsNullOrWhiteSpace($PreResumeSentinelPath)) {
            $PreResumeSentinelPath = [IO.Path]::GetFullPath($PreResumeSentinelPath)
            if ([IO.File]::Exists($PreResumeSentinelPath)) { throw 'pre-resume sentinel must start absent' }
            [Environment]::SetEnvironmentVariable('R1_PRE_RESUME_SENTINEL_PATH', $PreResumeSentinelPath, 'Process')
        }
        try {
            $Native = [R1Native.SuspendedJobProcess]::new(
                $Command[0], $CommandLine, [IO.Path]::GetFullPath($WorkingDirectory),
                $Stdout, $Stderr, $MaxPrivateCommittedBytes
            )
        } finally {
            [Environment]::SetEnvironmentVariable('R1_CHILD_MILESTONE_PATH', $OldMilestoneEnvironment, 'Process')
            [Environment]::SetEnvironmentVariable('R1_PRE_RESUME_SENTINEL_PATH', $OldSentinelEnvironment, 'Process')
            [Environment]::SetEnvironmentVariable('__PYVENV_LAUNCHER__', $OldPythonLauncherEnvironment, 'Process')
            [Environment]::SetEnvironmentVariable('PYTHONPATH', $OldPythonPathEnvironment, 'Process')
            [Environment]::SetEnvironmentVariable('PYTHONHOME', $OldPythonHomeEnvironment, 'Process')
            [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', $OldPythonNoUserSiteEnvironment, 'Process')
            foreach ($Name in @($OldOverrideEnvironment.Keys)) {
                [Environment]::SetEnvironmentVariable($Name, $OldOverrideEnvironment[$Name], 'Process')
            }
        }
        $RootPid = [int]$Native.ProcessId
        $AssignmentProof.create_process_flag_suspended = [bool]$Native.CreatedSuspended
        $AssignedIds = @($Native.GetProcessIds())
        $RootImage = [IO.Path]::GetFullPath($Native.GetImagePath())
        $ExpectedImage = [IO.Path]::GetFullPath($Command[0])
        $AssignmentProof.job_assignment_succeeded_before_resume = [bool]$Native.AssignedBeforeResume -and
            -not [bool]$Native.Resumed -and [int]$Native.ProcessId -in $AssignedIds
        $AssignmentProof.job_membership_verified_before_resume = [int]$Native.ProcessId -in $AssignedIds
        $AssignmentProof.root_image_matches_requested_executable = $RootImage.Equals(
            $ExpectedImage, [StringComparison]::OrdinalIgnoreCase
        )
        Add-R1JsonLine $MilestoneWriter ([ordered]@{
            milestone='root_created_suspended'
            at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; root_pid=[int]$Native.ProcessId
            created_suspended=$AssignmentProof.create_process_flag_suspended
        })
        Add-R1JsonLine $MilestoneWriter ([ordered]@{
            milestone='job_assigned_and_verified_before_resume'
            at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; root_pid=[int]$Native.ProcessId; proof=$AssignmentProof
        })
        if (-not $AssignmentProof.job_assignment_succeeded_before_resume -or
            -not $AssignmentProof.root_image_matches_requested_executable) {
            throw 'failed to prove Job assignment/image binding before resume'
        }
        if (-not [string]::IsNullOrWhiteSpace($PreResumeSentinelPath) -and
            [IO.File]::Exists($PreResumeSentinelPath)) {
            throw 'child code executed before ResumeThread'
        }
        $Sample = Get-R1ProcessTreeSample $Native $Sequence $Started
        Add-R1JsonLine $TelemetryWriter $Sample
        $Sequence += 1

        $Native.Resume()
        $AssignmentProof.resume_succeeded_after_assignment = [bool]$Native.Resumed
        Add-R1JsonLine $MilestoneWriter ([ordered]@{
            milestone='root_resumed_after_job_assignment'
            at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; root_pid=[int]$Native.ProcessId
            proof=$AssignmentProof
        })
        if (-not $AssignmentProof.resume_succeeded_after_assignment) { throw 'root process did not resume' }

        while ($true) {
            $Sample = Get-R1ProcessTreeSample $Native $Sequence $Started
            Add-R1JsonLine $TelemetryWriter $Sample
            $Sequence += 1
            [void](Sync-R1ExistingFile $ChildMilestonePath)
            if (-not $Sample.accounting_complete) {
                $StopReason = 'process-tree per-PID accounting incomplete'
                $Native.Terminate(139); break
            }
            $PeakWorking = [Math]::Max($PeakWorking, [int64]$Sample.process_tree_working_set_resident_bytes)
            $PeakPrivate = [Math]::Max($PeakPrivate, [int64]$Sample.process_tree_private_committed_bytes)
            $PeakJobMemory = [Math]::Max($PeakJobMemory, [uint64]$Sample.windows_job_peak_memory_used_bytes)
            $MaxProcessCount = [Math]::Max($MaxProcessCount, [int]$Sample.process_count)
            foreach ($Entry in @($Sample.processes)) {
                $PidKey = [string]$Entry.pid
                $OldWorking = if ($PeakPidWorking.Contains($PidKey)) { [int64]$PeakPidWorking[$PidKey] } else { [int64]0 }
                $OldPrivate = if ($PeakPidPrivate.Contains($PidKey)) { [int64]$PeakPidPrivate[$PidKey] } else { [int64]0 }
                $PeakPidWorking[$PidKey] = [Math]::Max($OldWorking, [int64]$Entry.working_set_resident_bytes)
                $PeakPidPrivate[$PidKey] = [Math]::Max($OldPrivate, [int64]$Entry.private_committed_bytes)
            }
            if ($Sample.process_tree_private_committed_bytes -gt $MaxPrivateCommittedBytes) {
                $StopReason = 'process-tree private committed memory limit exceeded'
                $Native.Terminate(137); break
            }
            if ($Sample.process_tree_working_set_resident_bytes -gt $MaxWorkingSetResidentBytes) {
                $StopReason = 'process-tree resident working-set limit exceeded'
                $Native.Terminate(138); break
            }
            if (([DateTimeOffset]::UtcNow - $RunStarted).TotalSeconds -gt $TotalHardWallSeconds) {
                $StopReason = 'R1 total hard wall exceeded'
                $Native.Terminate(124); break
            }
            $RootExited = $Native.Process.HasExited
            $LiveIds = @($Native.GetProcessIds())
            if ($RootExited -and $LiveIds.Count -eq 0) { break }
            Start-Sleep -Milliseconds 200
        }
        $Native.Process.WaitForExit()
        $Native.Process.Refresh()
        Add-R1JsonLine $MilestoneWriter ([ordered]@{
            milestone='root_exit_observed'; at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; root_pid=$RootPid; provisional_stop_reason=$StopReason
        })
        $EmptyDeadline = [DateTimeOffset]::UtcNow.AddSeconds(5)
        while (@($Native.GetProcessIds()).Count -ne 0 -and [DateTimeOffset]::UtcNow -lt $EmptyDeadline) {
            Start-Sleep -Milliseconds 50
        }
        $TreeEmptyVerified = @($Native.GetProcessIds()).Count -eq 0
        Add-R1JsonLine $MilestoneWriter ([ordered]@{
            milestone='job_tree_empty'; at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; verified=$TreeEmptyVerified
        })
        if (-not $TreeEmptyVerified -and $StopReason -eq 'completed') {
            $StopReason = 'Job process tree remained nonempty after root exit'
        }
        $ExitCode = if ($StopReason -eq 'completed') { [int]$Native.GetExitCode() } else { 1 }
        if ($ExitCode -ne 0 -and $StopReason -eq 'completed') { $StopReason = "process exit $ExitCode" }
    } catch {
        if ($null -ne $Native) {
            try { $Native.Terminate(125) } catch { }
        }
        $StopReason = "wrapper launch/monitor failure: $($_.Exception.Message)"
        $ExitCode = 1
    } finally {
        [Environment]::SetEnvironmentVariable('R1_CHILD_MILESTONE_PATH', $OldMilestoneEnvironment, 'Process')
        [Environment]::SetEnvironmentVariable('R1_PRE_RESUME_SENTINEL_PATH', $OldSentinelEnvironment, 'Process')
        [Environment]::SetEnvironmentVariable('__PYVENV_LAUNCHER__', $OldPythonLauncherEnvironment, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONPATH', $OldPythonPathEnvironment, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONHOME', $OldPythonHomeEnvironment, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', $OldPythonNoUserSiteEnvironment, 'Process')
        foreach ($Name in @($OldOverrideEnvironment.Keys)) {
            [Environment]::SetEnvironmentVariable($Name, $OldOverrideEnvironment[$Name], 'Process')
        }
        if ($null -ne $MilestoneWriter) {
            Add-R1JsonLine $MilestoneWriter ([ordered]@{
                milestone='process_monitor_closed'; at=[DateTimeOffset]::UtcNow.ToString('o')
                phase=$Phase; stop_reason=$StopReason; exit_code=$ExitCode
            })
            $MilestoneWriter.Dispose()
        }
        if ($null -ne $TelemetryWriter) { $TelemetryWriter.Dispose() }
        if ($null -ne $Native) { $Native.Dispose() }
    }
    $ChildMilestones = @()
    [void](Sync-R1ExistingFile $ChildMilestonePath)
    if ([IO.File]::Exists($ChildMilestonePath)) {
        try { $ChildMilestones = @(Get-Content -LiteralPath $ChildMilestonePath | ForEach-Object { $_ | ConvertFrom-Json }) }
        catch { $StopReason = "malformed child milestone JSONL: $($_.Exception.Message)"; $ExitCode = 1 }
    }
    if ($RequireChildMilestones) {
        $Names = @($ChildMilestones | ForEach-Object {
            if ($null -ne $_.PSObject.Properties['milestone']) { [string]$_.milestone }
            elseif ($null -ne $_.PSObject.Properties['stage']) { [string]$_.stage }
            elseif ($null -ne $_.PSObject.Properties['event']) { [string]$_.event }
            else { '' }
        })
        $ImportIndex = [array]::IndexOf($Names, 'imports_begin')
        $ImportCompleteIndices = @(@(
            [array]::IndexOf($Names, 'imports_complete'),
            [array]::IndexOf($Names, 'imports_end')
        ) | Where-Object { $_ -ge 0 })
        $TerminalNames = @('phase_complete', 'script_complete', 'artifact_published', 'assessment_complete')
        if ($Names.Count -lt 3 -or $Names[0] -cne 'script_enter' -or
            $ImportIndex -lt 1 -or
            ($ImportCompleteIndices.Count -gt 0 -and $ImportIndex -gt ($ImportCompleteIndices | Measure-Object -Minimum).Minimum) -or
            $Names[-1] -notin $TerminalNames) {
            $StopReason = 'required child milestones missing or out of order'
            $ExitCode = 1
        }
        if ($ChildMilestonePath -notin $Command) {
            $StopReason = 'production command is not bound to wrapper-owned child milestone path'
            $ExitCode = 1
        }
        $MilestonePids = @($ChildMilestones | ForEach-Object { [int]$_.pid } | Select-Object -Unique)
        if ($MilestonePids.Count -ne 1 -or $MilestonePids[0] -ne $RootPid) {
            $StopReason = 'child milestones are not bound to the monitored root PID'
            $ExitCode = 1
        }
    }
    $ExecutableShaAfter = (Get-FileHash -LiteralPath $Command[0] -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ExecutableShaAfter -cne $ExecutableShaBefore -or
        [int64](Get-Item -LiteralPath $Command[0]).Length -ne $ExecutableSizeBefore) {
        $StopReason = 'phase executable changed during execution'
        $ExitCode = 1
    }
    foreach ($Source in $ArgvSourceFiles) {
        if (-not [IO.File]::Exists([string]$Source.path) -or
            [int64](Get-Item -LiteralPath $Source.path).Length -ne [int64]$Source.bytes -or
            (Get-FileHash -LiteralPath $Source.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne [string]$Source.sha256) {
            $StopReason = "phase argv source changed during execution: $($Source.path)"
            $ExitCode = 1
        }
    }
    $FinalMilestoneStream = [IO.FileStream]::new(
        $MilestonePath, [IO.FileMode]::Append, [IO.FileAccess]::Write, [IO.FileShare]::Read
    )
    $FinalMilestoneWriter = [IO.StreamWriter]::new($FinalMilestoneStream, [Text.UTF8Encoding]::new($false))
    try {
        Add-R1JsonLine $FinalMilestoneWriter ([ordered]@{
            milestone='phase_result_ready'; at=[DateTimeOffset]::UtcNow.ToString('o')
            phase=$Phase; stop_reason=$StopReason; exit_code=$ExitCode
        })
    } finally { $FinalMilestoneWriter.Dispose() }
    $ArtifactRecords = @()
    foreach ($Path in @($Stdout, $Stderr, $TelemetryPath, $MilestonePath, $ChildMilestonePath)) {
        if ([IO.File]::Exists($Path)) {
            $ArtifactRecords += [ordered]@{
                path=[IO.Path]::GetFullPath($Path)
                bytes=[int64](Get-Item -LiteralPath $Path).Length
                sha256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }
    $PhaseProducts = @()
    foreach ($Path in $RequiredPhaseArtifacts) {
        if (-not [IO.File]::Exists($Path)) {
            $ProductFailure = "required phase product missing: $Path"
            $StopReason = if ($StopReason -ceq 'completed') { $ProductFailure } else { "$StopReason; $ProductFailure" }
            $ExitCode = 1
            continue
        }
        $PhaseProducts += [ordered]@{
            path=$Path; bytes=[int64](Get-Item -LiteralPath $Path).Length
            sha256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $Record = [ordered]@{
        phase=$Phase
        exit_code=$ExitCode
        stop_reason=$StopReason
        seconds=([DateTimeOffset]::UtcNow - $Started).TotalSeconds
        command=@($Command)
        root_executable=$Command[0]
        root_executable_size_bytes=$ExecutableSizeBefore
        root_executable_sha256_before=$ExecutableShaBefore
        root_executable_sha256_after=$ExecutableShaAfter
        root_executable_unchanged=$ExecutableShaAfter -ceq $ExecutableShaBefore
        argv_source_files=$ArgvSourceFiles
        root_pid=$RootPid
        assignment_proof=$AssignmentProof
        telemetry_schema='r1-process-tree-telemetry-v1'
        telemetry_interval_target_ms=200
        telemetry_samples=$Sequence
        maximum_observed_process_count=$MaxProcessCount
        telemetry_path=$TelemetryPath
        wrapper_milestones_path=$MilestonePath
        child_milestones_path=$ChildMilestonePath
        child_milestones_required=[bool]$RequireChildMilestones
        child_milestone_binding='exact argv path'
        child_milestone_count=$ChildMilestones.Count
        environment_sha256_handoffs=$NormalizedEnvironmentOverrides
        job_tree_empty_verified=$TreeEmptyVerified
        peak_process_tree_working_set_resident_bytes=$PeakWorking
        peak_process_tree_private_committed_bytes=$PeakPrivate
        windows_job_peak_memory_used_bytes=$PeakJobMemory
        per_pid_peak_working_set_resident_bytes=$PeakPidWorking
        per_pid_peak_private_committed_bytes=$PeakPidPrivate
        max_working_set_resident_bytes=$MaxWorkingSetResidentBytes
        max_private_committed_bytes=$MaxPrivateCommittedBytes
        stdout=$Stdout
        stderr=$Stderr
        closed_artifacts=@($ArtifactRecords)
        required_phase_product_count=$RequiredPhaseArtifacts.Count
        phase_products=@($PhaseProducts)
        memory_metric_scope='whole Windows Job Object process tree'
        os_job_private_commit_cap_enabled=$true
        private_committed_is_not_resident_ram=$true
        optional_development_phase_present=$false
        validation_seed_used=$false
        sealed_test_seed_used=$false
        game_run=$false
        pre_resume_sentinel_path=if ([string]::IsNullOrWhiteSpace($PreResumeSentinelPath)) { $null } else { $PreResumeSentinelPath }
        pre_resume_sentinel_absent_before_resume=if ([string]::IsNullOrWhiteSpace($PreResumeSentinelPath)) { $null } else { $true }
    }
    $PhaseResultPath = Join-Path $OutputDirectory "$Phase.phase.json"
    Write-R1JsonDurableAtomic $PhaseResultPath $Record
    $Record['phase_result_path'] = $PhaseResultPath
    $Record['phase_result_sha256'] = (Get-FileHash -LiteralPath $PhaseResultPath -Algorithm SHA256).Hash.ToLowerInvariant()
    return $Record
}

function New-R1AuthoritativeFailureAssessment([string]$Reason, [string]$Stage) {
    return [ordered]@{
        status='assessment-fail'
        success=$false
        failures=@($Reason)
        promotion='R1-calibration-rejected'
        authority='wrapper-fail-closed'
        failure_stage=$Stage
        game_strength_evidence=$false
        model_beat_claim=$false
        game_evaluation_authorized=$false
        development_authorized=$false
        validation_seed_used=$false
        sealed_test_seed_used=$false
        sealed_test_authorized=$false
        game_run=$false
    }
}

function Invoke-R1PhasePipeline {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Commands,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$OutputDirectory,
        [Parameter(Mandatory)][System.Collections.IDictionary]$PhaseArtifacts,
        [System.Collections.IDictionary]$PhaseEnvironment = ([ordered]@{}),
        [scriptblock]$BoundaryIntegrityCheck = $null,
        [double]$TotalHardWallSeconds = 3900.0
    )
    if (($Commands.Keys -join "`n") -cne ($script:R1ExpectedPhaseOrder -join "`n")) {
        throw 'R1 command plan must contain exactly the preregistered phase order'
    }
    if ($Commands.Keys -contains 'development' -or $Commands.Keys -contains 'evaluate_development') {
        throw 'development is disabled in the R1 calibration one-shot'
    }
    if (($PhaseArtifacts.Keys -join "`n") -cne ($script:R1ExpectedPhaseOrder -join "`n")) {
        throw 'R1 phase artifact plan must contain exactly the preregistered phase order'
    }
    foreach ($PhaseName in @($PhaseEnvironment.Keys)) {
        if ([string]$PhaseName -notin $script:R1ExpectedPhaseOrder -or
            $PhaseEnvironment[$PhaseName] -isnot [System.Collections.IDictionary]) {
            throw "invalid R1 per-phase environment plan entry: $PhaseName"
        }
    }
    $RunStarted = [DateTimeOffset]::UtcNow
    $Phases = @()
    $Skipped = @()
    $Failures = @()
    $ProducerLedger = [ordered]@{}
    $IntegrityBoundaryChecks = 0
    $PipelineOpen = $true
    $FinalizerDisposition = 'not_reached'
    $AuthoritativeFailureAssessment = $null
    foreach ($Phase in $script:R1ExpectedPhaseOrder) {
        if ($Phase -ne 'finalize_assessment' -and -not $PipelineOpen) {
            $Skipped += [ordered]@{ phase=$Phase; reason='prior required phase failed; no resume or rescue' }
            continue
        }
        if ($null -ne $BoundaryIntegrityCheck) {
            try {
                & $BoundaryIntegrityCheck $Phase
                $IntegrityBoundaryChecks += 1
            } catch {
                $IntegrityFailure = "phase $Phase boundary integrity check failed: $($_.Exception.Message)"
                $Failures += $IntegrityFailure
                $Skipped += [ordered]@{ phase=$Phase; reason=$IntegrityFailure }
                $PipelineOpen = $false
                if ($Phase -ceq 'finalize_assessment') {
                    $FinalizerDisposition = 'blocked_by_boundary_integrity_failure'
                    $AuthoritativeFailureAssessment = New-R1AuthoritativeFailureAssessment `
                        $IntegrityFailure 'before_finalize_assessment'
                }
                continue
            }
        }
        $CollectionLike = $Phase -in @('collect_training', 'collect_calibration')
        $PrivateLimit = if ($Phase -eq 'fit_ridge') { [int64](1GB) } else { [int64](4GB) }
        # The scientific contract caps collector resident memory at 2 GiB. Fit
        # has only a 1-GiB private-commit cap; its working set remains measured,
        # not silently promoted into a second preregistered rejection criterion.
        $WorkingLimit = if ($CollectionLike) { [int64](2GB) } else { [int64]::MaxValue }
        $ResolvedEnvironment = [ordered]@{}
        if ($PhaseEnvironment.Contains($Phase)) {
            try {
                $ResolvedEnvironment = Resolve-R1PhaseEnvironmentHandoffs `
                    -Phase $Phase -Plan $PhaseEnvironment[$Phase] -ProducerLedger $ProducerLedger
            } catch {
                $HandoffFailure = "phase $Phase SHA handoff resolution failed: $($_.Exception.Message)"
                $Failures += $HandoffFailure
                if ($Phase -ne 'finalize_assessment') {
                    $Skipped += [ordered]@{ phase=$Phase; reason=$HandoffFailure }
                    $PipelineOpen = $false
                    continue
                }
                # The strict finalizer is deliberately still launched without
                # an invalid/unproven handoff; it will publish assessment-fail.
                $ResolvedEnvironment = [ordered]@{}
            }
        }
        try {
            if ($Phase -ceq 'finalize_assessment') { $FinalizerDisposition = 'launch_attempted' }
            $Record = Invoke-R1MonitoredPhase -Phase $Phase -Command @($Commands[$Phase]) `
                -WorkingDirectory $WorkingDirectory -OutputDirectory $OutputDirectory `
                -RunStarted $RunStarted -TotalHardWallSeconds $TotalHardWallSeconds `
                -MaxPrivateCommittedBytes $PrivateLimit -MaxWorkingSetResidentBytes $WorkingLimit `
                -RequireChildMilestones -RequiredPhaseArtifacts @($PhaseArtifacts[$Phase]) `
                -EnvironmentOverrides $ResolvedEnvironment
        } catch {
            $LaunchFailure = "phase $Phase wrapper invocation failed before a phase record: $($_.Exception.Message)"
            $Failures += $LaunchFailure
            $Skipped += [ordered]@{ phase=$Phase; reason=$LaunchFailure }
            $PipelineOpen = $false
            if ($Phase -ceq 'finalize_assessment') {
                $FinalizerDisposition = 'invocation_failure'
                $AuthoritativeFailureAssessment = New-R1AuthoritativeFailureAssessment `
                    $LaunchFailure 'invoke_finalize_assessment'
            }
            continue
        }
        $Phases += $Record
        try {
            if ($Record.exit_code -eq 0 -and $Record.stop_reason -ceq 'completed') {
                # Validate the whole phase-product set before committing any
                # entry, so a wrapper-side postprocess failure cannot leave a
                # partially blessed producer ledger.
                $StagedProducts = @()
                $StagedPaths = [Collections.Generic.HashSet[string]]::new(
                    [StringComparer]::OrdinalIgnoreCase
                )
                foreach ($Product in @($Record.phase_products)) {
                    $ProductPath = [IO.Path]::GetFullPath([string]$Product.path)
                    if ($ProducerLedger.Contains($ProductPath) -or -not $StagedPaths.Add($ProductPath)) {
                        throw "duplicate producer-completion ledger path: $ProductPath"
                    }
                    $StagedProducts += [ordered]@{
                        phase=$Phase
                        path=$ProductPath
                        sha256=[string]$Product.sha256
                        captured_after_tree_empty=[bool]$Record.job_tree_empty_verified
                        phase_result_sha256=[string]$Record.phase_result_sha256
                    }
                }
                foreach ($StagedProduct in $StagedProducts) {
                    $ProducerLedger[[string]$StagedProduct.path] = $StagedProduct
                }
            }
            if ($Record.exit_code -ne 0) {
                $Failures += "phase $Phase failed: $($Record.stop_reason)"
                $PipelineOpen = $false
                if ($Phase -ceq 'finalize_assessment') {
                    $FinalizerDisposition = 'launched_failed'
                    $AuthoritativeFailureAssessment = New-R1AuthoritativeFailureAssessment `
                        "phase finalize_assessment failed: $($Record.stop_reason)" 'finalize_assessment_process'
                }
            } elseif ($Phase -ceq 'finalize_assessment') {
                $FinalizerDisposition = 'launched_completed'
            }
        } catch {
            $PostprocessFailure = "phase $Phase wrapper postprocess failed after phase record: $($_.Exception.Message)"
            $Failures += $PostprocessFailure
            $PipelineOpen = $false
            if ($Phase -ceq 'finalize_assessment') {
                $FinalizerDisposition = 'postprocess_failure'
                $AuthoritativeFailureAssessment = New-R1AuthoritativeFailureAssessment `
                    $PostprocessFailure 'finalize_assessment_postprocess'
            }
        }
    }
    if ($null -ne $BoundaryIntegrityCheck) {
        try {
            & $BoundaryIntegrityCheck 'terminal'
            $IntegrityBoundaryChecks += 1
        } catch {
            $TerminalIntegrityFailure = "terminal boundary integrity check failed: $($_.Exception.Message)"
            $Failures += $TerminalIntegrityFailure
            $FinalizerDisposition = 'terminal_integrity_failure'
            $AuthoritativeFailureAssessment = New-R1AuthoritativeFailureAssessment `
                $TerminalIntegrityFailure 'after_finalize_assessment'
        }
    }
    return [ordered]@{
        complete=$true
        expected_phase_order=@($script:R1ExpectedPhaseOrder)
        ended_at=[DateTimeOffset]::UtcNow.ToString('o')
        wall_seconds=([DateTimeOffset]::UtcNow - $RunStarted).TotalSeconds
        stop_reason=if ($Failures.Count -eq 0) { 'completed' } else { 'failed' }
        failures=@($Failures)
        phases=@($Phases)
        skipped_phases=@($Skipped)
        producer_completion_ledger=@($ProducerLedger.Values)
        finalizer_disposition=$FinalizerDisposition
        authoritative_failure_assessment=$AuthoritativeFailureAssessment
        integrity_boundary_checks=$IntegrityBoundaryChecks
        polling_interval_target_ms=200
        process_creation='CreateProcessW CREATE_SUSPENDED; assign Job; ResumeThread'
        memory_accounting='per-PID JSONL plus process-tree aggregate; private committed separated from resident working set'
        partial_phase_resume_allowed=$false
        rescue_allowed=$false
        optional_development_enabled=$false
        development_seed_used=$false
        validation_seed_used=$false
        sealed_test_seed_used=$false
        game_run=$false
    }
}
