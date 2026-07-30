[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9_.-]+$")]
    [string]$RunId,

    [ValidateRange(1, [int]::MaxValue)]
    [int]$ExpectedUpdates = 100000,

    [ValidateRange(2, 1024)]
    [int]$JuliaThreads = 20,

    [ValidateRange(1, 3600)]
    [int]$PollSeconds = 30,

    [ValidateRange(2, 86400)]
    [int]$StallSeconds = 600,

    [ValidateRange(2, 86400)]
    [int]$VerifierTimeoutSeconds = 3600,

    [ValidateSet("scratch", "resume", "finalize-only")]
    [string]$StartMode = "scratch",

    [switch]$VerifyOnly,
    [Alias("LaunchManifestSha256")]
    [string]$ExpectedLaunchManifestSha256 = "",
    [string]$ResumeCheckpoint = "",
    [string]$ResumeSha256 = "",
    [ValidateRange(-1, [int]::MaxValue)]
    [int]$ResumeUpdate = -1,
    [string]$JuliaExecutable = "",
    [string]$ProjectPath = "",
    [string]$TrainingScript = "",
    [string]$VerifierScript = "",
    [string]$DatasetPath = "",
    [string]$OutputRoot = "",
    [string]$WorkingDirectory = "",
    [string]$MutexName = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CanonicalRunIdPattern =
    "^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$"
if (
    $RunId -notmatch $CanonicalRunIdPattern -or
    $RunId -in @(".", "..") -or
    [IO.Path]::GetFileName($RunId) -cne $RunId -or
    $RunId -match
        "^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:[.].*)?$"
) {
    throw (
        "RunId must be a canonical Windows-safe basename of 1-128 ASCII " +
        "characters, beginning and ending with an alphanumeric character"
    )
}

function New-KillOnCloseJob {
    if ($null -eq ("SwsnnNativeJob" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

public static class SwsnnNativeJob {
    const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    const UInt32 CREATE_SUSPENDED = 0x00000004;
    const UInt32 CREATE_NO_WINDOW = 0x08000000;
    const UInt32 STARTF_USESTDHANDLES = 0x00000100;
    const UInt32 GENERIC_READ = 0x80000000;
    const UInt32 GENERIC_WRITE = 0x40000000;
    const UInt32 FILE_SHARE_READ = 0x00000001;
    const UInt32 FILE_SHARE_WRITE = 0x00000002;
    const UInt32 CREATE_ALWAYS = 2;
    const UInt32 OPEN_EXISTING = 3;
    const UInt32 FILE_ATTRIBUTE_NORMAL = 0x00000080;
    static readonly IntPtr INVALID_HANDLE_VALUE = new IntPtr(-1);
    static readonly Dictionary<UInt32, IntPtr> ProcessHandles =
        new Dictionary<UInt32, IntPtr>();

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public Int64 PerProcessUserTimeLimit;
        public Int64 PerJobUserTimeLimit;
        public UInt32 LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public UInt32 ActiveProcessLimit;
        public UIntPtr Affinity;
        public UInt32 PriorityClass;
        public UInt32 SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS {
        public UInt64 ReadOperationCount;
        public UInt64 WriteOperationCount;
        public UInt64 OtherOperationCount;
        public UInt64 ReadTransferCount;
        public UInt64 WriteTransferCount;
        public UInt64 OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_ACCOUNTING_INFORMATION {
        public Int64 TotalUserTime;
        public Int64 TotalKernelTime;
        public Int64 ThisPeriodTotalUserTime;
        public Int64 ThisPeriodTotalKernelTime;
        public UInt32 TotalPageFaultCount;
        public UInt32 TotalProcesses;
        public UInt32 ActiveProcesses;
        public UInt32 TotalTerminatedProcesses;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct SECURITY_ATTRIBUTES {
        public Int32 nLength;
        public IntPtr lpSecurityDescriptor;
        [MarshalAs(UnmanagedType.Bool)]
        public bool bInheritHandle;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct STARTUPINFO {
        public UInt32 cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public UInt32 dwX;
        public UInt32 dwY;
        public UInt32 dwXSize;
        public UInt32 dwYSize;
        public UInt32 dwXCountChars;
        public UInt32 dwYCountChars;
        public UInt32 dwFillAttribute;
        public UInt32 dwFlags;
        public UInt16 wShowWindow;
        public UInt16 cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_INFORMATION {
        public IntPtr hProcess;
        public IntPtr hThread;
        public UInt32 dwProcessId;
        public UInt32 dwThreadId;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObject(
        IntPtr jobAttributes,
        string name
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        UInt32 informationLength
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(
        IntPtr job,
        IntPtr process
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool TerminateJobObject(
        IntPtr job,
        UInt32 exitCode
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool QueryInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        UInt32 informationLength,
        out UInt32 returnLength
    );

    [DllImport(
        "kernel32.dll",
        CharSet = CharSet.Unicode,
        SetLastError = true
    )]
    static extern IntPtr CreateFile(
        string fileName,
        UInt32 desiredAccess,
        UInt32 shareMode,
        ref SECURITY_ATTRIBUTES securityAttributes,
        UInt32 creationDisposition,
        UInt32 flagsAndAttributes,
        IntPtr templateFile
    );

    [DllImport(
        "kernel32.dll",
        CharSet = CharSet.Unicode,
        SetLastError = true
    )]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool CreateProcess(
        string applicationName,
        StringBuilder commandLine,
        IntPtr processAttributes,
        IntPtr threadAttributes,
        [MarshalAs(UnmanagedType.Bool)] bool inheritHandles,
        UInt32 creationFlags,
        IntPtr environment,
        string currentDirectory,
        ref STARTUPINFO startupInfo,
        out PROCESS_INFORMATION processInformation
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern UInt32 ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern UInt32 WaitForSingleObject(
        IntPtr handle,
        UInt32 milliseconds
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool GetExitCodeProcess(
        IntPtr process,
        out UInt32 exitCode
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool TerminateProcess(
        IntPtr process,
        UInt32 exitCode
    );

    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr handle);

    public static IntPtr CreateKillOnCloseJob() {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "CreateJobObject failed"
            );
        }
        var information =
            new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        information.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        int size = Marshal.SizeOf(information);
        IntPtr pointer = Marshal.AllocHGlobal(size);
        try {
            Marshal.StructureToPtr(information, pointer, false);
            if (!SetInformationJobObject(job, 9, pointer, (UInt32)size)) {
                int error = Marshal.GetLastWin32Error();
                CloseHandle(job);
                throw new Win32Exception(
                    error,
                    "SetInformationJobObject failed"
                );
            }
        }
        finally {
            Marshal.FreeHGlobal(pointer);
        }
        return job;
    }

    public static void TerminateJobAndWaitForEmpty(
        IntPtr job,
        UInt32 exitCode,
        Int32 timeoutMilliseconds
    ) {
        if (job == IntPtr.Zero) {
            return;
        }
        if (!TerminateJobObject(job, exitCode)) {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "TerminateJobObject failed"
            );
        }
        int size = Marshal.SizeOf(
            typeof(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)
        );
        IntPtr pointer = Marshal.AllocHGlobal(size);
        try {
            DateTime deadline =
                DateTime.UtcNow.AddMilliseconds(timeoutMilliseconds);
            while (true) {
                UInt32 returnLength;
                if (!QueryInformationJobObject(
                    job,
                    1,
                    pointer,
                    (UInt32)size,
                    out returnLength
                )) {
                    throw new Win32Exception(
                        Marshal.GetLastWin32Error(),
                        "QueryInformationJobObject failed"
                    );
                }
                var accounting =
                    (JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)
                    Marshal.PtrToStructure(
                        pointer,
                        typeof(JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)
                    );
                if (accounting.ActiveProcesses == 0) {
                    return;
                }
                if (DateTime.UtcNow >= deadline) {
                    throw new TimeoutException(
                        "Job did not become quiescent after termination"
                    );
                }
                Thread.Sleep(10);
            }
        }
        finally {
            Marshal.FreeHGlobal(pointer);
        }
    }

    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "AssignProcessToJobObject failed"
            );
        }
    }

    static string QuoteArgument(string value) {
        if (value == null) {
            throw new ArgumentNullException("value");
        }
        if (value.Length == 0) {
            return "\"\"";
        }
        bool requiresQuotes = false;
        for (int index = 0; index < value.Length; ++index) {
            char current = value[index];
            if (Char.IsWhiteSpace(current) || current == '"') {
                requiresQuotes = true;
                break;
            }
        }
        if (!requiresQuotes) {
            return value;
        }

        var result = new StringBuilder();
        result.Append('"');
        int backslashes = 0;
        for (int index = 0; index < value.Length; ++index) {
            char current = value[index];
            if (current == '\\') {
                ++backslashes;
                continue;
            }
            if (current == '"') {
                result.Append('\\', backslashes * 2 + 1);
                result.Append('"');
                backslashes = 0;
                continue;
            }
            result.Append('\\', backslashes);
            backslashes = 0;
            result.Append(current);
        }
        result.Append('\\', backslashes * 2);
        result.Append('"');
        return result.ToString();
    }

    static StringBuilder BuildCommandLine(
        string executable,
        string[] arguments
    ) {
        var commandLine = new StringBuilder(QuoteArgument(executable));
        if (arguments != null) {
            foreach (string argument in arguments) {
                commandLine.Append(' ');
                commandLine.Append(QuoteArgument(argument));
            }
        }
        return commandLine;
    }

    static IntPtr OpenInheritedFile(
        string path,
        UInt32 desiredAccess,
        UInt32 creationDisposition,
        ref SECURITY_ATTRIBUTES securityAttributes
    ) {
        IntPtr handle = CreateFile(
            path,
            desiredAccess,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            ref securityAttributes,
            creationDisposition,
            FILE_ATTRIBUTE_NORMAL,
            IntPtr.Zero
        );
        if (handle == INVALID_HANDLE_VALUE) {
            throw new Win32Exception(
                Marshal.GetLastWin32Error(),
                "CreateFile failed for " + path
            );
        }
        return handle;
    }

    public static UInt32 StartAssignedProcess(
        IntPtr job,
        string executable,
        string[] arguments,
        string workingDirectory,
        string standardOutputPath,
        string standardErrorPath
    ) {
        var securityAttributes = new SECURITY_ATTRIBUTES();
        securityAttributes.nLength =
            Marshal.SizeOf(typeof(SECURITY_ATTRIBUTES));
        securityAttributes.lpSecurityDescriptor = IntPtr.Zero;
        securityAttributes.bInheritHandle = true;

        IntPtr standardInput = INVALID_HANDLE_VALUE;
        IntPtr standardOutput = INVALID_HANDLE_VALUE;
        IntPtr standardError = INVALID_HANDLE_VALUE;
        var processInformation = new PROCESS_INFORMATION();
        bool created = false;
        bool processHandleRetained = false;
        try {
            standardInput = OpenInheritedFile(
                "NUL",
                GENERIC_READ,
                OPEN_EXISTING,
                ref securityAttributes
            );
            standardOutput = OpenInheritedFile(
                standardOutputPath,
                GENERIC_WRITE,
                CREATE_ALWAYS,
                ref securityAttributes
            );
            standardError = OpenInheritedFile(
                standardErrorPath,
                GENERIC_WRITE,
                CREATE_ALWAYS,
                ref securityAttributes
            );

            var startupInfo = new STARTUPINFO();
            startupInfo.cb = (UInt32)Marshal.SizeOf(typeof(STARTUPINFO));
            startupInfo.dwFlags = STARTF_USESTDHANDLES;
            startupInfo.hStdInput = standardInput;
            startupInfo.hStdOutput = standardOutput;
            startupInfo.hStdError = standardError;
            StringBuilder commandLine =
                BuildCommandLine(executable, arguments);

            if (!CreateProcess(
                executable,
                commandLine,
                IntPtr.Zero,
                IntPtr.Zero,
                true,
                CREATE_SUSPENDED | CREATE_NO_WINDOW,
                IntPtr.Zero,
                workingDirectory,
                ref startupInfo,
                out processInformation
            )) {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "CreateProcess failed"
                );
            }
            created = true;
            if (!AssignProcessToJobObject(job, processInformation.hProcess)) {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "AssignProcessToJobObject failed before ResumeThread"
                );
            }
            if (ResumeThread(processInformation.hThread) == UInt32.MaxValue) {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "ResumeThread failed"
                );
            }
            lock (ProcessHandles) {
                ProcessHandles.Add(
                    processInformation.dwProcessId,
                    processInformation.hProcess
                );
            }
            processHandleRetained = true;
            return processInformation.dwProcessId;
        }
        catch {
            if (created && processInformation.hProcess != IntPtr.Zero) {
                TerminateProcess(processInformation.hProcess, 1);
            }
            throw;
        }
        finally {
            if (processInformation.hThread != IntPtr.Zero) {
                CloseHandle(processInformation.hThread);
            }
            if (
                processInformation.hProcess != IntPtr.Zero &&
                !processHandleRetained
            ) {
                CloseHandle(processInformation.hProcess);
            }
            if (standardInput != INVALID_HANDLE_VALUE) {
                CloseHandle(standardInput);
            }
            if (standardOutput != INVALID_HANDLE_VALUE) {
                CloseHandle(standardOutput);
            }
            if (standardError != INVALID_HANDLE_VALUE) {
                CloseHandle(standardError);
            }
        }
    }

    public static Int32 WaitForExitAndGetCode(UInt32 processId) {
        IntPtr process;
        lock (ProcessHandles) {
            if (!ProcessHandles.TryGetValue(processId, out process)) {
                throw new InvalidOperationException(
                    "No retained process handle for PID " + processId
                );
            }
        }
        try {
            UInt32 waitResult =
                WaitForSingleObject(process, UInt32.MaxValue);
            if (waitResult != 0) {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "WaitForSingleObject failed for PID " + processId
                );
            }
            UInt32 exitCode;
            if (!GetExitCodeProcess(process, out exitCode)) {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "GetExitCodeProcess failed for PID " + processId
                );
            }
            return unchecked((Int32)exitCode);
        }
        finally {
            lock (ProcessHandles) {
                ProcessHandles.Remove(processId);
            }
            CloseHandle(process);
        }
    }
}
"@
    }
    return [SwsnnNativeJob]::CreateKillOnCloseJob()
}

function Start-AssignedProcess {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Job,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StandardOutputPath,
        [Parameter(Mandatory = $true)][string]$StandardErrorPath
    )

    $processId = [SwsnnNativeJob]::StartAssignedProcess(
        $Job,
        $Executable,
        $Arguments,
        $WorkingDirectory,
        $StandardOutputPath,
        $StandardErrorPath
    )
    return [Diagnostics.Process]::GetProcessById([int]$processId)
}

function Add-ProcessToKillOnCloseJob {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Job,
        [Parameter(Mandatory = $true)]$Process
    )

    try {
        [SwsnnNativeJob]::Assign($Job, $Process.Handle)
    }
    catch {
        try {
            Stop-ChildProcessTree -Process $Process | Out-Null
        }
        catch {
        }
        throw
    }
}

function Resolve-ExecutablePath {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ([IO.Path]::IsPathRooted($Value)) {
        if (-not (Test-Path -LiteralPath $Value -PathType Leaf)) {
            throw "Executable does not exist: $Value"
        }
        return (Resolve-Path -LiteralPath $Value).Path
    }
    $command = Get-Command -Name $Value -CommandType Application `
        -ErrorAction Stop |
        Select-Object -First 1
    return $command.Source
}

function Move-AtomicReplace {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if ($null -eq ("SwsnnNativeFile" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class SwsnnNativeFile {
    const UInt32 MOVEFILE_REPLACE_EXISTING = 0x00000001;
    const UInt32 MOVEFILE_WRITE_THROUGH = 0x00000008;

    [DllImport(
        "kernel32.dll",
        CharSet = CharSet.Unicode,
        SetLastError = true
    )]
    static extern bool MoveFileEx(
        string existingFileName,
        string newFileName,
        UInt32 flags
    );

    public static void Replace(string source, string destination) {
        if (!MoveFileEx(
            source,
            destination,
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        )) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
    }
}
"@
    }
    [SwsnnNativeFile]::Replace(
        [IO.Path]::GetFullPath($Source),
        [IO.Path]::GetFullPath($Destination)
    )
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )

    $destination = [IO.Path]::GetFullPath($Path)
    $directory = [IO.Path]::GetDirectoryName($destination)
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $temporary = "$destination.tmp.$PID"
    $encoding = [Text.UTF8Encoding]::new($false)
    try {
        $json = $Value | ConvertTo-Json -Depth 32
        [IO.File]::WriteAllText($temporary, "$json`n", $encoding)
        Move-AtomicReplace `
            -Source $temporary `
            -Destination $destination
    }
    catch {
        if ([IO.File]::Exists($temporary)) {
            [IO.File]::Delete($temporary)
        }
        throw
    }
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $stream = [IO.File]::Open(
        $resolved,
        [IO.FileMode]::Open,
        [IO.FileAccess]::Read,
        [IO.FileShare]::Read
    )
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash($stream)
        return (
            [BitConverter]::ToString($digest).Replace("-", "")
        ).ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Get-Utf8Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Value)

    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Value)
        return -join (
            $sha256.ComputeHash($bytes) |
                ForEach-Object { $_.ToString("x2") }
        )
    }
    finally {
        $sha256.Dispose()
    }
}

function Quote-NativeArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function New-SuggestedRecoveryRunId {
    param([Parameter(Mandatory = $true)][string]$RecoveryStartMode)

    $suffix = (
        "-recovery-" +
        $RecoveryStartMode +
        "-" +
        [DateTime]::UtcNow.ToString(
            "yyyyMMddTHHmmssfffffffZ",
            [Globalization.CultureInfo]::InvariantCulture
        )
    )
    $maximumPrefixLength = 128 - $suffix.Length
    if ($maximumPrefixLength -lt 1) {
        throw "recovery RunId suffix exceeds the canonical length limit"
    }
    $prefix = if ($RunId.Length -le $maximumPrefixLength) {
        $RunId
    }
    else {
        $RunId.Substring(0, $maximumPrefixLength)
    }
    $prefix = $prefix.TrimEnd(".", "-", "_")
    if ([string]::IsNullOrWhiteSpace($prefix)) {
        $prefix = "run"
    }
    $candidate = $prefix + $suffix
    if (
        $candidate.Length -gt 128 -or
        $candidate -notmatch $CanonicalRunIdPattern
    ) {
        throw "could not construct a canonical recovery RunId"
    }
    return $candidate
}

function Assert-JsonProperties {
    param(
        [AllowNull()]$Value,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if ($null -eq $Value) {
        throw "$Context is null"
    }
    $availableNames = if ($Value -is [Collections.IDictionary]) {
        @($Value.Keys | ForEach-Object { [string]$_ })
    }
    else {
        @($Value.PSObject.Properties.Name)
    }
    foreach ($name in $Names) {
        if ($name -notin $availableNames) {
            throw "$Context lacks required property '$name'"
        }
    }
}

function Assert-ExactJsonProperties {
    param(
        [AllowNull()]$Value,
        [Parameter(Mandatory = $true)][string[]]$Names,
        [Parameter(Mandatory = $true)][string]$Context
    )

    Assert-JsonProperties -Value $Value -Names $Names -Context $Context
    $availableNames = if ($Value -is [Collections.IDictionary]) {
        @($Value.Keys | ForEach-Object { [string]$_ })
    }
    else {
        @($Value.PSObject.Properties.Name)
    }
    if ($availableNames.Count -ne $Names.Count) {
        throw "$Context has unexpected properties"
    }
    foreach ($availableName in $availableNames) {
        if ($availableName -cnotin $Names) {
            throw "$Context has unexpected property '$availableName'"
        }
    }
}

function Test-JsonInteger {
    param([AllowNull()]$Value)

    return (
        $Value -is [byte] -or
        $Value -is [sbyte] -or
        $Value -is [int16] -or
        $Value -is [uint16] -or
        $Value -is [int32] -or
        $Value -is [uint32] -or
        $Value -is [int64] -or
        $Value -is [uint64]
    )
}

function Assert-NotReparsePoint {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $item = Get-Item -LiteralPath $Path -Force
    if (
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw "$Context is a reparse point: $Path"
    }
    return $item
}

function Assert-NoReparsePathChain {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $fullPath = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetPathRoot($fullPath)
    if ([string]::IsNullOrWhiteSpace($root)) {
        throw "$Context has no filesystem root"
    }
    $current = $root
    $relative = $fullPath.Substring($root.Length)
    foreach ($component in @($relative -split "[\\/]")) {
        if ([string]::IsNullOrWhiteSpace($component)) {
            continue
        }
        $current = Join-Path $current $component
        if (Test-Path -LiteralPath $current) {
            Assert-NotReparsePoint `
                -Path $current `
                -Context "$Context path component" |
                Out-Null
        }
    }
    return $fullPath
}

function Assert-ExactCanonicalPathString {
    param(
        [AllowNull()]$RawPath,
        [Parameter(Mandatory = $true)][string]$ExpectedPath,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if (
        $RawPath -isnot [string] -or
        [string]::IsNullOrWhiteSpace([string]$RawPath) -or
        -not [IO.Path]::IsPathRooted([string]$RawPath)
    ) {
        throw "$Context is not an absolute path string"
    }
    $normalized = [IO.Path]::GetFullPath([string]$RawPath)
    if (
        -not [StringComparer]::Ordinal.Equals(
            [string]$RawPath,
            $normalized
        ) -or
        -not [StringComparer]::Ordinal.Equals(
            [string]$RawPath,
            [IO.Path]::GetFullPath($ExpectedPath)
        )
    ) {
        throw "$Context is not the exact canonical path"
    }
    return [string]$RawPath
}

function Get-FileArtifactSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Kind
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Kind does not exist: $Path"
    }
    Assert-NoReparsePathChain -Path $Path -Context $Kind | Out-Null
    $item = Assert-NotReparsePoint -Path $Path -Context $Kind
    return [ordered]@{
        kind = $Kind
        path = $item.FullName
        bytes = [long]$item.Length
        sha256 = Get-Sha256Hex -Path $item.FullName
    }
}

function Test-TrustedRegularFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    Assert-NoReparsePathChain -Path $Path -Context $Context | Out-Null
    $item = Assert-NotReparsePoint -Path $Path -Context $Context
    if ($item.PSIsContainer) {
        throw "$Context is not a regular file"
    }
    return $true
}

function New-VerificationRetryCommand {
    param(
        [Parameter(Mandatory = $true)][string]$ControllerScript,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][int]$ExpectedUpdates,
        [Parameter(Mandatory = $true)][int]$JuliaThreads,
        [Parameter(Mandatory = $true)][int]$PollSeconds,
        [Parameter(Mandatory = $true)][int]$StallSeconds,
        [Parameter(Mandatory = $true)][int]$VerifierTimeoutSeconds,
        [Parameter(Mandatory = $true)][string]$JuliaPath,
        [Parameter(Mandatory = $true)][string]$ProjectPath,
        [Parameter(Mandatory = $true)][string]$TrainingScript,
        [Parameter(Mandatory = $true)][string]$VerifierScript,
        [Parameter(Mandatory = $true)][string]$DatasetPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$MutexName,
        [Parameter(Mandatory = $true)][string]$LaunchManifestSha256
    )

    return (
        "& " +
        (Quote-NativeArgument $ControllerScript) +
        " -RunId " +
        (Quote-NativeArgument $RunId) +
        " -ExpectedUpdates $ExpectedUpdates" +
        " -JuliaThreads $JuliaThreads" +
        " -PollSeconds $PollSeconds" +
        " -StallSeconds $StallSeconds" +
        " -VerifierTimeoutSeconds $VerifierTimeoutSeconds" +
        " -JuliaExecutable " +
        (Quote-NativeArgument $JuliaPath) +
        " -ProjectPath " +
        (Quote-NativeArgument $ProjectPath) +
        " -TrainingScript " +
        (Quote-NativeArgument $TrainingScript) +
        " -VerifierScript " +
        (Quote-NativeArgument $VerifierScript) +
        " -DatasetPath " +
        (Quote-NativeArgument $DatasetPath) +
        " -OutputRoot " +
        (Quote-NativeArgument $OutputRoot) +
        " -WorkingDirectory " +
        (Quote-NativeArgument $WorkingDirectory) +
        " -MutexName " +
        (Quote-NativeArgument $MutexName) +
        " -VerifyOnly" +
        " -ExpectedLaunchManifestSha256 " +
        (Quote-NativeArgument $LaunchManifestSha256)
    )
}

function Get-SwsnnEnvironment {
    $snapshot = [ordered]@{}
    Get-ChildItem Env: |
        Where-Object { $_.Name -like "SWSNN_*" } |
        Sort-Object Name |
        ForEach-Object {
            $snapshot[$_.Name] = [string]$_.Value
        }
    return $snapshot
}

function Get-CheckpointUpdateFromPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $name = [IO.Path]::GetFileName($Path)
    if ($name -notmatch "^checkpoint_([0-9]{9})[.]jld2$") {
        throw (
            "Resume checkpoint filename must encode its update as " +
            "checkpoint_NNNNNNNNN.jld2: $name"
        )
    }
    return [int]$Matches[1]
}

function Get-RunTelemetry {
    param(
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [Parameter(Mandatory = $true)][string]$StandardOutputPath,
        [Parameter(Mandatory = $true)][string]$StandardErrorPath
    )

    $tracePath = Join-Path $RunDirectory "training_trace.tsv"
    $latestUpdate = $null
    $latestLoss = $null
    $statesPerSecond = $null
    $cpuPercent = $null
    $traceLastWrite = $null
    if (Test-Path -LiteralPath $tracePath -PathType Leaf) {
        try {
            $traceItem = Get-Item -LiteralPath $tracePath
            $traceLastWrite = $traceItem.LastWriteTimeUtc.ToString("o")
            $lastRecord = Import-Csv -LiteralPath $tracePath `
                -Delimiter "`t" |
                Select-Object -Last 1
            if ($null -ne $lastRecord) {
                $parsedUpdate = 0
                if (
                    [int]::TryParse(
                        [string]$lastRecord.update,
                        [ref]$parsedUpdate
                    )
                ) {
                    $latestUpdate = $parsedUpdate
                    $latestLoss = [string]$lastRecord.loss
                    $statesPerSecond = [string]$lastRecord.states_per_second
                    $cpuPercent = [string]$lastRecord.cpu_percent
                }
            }
        }
        catch {
            # A writer may be replacing or appending the trace. A later poll
            # will retry; completion is decided only by the verifier.
        }
    }

    $checkpointDirectory = Join-Path $RunDirectory "checkpoints"
    $checkpoints = @()
    if (Test-Path -LiteralPath $checkpointDirectory -PathType Container) {
        $checkpoints = @(
            Get-ChildItem -LiteralPath $checkpointDirectory -File |
                Where-Object {
                    $_.Name -match "^checkpoint_[0-9]{9}[.]jld2$"
                } |
                Sort-Object Name
        )
    }
    $latestCheckpoint = if ($checkpoints.Count -gt 0) {
        $checkpoints[-1].Name
    }
    else {
        $null
    }

    $stagePath = Join-Path $RunDirectory "stage_status.json"
    $stage = $null
    $stageObservedAt = $null
    $stageFileLastWriteUtc = $null
    $telemetryWarnings = @()
    if (Test-Path -LiteralPath $stagePath -PathType Leaf) {
        try {
            $stageItem = Get-Item -LiteralPath $stagePath
            $stageFileLastWriteUtc =
                $stageItem.LastWriteTimeUtc.ToString("o")
        }
        catch {
            $telemetryWarnings +=
                "stage_status.json metadata could not be read: " +
                $_.Exception.Message
        }

        try {
            $stageDocument = Get-Content -Raw -LiteralPath $stagePath |
                ConvertFrom-Json
            $stageProperty = $stageDocument.PSObject.Properties["phase"]
            if ($null -eq $stageProperty) {
                $stageProperty =
                    $stageDocument.PSObject.Properties["stage"]
            }
            if (
                $null -eq $stageProperty -or
                $stageProperty.Value -isnot [string] -or
                [string]::IsNullOrWhiteSpace(
                    [string]$stageProperty.Value
                )
            ) {
                throw (
                    "required non-empty string property 'phase' " +
                    "(or 'stage') is missing"
                )
            }
            $stage = [string]$stageProperty.Value

            $observedProperty =
                $stageDocument.PSObject.Properties["recorded_at"]
            if ($null -eq $observedProperty) {
                $observedProperty =
                    $stageDocument.PSObject.Properties["stage_observed_at"]
            }
            if ($null -eq $observedProperty) {
                $observedProperty =
                    $stageDocument.PSObject.Properties["observed_at"]
            }
            if (
                $null -ne $observedProperty -and
                -not [string]::IsNullOrWhiteSpace(
                    [string]$observedProperty.Value
                )
            ) {
                try {
                    $stageObservedAt = (
                        [datetimeoffset]$observedProperty.Value
                    ).ToString("o")
                }
                catch {
                    $stageObservedAt = $stageFileLastWriteUtc
                    $telemetryWarnings +=
                        "stage_status.json observed timestamp is malformed: " +
                        $_.Exception.Message
                }
            }
            else {
                $stageObservedAt = $stageFileLastWriteUtc
            }
        }
        catch {
            $stage = $null
            $stageObservedAt = $null
            $telemetryWarnings +=
                "stage_status.json is malformed: " +
                $_.Exception.Message
        }
    }

    return [ordered]@{
        latest_update = $latestUpdate
        latest_loss = $latestLoss
        states_per_second = $statesPerSecond
        cpu_percent = $cpuPercent
        trace_last_write_utc = $traceLastWrite
        checkpoint_count = $checkpoints.Count
        latest_checkpoint = $latestCheckpoint
        stage = $stage
        stage_observed_at = $stageObservedAt
        stage_file_last_write_utc = $stageFileLastWriteUtc
        warnings = @($telemetryWarnings)
        results_exists = Test-Path -LiteralPath (
            Join-Path $RunDirectory "results.json"
        ) -PathType Leaf
        stdout_bytes = if (
            Test-Path -LiteralPath $StandardOutputPath -PathType Leaf
        ) {
            (Get-Item -LiteralPath $StandardOutputPath).Length
        }
        else {
            0
        }
        stderr_bytes = if (
            Test-Path -LiteralPath $StandardErrorPath -PathType Leaf
        ) {
            (Get-Item -LiteralPath $StandardErrorPath).Length
        }
        else {
            0
        }
    }
}

function Get-ProgressTimeFromTelemetry {
    param(
        [Parameter(Mandatory = $true)]$Telemetry,
        [Parameter(Mandatory = $true)][datetime]$CurrentProgressAt
    )

    if (
        [string]::IsNullOrWhiteSpace(
            [string]$Telemetry.stage_file_last_write_utc
        )
    ) {
        return $CurrentProgressAt
    }
    try {
        $stageProgressAt = [datetime]::Parse(
            [string]$Telemetry.stage_file_last_write_utc,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind
        ).ToLocalTime()
        if ($stageProgressAt -gt $CurrentProgressAt) {
            return $stageProgressAt
        }
    }
    catch {
        # Get-RunTelemetry produces this timestamp from FileInfo. If a future
        # schema changes it, a malformed value must not terminate monitoring.
    }
    return $CurrentProgressAt
}

function Assert-StrictTopLevelJsonObject {
    param(
        [Parameter(Mandatory = $true)][string]$RawJson,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $text = $RawJson.Trim()
    if (
        $text.Length -lt 2 -or
        $text[0] -ne [char]'{' -or
        $text[$text.Length - 1] -ne [char]'}'
    ) {
        throw "$Context must be one top-level JSON object"
    }
    $objectKeySets = [Collections.Generic.List[object]]::new()
    $objectDepth = 0
    $arrayDepth = 0
    $index = 0
    while ($index -lt $text.Length) {
        $character = $text[$index]
        if ($character -eq [char]'"') {
            $stringStart = $index
            $index += 1
            $closed = $false
            while ($index -lt $text.Length) {
                if ($text[$index] -eq [char]'\') {
                    $index += 2
                    continue
                }
                if ($text[$index] -eq [char]'"') {
                    $closed = $true
                    break
                }
                $index += 1
            }
            if (-not $closed) {
                throw "$Context contains an unterminated JSON string"
            }
            if ($objectDepth -gt 0) {
                $afterString = $index + 1
                while (
                    $afterString -lt $text.Length -and
                    [char]::IsWhiteSpace($text[$afterString])
                ) {
                    $afterString += 1
                }
                if (
                    $afterString -lt $text.Length -and
                    $text[$afterString] -eq [char]':'
                ) {
                    $rawKey = $text.Substring(
                        $stringStart,
                        $index - $stringStart + 1
                    )
                    try {
                        $key = [string](
                            ConvertFrom-Json -InputObject $rawKey
                        )
                    }
                    catch {
                        throw "$Context contains an invalid property name: $_"
                    }
                    $currentKeys =
                        $objectKeySets[$objectKeySets.Count - 1]
                    if (-not $currentKeys.Add($key)) {
                        throw (
                            "$Context contains duplicate property '$key' " +
                            "at object depth $objectDepth"
                        )
                    }
                }
            }
        }
        else {
            switch ($character) {
                ([char]'{') {
                    $objectDepth += 1
                    $objectKeySets.Add(
                        [Collections.Generic.HashSet[string]]::new(
                            [StringComparer]::Ordinal
                        )
                    )
                    break
                }
                ([char]'}') {
                    if ($objectDepth -lt 1) {
                        throw (
                            "$Context has an invalid JSON object structure"
                        )
                    }
                    $objectKeySets.RemoveAt(
                        $objectKeySets.Count - 1
                    )
                    $objectDepth -= 1
                    break
                }
                ([char]'[') {
                    $arrayDepth += 1
                    break
                }
                ([char]']') {
                    $arrayDepth -= 1
                    break
                }
            }
            if ($objectDepth -lt 0 -or $arrayDepth -lt 0) {
                throw "$Context has an invalid JSON container structure"
            }
        }
        $index += 1
    }
    if ($objectDepth -ne 0 -or $arrayDepth -ne 0) {
        throw "$Context has an unbalanced JSON container structure"
    }
}

function Get-StrictCheckpointState {
    param(
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [Parameter(Mandatory = $true)][int]$MaximumUpdate,
        [int]$AllowedFinalizationUpdate = -1
    )

    $canonicalRunDirectory = [IO.Path]::GetFullPath($RunDirectory)
    if (-not (Test-Path -LiteralPath $canonicalRunDirectory -PathType Container)) {
        throw "checkpoint run directory does not exist: $canonicalRunDirectory"
    }
    Assert-NoReparsePathChain `
        -Path $canonicalRunDirectory `
        -Context "checkpoint run directory" |
        Out-Null
    $runItem = Assert-NotReparsePoint `
        -Path $canonicalRunDirectory `
        -Context "checkpoint run directory"
    if (-not [StringComparer]::Ordinal.Equals(
        $runItem.FullName,
        $canonicalRunDirectory
    )) {
        throw "checkpoint run directory is not in canonical spelling"
    }

    $checkpointDirectory =
        Join-Path $canonicalRunDirectory "checkpoints"
    $manifestPath =
        Join-Path $canonicalRunDirectory "checkpoint_manifest.jsonl"
    if (
        -not (Test-Path -LiteralPath $checkpointDirectory -PathType Container) -or
        -not (Test-Path -LiteralPath $manifestPath -PathType Leaf)
    ) {
        throw "checkpoint directory or manifest is missing"
    }
    $checkpointDirectoryItem = Assert-NotReparsePoint `
        -Path $checkpointDirectory `
        -Context "checkpoint directory"
    $manifestItem = Assert-NotReparsePoint `
        -Path $manifestPath `
        -Context "checkpoint manifest"

    $manifestRecords = @{}
    $manifestLines = @(
        [IO.File]::ReadAllLines($manifestItem.FullName)
    )
    if ($manifestLines.Count -eq 0) {
        throw "checkpoint manifest is empty"
    }
    for ($lineIndex = 0; $lineIndex -lt $manifestLines.Count; $lineIndex++) {
        $line = $manifestLines[$lineIndex]
        if ([string]::IsNullOrWhiteSpace($line)) {
            throw (
                "checkpoint manifest contains a blank line at " +
                ($lineIndex + 1)
            )
        }
        Assert-StrictTopLevelJsonObject `
            -RawJson $line `
            -Context "checkpoint manifest line $($lineIndex + 1)"
        try {
            $record = ConvertFrom-Json -InputObject $line
        }
        catch {
            throw (
                "checkpoint manifest line $($lineIndex + 1) is not JSON: $_"
            )
        }
        Assert-ExactJsonProperties `
            -Value $record `
            -Names @("kind", "path", "bytes", "sha256", "update") `
            -Context "checkpoint manifest line $($lineIndex + 1)"
        if (
            -not (Test-JsonInteger $record.update) -or
            -not (Test-JsonInteger $record.bytes) -or
            $record.kind -isnot [string] -or
            $record.path -isnot [string] -or
            $record.sha256 -isnot [string]
        ) {
            throw "checkpoint manifest record has a wrong JSON token type"
        }
        $update = [int64]$record.update
        $bytes = [int64]$record.bytes
        if (
            $update -lt 0 -or
            $update -gt $MaximumUpdate -or
            $update -gt [int]::MaxValue -or
            $bytes -lt 1 -or
            [string]$record.kind -cne "training" -or
            [string]$record.sha256 -cnotmatch "^[0-9a-f]{64}$" -or
            -not [IO.Path]::IsPathRooted([string]$record.path)
        ) {
            throw "checkpoint manifest record value is invalid"
        }
        $integerUpdate = [int]$update
        if ($manifestRecords.ContainsKey($integerUpdate)) {
            throw "checkpoint manifest contains duplicate update $integerUpdate"
        }
        $manifestRecords[$integerUpdate] = $record
    }

    $artifactsByUpdate = @{}
    $expectedFinalizationName = if ($AllowedFinalizationUpdate -ge 0) {
        "finalization_checkpoint_" +
            $AllowedFinalizationUpdate.ToString("000000000") +
            ".jld2"
    }
    else {
        $null
    }
    $finalizationArtifact = $null
    foreach (
        $entry in Get-ChildItem -LiteralPath $checkpointDirectoryItem.FullName -Force
    ) {
        if (
            ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $entry.PSIsContainer
        ) {
            throw "checkpoint directory contains a directory or reparse point"
        }
        if (
            $null -ne $expectedFinalizationName -and
            $entry.Name -ceq $expectedFinalizationName
        ) {
            if ($null -ne $finalizationArtifact) {
                throw "checkpoint directory contains duplicate finalization"
            }
            $finalizationArtifact = $entry
            continue
        }
        if ($entry.Name -cnotmatch "^checkpoint_([0-9]{9})[.]jld2$") {
            throw "checkpoint directory contains an unexpected entry"
        }
        $update = [int]$Matches[1]
        if (
            $update -gt $MaximumUpdate -or
            $artifactsByUpdate.ContainsKey($update)
        ) {
            throw "checkpoint directory contains an invalid update"
        }
        $artifactsByUpdate[$update] = $entry
    }
    if (
        $manifestRecords.Count -eq 0 -or
        $manifestRecords.Count -ne $artifactsByUpdate.Count
    ) {
        throw "checkpoint manifest and live training file set differ"
    }

    foreach ($update in $manifestRecords.Keys) {
        if (-not $artifactsByUpdate.ContainsKey([int]$update)) {
            throw "checkpoint manifest references a missing update"
        }
        $artifact = $artifactsByUpdate[[int]$update]
        $record = $manifestRecords[[int]$update]
        $recordPath = [string]$record.path
        $normalizedRecordPath = [IO.Path]::GetFullPath($recordPath)
        if (
            -not [StringComparer]::Ordinal.Equals(
                $recordPath,
                $normalizedRecordPath
            ) -or
            -not [StringComparer]::Ordinal.Equals(
                $recordPath,
                $artifact.FullName
            ) -or
            [int64]$record.bytes -ne [int64]$artifact.Length -or
            [string]$record.sha256 -cne
                (Get-Sha256Hex -Path $artifact.FullName)
        ) {
            throw "checkpoint manifest artifact identity differs at $update"
        }
    }

    $updates = @(
        $manifestRecords.Keys |
            ForEach-Object { [int]$_ } |
            Sort-Object
    )
    $latestUpdate = [int]$updates[-1]
    $latestArtifact = $artifactsByUpdate[$latestUpdate]
    $latestRecord = $manifestRecords[$latestUpdate]
    return [ordered]@{
        run_directory = $runItem.FullName
        checkpoint_directory = $checkpointDirectoryItem.FullName
        manifest = [ordered]@{
            kind = "checkpoint_manifest"
            path = $manifestItem.FullName
            bytes = [int64]$manifestItem.Length
            sha256 = Get-Sha256Hex -Path $manifestItem.FullName
        }
        records_by_update = $manifestRecords
        live_training_updates = $updates
        latest = [ordered]@{
            kind = "training"
            path = $latestArtifact.FullName
            bytes = [int64]$latestArtifact.Length
            sha256 = [string]$latestRecord.sha256
            update = $latestUpdate
        }
        finalization = if ($null -eq $finalizationArtifact) {
            $null
        }
        else {
            [ordered]@{
                kind = "finalization"
                path = $finalizationArtifact.FullName
                bytes = [int64]$finalizationArtifact.Length
                sha256 = Get-Sha256Hex -Path $finalizationArtifact.FullName
                update = $AllowedFinalizationUpdate
            }
        }
    }
}

function ConvertTo-ContractJson {
    param([AllowNull()]$Value)

    return ConvertTo-Json -InputObject $Value -Compress -Depth 100
}

function Assert-CheckpointCadence {
    param(
        [Parameter(Mandatory = $true)][int[]]$ObservedUpdates,
        [Parameter(Mandatory = $true)][string]$StartMode,
        [Parameter(Mandatory = $true)][int]$SegmentStartUpdate,
        [Parameter(Mandatory = $true)][int]$SelectedUpdate,
        [Parameter(Mandatory = $true)][int]$MaximumUpdate,
        [Parameter(Mandatory = $true)][int]$CheckpointInterval
    )

    if ($CheckpointInterval -lt 1) {
        throw "checkpoint interval must be positive"
    }
    $expected = [Collections.Generic.List[int]]::new()
    if ($StartMode -ceq "scratch") {
        if ($SegmentStartUpdate -ne 0) {
            throw "scratch checkpoint segment must start at update zero"
        }
        $expected.Add(0)
    }
    elseif ($StartMode -ceq "resume") {
        if ($SegmentStartUpdate -lt 0) {
            throw "resume checkpoint segment start is negative"
        }
    }
    else {
        throw "training checkpoint segment has invalid start mode"
    }
    $firstBoundary = (
        [math]::Floor($SegmentStartUpdate / $CheckpointInterval) + 1
    ) * $CheckpointInterval
    for (
        $boundary = [int64]$firstBoundary;
        $boundary -le $SelectedUpdate;
        $boundary += $CheckpointInterval
    ) {
        $expected.Add([int]$boundary)
    }
    if (
        $SelectedUpdate -eq $MaximumUpdate -and
        -not $expected.Contains($SelectedUpdate)
    ) {
        $expected.Add($SelectedUpdate)
    }
    $expectedArray = @($expected | Sort-Object -Unique)
    if ($ObservedUpdates.Count -ne $expectedArray.Count) {
        throw "checkpoint cadence count differs"
    }
    for ($index = 0; $index -lt $expectedArray.Count; $index++) {
        if ([int]$ObservedUpdates[$index] -ne [int]$expectedArray[$index]) {
            throw "checkpoint cadence differs at index $index"
        }
    }
}

function Get-ParentLineageSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$SelectedCheckpoint,
        [Parameter(Mandatory = $true)][string]$SelectedSha256,
        [Parameter(Mandatory = $true)][int]$SelectedUpdate,
        [Parameter(Mandatory = $true)][string]$ExpectedOutputRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedProject,
        [Parameter(Mandatory = $true)][string]$ExpectedTrainingScript,
        [Parameter(Mandatory = $true)][string]$ExpectedVerifierScript,
        [Parameter(Mandatory = $true)][string]$ExpectedControllerScript
    )

    $lineage = [Collections.Generic.List[object]]::new()
    $seen = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    $currentPath = [IO.Path]::GetFullPath($SelectedCheckpoint)
    $currentSha = $SelectedSha256.ToLowerInvariant()
    $currentUpdate = $SelectedUpdate
    $anchorBindings = $null
    while ($true) {
        if (
            $currentSha -cnotmatch "^[0-9a-f]{64}$" -or
            $currentUpdate -lt 0 -or
            -not $seen.Add("$currentPath|$currentSha")
        ) {
            throw "checkpoint lineage has an invalid reference or cycle"
        }
        if (-not (Test-Path -LiteralPath $currentPath -PathType Leaf)) {
            throw "checkpoint lineage file is missing: $currentPath"
        }
        $currentItem = Assert-NotReparsePoint `
            -Path $currentPath `
            -Context "checkpoint lineage file"
        if (
            -not [StringComparer]::Ordinal.Equals(
                $currentPath,
                $currentItem.FullName
            ) -or
            (Get-Sha256Hex -Path $currentItem.FullName) -cne $currentSha
        ) {
            throw "checkpoint lineage selected file identity differs"
        }
        if (
            $currentItem.Name -cnotmatch "^checkpoint_([0-9]{9})[.]jld2$" -or
            [int]$Matches[1] -ne $currentUpdate -or
            $currentItem.Directory.Name -cne "checkpoints"
        ) {
            throw "checkpoint lineage filename/update is invalid"
        }

        $runDirectory = $currentItem.Directory.Parent.FullName
        if (
            -not [StringComparer]::Ordinal.Equals(
                [IO.Path]::GetDirectoryName($runDirectory),
                $ExpectedOutputRoot
            )
        ) {
            throw "checkpoint lineage run escaped the expected output root"
        }
        $configPath = Join-Path $runDirectory "config.json"
        $configArtifact = Get-FileArtifactSnapshot `
            -Path $configPath `
            -Kind "config"
        try {
            $rawConfigDocument =
                [IO.File]::ReadAllText($configArtifact.path)
            Assert-StrictTopLevelJsonObject `
                -RawJson $rawConfigDocument `
                -Context "checkpoint lineage config.json"
            $configDocument =
                ConvertFrom-Json -InputObject $rawConfigDocument
        }
        catch {
            throw "checkpoint lineage config.json is invalid: $_"
        }
        Assert-ExactJsonProperties `
            -Value $configDocument `
            -Names @("config", "parent_checkpoint") `
            -Context "checkpoint lineage config.json"
        $config = $configDocument.config
        $parent = $configDocument.parent_checkpoint
        Assert-JsonProperties `
            -Value $config `
            -Names @(
                "experiment_id"
                "checkpoint_schema"
                "run_id"
                "start_mode"
                "maximum_updates"
                "checkpoint_interval"
                "dataset_content_sha256"
                "dataset_integrity"
                "source_fingerprint"
                "runtime_provenance"
                "launch_binding"
            ) `
            -Context "checkpoint lineage config"
        Assert-ExactJsonProperties `
            -Value $config.checkpoint_schema `
            -Names @("format", "version") `
            -Context "checkpoint lineage checkpoint schema"
        if (
            $config.run_id -isnot [string] -or
            $config.experiment_id -isnot [string] -or
            $config.start_mode -isnot [string] -or
            [string]$config.run_id -cne
                [IO.Path]::GetFileName($runDirectory) -or
            [string]$config.experiment_id -cne
                "serial_workspace_snn_arena_v3" -or
            $config.checkpoint_schema.format -isnot [string] -or
            [string]$config.checkpoint_schema.format -cne
                "serial-workspace-snn-arena-checkpoint" -or
            -not (
                Test-JsonInteger $config.checkpoint_schema.version
            ) -or
            [int]$config.checkpoint_schema.version -ne 3 -or
            -not (Test-JsonInteger $config.maximum_updates) -or
            -not (Test-JsonInteger $config.checkpoint_interval)
        ) {
            throw "checkpoint lineage config identity/type differs"
        }
        $maximumUpdate = [int]$config.maximum_updates
        $checkpointInterval = [int]$config.checkpoint_interval
        if (
            $maximumUpdate -lt $currentUpdate -or
            $checkpointInterval -lt 1
        ) {
            throw "checkpoint lineage config update range differs"
        }
        $startMode = [string]$config.start_mode
        if ($startMode -cnotin @("scratch", "resume")) {
            throw "checkpoint lineage segment is not a training segment"
        }

        Assert-ExactJsonProperties `
            -Value $config.launch_binding `
            -Names @("path", "sha256") `
            -Context "checkpoint lineage launch binding"
        if (
            $config.launch_binding.path -isnot [string] -or
            $config.launch_binding.sha256 -isnot [string] -or
            [string]$config.launch_binding.sha256 -cnotmatch
                "^[0-9a-f]{64}$"
        ) {
            throw "checkpoint lineage launch binding type differs"
        }
        $outputRoot = [IO.Path]::GetDirectoryName($runDirectory)
        $expectedLaunchPath = Join-Path (
            Join-Path (
                Join-Path $outputRoot "_controllers"
            ) ([string]$config.run_id)
        ) "launch_manifest.json"
        if (
            -not [StringComparer]::Ordinal.Equals(
                [string]$config.launch_binding.path,
                $expectedLaunchPath
            )
        ) {
            throw "checkpoint lineage launch binding path differs"
        }
        $launchArtifact = Get-FileArtifactSnapshot `
            -Path $expectedLaunchPath `
            -Kind "launch_manifest"
        if (
            $launchArtifact.sha256 -cne
                [string]$config.launch_binding.sha256
        ) {
            throw "checkpoint lineage launch binding SHA-256 differs"
        }
        try {
            $rawBoundLaunch =
                [IO.File]::ReadAllText($launchArtifact.path)
            Assert-StrictTopLevelJsonObject `
                -RawJson $rawBoundLaunch `
                -Context "checkpoint lineage launch manifest"
            $boundLaunch =
                ConvertFrom-Json -InputObject $rawBoundLaunch
        }
        catch {
            throw "checkpoint lineage launch manifest is invalid: $_"
        }
        Assert-JsonProperties `
            -Value $boundLaunch `
            -Names @(
                "format"
                "version"
                "run_id"
                "run_directory"
                "expected_updates"
                "start_mode"
                "project_path"
                "output_root"
                "training_script"
                "verifier_script"
                "controller_script"
                "code_artifacts"
                "parent_checkpoint"
                "mutex_name"
                "mutex_scope_sha256"
            ) `
            -Context "checkpoint lineage launch manifest"
        Assert-ExactJsonProperties `
            -Value $boundLaunch.code_artifacts `
            -Names @("controller", "training", "verifier") `
            -Context "checkpoint lineage launch code artifacts"
        if (
            $boundLaunch.format -isnot [string] -or
            $boundLaunch.run_id -isnot [string] -or
            $boundLaunch.run_directory -isnot [string] -or
            $boundLaunch.start_mode -isnot [string] -or
            $boundLaunch.output_root -isnot [string] -or
            $boundLaunch.mutex_name -isnot [string] -or
            $boundLaunch.mutex_scope_sha256 -isnot [string] -or
            -not (Test-JsonInteger $boundLaunch.version) -or
            -not (Test-JsonInteger $boundLaunch.expected_updates) -or
            [string]$boundLaunch.format -cne
                "serial-workspace-snn-arena-run-launch" -or
            [int]$boundLaunch.version -ne 2 -or
            [string]$boundLaunch.run_id -cne [string]$config.run_id -or
            [string]$boundLaunch.run_directory -cne $runDirectory -or
            [int]$boundLaunch.expected_updates -ne $maximumUpdate -or
            [string]$boundLaunch.start_mode -cne $startMode -or
            [string]$boundLaunch.output_root -cne $outputRoot
        ) {
            throw "checkpoint lineage launch identity differs"
        }
        $boundParent = $boundLaunch.parent_checkpoint
        if ($startMode -ceq "scratch") {
            if ($null -ne $boundParent) {
                throw "scratch lineage launch unexpectedly has a parent"
            }
        }
        else {
            Assert-ExactJsonProperties `
                -Value $boundParent `
                -Names @("path", "sha256", "update") `
                -Context "checkpoint lineage launch parent"
            if (
                $boundParent.path -isnot [string] -or
                $boundParent.sha256 -isnot [string] -or
                -not (Test-JsonInteger $boundParent.update) -or
                [string]$boundParent.path -cne [string]$parent.path -or
                [string]$boundParent.sha256 -cne [string]$parent.sha256 -or
                [int]$boundParent.update -ne [int]$parent.update
            ) {
                throw "checkpoint lineage launch parent differs"
            }
        }
        Assert-ExactCanonicalPathString `
            -RawPath $boundLaunch.project_path `
            -ExpectedPath $ExpectedProject `
            -Context "checkpoint lineage launch project path" |
            Out-Null
        Assert-ExactCanonicalPathString `
            -RawPath $boundLaunch.training_script `
            -ExpectedPath $ExpectedTrainingScript `
            -Context "checkpoint lineage launch training script" |
            Out-Null
        Assert-ExactCanonicalPathString `
            -RawPath $boundLaunch.verifier_script `
            -ExpectedPath $ExpectedVerifierScript `
            -Context "checkpoint lineage launch verifier script" |
            Out-Null
        Assert-ExactCanonicalPathString `
            -RawPath $boundLaunch.controller_script `
            -ExpectedPath $ExpectedControllerScript `
            -Context "checkpoint lineage launch controller script" |
            Out-Null
        $boundProject = $ExpectedProject
        $boundMutexScope = Get-Utf8Sha256Hex -Value (
            $boundProject.ToLowerInvariant() +
            "`n" +
            $outputRoot.ToLowerInvariant()
        )
        $boundMutexName = (
            "Local\OpenAI.SerialWorkspaceSNN.Arena100k.v1." +
            $boundMutexScope.Substring(0, 32)
        )
        if (
            [string]$boundLaunch.mutex_scope_sha256 -cne
                $boundMutexScope -or
            [string]$boundLaunch.mutex_name -cne $boundMutexName
        ) {
            throw "checkpoint lineage launch mutex scope differs"
        }
        foreach ($codeName in @("controller", "training", "verifier")) {
            $boundCodeArtifact = $boundLaunch.code_artifacts.$codeName
            $expectedCodePath = switch ($codeName) {
                "controller" { $ExpectedControllerScript; break }
                "training" { $ExpectedTrainingScript; break }
                "verifier" { $ExpectedVerifierScript; break }
            }
            Assert-ExactJsonProperties `
                -Value $boundCodeArtifact `
                -Names @("path", "bytes", "sha256") `
                -Context "checkpoint lineage $codeName artifact"
            if (
                $boundCodeArtifact.path -isnot [string] -or
                $boundCodeArtifact.sha256 -isnot [string] -or
                -not (Test-JsonInteger $boundCodeArtifact.bytes)
            ) {
                throw "checkpoint lineage code artifact type differs"
            }
            Assert-ExactCanonicalPathString `
                -RawPath $boundCodeArtifact.path `
                -ExpectedPath $expectedCodePath `
                -Context "checkpoint lineage $codeName artifact path" |
                Out-Null
            $actualCodeArtifact = Get-FileArtifactSnapshot `
                -Path $expectedCodePath `
                -Kind "code_$codeName"
            if (
                -not [StringComparer]::Ordinal.Equals(
                    [string]$actualCodeArtifact.path,
                    [string]$boundCodeArtifact.path
                ) -or
                [int64]$actualCodeArtifact.bytes -ne
                    [int64]$boundCodeArtifact.bytes -or
                [string]$actualCodeArtifact.sha256 -cne
                    [string]$boundCodeArtifact.sha256
            ) {
                throw "checkpoint lineage code artifact changed: $codeName"
            }
        }

        $allowedFinalizationUpdate = if (
            $currentUpdate -eq $maximumUpdate
        ) {
            $currentUpdate
        }
        else {
            -1
        }
        $checkpointState = Get-StrictCheckpointState `
            -RunDirectory $runDirectory `
            -MaximumUpdate $maximumUpdate `
            -AllowedFinalizationUpdate $allowedFinalizationUpdate
        if (
            [int]$checkpointState.latest.update -ne $currentUpdate -or
            [string]$checkpointState.latest.sha256 -cne $currentSha -or
            -not [StringComparer]::Ordinal.Equals(
                [string]$checkpointState.latest.path,
                $currentPath
            )
        ) {
            throw "selected checkpoint is not the latest live training record"
        }

        $segmentStartUpdate = if ($startMode -ceq "scratch") {
            if ($null -ne $parent) {
                throw "scratch checkpoint lineage unexpectedly has a parent"
            }
            0
        }
        else {
            Assert-ExactJsonProperties `
                -Value $parent `
                -Names @("kind", "path", "bytes", "sha256", "update") `
                -Context "checkpoint lineage parent reference"
            if (
                $parent.kind -isnot [string] -or
                $parent.path -isnot [string] -or
                $parent.sha256 -isnot [string] -or
                -not (Test-JsonInteger $parent.bytes) -or
                -not (Test-JsonInteger $parent.update) -or
                [string]$parent.kind -cne "training" -or
                [string]$parent.sha256 -cnotmatch "^[0-9a-f]{64}$" -or
                [int64]$parent.bytes -lt 1 -or
                [int]$parent.update -ge $currentUpdate
            ) {
                throw "checkpoint lineage parent reference is invalid"
            }
            [int]$parent.update
        }
        Assert-CheckpointCadence `
            -ObservedUpdates @($checkpointState.live_training_updates) `
            -StartMode $startMode `
            -SegmentStartUpdate $segmentStartUpdate `
            -SelectedUpdate $currentUpdate `
            -MaximumUpdate $maximumUpdate `
            -CheckpointInterval $checkpointInterval

        $bindings = [ordered]@{
            dataset_content_sha256 =
                [string]$config.dataset_content_sha256
            dataset_integrity =
                ConvertTo-ContractJson $config.dataset_integrity
            source_fingerprint =
                [string]$config.source_fingerprint
            runtime_provenance =
                ConvertTo-ContractJson $config.runtime_provenance
        }
        if ($null -eq $anchorBindings) {
            $anchorBindings = $bindings
        }
        elseif (
            $bindings.dataset_content_sha256 -cne
                $anchorBindings.dataset_content_sha256 -or
            $bindings.dataset_integrity -cne
                $anchorBindings.dataset_integrity -or
            $bindings.source_fingerprint -cne
                $anchorBindings.source_fingerprint -or
            $bindings.runtime_provenance -cne
                $anchorBindings.runtime_provenance
        ) {
            throw "checkpoint lineage source/dataset/runtime binding differs"
        }

        $lineage.Add([ordered]@{
            run_id = [string]$config.run_id
            run_directory = $runDirectory
            start_mode = $startMode
            maximum_updates = $maximumUpdate
            checkpoint_interval = $checkpointInterval
            selected_update = $currentUpdate
            segment_start_update = $segmentStartUpdate
            live_training_updates =
                @($checkpointState.live_training_updates)
            selected_checkpoint =
                $checkpointState.latest
            config = $configArtifact
            checkpoint_manifest = $checkpointState.manifest
            launch_manifest = $launchArtifact
            parent_checkpoint = $parent
        })
        if ($startMode -ceq "scratch") {
            if ($checkpointState.live_training_updates[0] -ne 0) {
                throw "scratch checkpoint lineage does not contain update zero"
            }
            break
        }
        $rawParentPath = [string]$parent.path
        $parentPath = [IO.Path]::GetFullPath($rawParentPath)
        if (
            -not [StringComparer]::Ordinal.Equals(
                $rawParentPath,
                $parentPath
            )
        ) {
            throw "checkpoint lineage parent path is noncanonical"
        }
        $parentItem = Get-Item -LiteralPath $parentPath -Force
        if (
            [int64]$parentItem.Length -ne [int64]$parent.bytes -or
            (Get-Sha256Hex -Path $parentItem.FullName) -cne
                [string]$parent.sha256
        ) {
            throw "checkpoint lineage parent artifact identity differs"
        }
        $currentPath = $parentPath
        $currentSha = [string]$parent.sha256
        $currentUpdate = [int]$parent.update
    }
    return @($lineage)
}

function Assert-PinnedLineageCurrent {
    param([Parameter(Mandatory = $true)]$PinnedLineage)

    foreach ($node in @($PinnedLineage)) {
        foreach (
            $artifactName in @(
                "selected_checkpoint"
                "config"
                "checkpoint_manifest"
                "launch_manifest"
            )
        ) {
            $artifact = $node.$artifactName
            Assert-ExactJsonProperties `
                -Value $artifact `
                -Names $(if ($artifactName -eq "selected_checkpoint") {
                    @("kind", "path", "bytes", "sha256", "update")
                }
                else {
                    @("kind", "path", "bytes", "sha256")
                }) `
                -Context "pinned lineage $artifactName"
            $current = Get-FileArtifactSnapshot `
                -Path ([string]$artifact.path) `
                -Kind ([string]$artifact.kind)
            if (
                [int64]$current.bytes -ne [int64]$artifact.bytes -or
                [string]$current.sha256 -cne [string]$artifact.sha256 -or
                -not [StringComparer]::Ordinal.Equals(
                    [string]$current.path,
                    [string]$artifact.path
                )
            ) {
                throw "pinned parent lineage artifact changed: $artifactName"
            }
        }
        $allowedFinalizationUpdate = if (
            [int]$node.selected_update -eq [int]$node.maximum_updates
        ) {
            [int]$node.selected_update
        }
        else {
            -1
        }
        $state = Get-StrictCheckpointState `
            -RunDirectory ([string]$node.run_directory) `
            -MaximumUpdate ([int]$node.maximum_updates) `
            -AllowedFinalizationUpdate $allowedFinalizationUpdate
        $observedUpdates = @($state.live_training_updates)
        $pinnedUpdates = @($node.live_training_updates)
        if ($observedUpdates.Count -ne $pinnedUpdates.Count) {
            throw "pinned parent lineage live checkpoint set changed"
        }
        for ($index = 0; $index -lt $observedUpdates.Count; $index++) {
            if ([int]$observedUpdates[$index] -ne [int]$pinnedUpdates[$index]) {
                throw "pinned parent lineage live checkpoint set changed"
            }
        }
    }
}

function Assert-PinnedLineageCurrentFailClosed {
    param([Parameter(Mandatory = $true)]$PinnedLineage)

    try {
        Assert-PinnedLineageCurrent -PinnedLineage $PinnedLineage
    }
    catch {
        $script:ParentLineageTrustLost = $true
        throw
    }
}

function Get-RecoveryCheckpoint {
    param([Parameter(Mandatory = $true)][string]$RunDirectory)

    try {
        $state = Get-StrictCheckpointState `
            -RunDirectory $RunDirectory `
            -MaximumUpdate $ExpectedUpdates `
            -AllowedFinalizationUpdate $ExpectedUpdates
        $latest = $state.latest
        $lineage = @(
            Get-ParentLineageSnapshot `
                -SelectedCheckpoint ([string]$latest.path) `
                -SelectedSha256 ([string]$latest.sha256) `
                -SelectedUpdate ([int]$latest.update) `
                -ExpectedOutputRoot $resolvedOutputRoot `
                -ExpectedProject $canonicalProject `
                -ExpectedTrainingScript $canonicalTrainingScript `
                -ExpectedVerifierScript $canonicalVerifierScript `
                -ExpectedControllerScript $resolvedControllerScript
        )
        $currentNode = @($lineage)[0]
        if (
            [string]$currentNode.run_id -cne $RunId -or
            [int]$currentNode.maximum_updates -ne $ExpectedUpdates -or
            [string]$currentNode.start_mode -cne $effectiveStartMode
        ) {
            throw "recovery checkpoint differs from the active launch"
        }
        return [ordered]@{
            kind = "training"
            path = [string]$latest.path
            sha256 = [string]$latest.sha256
            bytes = [int64]$latest.bytes
            update = [int]$latest.update
            manifest_path = [string]$state.manifest.path
            candidate_status = "payload_unverified"
            controller_artifact_lineage_verified = $true
            payload_state_verified = $false
            resume_requires_fail_closed_payload_validation = $true
            lineage = $lineage
        }
    }
    catch {
        return $null
    }
}

function Stop-ChildProcessTree {
    param([Parameter(Mandatory = $true)]$Process)

    if ($Process.HasExited) {
        return $false
    }
    try {
        $Process.Kill($true)
    }
    catch {
        # Older runtimes may not expose Kill(entireProcessTree).  The direct
        # child is still owned by this controller and must not be orphaned.
        $Process.Kill()
    }
    if (-not $Process.WaitForExit(5000)) {
        throw (
            "direct child did not exit within 5000 ms after termination; " +
            "Job-object termination is required"
        )
    }
    return $true
}

function New-ControllerStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [Parameter(Mandatory = $true)]$Telemetry,
        [Parameter(Mandatory = $true)][datetime]$LastProgressAt,
        [AllowNull()]$ChildProcessId,
        [AllowNull()]$ChildStartTime,
        [AllowNull()]$ChildExitCode,
        [AllowNull()]$VerifierExitCode,
        [AllowNull()]$Failure
    )

    $idleSeconds = [math]::Max(
        0.0,
        [math]::Round(
            ((Get-Date) - $LastProgressAt).TotalSeconds,
            1
        )
    )
    return [ordered]@{
        format = "serial-workspace-snn-arena-run-controller"
        version = 1
        state = $State
        verified = $State -eq "verified_complete"
        observed_at = (Get-Date).ToString("o")
        notification_capability = "none"
        controller_mode = if ($VerifyOnly) {
            "verify-only"
        }
        else {
            "train-and-verify"
        }
        controller_pid = $PID
        child_pid = $ChildProcessId
        child_start_time = $ChildStartTime
        child_exit_code = $ChildExitCode
        verifier_exit_code = $VerifierExitCode
        verifier_pid = $script:VerifierProcessId
        verifier_timeout_seconds = $VerifierTimeoutSeconds
        last_progress_at = $LastProgressAt.ToString("o")
        progress_idle_seconds = $idleSeconds
        stall_threshold_seconds = $StallSeconds
        stall_detected = (
            $State -like "running*" -and
            $idleSeconds -ge $StallSeconds
        )
        run_id = $RunId
        run_directory = $script:RunDirectory
        expected_updates = $ExpectedUpdates
        launch_manifest_path = $launchManifestPath
        launch_manifest_sha256 = $launchManifestSha256
        telemetry = $Telemetry
        recovery_path = if (
            -not [string]::IsNullOrWhiteSpace(
                [string]$script:RecoveryPath
            ) -and
            (Test-Path -LiteralPath $script:RecoveryPath -PathType Leaf)
        ) {
            $script:RecoveryPath
        }
        else {
            $null
        }
        failure = $Failure
    }
}

$scriptDirectory = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
    $ProjectPath = Split-Path -Parent $scriptDirectory
}
if ([string]::IsNullOrWhiteSpace($TrainingScript)) {
    $TrainingScript = Join-Path $scriptDirectory "train_arena_100k.jl"
}
if ([string]::IsNullOrWhiteSpace($VerifierScript)) {
    $VerifierScript = Join-Path $scriptDirectory "verify_arena_run.jl"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $scriptDirectory "trained"
}
if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $WorkingDirectory = (
        Resolve-Path -LiteralPath (
            Join-Path $scriptDirectory "..\..\.."
        )
    ).Path
}
if ([string]::IsNullOrWhiteSpace($JuliaExecutable)) {
    $JuliaExecutable = if (
        -not [string]::IsNullOrWhiteSpace($env:SWSNN_JULIA_EXECUTABLE)
    ) {
        $env:SWSNN_JULIA_EXECUTABLE
    }
    else {
        "julia"
    }
}
if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
    $DatasetPath =
        "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
}

$juliaPath = Resolve-ExecutablePath -Value $JuliaExecutable
$resolvedProject = (Resolve-Path -LiteralPath $ProjectPath).Path
$resolvedTrainingScript = (Resolve-Path -LiteralPath $TrainingScript).Path
$resolvedVerifierScript = (Resolve-Path -LiteralPath $VerifierScript).Path
$resolvedControllerScript = (Resolve-Path -LiteralPath $PSCommandPath).Path
$canonicalProject = (
    Resolve-Path -LiteralPath (
        Split-Path -Parent $scriptDirectory
    )
).Path
$canonicalTrainingScript = (
    Resolve-Path -LiteralPath (
        Join-Path $scriptDirectory "train_arena_100k.jl"
    )
).Path
$canonicalVerifierScript = (
    Resolve-Path -LiteralPath (
        Join-Path $scriptDirectory "verify_arena_run.jl"
    )
).Path
if ($resolvedTrainingScript -ne $canonicalTrainingScript) {
    throw "Production controller rejects a noncanonical training script"
}
if ($resolvedVerifierScript -ne $canonicalVerifierScript) {
    throw "Production controller rejects a noncanonical verifier script"
}
if ($resolvedProject -ne $canonicalProject) {
    throw (
        "Production controller rejects a noncanonical project path: " +
        "$resolvedProject"
    )
}
$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$resolvedDatasetPath = (Resolve-Path -LiteralPath $DatasetPath).Path
$requestedOutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$mutexScope = (
    $canonicalProject.ToLowerInvariant() +
    "`n" +
    $requestedOutputRoot.ToLowerInvariant()
)
$mutexScopeSha256 = Get-Utf8Sha256Hex -Value $mutexScope
$canonicalMutexName = (
    "Local\OpenAI.SerialWorkspaceSNN.Arena100k.v1." +
    $mutexScopeSha256.Substring(0, 32)
)
if (
    -not [string]::IsNullOrWhiteSpace($MutexName) -and
    $MutexName -cne $canonicalMutexName
) {
    throw (
        "MutexName is controller-derived and cannot be overridden; " +
        "expected $canonicalMutexName"
    )
}
$MutexName = $canonicalMutexName
Assert-NoReparsePathChain `
    -Path $requestedOutputRoot `
    -Context "requested output root before creation" |
    Out-Null
if ($VerifyOnly) {
    if (
        -not (
            Test-Path -LiteralPath $requestedOutputRoot -PathType Container
        )
    ) {
        throw "Verify-only output root does not exist: $requestedOutputRoot"
    }
}
else {
    [IO.Directory]::CreateDirectory($requestedOutputRoot) | Out-Null
}
$resolvedOutputRoot = (
    Resolve-Path -LiteralPath $requestedOutputRoot
).Path
if (
    -not [StringComparer]::Ordinal.Equals(
        $requestedOutputRoot,
        $resolvedOutputRoot
    )
) {
    throw "OutputRoot is not a canonical non-aliased path"
}
Assert-NoReparsePathChain `
    -Path $resolvedOutputRoot `
    -Context "output root" |
    Out-Null
$outputRootItem = Assert-NotReparsePoint `
    -Path $resolvedOutputRoot `
    -Context "output root"
if (-not $outputRootItem.PSIsContainer) {
    throw "OutputRoot is not a directory"
}
$expectedRunDirectory = [IO.Path]::Combine($resolvedOutputRoot, $RunId)
$script:RunDirectory = [IO.Path]::GetFullPath($expectedRunDirectory)
$controllerRoot = Join-Path $resolvedOutputRoot "_controllers"
$controllerDirectory = [IO.Path]::GetFullPath(
    [IO.Path]::Combine($controllerRoot, $RunId)
)
if (
    -not [StringComparer]::Ordinal.Equals(
        [IO.Path]::GetDirectoryName($script:RunDirectory),
        $resolvedOutputRoot
    ) -or
    -not [StringComparer]::Ordinal.Equals(
        [IO.Path]::GetFileName($script:RunDirectory),
        $RunId
    ) -or
    -not [StringComparer]::Ordinal.Equals(
        [IO.Path]::GetDirectoryName($controllerDirectory),
        $controllerRoot
    ) -or
    -not [StringComparer]::Ordinal.Equals(
        [IO.Path]::GetFileName($controllerDirectory),
        $RunId
    )
) {
    throw "RunId escaped its canonical immediate-child directories"
}
if (Test-Path -LiteralPath $controllerRoot) {
    $controllerRootItem = Assert-NotReparsePoint `
        -Path $controllerRoot `
        -Context "controller root"
    if (-not $controllerRootItem.PSIsContainer) {
        throw "controller root is not a directory"
    }
}
$statusPath = Join-Path $controllerDirectory "controller_status.json"
$launchManifestPath = Join-Path $controllerDirectory "launch_manifest.json"
$stdoutPath = Join-Path $controllerDirectory "training.stdout.log"
$stderrPath = Join-Path $controllerDirectory "training.stderr.log"
$verifierStdoutPath = Join-Path $controllerDirectory "verifier.stdout.log"
$verifierStderrPath = Join-Path $controllerDirectory "verifier.stderr.log"
$recoveryPath = Join-Path $controllerDirectory "recovery.json"
$script:RecoveryPath = $recoveryPath
$originalControllerDirectory = $controllerDirectory
$originalStatusPath = $statusPath
$originalRecoveryPath = $recoveryPath

$mutex = $null
$mutexAcquired = $false
$controllerDirectoryCreated = $false
$child = $null
$verifier = $null
$script:VerifierProcessId = $null
$childExitCode = $null
$verifierExitCode = $null
$jobHandle = [IntPtr]::Zero
$jobQuiescenceConfirmed = $false
$controllerExitCode = 1
$lastProgressAt = Get-Date
$lastObservedUpdate = -1
$failureState = "failed"
$stallWarningAt = $null
$launchManifestSha256 = $null
$parentCheckpointContract = $null
$parentLineageSnapshot = @()
$script:ParentLineageTrustLost = $false
$effectiveStartMode = $StartMode
$verifyOnlyLaunchAuthenticated = $false
$retryAttemptId = $null
$retryControllerDirectory = $null
$archivedVerificationPath = $null
$childStartTime = $null
$exitTelemetry = $null
$newVerificationRetryCommand = {
    if (
        [string]::IsNullOrWhiteSpace($launchManifestSha256) -or
        $launchManifestSha256 -notmatch "^[0-9a-f]{64}$"
    ) {
        throw "Cannot build a verification retry without a pinned manifest"
    }
    New-VerificationRetryCommand `
        -ControllerScript $resolvedControllerScript `
        -RunId $RunId `
        -ExpectedUpdates $ExpectedUpdates `
        -JuliaThreads $JuliaThreads `
        -PollSeconds $PollSeconds `
        -StallSeconds $StallSeconds `
        -VerifierTimeoutSeconds $VerifierTimeoutSeconds `
        -JuliaPath $juliaPath `
        -ProjectPath $resolvedProject `
        -TrainingScript $resolvedTrainingScript `
        -VerifierScript $resolvedVerifierScript `
        -DatasetPath $resolvedDatasetPath `
        -OutputRoot $resolvedOutputRoot `
        -WorkingDirectory $resolvedWorkingDirectory `
        -MutexName $MutexName `
        -LaunchManifestSha256 $launchManifestSha256
}

try {
    $createdNew = $false
    $mutex = [Threading.Mutex]::new($false, $MutexName, [ref]$createdNew)
    try {
        $mutexAcquired = $mutex.WaitOne(0)
    }
    catch [Threading.AbandonedMutexException] {
        $mutexAcquired = $true
    }
    if (-not $mutexAcquired) {
        throw "Another arena 100k controller holds named mutex $MutexName"
    }

    $juliaRuntimeArguments = @(
        "--startup-file=no"
        "--history-file=no"
    )
    :TrainingPhase do {
        if ($VerifyOnly) {
            if ($PSBoundParameters.ContainsKey("StartMode")) {
                throw (
                    "Verify-only reads the original start mode from the " +
                    "pinned launch manifest; do not pass -StartMode"
                )
            }
            if (
                -not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -or
                -not [string]::IsNullOrWhiteSpace($ResumeSha256) -or
                $ResumeUpdate -ne -1
            ) {
                throw (
                    "Verify-only reads the original parent checkpoint " +
                    "contract from the pinned launch manifest"
                )
            }
            if (
                [string]::IsNullOrWhiteSpace(
                    $ExpectedLaunchManifestSha256
                ) -or
                $ExpectedLaunchManifestSha256 -notmatch
                    "^[A-Fa-f0-9]{64}$"
            ) {
                throw (
                    "Verify-only requires a 64-digit " +
                    "-ExpectedLaunchManifestSha256"
                )
            }
            if (
                -not (
                    Test-Path -LiteralPath $script:RunDirectory `
                        -PathType Container
                )
            ) {
                throw (
                    "Verify-only run directory does not exist: " +
                    $script:RunDirectory
                )
            }
            if (
                -not (
                    Test-Path -LiteralPath $controllerDirectory `
                        -PathType Container
                )
            ) {
                throw (
                    "Verify-only controller directory does not exist: " +
                    $controllerDirectory
                )
            }
            foreach (
                $requiredDirectory in @(
                    $script:RunDirectory
                    $controllerDirectory
                )
            ) {
                $requiredDirectoryItem =
                    Get-Item -LiteralPath $requiredDirectory -Force
                Assert-NoReparsePathChain `
                    -Path $requiredDirectory `
                    -Context "Verify-only required directory" |
                    Out-Null
                if (
                    ($requiredDirectoryItem.Attributes -band
                        [IO.FileAttributes]::ReparsePoint) -ne 0
                ) {
                    throw (
                        "Verify-only rejects a reparse-point directory: " +
                        $requiredDirectory
                    )
                }
            }
            if (
                -not (
                    Test-Path -LiteralPath $launchManifestPath `
                        -PathType Leaf
                )
            ) {
                throw (
                    "Verify-only launch manifest does not exist: " +
                    $launchManifestPath
                )
            }
            $launchManifestItem =
                Get-Item -LiteralPath $launchManifestPath -Force
            if (
                ($launchManifestItem.Attributes -band
                    [IO.FileAttributes]::ReparsePoint) -ne 0
            ) {
                throw "Verify-only rejects a reparse-point launch manifest"
            }
            $launchManifestSha256 =
                Get-Sha256Hex -Path $launchManifestPath
            if (
                $launchManifestSha256 -ne
                    $ExpectedLaunchManifestSha256.ToLowerInvariant()
            ) {
                throw "Verify-only launch manifest SHA-256 differs"
            }
            $env:SWSNN_LAUNCH_MANIFEST_PATH = $launchManifestPath
            $env:SWSNN_LAUNCH_MANIFEST_SHA256 =
                $launchManifestSha256
            try {
                $rawLaunchManifest =
                    Get-Content -Raw -LiteralPath $launchManifestPath
                Assert-StrictTopLevelJsonObject `
                    -RawJson $rawLaunchManifest `
                    -Context "Verify-only launch manifest"
                $launchManifest =
                    ConvertFrom-Json -InputObject $rawLaunchManifest
            }
            catch {
                throw "Verify-only could not parse launch manifest: $_"
            }
            Assert-JsonProperties `
                -Value $launchManifest `
                -Context "Verify-only launch manifest" `
                -Names @(
                    "format"
                    "version"
                    "run_id"
                    "run_directory"
                    "expected_updates"
                    "start_mode"
                    "expected_contract"
                    "parent_checkpoint"
                    "julia_executable"
                    "julia_threads"
                    "julia_runtime_arguments"
                    "project_path"
                    "training_script"
                    "verifier_script"
                    "controller_script"
                    "code_artifacts"
                    "working_directory"
                    "output_root"
                    "mutex_name"
                    "mutex_scope_sha256"
                    "parent_lineage"
                    "stdout_path"
                    "stderr_path"
                )
            Assert-JsonProperties `
                -Value $launchManifest.expected_contract `
                -Context "Verify-only expected contract" `
                -Names @(
                    "start_mode"
                    "scratch"
                    "canonical_project_path"
                    "startup_file"
                    "history_file"
                    "dataset_path"
                    "active_workers"
                    "eprop_reducers"
                )
            Assert-JsonProperties `
                -Value $launchManifest.code_artifacts `
                -Context "Verify-only code artifacts" `
                -Names @(
                    "controller"
                    "training"
                    "verifier"
                )
            if (
                [string]$launchManifest.format -cne
                    "serial-workspace-snn-arena-run-launch" -or
                [int]$launchManifest.version -ne 2
            ) {
                throw "Verify-only launch manifest format/version differs"
            }
            if ([string]$launchManifest.run_id -cne $RunId) {
                throw "Verify-only launch manifest run identity differs"
            }
            $resolvedRunDirectory = (
                Resolve-Path -LiteralPath $script:RunDirectory
            ).Path
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.run_directory `
                -ExpectedPath $resolvedRunDirectory `
                -Context "Verify-only launch manifest run directory" |
                Out-Null
            if (
                [int]$launchManifest.expected_updates -ne
                    $ExpectedUpdates
            ) {
                throw (
                    "Verify-only expected update count differs from the " +
                    "launch manifest"
                )
            }
            $effectiveStartMode = [string]$launchManifest.start_mode
            if (
                $effectiveStartMode -notin @(
                    "scratch"
                    "resume"
                    "finalize-only"
                )
            ) {
                throw "Verify-only launch start mode is invalid"
            }
            if (
                [string]$launchManifest.expected_contract.start_mode -cne
                    $effectiveStartMode -or
                [bool]$launchManifest.expected_contract.scratch -ne
                    ($effectiveStartMode -eq "scratch")
            ) {
                throw "Verify-only launch start-mode contract differs"
            }
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.project_path `
                -ExpectedPath $canonicalProject `
                -Context "Verify-only launch project path" |
                Out-Null
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.expected_contract.
                    canonical_project_path `
                -ExpectedPath $canonicalProject `
                -Context "Verify-only expected canonical project path" |
                Out-Null
            if (
                [bool]$launchManifest.expected_contract.startup_file -or
                [bool]$launchManifest.expected_contract.history_file
            ) {
                throw "Verify-only launch project/runtime contract differs"
            }
            $manifestRuntimeArguments = @(
                $launchManifest.julia_runtime_arguments
            )
            if (
                $manifestRuntimeArguments.Count -ne 2 -or
                [string]$manifestRuntimeArguments[0] -ne
                    "--startup-file=no" -or
                [string]$manifestRuntimeArguments[1] -ne
                    "--history-file=no"
            ) {
                throw "Verify-only launch Julia runtime arguments differ"
            }
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.julia_executable `
                -ExpectedPath $juliaPath `
                -Context "Verify-only launch Julia executable" |
                Out-Null
            if ([int]$launchManifest.julia_threads -ne $JuliaThreads) {
                throw "Verify-only launch Julia executable/threads differ"
            }
            foreach (
                $verifyOnlyScriptPath in @(
                    [ordered]@{
                        name = "training"
                        raw = $launchManifest.training_script
                        expected = $canonicalTrainingScript
                    }
                    [ordered]@{
                        name = "verifier"
                        raw = $launchManifest.verifier_script
                        expected = $canonicalVerifierScript
                    }
                    [ordered]@{
                        name = "controller"
                        raw = $launchManifest.controller_script
                        expected = $resolvedControllerScript
                    }
                )
            ) {
                Assert-ExactCanonicalPathString `
                    -RawPath $verifyOnlyScriptPath.raw `
                    -ExpectedPath $verifyOnlyScriptPath.expected `
                    -Context (
                        "Verify-only launch " +
                        $verifyOnlyScriptPath.name +
                        " script"
                    ) |
                    Out-Null
            }
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.working_directory `
                -ExpectedPath $resolvedWorkingDirectory `
                -Context "Verify-only launch working directory" |
                Out-Null
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.output_root `
                -ExpectedPath $resolvedOutputRoot `
                -Context "Verify-only launch output root" |
                Out-Null
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.expected_contract.dataset_path `
                -ExpectedPath $resolvedDatasetPath `
                -Context "Verify-only launch dataset path" |
                Out-Null
            if (
                [string]$launchManifest.mutex_name -cne $MutexName -or
                [string]$launchManifest.mutex_scope_sha256 -cne
                    $mutexScopeSha256
            ) {
                throw (
                    "Verify-only launch working/output/dataset/mutex " +
                    "contract differs"
                )
            }
            if (
                [int]$launchManifest.expected_contract.active_workers -ne
                    $JuliaThreads -or
                [int]$launchManifest.expected_contract.eprop_reducers -ne
                    $JuliaThreads
            ) {
                throw "Verify-only launch worker contract differs"
            }
            foreach (
                $codeArtifact in @(
                    [ordered]@{
                        name = "controller"
                        manifest = $launchManifest.code_artifacts.controller
                        path = $resolvedControllerScript
                    }
                    [ordered]@{
                        name = "training"
                        manifest = $launchManifest.code_artifacts.training
                        path = $resolvedTrainingScript
                    }
                    [ordered]@{
                        name = "verifier"
                        manifest = $launchManifest.code_artifacts.verifier
                        path = $resolvedVerifierScript
                    }
                )
            ) {
                Assert-JsonProperties `
                    -Value $codeArtifact.manifest `
                    -Context (
                        "Verify-only $($codeArtifact.name) code artifact"
                    ) `
                    -Names @(
                        "path"
                        "bytes"
                        "sha256"
                    )
                $actualCodeItem =
                    Get-Item -LiteralPath $codeArtifact.path
                Assert-ExactCanonicalPathString `
                    -RawPath $codeArtifact.manifest.path `
                    -ExpectedPath $codeArtifact.path `
                    -Context (
                        "Verify-only $($codeArtifact.name) code artifact path"
                    ) |
                    Out-Null
                if (
                    [long]$codeArtifact.manifest.bytes -ne
                        $actualCodeItem.Length -or
                    ([string]$codeArtifact.manifest.sha256).
                        ToLowerInvariant() -ne
                        (Get-Sha256Hex -Path $codeArtifact.path)
                ) {
                    throw (
                        "Verify-only launch code artifact differs: " +
                        $codeArtifact.name
                    )
                }
            }
            $verifyOnlyLaunchAuthenticated = $true
            $retryRoot =
                Join-Path $controllerDirectory "verification_retries"
            if (Test-Path -LiteralPath $retryRoot) {
                Assert-NoReparsePathChain `
                    -Path $retryRoot `
                    -Context "verification retry root" |
                    Out-Null
                $retryRootItem = Assert-NotReparsePoint `
                    -Path $retryRoot `
                    -Context "verification retry root"
                if (-not $retryRootItem.PSIsContainer) {
                    throw "verification retry root is not a directory"
                }
            }
            else {
                [IO.Directory]::CreateDirectory($retryRoot) | Out-Null
                Assert-NoReparsePathChain `
                    -Path $retryRoot `
                    -Context "verification retry root" |
                    Out-Null
                $retryRootItem = Assert-NotReparsePoint `
                    -Path $retryRoot `
                    -Context "verification retry root"
            }
            $retryAttemptId = (
                "verification_retry_" +
                [DateTime]::UtcNow.ToString(
                    "yyyyMMddTHHmmssfffffffZ",
                    [Globalization.CultureInfo]::InvariantCulture
                ) +
                "_pid$PID"
            )
            $retryControllerDirectory =
                Join-Path $retryRoot $retryAttemptId
            if (Test-Path -LiteralPath $retryControllerDirectory) {
                throw (
                    "Verify-only retry controller directory already " +
                    "exists: $retryControllerDirectory"
                )
            }
            [IO.Directory]::CreateDirectory(
                $retryControllerDirectory
            ) | Out-Null
            Assert-NoReparsePathChain `
                -Path $retryControllerDirectory `
                -Context "verification retry attempt directory" |
                Out-Null
            $retryControllerDirectoryItem = Assert-NotReparsePoint `
                -Path $retryControllerDirectory `
                -Context "verification retry attempt directory"
            if (-not $retryControllerDirectoryItem.PSIsContainer) {
                throw "verification retry attempt path is not a directory"
            }
            $controllerDirectoryCreated = $true
            $statusPath =
                Join-Path $retryControllerDirectory "controller_status.json"
            $verifierStdoutPath =
                Join-Path $retryControllerDirectory "verifier.stdout.log"
            $verifierStderrPath =
                Join-Path $retryControllerDirectory "verifier.stderr.log"
            $recoveryPath =
                Join-Path $retryControllerDirectory "recovery.json"
            $script:RecoveryPath = $recoveryPath

            $parentCheckpointContract = $launchManifest.parent_checkpoint
            $parentLineageSnapshot = @($launchManifest.parent_lineage)
            if ($effectiveStartMode -eq "scratch") {
                if (
                    $null -ne $parentCheckpointContract -or
                    $parentLineageSnapshot.Count -ne 0
                ) {
                    throw (
                        "Verify-only scratch launch unexpectedly has a " +
                        "parent checkpoint"
                    )
                }
            }
            else {
                if ($null -eq $parentCheckpointContract) {
                    throw (
                        "Verify-only non-scratch launch lacks a parent " +
                        "checkpoint"
                    )
                }
                Assert-JsonProperties `
                    -Value $parentCheckpointContract `
                    -Context "Verify-only parent checkpoint" `
                    -Names @(
                        "path"
                        "sha256"
                        "update"
                    )
                if (
                    [string]$parentCheckpointContract.sha256 -notmatch
                        "^[A-Fa-f0-9]{64}$" -or
                    [int]$parentCheckpointContract.update -lt 0
                ) {
                    throw (
                        "Verify-only parent checkpoint triple is invalid"
                    )
                }
                $resolvedParentCheckpoint = (
                    Resolve-Path -LiteralPath (
                        [string]$parentCheckpointContract.path
                    )
                ).Path
                $parentCheckpointSha256 =
                    Get-Sha256Hex -Path $resolvedParentCheckpoint
                if (
                    $parentCheckpointSha256 -ne
                        ([string]$parentCheckpointContract.sha256).
                            ToLowerInvariant()
                ) {
                    throw "Verify-only parent checkpoint SHA-256 differs"
                }
                if (
                    (
                        $effectiveStartMode -eq "resume" -and
                        [int]$parentCheckpointContract.update -ge
                            $ExpectedUpdates
                    ) -or
                    (
                        $effectiveStartMode -eq "finalize-only" -and
                        [int]$parentCheckpointContract.update -ne
                            $ExpectedUpdates
                    )
                ) {
                    throw "Verify-only parent checkpoint update differs"
                }
                $parentCheckpointContract = [ordered]@{
                    path = $resolvedParentCheckpoint
                    sha256 = $parentCheckpointSha256
                    update = [int]$parentCheckpointContract.update
                }
                if ($parentLineageSnapshot.Count -eq 0) {
                    throw "Verify-only launch lacks a pinned parent lineage"
                }
                $pinnedSelected =
                    $parentLineageSnapshot[0].selected_checkpoint
                if (
                    [string]$pinnedSelected.path -cne
                        [string]$parentCheckpointContract.path -or
                    [string]$pinnedSelected.sha256 -cne
                        [string]$parentCheckpointContract.sha256 -or
                    [int]$pinnedSelected.update -ne
                        [int]$parentCheckpointContract.update
                ) {
                    throw "Verify-only pinned parent lineage identity differs"
                }
                Assert-PinnedLineageCurrentFailClosed `
                    -PinnedLineage $parentLineageSnapshot
            }
            if (
                -not (
                    Test-Path -LiteralPath (
                        Join-Path $script:RunDirectory "results.json"
                    ) -PathType Leaf
                )
            ) {
                throw "Verify-only requires an existing results.json"
            }
            $verifyOnlyResultsPath =
                Join-Path $script:RunDirectory "results.json"
            Assert-NoReparsePathChain `
                -Path $verifyOnlyResultsPath `
                -Context "Verify-only results.json" |
                Out-Null
            $verifyOnlyResultsItem = Assert-NotReparsePoint `
                -Path $verifyOnlyResultsPath `
                -Context "Verify-only results.json"
            if ($verifyOnlyResultsItem.PSIsContainer) {
                throw "Verify-only results.json is not a regular file"
            }
            $expectedTrainingStdoutPath =
                Join-Path $controllerDirectory "training.stdout.log"
            $expectedTrainingStderrPath =
                Join-Path $controllerDirectory "training.stderr.log"
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.stdout_path `
                -ExpectedPath $expectedTrainingStdoutPath `
                -Context "Verify-only launch training stdout path" |
                Out-Null
            Assert-ExactCanonicalPathString `
                -RawPath $launchManifest.stderr_path `
                -ExpectedPath $expectedTrainingStderrPath `
                -Context "Verify-only launch training stderr path" |
                Out-Null
            $stdoutPath = $expectedTrainingStdoutPath
            $stderrPath = $expectedTrainingStderrPath

            $env:SWSNN_LEARNING_MODE = "local_hybrid"
            $env:SWSNN_MODEL_PRESET = "scaled_v2"
            $env:SWSNN_STATE_BATCH = "8"
            $env:SWSNN_ACTIVE_WORKERS = [string]$JuliaThreads
            $env:SWSNN_EPROP_REDUCERS = [string]$JuliaThreads
            $env:SWSNN_CPUSET_MODE = "all"
            $env:SWSNN_LEARNING_RATE = "0.0005"
            $env:SWSNN_WEIGHT_DECAY = "0.00001"
            $env:SWSNN_STRUCTURE_WEIGHT = "0.01"
            $env:SWSNN_STRUCTURAL_INTERVAL = "25"
            $env:SWSNN_CHECKPOINT_INTERVAL = "10000"
            $env:SWSNN_LOG_INTERVAL = "1000"
            $env:SWSNN_EVAL_STATES = "128"
            $env:SWSNN_UTILITY_DECAY = "0.99"
            $env:SWSNN_UTILITY_CONNECTION_COST = "0.000001"
            $env:SWSNN_UTILITY_KEEP_FRACTION = "0.5"
            $env:SWSNN_UTILITY_TURNOVER_PERIOD = "128"
            $env:SWSNN_MAX_HOT_ALLOCATION_BYTES = "0"
            $env:SWSNN_DATASET = $resolvedDatasetPath
            $env:SWSNN_RUN_ID = $RunId
            $env:SWSNN_OUTPUT = $resolvedOutputRoot
            $env:SWSNN_MAX_UPDATES = [string]$ExpectedUpdates
            $env:SWSNN_START_MODE =
                $effectiveStartMode.Replace("-", "_")
            if ($null -eq $parentCheckpointContract) {
                Remove-Item Env:SWSNN_RESUME_CHECKPOINT `
                    -ErrorAction SilentlyContinue
                Remove-Item Env:SWSNN_RESUME_SHA256 `
                    -ErrorAction SilentlyContinue
                $env:SWSNN_SCRATCH = "true"
            }
            else {
                $env:SWSNN_RESUME_CHECKPOINT =
                    [string]$parentCheckpointContract.path
                $env:SWSNN_RESUME_SHA256 =
                    [string]$parentCheckpointContract.sha256
                $env:SWSNN_SCRATCH = "false"
            }
            $exitTelemetry = Get-RunTelemetry `
                -RunDirectory $script:RunDirectory `
                -StandardOutputPath $stdoutPath `
                -StandardErrorPath $stderrPath
            $lastProgressAt = Get-ProgressTimeFromTelemetry `
                -Telemetry $exitTelemetry `
                -CurrentProgressAt $lastProgressAt
            break TrainingPhase
        }
        if (
            -not [string]::IsNullOrWhiteSpace(
                $ExpectedLaunchManifestSha256
            )
        ) {
            throw (
                "-ExpectedLaunchManifestSha256 is valid only with " +
                "-VerifyOnly"
            )
        }

    # A controller launch is a production contract, not an invitation for
    # inherited shell state to select a different learner.  Set every
    # correctness-relevant production value explicitly before snapshotting the
    # launch manifest or starting Julia.
    $env:SWSNN_LEARNING_MODE = "local_hybrid"
    $env:SWSNN_MODEL_PRESET = "scaled_v2"
    $env:SWSNN_STATE_BATCH = "8"
    $env:SWSNN_ACTIVE_WORKERS = [string]$JuliaThreads
    $env:SWSNN_EPROP_REDUCERS = [string]$JuliaThreads
    $env:SWSNN_CPUSET_MODE = "all"
    $env:SWSNN_LEARNING_RATE = "0.0005"
    $env:SWSNN_WEIGHT_DECAY = "0.00001"
    $env:SWSNN_STRUCTURE_WEIGHT = "0.01"
    $env:SWSNN_STRUCTURAL_INTERVAL = "25"
    $env:SWSNN_CHECKPOINT_INTERVAL = "10000"
    $env:SWSNN_LOG_INTERVAL = "1000"
    $env:SWSNN_EVAL_STATES = "128"
    $env:SWSNN_UTILITY_DECAY = "0.99"
    $env:SWSNN_UTILITY_CONNECTION_COST = "0.000001"
    $env:SWSNN_UTILITY_KEEP_FRACTION = "0.5"
    $env:SWSNN_UTILITY_TURNOVER_PERIOD = "128"
    $env:SWSNN_MAX_HOT_ALLOCATION_BYTES = "0"
    $env:SWSNN_DATASET = $resolvedDatasetPath

    $parentCheckpointContract = $null
    if ($StartMode -eq "scratch") {
        if (
            -not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -or
            -not [string]::IsNullOrWhiteSpace($ResumeSha256) -or
            $ResumeUpdate -ne -1
        ) {
            throw "Scratch mode cannot accept resume checkpoint values"
        }
        Remove-Item Env:SWSNN_RESUME_CHECKPOINT -ErrorAction SilentlyContinue
        Remove-Item Env:SWSNN_RESUME_SHA256 -ErrorAction SilentlyContinue
        $env:SWSNN_SCRATCH = "true"
    }
    else {
        if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
            throw "$StartMode mode requires -ResumeCheckpoint"
        }
        if (
            [string]::IsNullOrWhiteSpace($ResumeSha256) -or
            $ResumeSha256 -notmatch "^[A-Fa-f0-9]{64}$"
        ) {
            throw "$StartMode mode requires a 64-digit -ResumeSha256"
        }
        $resolvedResume = (
            Resolve-Path -LiteralPath $ResumeCheckpoint
        ).Path
        $resumeItem = Assert-NotReparsePoint `
            -Path $resolvedResume `
            -Context "resume checkpoint"
        if ($resumeItem.PSIsContainer) {
            throw "Resume checkpoint is not a regular file"
        }
        $actualResumeSha256 = Get-Sha256Hex -Path $resolvedResume
        if ($actualResumeSha256 -ne $ResumeSha256.ToLowerInvariant()) {
            throw "Resume checkpoint SHA-256 differs"
        }
        $encodedResumeUpdate =
            Get-CheckpointUpdateFromPath -Path $resolvedResume
        if (
            $ResumeUpdate -ne -1 -and
            $ResumeUpdate -ne $encodedResumeUpdate
        ) {
            throw (
                "Resume checkpoint filename update differs from " +
                "-ResumeUpdate"
            )
        }
        $ResumeUpdate = $encodedResumeUpdate
        if (
            $StartMode -eq "resume" -and
            $ResumeUpdate -ge $ExpectedUpdates
        ) {
            throw "Resume parent update must be less than expected updates"
        }
        if (
            $StartMode -eq "finalize-only" -and
            $ResumeUpdate -ne $ExpectedUpdates
        ) {
            throw (
                "Finalize-only parent update must equal expected updates"
            )
        }
        if ($StartMode -eq "finalize-only") {
            $parentRunDirectory = Split-Path -Parent (
                Split-Path -Parent $resolvedResume
            )
            $parentResultsPath =
                Join-Path $parentRunDirectory "results.json"
            $parentVerificationPath =
                Join-Path $parentRunDirectory "verification.json"
            if (
                Test-Path -LiteralPath $parentResultsPath -PathType Leaf
            ) {
                throw (
                    "Finalize-only recovery is not allowed when the parent " +
                    "run already has results.json"
                )
            }
            if (
                Test-Path -LiteralPath $parentVerificationPath -PathType Leaf
            ) {
                Assert-NoReparsePathChain `
                    -Path $parentVerificationPath `
                    -Context "finalize-only parent verification.json" |
                    Out-Null
                $parentVerificationItem = Assert-NotReparsePoint `
                    -Path $parentVerificationPath `
                    -Context "finalize-only parent verification.json"
                if ($parentVerificationItem.PSIsContainer) {
                    throw (
                        "Finalize-only parent verification.json is not " +
                        "a regular file"
                    )
                }
                $rawParentVerification =
                    Get-Content -Raw -LiteralPath $parentVerificationPath
                Assert-StrictTopLevelJsonObject `
                    -RawJson $rawParentVerification `
                    -Context "finalize-only parent verification.json"
                $parentVerification =
                    ConvertFrom-Json -InputObject $rawParentVerification
                if (
                    $parentVerification.verified -eq $true -or
                    $parentVerification.status -eq
                        "verified_complete"
                ) {
                    throw (
                        "Finalize-only recovery is not allowed for an " +
                        "already verified parent run"
                    )
                }
            }
        }
        $env:SWSNN_RESUME_CHECKPOINT = $resolvedResume
        $env:SWSNN_RESUME_SHA256 = $actualResumeSha256
        $env:SWSNN_SCRATCH = "false"
        $parentCheckpointContract = [ordered]@{
            path = $resolvedResume
            sha256 = $actualResumeSha256
            update = $ResumeUpdate
        }
        $parentLineageSnapshot = @(
            Get-ParentLineageSnapshot `
                -SelectedCheckpoint $resolvedResume `
                -SelectedSha256 $actualResumeSha256 `
                -SelectedUpdate $ResumeUpdate `
                -ExpectedOutputRoot $resolvedOutputRoot `
                -ExpectedProject $canonicalProject `
                -ExpectedTrainingScript $canonicalTrainingScript `
                -ExpectedVerifierScript $canonicalVerifierScript `
                -ExpectedControllerScript $resolvedControllerScript
        )
    }

    if (Test-Path -LiteralPath $script:RunDirectory) {
        throw "Run directory already exists: $script:RunDirectory"
    }
    if (Test-Path -LiteralPath $controllerDirectory) {
        throw "Controller directory already exists: $controllerDirectory"
    }
    [IO.Directory]::CreateDirectory($controllerDirectory) | Out-Null
    $controllerDirectoryCreated = $true
    Assert-NoReparsePathChain `
        -Path $controllerDirectory `
        -Context "controller directory" |
        Out-Null
    $createdControllerRoot = Assert-NotReparsePoint `
        -Path $controllerRoot `
        -Context "controller root"
    $createdControllerDirectory = Assert-NotReparsePoint `
        -Path $controllerDirectory `
        -Context "controller directory"
    if (
        -not $createdControllerRoot.PSIsContainer -or
        -not $createdControllerDirectory.PSIsContainer
    ) {
        throw "controller artifact path is not a directory"
    }

    $env:SWSNN_RUN_ID = $RunId
    $env:SWSNN_OUTPUT = $resolvedOutputRoot
    $env:SWSNN_MAX_UPDATES = [string]$ExpectedUpdates
    $env:SWSNN_START_MODE = $StartMode.Replace("-", "_")
    $juliaRuntimeArguments = @(
        "--startup-file=no"
        "--history-file=no"
    )

    $launchManifest = [ordered]@{
        format = "serial-workspace-snn-arena-run-launch"
        version = 2
        created_at = (Get-Date).ToString("o")
        notification_capability = "none"
        mutex_name = $MutexName
        mutex_scope_sha256 = $mutexScopeSha256
        controller_pid = $PID
        run_id = $RunId
        run_directory = $script:RunDirectory
        expected_updates = $ExpectedUpdates
        start_mode = $StartMode
        training_stall_seconds = $StallSeconds
        verifier_timeout_seconds = $VerifierTimeoutSeconds
        expected_contract = [ordered]@{
            experiment_id = "serial_workspace_snn_arena_v3"
            checkpoint_format =
                "serial-workspace-snn-arena-checkpoint"
            checkpoint_version = 3
            canonical_project_path = $canonicalProject
            startup_file = $false
            history_file = $false
            start_mode = $StartMode
            scratch = $StartMode -eq "scratch"
            learning_mode = "local_hybrid"
            model_preset = "scaled_v2"
            state_batch = 8
            active_workers = $JuliaThreads
            eprop_reducers = $JuliaThreads
            cpuset_mode = "all"
            structural_interval = 25
            checkpoint_interval = 10000
            log_interval = 1000
            evaluation_states = 128
            learning_rate = 0.0005
            weight_decay = 0.00001
            structure_weight = 0.01
            maximum_hot_allocation_bytes = 0
            full_eprop = [ordered]@{
                analytic_vjp = $false
                supervised_head_vjp = $true
                recurrent_credit_assignment =
                    "eprop_decolle_block_local_three_factor"
                feedback_mode = "block_local"
                edge_parameter_mode = "weight_gate_delay"
                node_parameter_mode = "full_state"
                routing_parameter_mode = "three_factor"
                signal_schedule = "all_cycles"
                third_factor_mode = "aligned"
                time_order = "forward"
            }
            routing = [ordered]@{
                inference_selection =
                    "deterministic_hard_top_k"
                training_selection =
                    "stochastic_hard_top_k_without_replacement"
                parameter_update =
                    "ordered_plackett_luce_score_eligibility_three_factor"
                learning_signal_semantics =
                    "supervised_reward_surrogate"
                reward_source =
                    "candidate_centered_listnet_and_auxiliary_loss_advantage"
                exploration_probability = 0.05
                entropy_weight = 0.002
                entropy_floor = 0.7
                load_balance_weight = 0.002
            }
            utility = [ordered]@{
                mode = "utility"
                decay = 0.99
                connection_cost = 0.000001
                keep_fraction = 0.5
                turnover_period = 128
                responsibility =
                    "normalized_abs_block_signal_times_eligibility"
            }
            model = [ordered]@{
                blocks = 96
                nodes = 4608
                candidate_synapses = 110592
                enabled_synapses = 55296
                fanout = 24
                cycles = 4
                workspace_capacity = 8
                input_rails = 1298
                parameter_count = 444599
                candidate_width = 80
            }
            require_dataset_content_sha256 = $true
            dataset_path = $resolvedDatasetPath
            require_dataset_integrity = $true
            require_runtime_provenance = $true
            require_source_fingerprint = $true
        }
        parent_checkpoint = $parentCheckpointContract
        parent_lineage = @($parentLineageSnapshot)
        julia_executable = $juliaPath
        julia_threads = $JuliaThreads
        julia_runtime_arguments = $juliaRuntimeArguments
        project_path = $resolvedProject
        training_script = $resolvedTrainingScript
        verifier_script = $resolvedVerifierScript
        controller_script = $resolvedControllerScript
        code_artifacts = [ordered]@{
            controller = [ordered]@{
                path = $resolvedControllerScript
                bytes = (Get-Item -LiteralPath $resolvedControllerScript).Length
                sha256 = Get-Sha256Hex -Path $resolvedControllerScript
            }
            training = [ordered]@{
                path = $resolvedTrainingScript
                bytes = (Get-Item -LiteralPath $resolvedTrainingScript).Length
                sha256 = Get-Sha256Hex -Path $resolvedTrainingScript
            }
            verifier = [ordered]@{
                path = $resolvedVerifierScript
                bytes = (Get-Item -LiteralPath $resolvedVerifierScript).Length
                sha256 = Get-Sha256Hex -Path $resolvedVerifierScript
            }
        }
        working_directory = $resolvedWorkingDirectory
        output_root = $resolvedOutputRoot
        environment = Get-SwsnnEnvironment
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
    }
    Write-AtomicJson -Path $launchManifestPath -Value $launchManifest
    $launchManifestSha256 = Get-Sha256Hex -Path $launchManifestPath
    $env:SWSNN_LAUNCH_MANIFEST_PATH = $launchManifestPath
    $env:SWSNN_LAUNCH_MANIFEST_SHA256 = $launchManifestSha256
    if ($parentLineageSnapshot.Count -gt 0) {
        Assert-PinnedLineageCurrentFailClosed `
            -PinnedLineage $parentLineageSnapshot
    }

    $emptyTelemetry = Get-RunTelemetry `
        -RunDirectory $script:RunDirectory `
        -StandardOutputPath $stdoutPath `
        -StandardErrorPath $stderrPath
    Write-AtomicJson -Path $statusPath -Value (
        New-ControllerStatus `
            -State "starting" `
            -Telemetry $emptyTelemetry `
            -LastProgressAt $lastProgressAt `
            -ChildProcessId $null `
            -ChildStartTime $null `
            -ChildExitCode $null `
            -VerifierExitCode $null `
            -Failure $null
    )

    $trainingArguments = @(
        $juliaRuntimeArguments
        "--project=$resolvedProject"
        "--threads=$JuliaThreads,0"
        $resolvedTrainingScript
    )
    if ($parentLineageSnapshot.Count -gt 0) {
        Assert-PinnedLineageCurrentFailClosed `
            -PinnedLineage $parentLineageSnapshot
    }
    $jobHandle = New-KillOnCloseJob
    $child = Start-AssignedProcess `
        -Job $jobHandle `
        -Executable $juliaPath `
        -Arguments $trainingArguments `
        -WorkingDirectory $resolvedWorkingDirectory `
        -StandardOutputPath $stdoutPath `
        -StandardErrorPath $stderrPath
    $childStartTime = $child.StartTime.ToString("o")

    while (-not $child.WaitForExit($PollSeconds * 1000)) {
        $telemetry = Get-RunTelemetry `
            -RunDirectory $script:RunDirectory `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath
        if (
            $null -ne $telemetry.latest_update -and
            [int]$telemetry.latest_update -gt $lastObservedUpdate
        ) {
            $lastObservedUpdate = [int]$telemetry.latest_update
            $lastProgressAt = Get-Date
        }
        $lastProgressAt = Get-ProgressTimeFromTelemetry `
            -Telemetry $telemetry `
            -CurrentProgressAt $lastProgressAt
        $idleSeconds = ((Get-Date) - $lastProgressAt).TotalSeconds
        $state = if ($idleSeconds -ge (2 * $StallSeconds)) {
            "running_stalled_terminating"
        }
        elseif ($idleSeconds -ge $StallSeconds) {
            if ($null -eq $stallWarningAt) {
                $stallWarningAt = Get-Date
            }
            "running_stalled_warning"
        }
        else {
            $stallWarningAt = $null
            "running"
        }
        Write-AtomicJson -Path $statusPath -Value (
            New-ControllerStatus `
                -State $state `
                -Telemetry $telemetry `
                -LastProgressAt $lastProgressAt `
                -ChildProcessId $child.Id `
                -ChildStartTime $childStartTime `
                -ChildExitCode $null `
                -VerifierExitCode $null `
                -Failure $null
        )
        if ($idleSeconds -ge (2 * $StallSeconds)) {
            $failureState = "failed_stalled"
            $terminated = Stop-ChildProcessTree -Process $child
            $childExitCode = [SwsnnNativeJob]::WaitForExitAndGetCode(
                [uint32]$child.Id
            )
            [SwsnnNativeJob]::TerminateJobAndWaitForEmpty(
                $jobHandle,
                [uint32]1,
                5000
            )
            $jobQuiescenceConfirmed = $true
            if ($parentLineageSnapshot.Count -gt 0) {
                Assert-PinnedLineageCurrentFailClosed `
                    -PinnedLineage $parentLineageSnapshot
            }
            $recoveryCheckpoint =
                Get-RecoveryCheckpoint -RunDirectory $script:RunDirectory
            $resultsAlreadyExist = Test-TrustedRegularFile `
                -Path (
                    Join-Path $script:RunDirectory "results.json"
                ) `
                -Context "stalled-run results.json"
            $recoveryStartMode = if (
                $null -ne $recoveryCheckpoint -and
                [int]$recoveryCheckpoint.update -eq $ExpectedUpdates -and
                -not $resultsAlreadyExist
            ) {
                "finalize-only"
            }
            elseif (
                $null -ne $recoveryCheckpoint -and
                [int]$recoveryCheckpoint.update -lt $ExpectedUpdates
            ) {
                "resume"
            }
            else {
                $null
            }
            $suggestedRecoveryRunId = if (
                $null -eq $recoveryStartMode
            ) {
                $null
            }
            else {
                New-SuggestedRecoveryRunId `
                    -RecoveryStartMode $recoveryStartMode
            }
            $resumeCommand = if (
                $null -eq $recoveryCheckpoint -or
                $null -eq $recoveryStartMode
            ) {
                $null
            }
            else {
                (
                    "& " +
                    (Quote-NativeArgument $PSCommandPath) +
                    " -RunId " +
                    (Quote-NativeArgument $suggestedRecoveryRunId) +
                    " -ExpectedUpdates $ExpectedUpdates" +
                    " -JuliaThreads $JuliaThreads" +
                    " -PollSeconds $PollSeconds" +
                    " -StallSeconds $StallSeconds" +
                    " -VerifierTimeoutSeconds $VerifierTimeoutSeconds" +
                    " -JuliaExecutable " +
                    (Quote-NativeArgument $juliaPath) +
                    " -ProjectPath " +
                    (Quote-NativeArgument $resolvedProject) +
                    " -TrainingScript " +
                    (Quote-NativeArgument $resolvedTrainingScript) +
                    " -VerifierScript " +
                    (Quote-NativeArgument $resolvedVerifierScript) +
                    " -DatasetPath " +
                    (Quote-NativeArgument $resolvedDatasetPath) +
                    " -OutputRoot " +
                    (Quote-NativeArgument $resolvedOutputRoot) +
                    " -WorkingDirectory " +
                    (Quote-NativeArgument $resolvedWorkingDirectory) +
                    " -MutexName " +
                    (Quote-NativeArgument $MutexName) +
                    " -StartMode $recoveryStartMode" +
                    " -ResumeCheckpoint " +
                    (Quote-NativeArgument (
                        [string]$recoveryCheckpoint.path
                    )) +
                    " -ResumeSha256 " +
                    (Quote-NativeArgument (
                        [string]$recoveryCheckpoint.sha256
                    )) +
                    " -ResumeUpdate " +
                    [string]$recoveryCheckpoint.update
                )
            }
            $verificationRetryCommand = if (
                $resultsAlreadyExist -and
                $null -ne $launchManifestSha256
            ) {
                & $newVerificationRetryCommand
            }
            else {
                $null
            }
            Write-AtomicJson -Path $recoveryPath -Value ([ordered]@{
                format =
                    "serial-workspace-snn-arena-run-recovery"
                version = 1
                state = "failed_stalled"
                recorded_at = (Get-Date).ToString("o")
                run_id = $RunId
                run_directory = $script:RunDirectory
                expected_updates = $ExpectedUpdates
                child_pid = $child.Id
                child_terminated = $terminated
                child_exit_code = $childExitCode
                last_progress_at = $lastProgressAt.ToString("o")
                first_stall_warning_at = if (
                    $null -eq $stallWarningAt
                ) {
                    $null
                }
                else {
                    $stallWarningAt.ToString("o")
                }
                stall_threshold_seconds = $StallSeconds
                observed_idle_seconds =
                    [math]::Round($idleSeconds, 1)
                telemetry = $telemetry
                launch_manifest_path = $launchManifestPath
                launch_manifest_sha256 = $launchManifestSha256
                recovery_checkpoint = $recoveryCheckpoint
                suggested_recovery_run_id = $suggestedRecoveryRunId
                resume_command = $resumeCommand
                verification_retry_command =
                    $verificationRetryCommand
            })
            throw (
                "Arena training made no heartbeat or update progress for " +
                "$([math]::Round($idleSeconds, 1)) seconds; child process " +
                "$($child.Id) was terminated"
            )
        }
    }
    $childExitCode = [SwsnnNativeJob]::WaitForExitAndGetCode(
        [uint32]$child.Id
    )
    [SwsnnNativeJob]::TerminateJobAndWaitForEmpty(
        $jobHandle,
        [uint32]1,
        5000
    )
    $jobQuiescenceConfirmed = $true
    if ($parentLineageSnapshot.Count -gt 0) {
        Assert-PinnedLineageCurrentFailClosed `
            -PinnedLineage $parentLineageSnapshot
    }
    $exitTelemetry = Get-RunTelemetry `
        -RunDirectory $script:RunDirectory `
        -StandardOutputPath $stdoutPath `
        -StandardErrorPath $stderrPath
    $lastProgressAt = Get-ProgressTimeFromTelemetry `
        -Telemetry $exitTelemetry `
        -CurrentProgressAt $lastProgressAt
    $childExitState = if ($childExitCode -eq 0) {
        "child_exited"
    }
    else {
        "child_failed"
    }
    Write-AtomicJson -Path $statusPath -Value (
        New-ControllerStatus `
            -State $childExitState `
            -Telemetry $exitTelemetry `
            -LastProgressAt $lastProgressAt `
            -ChildProcessId $child.Id `
            -ChildStartTime $childStartTime `
            -ChildExitCode $childExitCode `
            -VerifierExitCode $null `
            -Failure $null
    )
    if ($childExitCode -ne 0) {
        throw "Arena training child exited with code $childExitCode"
    }
    } while ($false)

    $statusChildProcessId = if ($null -eq $child) {
        $null
    }
    else {
        $child.Id
    }
    $statusChildStartTime = if ($null -eq $child) {
        $null
    }
    else {
        $childStartTime
    }
    if ($jobHandle -eq [IntPtr]::Zero) {
        $jobHandle = New-KillOnCloseJob
    }
    Write-AtomicJson -Path $statusPath -Value (
        New-ControllerStatus `
            -State "verifying" `
            -Telemetry $exitTelemetry `
            -LastProgressAt $lastProgressAt `
            -ChildProcessId $statusChildProcessId `
            -ChildStartTime $statusChildStartTime `
            -ChildExitCode $childExitCode `
            -VerifierExitCode $null `
            -Failure $null
    )
    $verifierArguments = @(
        $juliaRuntimeArguments
        "--project=$resolvedProject"
        "--threads=1,0"
        $resolvedVerifierScript
        "--run-dir=$($script:RunDirectory)"
        "--expected-updates=$ExpectedUpdates"
        "--expected-run-id=$RunId"
        "--expected-start-mode=$effectiveStartMode"
        "--launch-manifest=$launchManifestPath"
        "--launch-manifest-sha256=$launchManifestSha256"
    )
    if ($null -ne $parentCheckpointContract) {
        $verifierArguments += @(
            "--parent-checkpoint=$([string]$parentCheckpointContract.path)"
            "--parent-sha256=$([string]$parentCheckpointContract.sha256)"
            "--parent-update=$([int]$parentCheckpointContract.update)"
        )
    }
    $preexistingVerificationPath =
        Join-Path $script:RunDirectory "verification.json"
    if (Test-Path -LiteralPath $preexistingVerificationPath) {
        Assert-NoReparsePathChain `
            -Path $preexistingVerificationPath `
            -Context "pre-verifier verification.json" |
            Out-Null
        $preexistingVerificationItem = Assert-NotReparsePoint `
            -Path $preexistingVerificationPath `
            -Context "pre-verifier verification.json"
        if ($preexistingVerificationItem.PSIsContainer) {
            throw "pre-verifier verification.json is not a regular file"
        }
        if (-not $VerifyOnly) {
            throw (
                "new training run unexpectedly has a preexisting " +
                "verification.json"
            )
        }
        if (
            [string]::IsNullOrWhiteSpace($retryControllerDirectory) -or
            -not (
                Test-Path -LiteralPath $retryControllerDirectory `
                    -PathType Container
            )
        ) {
            throw (
                "Verify-only cannot archive the stale verification report " +
                "without a validated retry directory"
            )
        }
        $archivedVerificationPath =
            Join-Path $retryControllerDirectory (
                "preexisting_verification.json"
            )
        if (Test-Path -LiteralPath $archivedVerificationPath) {
            throw "verification archive target already exists"
        }
        [IO.File]::Move(
            $preexistingVerificationItem.FullName,
            $archivedVerificationPath
        )
        Assert-NoReparsePathChain `
            -Path $archivedVerificationPath `
            -Context "archived preexisting verification.json" |
            Out-Null
        $archivedVerificationItem = Assert-NotReparsePoint `
            -Path $archivedVerificationPath `
            -Context "archived preexisting verification.json"
        if (
            $archivedVerificationItem.PSIsContainer -or
            (
                Test-Path -LiteralPath $preexistingVerificationPath
            )
        ) {
            throw "preexisting verification.json archive operation failed"
        }
    }
    $verifier = Start-AssignedProcess `
        -Job $jobHandle `
        -Executable $juliaPath `
        -Arguments $verifierArguments `
        -WorkingDirectory $resolvedWorkingDirectory `
        -StandardOutputPath $verifierStdoutPath `
        -StandardErrorPath $verifierStderrPath
    $script:VerifierProcessId = $verifier.Id
    $verifierStartedAt = Get-Date
    while (-not $verifier.HasExited) {
        $verifierElapsedBeforeWait =
            ((Get-Date) - $verifierStartedAt).TotalSeconds
        $remainingVerifierMilliseconds = [int][math]::Max(
            0,
            [math]::Ceiling(
                (
                    $VerifierTimeoutSeconds -
                    $verifierElapsedBeforeWait
                ) * 1000
            )
        )
        $boundedVerifierWaitMilliseconds = [int][math]::Min(
            $PollSeconds * 1000,
            $remainingVerifierMilliseconds
        )
        if (
            $boundedVerifierWaitMilliseconds -gt 0 -and
            $verifier.WaitForExit($boundedVerifierWaitMilliseconds)
        ) {
            break
        }
        $verifyTelemetry = Get-RunTelemetry `
            -RunDirectory $script:RunDirectory `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath
        Write-AtomicJson -Path $statusPath -Value (
            New-ControllerStatus `
                -State "verifying" `
                -Telemetry $verifyTelemetry `
                -LastProgressAt $lastProgressAt `
                -ChildProcessId $statusChildProcessId `
                -ChildStartTime $statusChildStartTime `
                -ChildExitCode $childExitCode `
                -VerifierExitCode $null `
                -Failure $null
        )
        $verifierElapsed =
            ((Get-Date) - $verifierStartedAt).TotalSeconds
        if ($verifierElapsed -ge $VerifierTimeoutSeconds) {
            $failureState = "failed_verifier_stalled"
            $terminated = Stop-ChildProcessTree -Process $verifier
            $verifierExitCode = [SwsnnNativeJob]::WaitForExitAndGetCode(
                [uint32]$verifier.Id
            )
            [SwsnnNativeJob]::TerminateJobAndWaitForEmpty(
                $jobHandle,
                [uint32]1,
                5000
            )
            $jobQuiescenceConfirmed = $true
            if ($parentLineageSnapshot.Count -gt 0) {
                Assert-PinnedLineageCurrentFailClosed `
                    -PinnedLineage $parentLineageSnapshot
            }
            $verificationRetryCommand = if (
                (Test-TrustedRegularFile `
                    -Path (
                        Join-Path $script:RunDirectory "results.json"
                    ) `
                    -Context "verifier-timeout results.json") -and
                $null -ne $launchManifestSha256
            ) {
                & $newVerificationRetryCommand
            }
            else {
                $null
            }
            Write-AtomicJson -Path $recoveryPath -Value ([ordered]@{
                format =
                    "serial-workspace-snn-arena-run-recovery"
                version = 1
                state = "failed_verifier_stalled"
                recorded_at = (Get-Date).ToString("o")
                run_id = $RunId
                run_directory = $script:RunDirectory
                expected_updates = $ExpectedUpdates
                verifier_pid = $verifier.Id
                verifier_terminated = $terminated
                verifier_exit_code = $verifierExitCode
                verifier_elapsed_seconds =
                    [math]::Round($verifierElapsed, 1)
                verifier_timeout_seconds =
                    $VerifierTimeoutSeconds
                launch_manifest_path = $launchManifestPath
                launch_manifest_sha256 = $launchManifestSha256
                final_training_checkpoint =
                    Get-RecoveryCheckpoint `
                        -RunDirectory $script:RunDirectory
                resume_command = $null
                verification_retry_command =
                    $verificationRetryCommand
            })
            throw (
                "Arena run verifier exceeded " +
                "$VerifierTimeoutSeconds seconds and was terminated"
            )
        }
    }
    $verifierExitCode = [SwsnnNativeJob]::WaitForExitAndGetCode(
        [uint32]$verifier.Id
    )
    [SwsnnNativeJob]::TerminateJobAndWaitForEmpty(
        $jobHandle,
        [uint32]1,
        5000
    )
    $jobQuiescenceConfirmed = $true
    if ($parentLineageSnapshot.Count -gt 0) {
        Assert-PinnedLineageCurrentFailClosed `
            -PinnedLineage $parentLineageSnapshot
    }
    if ($verifierExitCode -ne 0) {
        throw "Arena run verifier exited with code $verifierExitCode"
    }

    $verificationPath = Join-Path $script:RunDirectory "verification.json"
    if (-not (Test-Path -LiteralPath $verificationPath -PathType Leaf)) {
        throw "Verifier succeeded without verification.json"
    }
    Assert-NoReparsePathChain `
        -Path $verificationPath `
        -Context "produced verification.json" |
        Out-Null
    $verificationItem = Assert-NotReparsePoint `
        -Path $verificationPath `
        -Context "produced verification.json"
    if ($verificationItem.PSIsContainer) {
        throw "produced verification.json is not a regular file"
    }
    $rawVerification =
        Get-Content -Raw -LiteralPath $verificationPath
    Assert-StrictTopLevelJsonObject `
        -RawJson $rawVerification `
        -Context "produced verification.json"
    $verification = ConvertFrom-Json -InputObject $rawVerification
    if (
        $verification.verified -ne $true -or
        [string]$verification.status -cne "verified_complete" -or
        [string]$verification.format -cne
            "serial-workspace-snn-arena-run-verification" -or
        [int]$verification.version -ne 2 -or
        $verification.metrics_verified -ne $true -or
        $verification.results.metrics_verified -ne $true -or
        $verification.results.fixed_panel_recomputation.verified -ne
            $true
    ) {
        throw (
            "verification.json does not report a v2 metrics-verified " +
            "completion"
        )
    }
    if ([int]$verification.expected_updates -ne $ExpectedUpdates) {
        throw "verification.json expected update count differs"
    }
    if (
        [string]$verification.run_id -cne $RunId -or
        [string]$verification.launch_manifest.sha256 -cne
            $launchManifestSha256
    ) {
        throw "verification.json launch identity differs"
    }

    $finalTelemetry = Get-RunTelemetry `
        -RunDirectory $script:RunDirectory `
        -StandardOutputPath $stdoutPath `
        -StandardErrorPath $stderrPath
    $lastProgressAt = Get-ProgressTimeFromTelemetry `
        -Telemetry $finalTelemetry `
        -CurrentProgressAt $lastProgressAt
    $verifiedStatus = New-ControllerStatus `
        -State "verified_complete" `
        -Telemetry $finalTelemetry `
        -LastProgressAt $lastProgressAt `
        -ChildProcessId $statusChildProcessId `
        -ChildStartTime $statusChildStartTime `
        -ChildExitCode $childExitCode `
        -VerifierExitCode $verifierExitCode `
        -Failure $null
    if ($VerifyOnly) {
        $verifiedStatus["latest_verification_retry"] = [ordered]@{
            attempt_id = $retryAttemptId
            attempt_directory = $retryControllerDirectory
            status_path = $statusPath
            recovery_path = if (
                Test-Path -LiteralPath $recoveryPath -PathType Leaf
            ) {
                $recoveryPath
            }
            else {
                $null
            }
            verification_path = $verificationPath
            verification_sha256 =
                Get-Sha256Hex -Path $verificationPath
            previous_verification_archive =
                $archivedVerificationPath
            verifier_pid = $script:VerifierProcessId
            verifier_exit_code = $verifierExitCode
            verified = $true
        }
    }
    Write-AtomicJson -Path $statusPath -Value $verifiedStatus
    if ($VerifyOnly) {
        if (Test-Path -LiteralPath $originalStatusPath -PathType Leaf) {
            Assert-NotReparsePoint `
                -Path $originalStatusPath `
                -Context "original controller status" |
                Out-Null
        }
        Write-AtomicJson `
            -Path $originalStatusPath `
            -Value $verifiedStatus
    }
    $controllerExitCode = 0
}
catch {
    $failure = ($_ | Out-String).Trim()
    $failedStatusForOriginal = $null
    if ($null -ne $verifier -and -not $verifier.HasExited) {
        try {
            Stop-ChildProcessTree -Process $verifier | Out-Null
            $verifierExitCode = [SwsnnNativeJob]::WaitForExitAndGetCode(
                [uint32]$verifier.Id
            )
        }
        catch {
            $failure += "`nVerifier termination before recovery scan failed: $_"
        }
    }
    if ($null -ne $child -and -not $child.HasExited) {
        try {
            Stop-ChildProcessTree -Process $child | Out-Null
            $childExitCode = [SwsnnNativeJob]::WaitForExitAndGetCode(
                [uint32]$child.Id
            )
        }
        catch {
            $failure += "`nTraining termination before recovery scan failed: $_"
        }
    }
    if ($jobHandle -ne [IntPtr]::Zero) {
        try {
            [SwsnnNativeJob]::TerminateJobAndWaitForEmpty(
                $jobHandle,
                [uint32]1,
                5000
            )
            $jobQuiescenceConfirmed = $true
        }
        catch {
            $jobQuiescenceConfirmed = $false
            $failure += "`nJob quiescence before recovery scan failed: $_"
        }
    }
    if (
        $parentLineageSnapshot.Count -gt 0 -and
        -not $script:ParentLineageTrustLost
    ) {
        try {
            Assert-PinnedLineageCurrentFailClosed `
                -PinnedLineage $parentLineageSnapshot
        }
        catch {
            $failure += (
                "`nPinned parent lineage revalidation before recovery " +
                "failed: $_"
            )
        }
    }
    if ($controllerDirectoryCreated) {
        if (-not (Test-Path -LiteralPath $recoveryPath -PathType Leaf)) {
            try {
                $recoveryCheckpoint = if (
                    $VerifyOnly -or
                    -not $jobQuiescenceConfirmed -or
                    $script:ParentLineageTrustLost
                ) {
                    $null
                }
                else {
                    Get-RecoveryCheckpoint `
                        -RunDirectory $script:RunDirectory
                }
                $resultsAlreadyExist = Test-TrustedRegularFile `
                    -Path (
                        Join-Path $script:RunDirectory "results.json"
                    ) `
                    -Context "failure-recovery results.json"
                $recoveryStartMode = if (
                    -not $VerifyOnly -and
                    $null -ne $recoveryCheckpoint -and
                    [int]$recoveryCheckpoint.update -eq $ExpectedUpdates -and
                    -not $resultsAlreadyExist
                ) {
                    "finalize-only"
                }
                elseif (
                    -not $VerifyOnly -and
                    $null -ne $recoveryCheckpoint -and
                    [int]$recoveryCheckpoint.update -lt $ExpectedUpdates
                ) {
                    "resume"
                }
                else {
                    $null
                }
                $suggestedRecoveryRunId = if (
                    $null -eq $recoveryStartMode
                ) {
                    $null
                }
                else {
                    New-SuggestedRecoveryRunId `
                        -RecoveryStartMode $recoveryStartMode
                }
                $resumeCommand = if (
                    $null -eq $recoveryCheckpoint -or
                    $null -eq $recoveryStartMode
                ) {
                    $null
                }
                else {
                    (
                        "& " +
                        (Quote-NativeArgument $PSCommandPath) +
                        " -RunId " +
                        (Quote-NativeArgument $suggestedRecoveryRunId) +
                        " -ExpectedUpdates $ExpectedUpdates" +
                        " -JuliaThreads $JuliaThreads" +
                        " -PollSeconds $PollSeconds" +
                        " -StallSeconds $StallSeconds" +
                        " -VerifierTimeoutSeconds " +
                        "$VerifierTimeoutSeconds" +
                        " -JuliaExecutable " +
                        (Quote-NativeArgument $juliaPath) +
                        " -ProjectPath " +
                        (Quote-NativeArgument $resolvedProject) +
                        " -TrainingScript " +
                        (Quote-NativeArgument $resolvedTrainingScript) +
                        " -VerifierScript " +
                        (Quote-NativeArgument $resolvedVerifierScript) +
                        " -DatasetPath " +
                        (Quote-NativeArgument $resolvedDatasetPath) +
                        " -OutputRoot " +
                        (Quote-NativeArgument $resolvedOutputRoot) +
                        " -WorkingDirectory " +
                        (Quote-NativeArgument $resolvedWorkingDirectory) +
                        " -MutexName " +
                        (Quote-NativeArgument $MutexName) +
                        " -StartMode $recoveryStartMode" +
                        " -ResumeCheckpoint " +
                        (Quote-NativeArgument (
                            [string]$recoveryCheckpoint.path
                        )) +
                        " -ResumeSha256 " +
                        (Quote-NativeArgument (
                            [string]$recoveryCheckpoint.sha256
                        )) +
                        " -ResumeUpdate " +
                        [string]$recoveryCheckpoint.update
                    )
                }
                $verificationRetryCommand = if (
                    $resultsAlreadyExist -and
                    $null -ne $launchManifestSha256 -and
                    $jobQuiescenceConfirmed -and
                    -not $script:ParentLineageTrustLost
                ) {
                    & $newVerificationRetryCommand
                }
                else {
                    $null
                }
                Write-AtomicJson -Path $recoveryPath -Value ([ordered]@{
                    format =
                        "serial-workspace-snn-arena-run-recovery"
                    version = 1
                    state = $failureState
                    recorded_at = (Get-Date).ToString("o")
                    run_id = $RunId
                    run_directory = $script:RunDirectory
                    expected_updates = $ExpectedUpdates
                    child_pid = if ($null -eq $child) {
                        $null
                    }
                    else {
                        $child.Id
                    }
                    child_exit_code = $childExitCode
                    verifier_pid = $script:VerifierProcessId
                    verifier_exit_code = $verifierExitCode
                    failure = $failure
                    launch_manifest_path = if (
                        Test-Path -LiteralPath $launchManifestPath `
                            -PathType Leaf
                    ) {
                        $launchManifestPath
                    }
                    else {
                        $null
                    }
                    launch_manifest_sha256 =
                        $launchManifestSha256
                    recovery_checkpoint = $recoveryCheckpoint
                    suggested_recovery_run_id =
                        $suggestedRecoveryRunId
                    resume_command = $resumeCommand
                    verification_retry_command =
                        $verificationRetryCommand
                })
            }
            catch {
                [Console]::Error.WriteLine(
                    "Could not persist controller recovery artifact: $_"
                )
            }
        }
        try {
            $telemetry = Get-RunTelemetry `
                -RunDirectory $script:RunDirectory `
                -StandardOutputPath $stdoutPath `
                -StandardErrorPath $stderrPath
            $failedChildProcessId = if ($null -eq $child) {
                $null
            }
            else {
                $child.Id
            }
            $failedChildStartTime = if ($null -eq $child) {
                $null
            }
            else {
                try {
                    $child.StartTime.ToString("o")
                }
                catch {
                    $null
                }
            }
            $failedStatus = New-ControllerStatus `
                -State $failureState `
                -Telemetry $telemetry `
                -LastProgressAt $lastProgressAt `
                -ChildProcessId $failedChildProcessId `
                -ChildStartTime $failedChildStartTime `
                -ChildExitCode $childExitCode `
                -VerifierExitCode $verifierExitCode `
                -Failure $failure
            if ($VerifyOnly -and $verifyOnlyLaunchAuthenticated) {
                $failedStatus["latest_verification_retry"] = [ordered]@{
                    attempt_id = $retryAttemptId
                    attempt_directory = $retryControllerDirectory
                    status_path = $statusPath
                    recovery_path = if (
                        Test-Path -LiteralPath $recoveryPath -PathType Leaf
                    ) {
                        $recoveryPath
                    }
                    else {
                        $null
                    }
                    verifier_pid = $script:VerifierProcessId
                    verifier_exit_code = $verifierExitCode
                    previous_verification_archive =
                        $archivedVerificationPath
                    verified = $false
                    failure = $failure
                }
                $failedStatusForOriginal = $failedStatus
            }
            Write-AtomicJson -Path $statusPath -Value $failedStatus
        }
        catch {
            [Console]::Error.WriteLine(
                "Could not persist controller failure status: $_"
            )
        }
    }
    if ($VerifyOnly -and $verifyOnlyLaunchAuthenticated) {
        try {
            if ($null -eq $failedStatusForOriginal) {
                $authenticatedFailureTelemetry = Get-RunTelemetry `
                    -RunDirectory $script:RunDirectory `
                    -StandardOutputPath $stdoutPath `
                    -StandardErrorPath $stderrPath
                $failedStatusForOriginal = New-ControllerStatus `
                    -State $failureState `
                    -Telemetry $authenticatedFailureTelemetry `
                    -LastProgressAt $lastProgressAt `
                    -ChildProcessId $null `
                    -ChildStartTime $null `
                    -ChildExitCode $childExitCode `
                    -VerifierExitCode $verifierExitCode `
                    -Failure $failure
                $failedStatusForOriginal["latest_verification_retry"] =
                    [ordered]@{
                        attempt_id = $retryAttemptId
                        attempt_directory = $retryControllerDirectory
                        status_path = if ($controllerDirectoryCreated) {
                            $statusPath
                        }
                        else {
                            $null
                        }
                        recovery_path = if (
                            $controllerDirectoryCreated -and
                            (
                                Test-Path -LiteralPath $recoveryPath `
                                    -PathType Leaf
                            )
                        ) {
                            $recoveryPath
                        }
                        else {
                            $null
                        }
                        verifier_pid = $script:VerifierProcessId
                        verifier_exit_code = $verifierExitCode
                        previous_verification_archive =
                            $archivedVerificationPath
                        verified = $false
                        failure = $failure
                    }
            }
            if (Test-Path -LiteralPath $originalStatusPath -PathType Leaf) {
                Assert-NotReparsePoint `
                    -Path $originalStatusPath `
                    -Context "original controller status" |
                    Out-Null
            }
            Write-AtomicJson `
                -Path $originalStatusPath `
                -Value $failedStatusForOriginal
        }
        catch {
            [Console]::Error.WriteLine(
                "Could not reflect authenticated Verify-only failure in " +
                "original controller status: $_"
            )
        }
    }
    [Console]::Error.WriteLine($failure)
    $controllerExitCode = 1
}
finally {
    if ($null -ne $verifier) {
        try {
            if (-not $verifier.HasExited) {
                Stop-ChildProcessTree -Process $verifier | Out-Null
            }
        }
        catch {
            [Console]::Error.WriteLine(
                "Could not terminate verifier process tree during cleanup: $_"
            )
        }
    }
    if ($null -ne $child) {
        try {
            if (-not $child.HasExited) {
                Stop-ChildProcessTree -Process $child | Out-Null
            }
        }
        catch {
            [Console]::Error.WriteLine(
                "Could not terminate child process tree during cleanup: $_"
            )
        }
    }
    if ($jobHandle -ne [IntPtr]::Zero) {
        [SwsnnNativeJob]::CloseHandle($jobHandle) | Out-Null
        $jobHandle = [IntPtr]::Zero
    }
    if ($mutexAcquired -and $null -ne $mutex) {
        try {
            $mutex.ReleaseMutex()
        }
        catch {
        }
    }
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
}

exit $controllerExitCode
