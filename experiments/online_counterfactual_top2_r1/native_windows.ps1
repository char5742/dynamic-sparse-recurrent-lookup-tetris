Set-StrictMode -Version Latest

function ConvertTo-R1NativeArgument([AllowEmptyString()][string]$Argument) {
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }
    $Builder = [Text.StringBuilder]::new()
    [void]$Builder.Append('"')
    $Backslashes = 0
    foreach ($Character in $Argument.ToCharArray()) {
        if ($Character -eq '\') {
            $Backslashes += 1
            continue
        }
        if ($Character -eq '"') {
            for ($Index = 0; $Index -lt (2 * $Backslashes + 1); $Index += 1) {
                [void]$Builder.Append('\')
            }
            [void]$Builder.Append('"')
        } else {
            for ($Index = 0; $Index -lt $Backslashes; $Index += 1) {
                [void]$Builder.Append('\')
            }
            [void]$Builder.Append($Character)
        }
        $Backslashes = 0
    }
    for ($Index = 0; $Index -lt (2 * $Backslashes); $Index += 1) {
        [void]$Builder.Append('\')
    }
    [void]$Builder.Append('"')
    return $Builder.ToString()
}

function Join-R1NativeArguments([string[]]$ArgumentList) {
    return (($ArgumentList | ForEach-Object { ConvertTo-R1NativeArgument $_ }) -join ' ')
}

# R1 must not repeat Q1's Start-Process -> Assign race.  CreateProcessW starts
# the child suspended, the child is assigned to the kill-on-close Job Object,
# and only then is its primary thread resumed.  Redirection handles are opened
# before CreateProcess and inherited only by that child.
if (-not ('R1Native.SuspendedJobProcess' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace R1Native
{
    internal static class NativeMethods
    {
        internal const uint CREATE_SUSPENDED = 0x00000004;
        internal const uint CREATE_NO_WINDOW = 0x08000000;
        internal const uint STARTF_USESHOWWINDOW = 0x00000001;
        internal const uint STARTF_USESTDHANDLES = 0x00000100;
        internal const short SW_HIDE = 0;
        internal const uint GENERIC_WRITE = 0x40000000;
        internal const uint FILE_SHARE_READ = 0x00000001;
        internal const uint FILE_SHARE_WRITE = 0x00000002;
        internal const uint CREATE_ALWAYS = 2;
        internal const uint FILE_ATTRIBUTE_NORMAL = 0x00000080;
        internal const int STD_INPUT_HANDLE = -10;
        internal const int JobObjectExtendedLimitInformation = 9;
        internal const int JobObjectBasicProcessIdList = 3;
        internal const uint JobObjectLimitKillOnJobClose = 0x00002000;
        internal const uint JobObjectLimitJobMemory = 0x00000200;
        internal const int ErrorMoreData = 234;

        [StructLayout(LayoutKind.Sequential)]
        internal struct SecurityAttributes
        {
            internal int Length;
            internal IntPtr SecurityDescriptor;
            [MarshalAs(UnmanagedType.Bool)] internal bool InheritHandle;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        internal struct StartupInfo
        {
            internal int Size;
            internal string Reserved;
            internal string Desktop;
            internal string Title;
            internal int X;
            internal int Y;
            internal int XSize;
            internal int YSize;
            internal int XCountChars;
            internal int YCountChars;
            internal int FillAttribute;
            internal int Flags;
            internal short ShowWindow;
            internal short Reserved2Count;
            internal IntPtr Reserved2;
            internal IntPtr StdInput;
            internal IntPtr StdOutput;
            internal IntPtr StdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ProcessInformation
        {
            internal IntPtr Process;
            internal IntPtr Thread;
            internal int ProcessId;
            internal int ThreadId;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct IoCounters
        {
            internal ulong ReadOperationCount;
            internal ulong WriteOperationCount;
            internal ulong OtherOperationCount;
            internal ulong ReadTransferCount;
            internal ulong WriteTransferCount;
            internal ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct BasicLimitInformation
        {
            internal long PerProcessUserTimeLimit;
            internal long PerJobUserTimeLimit;
            internal uint LimitFlags;
            internal UIntPtr MinimumWorkingSetSize;
            internal UIntPtr MaximumWorkingSetSize;
            internal uint ActiveProcessLimit;
            internal UIntPtr Affinity;
            internal uint PriorityClass;
            internal uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ExtendedLimitInformation
        {
            internal BasicLimitInformation BasicLimitInformation;
            internal IoCounters IoInfo;
            internal UIntPtr ProcessMemoryLimit;
            internal UIntPtr JobMemoryLimit;
            internal UIntPtr PeakProcessMemoryUsed;
            internal UIntPtr PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr CreateFile(
            string fileName, uint desiredAccess, uint shareMode,
            ref SecurityAttributes securityAttributes, uint creationDisposition,
            uint flagsAndAttributes, IntPtr templateFile);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool CreateProcess(
            string applicationName, StringBuilder commandLine,
            IntPtr processAttributes, IntPtr threadAttributes,
            [MarshalAs(UnmanagedType.Bool)] bool inheritHandles,
            uint creationFlags, IntPtr environment, string currentDirectory,
            ref StartupInfo startupInfo, out ProcessInformation processInformation);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern IntPtr CreateJobObject(IntPtr securityAttributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool SetInformationJobObject(
            IntPtr job, int informationClass, IntPtr information,
            uint informationLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool QueryInformationJobObject(
            IntPtr job, int informationClass, IntPtr information,
            uint informationLength, out uint returnLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool TerminateJobObject(IntPtr job, uint exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern uint ResumeThread(IntPtr thread);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool TerminateProcess(IntPtr process, uint exitCode);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool QueryFullProcessImageName(
            IntPtr process, uint flags, StringBuilder executableName,
            ref uint size);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool CloseHandle(IntPtr handle);

        [DllImport("kernel32.dll")]
        internal static extern IntPtr GetStdHandle(int standardHandle);
    }

    public sealed class SuspendedJobProcess : IDisposable
    {
        private IntPtr job;
        private IntPtr processHandle;
        private IntPtr threadHandle;
        private Process process;
        private bool resumed;

        public int ProcessId { get; private set; }
        public bool CreatedSuspended { get { return true; } }
        public bool AssignedBeforeResume { get; private set; }
        public bool Resumed { get { return resumed; } }
        public Process Process { get { return process; } }

        public SuspendedJobProcess(
            string executable, string commandLine, string workingDirectory,
            string stdoutPath, string stderrPath, long jobMemoryLimitBytes)
        {
            if (String.IsNullOrWhiteSpace(executable) || !System.IO.Path.IsPathRooted(executable))
                throw new ArgumentException("executable must be an absolute path", "executable");
            job = NativeMethods.CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject failed");
            if (jobMemoryLimitBytes <= 0)
                throw new ArgumentOutOfRangeException("jobMemoryLimitBytes");
            ConfigureLimits(jobMemoryLimitBytes);

            NativeMethods.SecurityAttributes security = new NativeMethods.SecurityAttributes();
            security.Length = Marshal.SizeOf(typeof(NativeMethods.SecurityAttributes));
            security.InheritHandle = true;
            IntPtr stdoutHandle = IntPtr.Zero;
            IntPtr stderrHandle = IntPtr.Zero;
            IntPtr stdinHandle = IntPtr.Zero;
            try
            {
                stdoutHandle = NativeMethods.CreateFile(
                    stdoutPath, NativeMethods.GENERIC_WRITE,
                    NativeMethods.FILE_SHARE_READ | NativeMethods.FILE_SHARE_WRITE,
                    ref security, NativeMethods.CREATE_ALWAYS,
                    NativeMethods.FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
                if (stdoutHandle == new IntPtr(-1))
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "stdout CreateFile failed");
                stderrHandle = NativeMethods.CreateFile(
                    stderrPath, NativeMethods.GENERIC_WRITE,
                    NativeMethods.FILE_SHARE_READ | NativeMethods.FILE_SHARE_WRITE,
                    ref security, NativeMethods.CREATE_ALWAYS,
                    NativeMethods.FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
                if (stderrHandle == new IntPtr(-1))
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "stderr CreateFile failed");
                stdinHandle = NativeMethods.CreateFile(
                    "NUL", 0x80000000, NativeMethods.FILE_SHARE_READ | NativeMethods.FILE_SHARE_WRITE,
                    ref security, 3, NativeMethods.FILE_ATTRIBUTE_NORMAL, IntPtr.Zero);
                if (stdinHandle == new IntPtr(-1))
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "stdin NUL CreateFile failed");

                NativeMethods.StartupInfo startup = new NativeMethods.StartupInfo();
                startup.Size = Marshal.SizeOf(typeof(NativeMethods.StartupInfo));
                startup.Flags = (int)(NativeMethods.STARTF_USESTDHANDLES | NativeMethods.STARTF_USESHOWWINDOW);
                startup.ShowWindow = NativeMethods.SW_HIDE;
                startup.StdInput = stdinHandle;
                startup.StdOutput = stdoutHandle;
                startup.StdError = stderrHandle;
                NativeMethods.ProcessInformation info;
                bool created = NativeMethods.CreateProcess(
                    executable, new StringBuilder(commandLine), IntPtr.Zero,
                    IntPtr.Zero, true,
                    NativeMethods.CREATE_SUSPENDED | NativeMethods.CREATE_NO_WINDOW,
                    IntPtr.Zero, workingDirectory, ref startup, out info);
                if (!created)
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateProcessW failed");
                processHandle = info.Process;
                threadHandle = info.Thread;
                ProcessId = info.ProcessId;
                process = Process.GetProcessById(ProcessId);
                if (!NativeMethods.AssignProcessToJobObject(job, processHandle))
                    throw new Win32Exception(Marshal.GetLastWin32Error(), "AssignProcessToJobObject failed");
                AssignedBeforeResume = true;
            }
            catch
            {
                if (processHandle != IntPtr.Zero)
                    NativeMethods.TerminateProcess(processHandle, 125);
                Dispose();
                throw;
            }
            finally
            {
                if (stdoutHandle != IntPtr.Zero && stdoutHandle != new IntPtr(-1))
                    NativeMethods.CloseHandle(stdoutHandle);
                if (stderrHandle != IntPtr.Zero && stderrHandle != new IntPtr(-1))
                    NativeMethods.CloseHandle(stderrHandle);
                if (stdinHandle != IntPtr.Zero && stdinHandle != new IntPtr(-1))
                    NativeMethods.CloseHandle(stdinHandle);
            }
        }

        private void ConfigureLimits(long jobMemoryLimitBytes)
        {
            NativeMethods.ExtendedLimitInformation limits = new NativeMethods.ExtendedLimitInformation();
            limits.BasicLimitInformation.LimitFlags =
                NativeMethods.JobObjectLimitKillOnJobClose |
                NativeMethods.JobObjectLimitJobMemory;
            limits.JobMemoryLimit = new UIntPtr(unchecked((ulong)jobMemoryLimitBytes));
            int size = Marshal.SizeOf(typeof(NativeMethods.ExtendedLimitInformation));
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                Marshal.StructureToPtr(limits, buffer, false);
                if (!NativeMethods.SetInformationJobObject(
                        job, NativeMethods.JobObjectExtendedLimitInformation,
                        buffer, (uint)size))
                    throw new Win32Exception(
                        Marshal.GetLastWin32Error(), "SetInformationJobObject failed");
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }

        public void Resume()
        {
            if (!AssignedBeforeResume || threadHandle == IntPtr.Zero)
                throw new InvalidOperationException("process is not assigned and suspended");
            if (resumed)
                throw new InvalidOperationException("process already resumed");
            uint previous = NativeMethods.ResumeThread(threadHandle);
            if (previous == UInt32.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "ResumeThread failed");
            resumed = true;
            NativeMethods.CloseHandle(threadHandle);
            threadHandle = IntPtr.Zero;
        }

        public string GetImagePath()
        {
            if (processHandle == IntPtr.Zero)
                throw new ObjectDisposedException("SuspendedJobProcess");
            uint size = 32768;
            StringBuilder buffer = new StringBuilder((int)size);
            if (!NativeMethods.QueryFullProcessImageName(
                    processHandle, 0, buffer, ref size))
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(), "QueryFullProcessImageName failed");
            return buffer.ToString();
        }

        public int GetExitCode()
        {
            if (processHandle == IntPtr.Zero)
                throw new ObjectDisposedException("SuspendedJobProcess");
            uint code;
            if (!NativeMethods.GetExitCodeProcess(processHandle, out code))
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(), "GetExitCodeProcess failed");
            return unchecked((int)code);
        }

        public int[] GetProcessIds()
        {
            if (job == IntPtr.Zero) return new int[0];
            const int size = 1024 * 1024;
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                uint returned;
                if (!NativeMethods.QueryInformationJobObject(
                        job, NativeMethods.JobObjectBasicProcessIdList,
                        buffer, size, out returned))
                {
                    int error = Marshal.GetLastWin32Error();
                    if (error != NativeMethods.ErrorMoreData)
                        throw new Win32Exception(error, "QueryInformationJobObject failed");
                }
                uint count = unchecked((uint)Marshal.ReadInt32(buffer, sizeof(uint)));
                List<int> result = new List<int>((int)count);
                int offset = 2 * sizeof(uint);
                for (uint index = 0; index < count; ++index)
                {
                    long id = IntPtr.Size == 8
                        ? Marshal.ReadInt64(buffer, offset + (int)index * IntPtr.Size)
                        : Marshal.ReadInt32(buffer, offset + (int)index * IntPtr.Size);
                    if (id > 0 && id <= Int32.MaxValue) result.Add((int)id);
                }
                return result.ToArray();
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }

        public ulong GetPeakJobMemoryUsed()
        {
            if (job == IntPtr.Zero) return 0;
            int size = Marshal.SizeOf(typeof(NativeMethods.ExtendedLimitInformation));
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                uint returned;
                if (!NativeMethods.QueryInformationJobObject(
                        job, NativeMethods.JobObjectExtendedLimitInformation,
                        buffer, (uint)size, out returned))
                    throw new Win32Exception(
                        Marshal.GetLastWin32Error(),
                        "QueryInformationJobObject(ExtendedLimitInformation) failed");
                NativeMethods.ExtendedLimitInformation value =
                    (NativeMethods.ExtendedLimitInformation)Marshal.PtrToStructure(
                        buffer, typeof(NativeMethods.ExtendedLimitInformation));
                return value.PeakJobMemoryUsed.ToUInt64();
            }
            finally { Marshal.FreeHGlobal(buffer); }
        }

        public void Terminate(uint exitCode)
        {
            if (job != IntPtr.Zero && !NativeMethods.TerminateJobObject(job, exitCode))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "TerminateJobObject failed");
        }

        public void Dispose()
        {
            if (threadHandle != IntPtr.Zero)
            {
                if (!resumed && processHandle != IntPtr.Zero)
                    NativeMethods.TerminateProcess(processHandle, 125);
                NativeMethods.CloseHandle(threadHandle);
                threadHandle = IntPtr.Zero;
            }
            if (processHandle != IntPtr.Zero)
            {
                NativeMethods.CloseHandle(processHandle);
                processHandle = IntPtr.Zero;
            }
            if (job != IntPtr.Zero)
            {
                NativeMethods.CloseHandle(job);
                job = IntPtr.Zero;
            }
            if (process != null) { process.Dispose(); process = null; }
            GC.SuppressFinalize(this);
        }

        ~SuspendedJobProcess() { Dispose(); }
    }
}
'@
}
