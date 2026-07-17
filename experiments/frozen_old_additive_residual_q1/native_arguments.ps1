Set-StrictMode -Version Latest

function ConvertTo-NativeArgument([AllowEmptyString()][string]$Argument) {
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $Builder = [System.Text.StringBuilder]::new()
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
    # The Windows CRT consumes pairs of backslashes immediately before a
    # closing quote, so every trailing backslash must be doubled here.
    for ($Index = 0; $Index -lt (2 * $Backslashes); $Index += 1) {
        [void]$Builder.Append('\')
    }
    [void]$Builder.Append('"')
    return $Builder.ToString()
}

function Join-NativeArguments([string[]]$ArgumentList) {
    return (($ArgumentList | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
}

# Every external Q1 phase is assigned to a Windows Job Object. Kill-on-close
# handles an abnormal wrapper exit; the wrapper can call Terminate() for its
# explicit wall-time and process-tree memory gates.
if (-not ('Q1Native.JobObject' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Q1Native
{
    public sealed class JobObject : IDisposable
    {
        private const int JobObjectExtendedLimitInformation = 9;
        private const int JobObjectBasicProcessIdList = 3;
        private const uint JobObjectLimitKillOnJobClose = 0x00002000;
        private const int ErrorMoreData = 234;

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimitInformation
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimitInformation
        {
            public BasicLimitInformation BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr securityAttributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(
            IntPtr job,
            int informationClass,
            IntPtr information,
            uint informationLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool QueryInformationJobObject(
            IntPtr job,
            int informationClass,
            IntPtr information,
            uint informationLength,
            out uint returnLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        private IntPtr handle;

        public JobObject()
        {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateJobObject failed");

            ExtendedLimitInformation limits = new ExtendedLimitInformation();
            limits.BasicLimitInformation.LimitFlags = JobObjectLimitKillOnJobClose;
            int size = Marshal.SizeOf(typeof(ExtendedLimitInformation));
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                Marshal.StructureToPtr(limits, buffer, false);
                if (!SetInformationJobObject(
                        handle, JobObjectExtendedLimitInformation, buffer, (uint)size))
                    throw new Win32Exception(
                        Marshal.GetLastWin32Error(), "SetInformationJobObject failed");
            }
            catch
            {
                CloseHandle(handle);
                handle = IntPtr.Zero;
                throw;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        public void Assign(Process process)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException("JobObject");
            if (!AssignProcessToJobObject(handle, process.Handle))
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(), "AssignProcessToJobObject failed");
        }

        public int[] GetProcessIds()
        {
            if (handle == IntPtr.Zero)
                return new int[0];

            // A fixed one-MiB buffer avoids a racy query/reallocate sequence
            // and is far larger than any plausible Q1 process tree.
            const int size = 1024 * 1024;
            IntPtr buffer = Marshal.AllocHGlobal(size);
            try
            {
                uint returned;
                if (!QueryInformationJobObject(
                        handle, JobObjectBasicProcessIdList, buffer, size, out returned))
                {
                    int error = Marshal.GetLastWin32Error();
                    if (error != ErrorMoreData)
                        throw new Win32Exception(error, "QueryInformationJobObject failed");
                }
                uint count = unchecked((uint)Marshal.ReadInt32(buffer, sizeof(uint)));
                List<int> ids = new List<int>((int)count);
                int offset = 2 * sizeof(uint);
                for (uint index = 0; index < count; ++index)
                {
                    long id = IntPtr.Size == 8
                        ? Marshal.ReadInt64(buffer, offset + (int)index * IntPtr.Size)
                        : Marshal.ReadInt32(buffer, offset + (int)index * IntPtr.Size);
                    if (id > 0 && id <= Int32.MaxValue)
                        ids.Add((int)id);
                }
                return ids.ToArray();
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        public void Terminate(uint exitCode)
        {
            if (handle != IntPtr.Zero && !TerminateJobObject(handle, exitCode))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "TerminateJobObject failed");
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                CloseHandle(handle);
                handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~JobObject()
        {
            Dispose();
        }
    }
}
'@
}
