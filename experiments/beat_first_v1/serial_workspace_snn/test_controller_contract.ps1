[CmdletBinding()]
param(
    [string]$ControllerPath = "",
    [switch]$ParseOnly,
    [switch]$RunWatchdogProbes,
    [string]$ProbeRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ControllerPath)) {
    $ControllerPath =
        Join-Path $PSScriptRoot "run_arena_100k_controller.ps1"
}

if (
    $RunWatchdogProbes -and
    [string]$PSVersionTable.PSEdition -eq "Core"
) {
    $windowsPowerShell = Join-Path (
        [Environment]::GetFolderPath("System")
    ) "WindowsPowerShell\v1.0\powershell.exe"
    if (-not (Test-Path -LiteralPath $windowsPowerShell -PathType Leaf)) {
        throw (
            "watchdog probes require Windows PowerShell 5.1 because " +
            "pwsh Add-Type cannot emit the helper console executable"
        )
    }
    $relaunchArguments = @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $PSCommandPath,
        "-ControllerPath",
        $ControllerPath,
        "-RunWatchdogProbes"
    )
    if ($ParseOnly) {
        $relaunchArguments += "-ParseOnly"
    }
    if (-not [string]::IsNullOrWhiteSpace($ProbeRoot)) {
        $relaunchArguments += @("-ProbeRoot", $ProbeRoot)
    }
    & $windowsPowerShell @relaunchArguments
    exit $LASTEXITCODE
}

function Assert-Contract {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw "CONTRACT ASSERTION FAILED: $Message"
    }
}

function Assert-Contains {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Needle,
        [Parameter(Mandatory = $true)][string]$Message
    )
    Assert-Contract -Condition $Source.Contains($Needle) -Message $Message
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            return (
                [BitConverter]::ToString($sha.ComputeHash($stream))
            ).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Get-Utf8Sha256 {
    param([Parameter(Mandatory = $true)][string]$Value)
    $utf8 = [Text.UTF8Encoding]::new($false)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return (
            [BitConverter]::ToString(
                $sha.ComputeHash($utf8.GetBytes($Value))
            )
        ).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Quote-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Quote-ManagedCommandLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Get-ParsedControllerCommand {
    param([Parameter(Mandatory = $true)][string]$Command)

    $tokens = $null
    $parseErrors = $null
    $commandAst = [Management.Automation.Language.Parser]::ParseInput(
        $Command,
        [ref]$tokens,
        [ref]$parseErrors
    )
    if ($parseErrors.Count -ne 0) {
        throw "managed command is not valid PowerShell: $Command"
    }
    $commands = @(
        $commandAst.FindAll(
            {
                param($node)
                $node -is [Management.Automation.Language.CommandAst]
            },
            $true
        )
    )
    if ($commands.Count -ne 1) {
        throw "managed command must contain exactly one command AST"
    }
    $elements = @($commands[0].CommandElements)
    if (
        $elements.Count -lt 1 -or
        $elements[0] -isnot
            [Management.Automation.Language.ExpressionAst]
    ) {
        throw "managed command executable is missing"
    }
    try {
        $executable = [string]$elements[0].SafeGetValue()
    }
    catch {
        throw "managed command executable is not a literal"
    }
    $parameters =
        [Collections.Generic.Dictionary[string, object]]::new(
            [StringComparer]::Ordinal
        )
    $index = 1
    while ($index -lt $elements.Count) {
        $parameter = $elements[$index]
        if (
            $parameter -isnot
                [Management.Automation.Language.CommandParameterAst]
        ) {
            throw (
                "managed command contains a positional/unparsed element: " +
                $parameter.Extent.Text
            )
        }
        $name = [string]$parameter.ParameterName
        if ($parameters.ContainsKey($name)) {
            throw "managed command repeats parameter -$name"
        }
        $value = $true
        if (
            ($index + 1) -lt $elements.Count -and
            $elements[$index + 1] -isnot
                [Management.Automation.Language.CommandParameterAst]
        ) {
            try {
                $value = $elements[$index + 1].SafeGetValue()
            }
            catch {
                throw "managed command parameter -$name is not literal"
            }
            $index += 1
        }
        $parameters.Add($name, $value)
        $index += 1
    }
    return [pscustomobject]@{
        executable = $executable
        parameters = $parameters
    }
}

function Assert-ManagedControllerCommandExact {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$ExpectedExecutable,
        [Parameter(Mandatory = $true)]
        [Collections.IDictionary]$ExpectedParameters,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $parsed = Get-ParsedControllerCommand -Command $Command
    Assert-Contract `
        -Condition (
            [IO.Path]::GetFullPath([string]$parsed.executable) -ceq
                [IO.Path]::GetFullPath($ExpectedExecutable)
        ) `
        -Message "$Context executable differs"
    $actualNames = @($parsed.parameters.Keys | Sort-Object)
    $expectedNames = @($ExpectedParameters.Keys | Sort-Object)
    Assert-Contract `
        -Condition (
            ($actualNames -join "`n") -ceq
                ($expectedNames -join "`n")
        ) `
        -Message (
            "$Context parameter set differs; actual=" +
            ($actualNames -join ",")
        )
    foreach ($name in $ExpectedParameters.Keys) {
        $actualValue = $parsed.parameters[[string]$name]
        $expectedValue = $ExpectedParameters[$name]
        $matches = if ($expectedValue -is [bool]) {
            $actualValue -is [bool] -and
                [bool]$actualValue -eq [bool]$expectedValue
        }
        else {
            [string]$actualValue -ceq [string]$expectedValue
        }
        Assert-Contract `
            -Condition $matches `
            -Message (
                "$Context -$name differs; actual='$actualValue' " +
                "expected='$expectedValue'"
            )
    }
}

function Invoke-ProcessWithDeadline {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)][string]$Context,
        [string]$StandardOutputPath = "",
        [string]$StandardErrorPath = ""
    )

    if (
        -not [string]::IsNullOrWhiteSpace($StandardOutputPath) -and
        -not [string]::IsNullOrWhiteSpace($StandardErrorPath) -and
        [IO.Path]::GetFullPath($StandardOutputPath).Equals(
            [IO.Path]::GetFullPath($StandardErrorPath),
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "$Context stdout and stderr paths must differ"
    }
    $startParameters = @{
        FilePath = $FilePath
        ArgumentList = $ArgumentList
        PassThru = $true
        WindowStyle = "Hidden"
    }
    if (-not [string]::IsNullOrWhiteSpace($StandardOutputPath)) {
        $startParameters.RedirectStandardOutput = $StandardOutputPath
    }
    if (-not [string]::IsNullOrWhiteSpace($StandardErrorPath)) {
        $startParameters.RedirectStandardError = $StandardErrorPath
    }
    $process = Start-Process @startParameters
    try {
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            if (-not $process.WaitForExit(5000)) {
                throw "$Context could not terminate after its outer deadline"
            }
            throw "$Context exceeded outer deadline $TimeoutSeconds seconds"
        }
        return [int]$process.ExitCode
    }
    finally {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            if (-not $process.WaitForExit(5000)) {
                throw "$Context process survived forced termination"
            }
        }
        $process.Dispose()
    }
}

function Test-IsUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )
    $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar
    )
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar
    )
    return (
        $fullPath.Equals(
            $fullRoot,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        $fullPath.StartsWith(
            $fullRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    )
}

function New-HangHelper {
    param([Parameter(Mandatory = $true)][string]$Destination)

    $source = @"
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

public static class SwsnnContractHangHelper {
    const UInt32 SYMBOLIC_LINK_FLAG_FILE = 0x0;
    const UInt32 SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE = 0x2;

    [DllImport(
        "kernel32.dll",
        CharSet = CharSet.Unicode,
        SetLastError = true
    )]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool CreateSymbolicLink(
        string symbolicLink,
        string target,
        UInt32 flags
    );

    static string JsonEscape(string value) {
        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"");
    }

    static string Sha256Hex(string path) {
        using (FileStream stream = File.OpenRead(path))
        using (SHA256 sha = SHA256.Create()) {
            byte[] digest = sha.ComputeHash(stream);
            StringBuilder builder = new StringBuilder(64);
            foreach (byte value in digest) {
                builder.Append(value.ToString("x2"));
            }
            return builder.ToString();
        }
    }

    static string RunDirectory() {
        string output = Environment.GetEnvironmentVariable("SWSNN_OUTPUT");
        string runId = Environment.GetEnvironmentVariable("SWSNN_RUN_ID");
        if (String.IsNullOrWhiteSpace(output) ||
            String.IsNullOrWhiteSpace(runId)) {
            throw new InvalidOperationException(
                "SWSNN_OUTPUT and SWSNN_RUN_ID are required"
            );
        }
        return Path.Combine(output, runId);
    }

    static void WriteConfig() {
        string runDirectory = RunDirectory();
        string runId = Environment.GetEnvironmentVariable("SWSNN_RUN_ID");
        string maximumUpdates = Environment.GetEnvironmentVariable(
            "SWSNN_MAX_UPDATES"
        );
        string startMode = Environment.GetEnvironmentVariable(
            "SWSNN_START_MODE"
        );
        string launchPath = Environment.GetEnvironmentVariable(
            "SWSNN_LAUNCH_MANIFEST_PATH"
        );
        string launchSha = Environment.GetEnvironmentVariable(
            "SWSNN_LAUNCH_MANIFEST_SHA256"
        );
        string resumeCheckpoint = Environment.GetEnvironmentVariable(
            "SWSNN_RESUME_CHECKPOINT"
        );
        if (String.IsNullOrWhiteSpace(launchPath) ||
            String.IsNullOrWhiteSpace(launchSha)) {
            throw new InvalidOperationException(
                "controller did not inject the launch binding"
            );
        }
        string parentCheckpoint = "null";
        if (!String.IsNullOrWhiteSpace(resumeCheckpoint)) {
            string canonicalParent = Path.GetFullPath(resumeCheckpoint);
            string fileName = Path.GetFileNameWithoutExtension(
                canonicalParent
            );
            int separator = fileName.LastIndexOf('_');
            int parentUpdate = Int32.Parse(
                fileName.Substring(separator + 1)
            );
            parentCheckpoint =
                "{\"kind\":\"training\"," +
                "\"path\":\"" + JsonEscape(canonicalParent) + "\"," +
                "\"bytes\":" +
                    new FileInfo(canonicalParent).Length.ToString() + "," +
                "\"sha256\":\"" + Sha256Hex(canonicalParent) + "\"," +
                "\"update\":" + parentUpdate.ToString() + "}";
        }
        string config =
            "{\"config\":{" +
            "\"experiment_id\":\"serial_workspace_snn_arena_v3\"," +
            "\"checkpoint_schema\":{" +
                "\"format\":\"serial-workspace-snn-arena-checkpoint\"," +
                "\"version\":3}," +
            "\"run_id\":\"" + JsonEscape(runId) + "\"," +
            "\"start_mode\":\"" + JsonEscape(startMode) + "\"," +
            "\"maximum_updates\":" + maximumUpdates + "," +
            "\"checkpoint_interval\":10000," +
            "\"dataset_content_sha256\":\"" +
                new String('a', 64) + "\"," +
            "\"dataset_integrity\":{\"fixture\":true}," +
            "\"source_fingerprint\":\"" + new String('b', 64) + "\"," +
            "\"runtime_provenance\":{\"fixture\":true}," +
            "\"launch_binding\":{" +
                "\"path\":\"" + JsonEscape(Path.GetFullPath(launchPath)) +
                "\",\"sha256\":\"" + launchSha + "\"}" +
            "},\"parent_checkpoint\":" + parentCheckpoint + "}";
        File.WriteAllText(
            Path.Combine(runDirectory, "config.json"),
            config + Environment.NewLine,
            new UTF8Encoding(false)
        );
    }

    static string CheckpointRecord(string checkpointPath, int update) {
        return
            "{\"kind\":\"training\"" +
            ",\"path\":\"" + JsonEscape(
                Path.GetFullPath(checkpointPath)
            ) + "\"" +
            ",\"bytes\":" +
            new FileInfo(checkpointPath).Length.ToString() +
            ",\"sha256\":\"" + Sha256Hex(checkpointPath) + "\"" +
            ",\"update\":" + update.ToString() +
            "}";
    }

    static int[] ExplicitUpdatesOrDefault(int update, bool scratch) {
        string raw = Environment.GetEnvironmentVariable(
            "SWSNN_CONTRACT_TEST_UPDATES"
        );
        if (String.IsNullOrWhiteSpace(raw)) {
            return update == 0 ?
                new int[] { 0 } :
                (scratch ? new int[] { 0, update } : new int[] { update });
        }
        string[] fields = raw.Split(',');
        int[] updates = new int[fields.Length];
        int previous = -1;
        for (int index = 0; index < fields.Length; index++) {
            int parsed;
            if (!Int32.TryParse(fields[index], out parsed) ||
                parsed < 0 ||
                parsed <= previous) {
                throw new InvalidOperationException(
                    "SWSNN_CONTRACT_TEST_UPDATES must be strictly " +
                    "increasing non-negative integers"
                );
            }
            updates[index] = parsed;
            previous = parsed;
        }
        return updates;
    }

    static void WriteCheckpoint(int update, bool writeResults) {
        string runDirectory = RunDirectory();
        string checkpointDirectory =
            Path.Combine(runDirectory, "checkpoints");
        Directory.CreateDirectory(checkpointDirectory);
        WriteConfig();
        StringBuilder manifest = new StringBuilder();
        bool scratch = String.Equals(
            Environment.GetEnvironmentVariable("SWSNN_START_MODE"),
            "scratch",
            StringComparison.Ordinal
        );
        int[] updates = ExplicitUpdatesOrDefault(update, scratch);
        foreach (int checkpointUpdate in updates) {
            string checkpointName = String.Format(
                "checkpoint_{0:D9}.jld2",
                checkpointUpdate
            );
            string checkpointPath =
                Path.Combine(checkpointDirectory, checkpointName);
            File.WriteAllText(
                checkpointPath,
                "contract-test-only fake checkpoint " +
                checkpointUpdate.ToString() + " " +
                Process.GetCurrentProcess().Id.ToString(),
                new UTF8Encoding(false)
            );
            manifest.Append(
                CheckpointRecord(checkpointPath, checkpointUpdate)
            );
            manifest.Append(Environment.NewLine);
        }
        File.WriteAllText(
            Path.Combine(runDirectory, "checkpoint_manifest.jsonl"),
            manifest.ToString(),
            new UTF8Encoding(false)
        );
        if (writeResults) {
            File.WriteAllText(
                Path.Combine(runDirectory, "results.json"),
                "{}" + Environment.NewLine,
                new UTF8Encoding(false)
            );
        }
    }

    static string TogglePathCase(string path) {
        char[] characters = path.ToCharArray();
        for (int index = 0; index < characters.Length; index++) {
            if (Char.IsLetter(characters[index])) {
                characters[index] = Char.IsUpper(characters[index]) ?
                    Char.ToLowerInvariant(characters[index]) :
                    Char.ToUpperInvariant(characters[index]);
                return new String(characters);
            }
        }
        throw new InvalidOperationException(
            "could not construct a case-mutated path"
        );
    }

    static void ReplaceManifestPath(string replacement) {
        string runDirectory = RunDirectory();
        string manifestPath =
            Path.Combine(runDirectory, "checkpoint_manifest.jsonl");
        string[] lines = File.ReadAllLines(manifestPath);
        if (lines.Length < 1) {
            throw new InvalidOperationException(
                "path mutation requires one manifest record"
            );
        }
        string checkpointPath = Path.Combine(
            runDirectory,
            "checkpoints",
            "checkpoint_000000000.jld2"
        );
        string escaped = JsonEscape(Path.GetFullPath(checkpointPath));
        lines[0] = lines[0].Replace(
            "\"path\":\"" + escaped + "\"",
            "\"path\":\"" + JsonEscape(replacement) + "\""
        );
        File.WriteAllText(
            manifestPath,
            String.Join(Environment.NewLine, lines) +
                Environment.NewLine,
            new UTF8Encoding(false)
        );
    }

    static void CreateRequiredFileSymlink(
        string linkPath,
        string targetPath
    ) {
        if (File.Exists(linkPath)) {
            File.Delete(linkPath);
        }
        if (!CreateSymbolicLink(
            linkPath,
            targetPath,
            SYMBOLIC_LINK_FLAG_FILE |
                SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE
        )) {
            int error = Marshal.GetLastWin32Error();
            if (
                error == 87 &&
                CreateSymbolicLink(
                    linkPath,
                    targetPath,
                    SYMBOLIC_LINK_FLAG_FILE
                )
            ) {
                return;
            }
            throw new Win32Exception(
                error,
                "REPARSE CONTRACT PREREQUISITE FAILED: could not create " +
                "a required file symbolic link"
            );
        }
    }

    static void ReplaceWithRequiredFileSymlink(
        string sourcePath,
        string label
    ) {
        string targetDirectory = Path.Combine(
            RunDirectory(),
            ".fixture_reparse_targets"
        );
        Directory.CreateDirectory(targetDirectory);
        string targetPath = Path.Combine(
            targetDirectory,
            label + "_" + Guid.NewGuid().ToString("N")
        );
        File.Move(sourcePath, targetPath);
        CreateRequiredFileSymlink(sourcePath, targetPath);
        File.WriteAllText(
            Path.Combine(
                RunDirectory(),
                "reparse_fixture_" + label + ".json"
            ),
            "{\"link\":\"" + JsonEscape(Path.GetFullPath(sourcePath)) +
            "\",\"target\":\"" + JsonEscape(Path.GetFullPath(targetPath)) +
            "\"}" + Environment.NewLine,
            new UTF8Encoding(false)
        );
    }

    static void CorruptRecoveryEvidence(string mode) {
        string runDirectory = RunDirectory();
        string manifestPath =
            Path.Combine(runDirectory, "checkpoint_manifest.jsonl");
        if (String.Equals(
            mode,
            "training_stall_invalid_blank",
            StringComparison.Ordinal
        )) {
            File.AppendAllText(
                manifestPath,
                Environment.NewLine,
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_duplicate",
            StringComparison.Ordinal
        )) {
            string first = File.ReadAllLines(manifestPath)[0];
            File.AppendAllText(
                manifestPath,
                first + Environment.NewLine,
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_float_token",
            StringComparison.Ordinal
        )) {
            string manifest = File.ReadAllText(manifestPath);
            File.WriteAllText(
                manifestPath,
                manifest.Replace("\"update\":0", "\"update\":0.0"),
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_duplicate_key",
            StringComparison.Ordinal
        )) {
            string manifest = File.ReadAllText(manifestPath);
            File.WriteAllText(
                manifestPath,
                manifest.Replace(
                    "\"update\":0}",
                    "\"update\":0,\"update\":0}"
                ),
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_array_record",
            StringComparison.Ordinal
        )) {
            string first = File.ReadAllLines(manifestPath)[0];
            File.WriteAllText(
                manifestPath,
                "[" + first + "]" + Environment.NewLine,
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_extra_file",
            StringComparison.Ordinal
        )) {
            File.WriteAllText(
                Path.Combine(runDirectory, "checkpoints", "alias.jld2"),
                "unexpected",
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_foreign_binding",
            StringComparison.Ordinal
        )) {
            string configPath = Path.Combine(runDirectory, "config.json");
            string config = File.ReadAllText(configPath);
            string launchSha = Environment.GetEnvironmentVariable(
                "SWSNN_LAUNCH_MANIFEST_SHA256"
            );
            File.WriteAllText(
                configPath,
                config.Replace(launchSha, new String('c', 64)),
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_nested_duplicate",
            StringComparison.Ordinal
        )) {
            string configPath = Path.Combine(runDirectory, "config.json");
            string config = File.ReadAllText(configPath);
            File.WriteAllText(
                configPath,
                config.Replace(
                    "\"launch_binding\":{\"path\":",
                    "\"launch_binding\":{\"path\":\"duplicate\"," +
                    "\"path\":"
                ),
                new UTF8Encoding(false)
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_path_relative",
            StringComparison.Ordinal
        )) {
            ReplaceManifestPath(
                Path.Combine(
                    "checkpoints",
                    "checkpoint_000000000.jld2"
                )
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_path_alias",
            StringComparison.Ordinal
        )) {
            ReplaceManifestPath(
                Path.Combine(
                    RunDirectory(),
                    "checkpoints",
                    "..",
                    "checkpoints",
                    "checkpoint_000000000.jld2"
                )
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_path_case",
            StringComparison.Ordinal
        )) {
            ReplaceManifestPath(
                TogglePathCase(
                    Path.Combine(
                        RunDirectory(),
                        "checkpoints",
                        "checkpoint_000000000.jld2"
                    )
                )
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_path_external",
            StringComparison.Ordinal
        )) {
            string externalDirectory = Path.Combine(
                Directory.GetParent(RunDirectory()).FullName,
                ".fixture_external_paths"
            );
            Directory.CreateDirectory(externalDirectory);
            string externalPath = Path.Combine(
                externalDirectory,
                "checkpoint_000000000.jld2"
            );
            File.Copy(
                Path.Combine(
                    RunDirectory(),
                    "checkpoints",
                    "checkpoint_000000000.jld2"
                ),
                externalPath,
                true
            );
            ReplaceManifestPath(externalPath);
        } else if (String.Equals(
            mode,
            "training_stall_invalid_reparse_manifest",
            StringComparison.Ordinal
        )) {
            ReplaceWithRequiredFileSymlink(
                manifestPath,
                "manifest"
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_reparse_config",
            StringComparison.Ordinal
        )) {
            ReplaceWithRequiredFileSymlink(
                Path.Combine(runDirectory, "config.json"),
                "config"
            );
        } else if (String.Equals(
            mode,
            "training_stall_invalid_reparse_checkpoint",
            StringComparison.Ordinal
        )) {
            ReplaceWithRequiredFileSymlink(
                Path.Combine(
                    runDirectory,
                    "checkpoints",
                    "checkpoint_000000000.jld2"
                ),
                "checkpoint"
            );
        }
    }

    static int LateDescendantMain(string[] arguments) {
        if (arguments.Length != 4) {
            throw new InvalidOperationException(
                "late descendant arguments differ"
            );
        }
        string manifestPath = arguments[1];
        string readyPath = arguments[2];
        string heartbeatPath = arguments[3];
        using (FileStream locked = new FileStream(
            manifestPath,
            FileMode.Open,
            FileAccess.ReadWrite,
            FileShare.None
        )) {
            File.WriteAllText(
                readyPath,
                "{\"pid\":" +
                    Process.GetCurrentProcess().Id.ToString() + "}" +
                    Environment.NewLine,
                new UTF8Encoding(false)
            );
            int counter = 0;
            while (true) {
                locked.Position = 0;
                int first = locked.ReadByte();
                if (first >= 0) {
                    locked.Position = 0;
                    locked.WriteByte((byte)first);
                    locked.Flush(true);
                }
                File.WriteAllText(
                    heartbeatPath,
                    "{\"pid\":" +
                        Process.GetCurrentProcess().Id.ToString() +
                        ",\"writes\":" + (++counter).ToString() + "}" +
                        Environment.NewLine,
                    new UTF8Encoding(false)
                );
                Thread.Sleep(10);
            }
        }
    }

    static Process StartLateDescendant() {
        string runDirectory = RunDirectory();
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = Process.GetCurrentProcess().MainModule.FileName;
        start.Arguments =
            "--late-descendant " +
            "\"" + Path.Combine(
                runDirectory,
                "checkpoint_manifest.jsonl"
            ) + "\" " +
            "\"" + Path.Combine(
                runDirectory,
                "late_descendant_lock_ready.json"
            ) + "\" " +
            "\"" + Path.Combine(
                runDirectory,
                "late_descendant_heartbeat.json"
            ) + "\"";
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        return Process.Start(start);
    }

    static void WriteVerification() {
        string runId = Environment.GetEnvironmentVariable("SWSNN_RUN_ID");
        string expectedUpdates = Environment.GetEnvironmentVariable(
            "SWSNN_MAX_UPDATES"
        );
        string launchSha = Environment.GetEnvironmentVariable(
            "SWSNN_LAUNCH_MANIFEST_SHA256"
        );
        string verification =
            "{\"format\":\"serial-workspace-snn-arena-run-verification\"," +
            "\"version\":2,\"status\":\"verified_complete\"," +
            "\"verified\":true,\"metrics_verified\":true," +
            "\"expected_updates\":" + expectedUpdates + "," +
            "\"run_id\":\"" + JsonEscape(runId) + "\"," +
            "\"launch_manifest\":{\"sha256\":\"" + launchSha + "\"}," +
            "\"results\":{\"metrics_verified\":true," +
                "\"fixed_panel_recomputation\":{\"verified\":true}}}";
        File.WriteAllText(
            Path.Combine(RunDirectory(), "verification.json"),
            verification + Environment.NewLine,
            new UTF8Encoding(false)
        );
    }

    static bool IsVerifier(string[] arguments) {
        foreach (string argument in arguments) {
            if (argument.IndexOf(
                "verify_arena_run.jl",
                StringComparison.OrdinalIgnoreCase
            ) >= 0) {
                return true;
            }
        }
        return false;
    }

    static Process StartGrandchild() {
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = Path.Combine(
            Environment.SystemDirectory,
            "PING.EXE"
        );
        start.Arguments = "127.0.0.1 -t";
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        return Process.Start(start);
    }

    static void WritePidState(Process grandchild, string phase) {
        string state =
            "{\"phase\":\"" + JsonEscape(phase) + "\"" +
            ",\"helper_pid\":" +
            Process.GetCurrentProcess().Id.ToString() +
            ",\"grandchild_pid\":" + grandchild.Id.ToString() +
            "}";
        File.WriteAllText(
            Path.Combine(RunDirectory(), "contract_helper_pids.json"),
            state + Environment.NewLine,
            new UTF8Encoding(false)
        );
    }

    static void AppendInvocation(string phase, string[] arguments) {
        Directory.CreateDirectory(RunDirectory());
        StringBuilder record = new StringBuilder();
        record.Append("{\"phase\":\"");
        record.Append(JsonEscape(phase));
        record.Append("\",\"pid\":");
        record.Append(Process.GetCurrentProcess().Id.ToString());
        record.Append(",\"arguments\":[");
        for (int index = 0; index < arguments.Length; index++) {
            if (index > 0) {
                record.Append(",");
            }
            record.Append("\"");
            record.Append(JsonEscape(arguments[index]));
            record.Append("\"");
        }
        record.Append("]}");
        File.AppendAllText(
            Path.Combine(
                RunDirectory(),
                "contract_helper_invocations.jsonl"
            ),
            record.ToString() + Environment.NewLine,
            new UTF8Encoding(false)
        );
    }

    public static int Main(string[] arguments) {
        if (
            arguments.Length > 0 &&
            String.Equals(
                arguments[0],
                "--late-descendant",
                StringComparison.Ordinal
            )
        ) {
            return LateDescendantMain(arguments);
        }
        string mode = Environment.GetEnvironmentVariable(
            "SWSNN_CONTRACT_TEST_HELPER_MODE"
        );
        bool verifier = IsVerifier(arguments);
        AppendInvocation(verifier ? "verifier" : "training", arguments);
        if (String.Equals(
            mode,
            "verifier_success",
            StringComparison.Ordinal
        )) {
            if (verifier) {
                WriteVerification();
            } else {
                WriteCheckpoint(2, true);
            }
            return 0;
        }
        if (String.Equals(
            mode,
            "verifier_zero_write_success",
            StringComparison.Ordinal
        )) {
            if (!verifier) {
                WriteCheckpoint(2, true);
            }
            return 0;
        }
        if (String.Equals(
            mode,
            "verifier_stall",
            StringComparison.Ordinal
        ) && !verifier) {
            WriteCheckpoint(2, true);
            return 0;
        }
        if (String.Equals(
            mode,
            "finalize_success_verifier_stall",
            StringComparison.Ordinal
        ) && !verifier) {
            WriteCheckpoint(
                Int32.Parse(
                    Environment.GetEnvironmentVariable("SWSNN_MAX_UPDATES")
                ),
                true
            );
            return 0;
        }

        int update = 0;
        bool writeResults = false;
        if (String.Equals(
            mode,
            "training_stall_target",
            StringComparison.Ordinal
        )) {
            update = Int32.Parse(
                Environment.GetEnvironmentVariable("SWSNN_MAX_UPDATES")
            );
        } else if (String.Equals(
            mode,
            "training_stall_target_results",
            StringComparison.Ordinal
        )) {
            update = Int32.Parse(
                Environment.GetEnvironmentVariable("SWSNN_MAX_UPDATES")
            );
            writeResults = true;
        } else if (String.Equals(
            mode,
            "training_stall_boundary_10000",
            StringComparison.Ordinal
        )) {
            update = 10000;
        } else if (String.Equals(
            mode,
            "training_stall_custom",
            StringComparison.Ordinal
        )) {
            update = Int32.Parse(
                Environment.GetEnvironmentVariable("SWSNN_MAX_UPDATES")
            );
        } else if (String.Equals(
            mode,
            "training_stall_custom_results",
            StringComparison.Ordinal
        )) {
            update = Int32.Parse(
                Environment.GetEnvironmentVariable("SWSNN_MAX_UPDATES")
            );
            writeResults = true;
        } else if (String.Equals(
            mode,
            "training_stall_late_descendant",
            StringComparison.Ordinal
        )) {
            update = 0;
        } else if (String.Equals(
            mode,
            "finalize_success_verifier_stall",
            StringComparison.Ordinal
        )) {
            update = Int32.Parse(
                Environment.GetEnvironmentVariable("SWSNN_MAX_UPDATES")
            );
            writeResults = true;
        } else if (!String.Equals(
            mode,
            "training_stall_resume",
            StringComparison.Ordinal
        ) && !String.Equals(
            mode,
            "verifier_stall",
            StringComparison.Ordinal
        ) && !mode.StartsWith(
            "training_stall_invalid_",
            StringComparison.Ordinal
        )) {
            throw new InvalidOperationException(
                "unknown SWSNN contract helper mode: " + mode
            );
        }

        if (!verifier) {
            WriteCheckpoint(update, writeResults);
            if (mode.StartsWith(
                "training_stall_invalid_",
                StringComparison.Ordinal
            )) {
                CorruptRecoveryEvidence(mode);
            }
        }
        Process grandchild = String.Equals(
            mode,
            "training_stall_late_descendant",
            StringComparison.Ordinal
        ) ? StartLateDescendant() : StartGrandchild();
        if (String.Equals(
            mode,
            "training_stall_late_descendant",
            StringComparison.Ordinal
        )) {
            string readyPath = Path.Combine(
                RunDirectory(),
                "late_descendant_lock_ready.json"
            );
            for (int attempt = 0; attempt < 500; attempt++) {
                if (File.Exists(readyPath)) {
                    break;
                }
                if (grandchild.HasExited) {
                    throw new InvalidOperationException(
                        "late descendant exited before acquiring its lock"
                    );
                }
                Thread.Sleep(10);
            }
            if (!File.Exists(readyPath)) {
                throw new InvalidOperationException(
                    "late descendant did not acquire its manifest lock"
                );
            }
        }
        WritePidState(
            grandchild,
            verifier ? "verifier" : "training"
        );
        Console.Out.WriteLine(
            "contract hang helper ready; mode=" + mode +
            " helper_pid=" + Process.GetCurrentProcess().Id.ToString() +
            " grandchild_pid=" + grandchild.Id.ToString()
        );
        Console.Out.Flush();
        Thread.Sleep(Timeout.Infinite);
        return 0;
    }
}
"@

    Add-Type `
        -TypeDefinition $source `
        -Language CSharp `
        -OutputAssembly $Destination `
        -OutputType ConsoleApplication
    return (Resolve-Path -LiteralPath $Destination).Path
}

function Initialize-ReparseFixtureType {
    if ($null -ne ("SwsnnRequiredReparseFixture" -as [type])) {
        return
    }
    Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class SwsnnRequiredReparseFixture {
    [DllImport(
        "kernel32.dll",
        CharSet = CharSet.Unicode,
        SetLastError = true
    )]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool CreateSymbolicLink(
        string symbolicLink,
        string target,
        UInt32 flags
    );

    public static void Create(
        string symbolicLink,
        string target,
        bool directory
    ) {
        UInt32 typeFlag = directory ? 0x1u : 0x0u;
        if (CreateSymbolicLink(symbolicLink, target, typeFlag | 0x2u)) {
            return;
        }
        int error = Marshal.GetLastWin32Error();
        if (
            error == 87 &&
            CreateSymbolicLink(symbolicLink, target, typeFlag)
        ) {
            return;
        }
        throw new Win32Exception(
            error,
            "REPARSE CONTRACT PREREQUISITE FAILED: CreateSymbolicLink"
        );
    }
}
"@
}

function New-RequiredReparseLink {
    param(
        [Parameter(Mandatory = $true)][string]$LinkPath,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [switch]$Directory
    )

    Initialize-ReparseFixtureType
    [SwsnnRequiredReparseFixture]::Create(
        [IO.Path]::GetFullPath($LinkPath),
        [IO.Path]::GetFullPath($TargetPath),
        [bool]$Directory
    )
    $linkItem = Get-Item -LiteralPath $LinkPath -Force
    Assert-Contract `
        -Condition (
            ($linkItem.Attributes -band
                [IO.FileAttributes]::ReparsePoint) -ne 0
        ) `
        -Message (
            "REPARSE CONTRACT PREREQUISITE FAILED: link lacks " +
            "ReparsePoint attribute: $LinkPath"
        )
}

function Invoke-WithRequiredFileReparse {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $fullPath = [IO.Path]::GetFullPath($Path)
    $originalSha256 = Get-Sha256 -Path $fullPath
    $targetPath = (
        $fullPath +
        ".contract_reparse_target_" +
        [guid]::NewGuid().ToString("N")
    )
    [IO.File]::Move($fullPath, $targetPath)
    try {
        New-RequiredReparseLink `
            -LinkPath $fullPath `
            -TargetPath $targetPath
        & $Action
    }
    finally {
        if (Test-Path -LiteralPath $fullPath) {
            [IO.File]::Delete($fullPath)
        }
        if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
            [IO.File]::Move($targetPath, $fullPath)
        }
    }
    Assert-Contract `
        -Condition (
            (Get-Sha256 -Path $fullPath) -ceq $originalSha256
        ) `
        -Message "$Context file restoration hash differs"
}

function Invoke-WithRequiredDirectoryReparse {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $fullPath = [IO.Path]::GetFullPath($Path)
    $targetPath = (
        $fullPath +
        ".contract_reparse_target_" +
        [guid]::NewGuid().ToString("N")
    )
    [IO.Directory]::Move($fullPath, $targetPath)
    try {
        New-RequiredReparseLink `
            -LinkPath $fullPath `
            -TargetPath $targetPath `
            -Directory
        & $Action
    }
    finally {
        if (Test-Path -LiteralPath $fullPath) {
            [IO.Directory]::Delete($fullPath)
        }
        if (Test-Path -LiteralPath $targetPath -PathType Container) {
            [IO.Directory]::Move($targetPath, $fullPath)
        }
    }
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $fullPath -PathType Container) `
        -Message "$Context directory restoration failed"
}

function Write-Utf8NoBomText {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )

    [IO.File]::WriteAllText(
        [IO.Path]::GetFullPath($Path),
        $Value,
        [Text.UTF8Encoding]::new($false)
    )
}

function Get-HelperInvocationCount {
    param([Parameter(Mandatory = $true)][string]$OutputRoot)

    return @(
        Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter (
            "contract_helper_invocations.jsonl"
        ) -ErrorAction SilentlyContinue |
            ForEach-Object {
                Get-Content -LiteralPath $_.FullName |
                    Where-Object {
                        -not [string]::IsNullOrWhiteSpace($_)
                    }
            }
    ).Count
}

function Wait-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$TimeoutMilliseconds,
        [Parameter(Mandatory = $true)][string]$Context
    )

    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $Path) {
            return
        }
        Start-Sleep -Milliseconds 25
    }
    throw "$Context did not appear before its deadline: $Path"
}

function Test-ProcessAbsent {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][string]$Location,
        [Parameter(Mandatory = $true)][string]$ExpectedProcessName
    )
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            return
        }
        Start-Sleep -Milliseconds 100
    }
    $survivor = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (
        $null -ne $survivor -and
        [string]$survivor.ProcessName -eq $ExpectedProcessName
    ) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
    throw (
        "$Location process $ProcessId survived controller teardown " +
        "(observed_name=$($survivor.ProcessName))"
    )
}

function Invoke-ControllerProbe {
    param(
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ExpectedState,
        [Parameter(Mandatory = $true)][string]$ExpectedRecoveryKind,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [ValidateSet("scratch", "resume", "finalize-only")]
        [string]$StartMode = "scratch",
        [int]$ExpectedUpdates = 2,
        [string]$ResumeCheckpoint = "",
        [string]$ResumeSha256 = "",
        [int]$ResumeUpdate = -1,
        [string]$FixtureUpdates = ""
    )

    $runId = (
        "contract_probe_" +
        $Mode +
        "_" +
        [guid]::NewGuid().ToString("N")
    )
    $powershell = Join-Path $PSHOME "powershell.exe"
    $argumentList = @(
        "-NoProfile"
        "-NonInteractive"
        "-ExecutionPolicy"
        "Bypass"
        "-File"
        (Quote-ProcessArgument $ControllerPath)
        "-RunId"
        $runId
        "-ExpectedUpdates"
        [string]$ExpectedUpdates
        "-JuliaThreads"
        "2"
        "-PollSeconds"
        "1"
        "-StallSeconds"
        "2"
        "-VerifierTimeoutSeconds"
        "2"
        "-StartMode"
        $StartMode
        "-JuliaExecutable"
        (Quote-ProcessArgument $HelperPath)
        "-ProjectPath"
        (Quote-ProcessArgument $CanonicalProject)
        "-DatasetPath"
        (Quote-ProcessArgument $PSScriptRoot)
        "-OutputRoot"
        (Quote-ProcessArgument $OutputRoot)
        "-WorkingDirectory"
        (Quote-ProcessArgument $WorkingDirectory)
    )
    if ($StartMode -ne "scratch") {
        $argumentList += @(
            "-ResumeCheckpoint"
            (Quote-ProcessArgument $ResumeCheckpoint)
            "-ResumeSha256"
            $ResumeSha256
            "-ResumeUpdate"
            [string]$ResumeUpdate
        )
    }
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    $savedFixtureUpdates = $env:SWSNN_CONTRACT_TEST_UPDATES
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $Mode
        if ([string]::IsNullOrWhiteSpace($FixtureUpdates)) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_UPDATES `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_UPDATES = $FixtureUpdates
        }
        $controllerExit = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 15 `
            -Context "controller probe $Mode" `
            -ArgumentList $argumentList
    }
    finally {
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
        if ($null -eq $savedFixtureUpdates) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_UPDATES `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_UPDATES = $savedFixtureUpdates
        }
    }

    Assert-Contract `
        -Condition ($controllerExit -eq 1) `
        -Message "$Mode controller exit must be 1, got $controllerExit"

    $controllerDirectory =
        Join-Path (Join-Path $OutputRoot "_controllers") $runId
    $statusPath = Join-Path $controllerDirectory "controller_status.json"
    $recoveryPath = Join-Path $controllerDirectory "recovery.json"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $statusPath -PathType Leaf) `
        -Message "$Mode did not write controller_status.json"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $recoveryPath -PathType Leaf) `
        -Message "$Mode did not write recovery.json"

    $status = Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json
    $recovery = Get-Content -Raw -LiteralPath $recoveryPath |
        ConvertFrom-Json
    $launchPath = Join-Path $controllerDirectory "launch_manifest.json"
    $launch = Get-Content -Raw -LiteralPath $launchPath |
        ConvertFrom-Json
    Assert-Contract `
        -Condition ([string]$status.state -eq $ExpectedState) `
        -Message (
            "$Mode status differs: observed=$($status.state) " +
            "expected=$ExpectedState"
        )
    Assert-Contract `
        -Condition ([string]$recovery.state -eq $ExpectedState) `
        -Message (
            "$Mode recovery state differs: observed=$($recovery.state) " +
            "expected=$ExpectedState"
        )

    $retryProperty =
        $recovery.PSObject.Properties["verification_retry_command"]
    $resumeProperty = $recovery.PSObject.Properties["resume_command"]
    $selectedCommand = if (
        $ExpectedRecoveryKind -eq "verification-retry"
    ) {
        [string]$retryProperty.Value
    }
    else {
        [string]$resumeProperty.Value
    }
    if ($ExpectedRecoveryKind -ne "none") {
        $parsedCommand =
            Get-ParsedControllerCommand -Command $selectedCommand
        Assert-Contract `
            -Condition (
                [IO.Path]::GetFullPath(
                    [string]$parsedCommand.executable
                ) -ceq [IO.Path]::GetFullPath($ControllerPath)
            ) `
            -Message "$Mode recovery command executable differs"
        $expectedParameters = [ordered]@{
            RunId = if (
                $ExpectedRecoveryKind -eq "verification-retry"
            ) {
                $runId
            }
            else {
                [string]$recovery.suggested_recovery_run_id
            }
            ExpectedUpdates = [string]$ExpectedUpdates
            JuliaThreads = "2"
            PollSeconds = "1"
            StallSeconds = "2"
            VerifierTimeoutSeconds = "2"
            JuliaExecutable = $HelperPath
            ProjectPath = $CanonicalProject
            TrainingScript =
                Join-Path $PSScriptRoot "train_arena_100k.jl"
            VerifierScript =
                Join-Path $PSScriptRoot "verify_arena_run.jl"
            DatasetPath = $PSScriptRoot
            OutputRoot = $OutputRoot
            WorkingDirectory = $WorkingDirectory
            MutexName = [string]$launch.mutex_name
        }
        if ($ExpectedRecoveryKind -eq "verification-retry") {
            $expectedParameters["VerifyOnly"] = $true
            $expectedParameters["ExpectedLaunchManifestSha256"] =
                [string]$recovery.launch_manifest_sha256
        }
        else {
            $expectedParameters["StartMode"] = $ExpectedRecoveryKind
            $expectedParameters["ResumeCheckpoint"] =
                [string]$recovery.recovery_checkpoint.path
            $expectedParameters["ResumeSha256"] =
                [string]$recovery.recovery_checkpoint.sha256
            $expectedParameters["ResumeUpdate"] =
                [string]$recovery.recovery_checkpoint.update
        }
        $actualNames = @($parsedCommand.parameters.Keys | Sort-Object)
        $expectedNames = @($expectedParameters.Keys | Sort-Object)
        Assert-Contract `
            -Condition (
                ($actualNames -join "`n") -ceq
                    ($expectedNames -join "`n")
            ) `
            -Message (
                "$Mode recovery command parameter set differs; actual=" +
                ($actualNames -join ",")
            )
        foreach ($expectedName in $expectedParameters.Keys) {
            $actualValue = $parsedCommand.parameters[$expectedName]
            $expectedValue = $expectedParameters[$expectedName]
            $valueMatches = if ($expectedValue -is [bool]) {
                $actualValue -is [bool] -and
                    [bool]$actualValue -eq [bool]$expectedValue
            }
            else {
                [string]$actualValue -ceq [string]$expectedValue
            }
            Assert-Contract `
                -Condition $valueMatches `
                -Message (
                    "$Mode recovery command -$expectedName differs; " +
                    "actual='$actualValue' expected='$expectedValue'"
                )
        }
    }
    if ($ExpectedRecoveryKind -eq "resume") {
        Assert-Contract `
            -Condition (
                [string]$recovery.suggested_recovery_run_id -match
                    "^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}[A-Za-z0-9]$" -and
                $selectedCommand.Contains(
                    "-RunId " +
                    (Quote-ManagedCommandLiteral (
                        [string]$recovery.suggested_recovery_run_id
                    ))
                ) -and
                -not $selectedCommand.Contains("<new-run-id>")
            ) `
            -Message "$Mode recovery RunId is not concrete/canonical"
        Assert-Contract `
            -Condition (
                [string]$recovery.suggested_recovery_run_id -match
                    (
                        "^" +
                        [regex]::Escape($runId) +
                        "-recovery-resume-[0-9]{8}T[0-9]{13}Z$"
                    )
            ) `
            -Message "$Mode recovery RunId is not tied to its source run"
        Assert-Contract `
            -Condition (
                $null -ne $resumeProperty -and
                [string]$resumeProperty.Value -match
                    "-StartMode resume"
            ) `
            -Message "$Mode did not emit a resume recovery command"
        Assert-Contract `
            -Condition (
                $null -eq $retryProperty -or
                [string]::IsNullOrWhiteSpace(
                    [string]$retryProperty.Value
                )
            ) `
            -Message "$Mode unexpectedly emitted verifier retry"
        Assert-Contract `
            -Condition (
                $recovery.recovery_checkpoint.
                    controller_artifact_lineage_verified -eq $true -and
                $recovery.recovery_checkpoint.payload_state_verified -eq
                    $false -and
                $recovery.recovery_checkpoint.
                    resume_requires_fail_closed_payload_validation -eq $true
            ) `
            -Message "$Mode recovery candidate overclaims payload validation"
    }
    elseif ($ExpectedRecoveryKind -eq "finalize-only") {
        Assert-Contract `
            -Condition (
                [string]$recovery.suggested_recovery_run_id -match
                    "^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}[A-Za-z0-9]$" -and
                $selectedCommand.Contains(
                    "-RunId " +
                    (Quote-ManagedCommandLiteral (
                        [string]$recovery.suggested_recovery_run_id
                    ))
                ) -and
                -not $selectedCommand.Contains("<new-run-id>")
            ) `
            -Message "$Mode finalization RunId is not concrete/canonical"
        Assert-Contract `
            -Condition (
                [string]$recovery.suggested_recovery_run_id -match
                    (
                        "^" +
                        [regex]::Escape($runId) +
                        "-recovery-finalize-only-" +
                        "[0-9]{8}T[0-9]{13}Z$"
                    )
            ) `
            -Message "$Mode finalization RunId is not tied to its source run"
        Assert-Contract `
            -Condition (
                $null -ne $resumeProperty -and
                [string]$resumeProperty.Value -match
                    "-StartMode finalize-only"
            ) `
            -Message "$Mode did not emit a finalize-only recovery command"
        Assert-Contract `
            -Condition (
                $recovery.recovery_checkpoint.
                    candidate_status -eq "payload_unverified"
            ) `
            -Message "$Mode finalization candidate is not marked unverified"
    }
    elseif ($ExpectedRecoveryKind -eq "verification-retry") {
        Assert-Contract `
            -Condition (
                $null -ne $retryProperty -and
                -not [string]::IsNullOrWhiteSpace(
                    [string]$retryProperty.Value
                )
            ) `
            -Message "$Mode did not emit a verifier retry command"
        Assert-Contract `
            -Condition (
                $null -eq $resumeProperty -or
                [string]::IsNullOrWhiteSpace(
                    [string]$resumeProperty.Value
                )
            ) `
            -Message "$Mode emitted an invalid training resume command"
        $retryCommand = [string]$retryProperty.Value
        Assert-Contract `
            -Condition ($retryCommand -match "(^| )-VerifyOnly( |$)") `
            -Message "$Mode verifier retry is not controller-managed"
    }
    elseif ($ExpectedRecoveryKind -eq "none") {
        Assert-Contract `
            -Condition (
                ($null -eq $resumeProperty -or
                    [string]::IsNullOrWhiteSpace(
                        [string]$resumeProperty.Value
                    )) -and
                ($null -eq $retryProperty -or
                    [string]::IsNullOrWhiteSpace(
                        [string]$retryProperty.Value
                    )) -and
                $null -eq $recovery.recovery_checkpoint
            ) `
            -Message "$Mode emitted recovery from invalid evidence"
    }
    else {
        throw "unknown expected recovery kind: $ExpectedRecoveryKind"
    }

    $pidPath = Join-Path (
        Join-Path $OutputRoot $runId
    ) "contract_helper_pids.json"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $pidPath -PathType Leaf) `
        -Message "$Mode helper PID artifact is missing"
    $pidState = Get-Content -Raw -LiteralPath $pidPath | ConvertFrom-Json
    Test-ProcessAbsent `
        -ProcessId ([int]$pidState.helper_pid) `
        -Location "$Mode helper" `
        -ExpectedProcessName "swsnn_contract_hang_helper"
    Test-ProcessAbsent `
        -ProcessId ([int]$pidState.grandchild_pid) `
        -Location "$Mode grandchild" `
        -ExpectedProcessName $(if (
            $Mode -eq "training_stall_late_descendant"
        ) {
            "swsnn_contract_hang_helper"
        }
        else {
            "PING"
        })

    if ($ExpectedState -eq "failed_stalled") {
        Assert-Contract `
            -Condition ($recovery.child_terminated -eq $true) `
            -Message "$Mode did not report child_terminated=true"
        Assert-Contract `
            -Condition ($null -ne $recovery.child_exit_code) `
            -Message "$Mode did not persist the child exit code"
    }
    else {
        Assert-Contract `
            -Condition ($recovery.verifier_terminated -eq $true) `
            -Message "$Mode did not report verifier_terminated=true"
        Assert-Contract `
            -Condition ($null -ne $recovery.verifier_exit_code) `
            -Message "$Mode did not persist the verifier exit code"
    }

    if ($Mode -match "^training_stall_invalid_reparse_(.+)$") {
        $reparseLabel = [string]$Matches[1]
        $markerPath = Join-Path (
            Join-Path $OutputRoot $runId
        ) "reparse_fixture_$reparseLabel.json"
        Assert-Contract `
            -Condition (Test-Path -LiteralPath $markerPath -PathType Leaf) `
            -Message (
                "REPARSE CONTRACT PREREQUISITE FAILED: helper did not " +
                "create $reparseLabel marker"
            )
        $marker = Get-Content -Raw -LiteralPath $markerPath |
            ConvertFrom-Json
        $linkItem = Get-Item -LiteralPath ([string]$marker.link) -Force
        Assert-Contract `
            -Condition (
                ($linkItem.Attributes -band
                    [IO.FileAttributes]::ReparsePoint) -ne 0
            ) `
            -Message (
                "REPARSE CONTRACT PREREQUISITE FAILED: helper " +
                "$reparseLabel path is not a reparse point"
            )
    }
    if ($Mode -eq "training_stall_late_descendant") {
        $heartbeatPath = Join-Path (
            Join-Path $OutputRoot $runId
        ) "late_descendant_heartbeat.json"
        Assert-Contract `
            -Condition (
                Test-Path -LiteralPath $heartbeatPath -PathType Leaf
            ) `
            -Message "assigned late descendant did not write its heartbeat"
        $heartbeat = Get-Content -Raw -LiteralPath $heartbeatPath |
            ConvertFrom-Json
        Assert-Contract `
            -Condition ([int]$heartbeat.writes -gt 0) `
            -Message "assigned late descendant did not perform a write"
        Assert-Contract `
            -Condition (
                $null -ne $recovery.recovery_checkpoint -and
                (Get-Sha256 -Path (
                    [string]$recovery.recovery_checkpoint.manifest_path
                )) -ceq
                    [string]$recovery.recovery_checkpoint.lineage[0].
                        checkpoint_manifest.sha256
            ) `
            -Message (
                "controller scanned/published recovery before its assigned " +
                "late-writing descendant became quiescent"
            )
    }

    Write-Host (
        "PASS controller probe $Mode; artifacts retained at " +
        $controllerDirectory
    )
    return [pscustomobject]@{
        mode = $Mode
        run_id = $runId
        run_directory = Join-Path $OutputRoot $runId
        controller_directory = $controllerDirectory
        status_path = $statusPath
        recovery_path = $recoveryPath
        recovery = $recovery
        status = $status
        launch = $launch
        mutex_name = [string]$launch.mutex_name
        launch_path = $launchPath
        launch_sha256 = Get-Sha256 -Path $launchPath
        start_mode = $StartMode
        expected_updates = $ExpectedUpdates
    }
}

function Get-LineageArtifactSnapshots {
    param([Parameter(Mandatory = $true)]$RecoveryCheckpoint)

    $snapshots = [Collections.Generic.List[object]]::new()
    foreach ($node in @($RecoveryCheckpoint.lineage)) {
        foreach (
            $artifactName in @(
                "selected_checkpoint"
                "config"
                "checkpoint_manifest"
                "launch_manifest"
            )
        ) {
            $artifact = $node.$artifactName
            $path = [string]$artifact.path
            $item = Get-Item -LiteralPath $path -Force
            $snapshots.Add([pscustomobject]@{
                artifact_name = $artifactName
                path = $item.FullName
                bytes = [int64]$item.Length
                sha256 = Get-Sha256 -Path $item.FullName
            })
        }
    }
    return @($snapshots)
}

function Assert-LineageArtifactSnapshotsUnchanged {
    param(
        [Parameter(Mandatory = $true)]$Snapshots,
        [Parameter(Mandatory = $true)][string]$Context
    )

    foreach ($snapshot in @($Snapshots)) {
        Assert-Contract `
            -Condition (
                Test-Path -LiteralPath $snapshot.path -PathType Leaf
            ) `
            -Message "$Context lineage artifact disappeared: $($snapshot.path)"
        $current = Get-Item -LiteralPath $snapshot.path -Force
        Assert-Contract `
            -Condition (
                [string]$current.FullName -ceq [string]$snapshot.path -and
                [int64]$current.Length -eq [int64]$snapshot.bytes -and
                (Get-Sha256 -Path $current.FullName) -ceq
                    [string]$snapshot.sha256
            ) `
            -Message (
                "$Context lineage artifact changed: " +
                "$($snapshot.artifact_name) $($snapshot.path)"
            )
    }
}

function Invoke-LineageControllerProbe {
    param(
        [Parameter(Mandatory = $true)]$ParentProbe,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][string]$ExpectedState,
        [Parameter(Mandatory = $true)][string]$ExpectedRecoveryKind,
        [Parameter(Mandatory = $true)]
        [ValidateSet("resume", "finalize-only")]
        [string]$StartMode,
        [Parameter(Mandatory = $true)][int]$ExpectedUpdates,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [string]$FixtureUpdates = ""
    )

    $selected = $ParentProbe.recovery.recovery_checkpoint
    Assert-Contract `
        -Condition ($null -ne $selected) `
        -Message "$Mode parent probe has no recovery checkpoint"
    $parentSnapshots =
        Get-LineageArtifactSnapshots -RecoveryCheckpoint $selected
    Assert-LineageArtifactSnapshotsUnchanged `
        -Snapshots $parentSnapshots `
        -Context "$Mode pre-launch"

    $probe = Invoke-ControllerProbe `
        -Mode $Mode `
        -ExpectedState $ExpectedState `
        -ExpectedRecoveryKind $ExpectedRecoveryKind `
        -HelperPath $HelperPath `
        -OutputRoot $OutputRoot `
        -CanonicalProject $CanonicalProject `
        -WorkingDirectory $WorkingDirectory `
        -StartMode $StartMode `
        -ExpectedUpdates $ExpectedUpdates `
        -ResumeCheckpoint ([string]$selected.path) `
        -ResumeSha256 ([string]$selected.sha256) `
        -ResumeUpdate ([int]$selected.update) `
        -FixtureUpdates $FixtureUpdates

    $launch = Get-Content -Raw -LiteralPath $probe.launch_path |
        ConvertFrom-Json
    Assert-Contract `
        -Condition (
            [string]$launch.start_mode -ceq $StartMode -and
            [string]$launch.parent_checkpoint.path -ceq
                [string]$selected.path -and
            [string]$launch.parent_checkpoint.sha256 -ceq
                [string]$selected.sha256 -and
            [int]$launch.parent_checkpoint.update -eq
                [int]$selected.update
        ) `
        -Message "$Mode launch does not pin the exact parent triple"
    $expectedLineage = @($selected.lineage)
    $actualLineage = @($launch.parent_lineage)
    Assert-Contract `
        -Condition ($actualLineage.Count -eq $expectedLineage.Count) `
        -Message "$Mode launch lineage depth differs"
    for ($index = 0; $index -lt $expectedLineage.Count; $index++) {
        Assert-Contract `
            -Condition (
                [string]$actualLineage[$index].selected_checkpoint.path -ceq
                    [string]$expectedLineage[$index].
                        selected_checkpoint.path -and
                [string]$actualLineage[$index].selected_checkpoint.sha256 -ceq
                    [string]$expectedLineage[$index].
                        selected_checkpoint.sha256 -and
                [int]$actualLineage[$index].selected_checkpoint.update -eq
                    [int]$expectedLineage[$index].
                        selected_checkpoint.update
            ) `
            -Message "$Mode launch lineage node $index differs"
    }
    Assert-LineageArtifactSnapshotsUnchanged `
        -Snapshots $parentSnapshots `
        -Context "$Mode post-controller"
    return $probe
}

function Invoke-ResumePreflightRejectionProbe {
    param(
        [Parameter(Mandatory = $true)]$ParentProbe,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][int]$ExpectedUpdates,
        [Parameter(Mandatory = $true)][string]$ExpectedErrorNeedle,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $selected = $ParentProbe.recovery.recovery_checkpoint
    Assert-Contract `
        -Condition ($null -ne $selected) `
        -Message "$Label parent has no recovery checkpoint"
    $runId = (
        "contract_preflight_" +
        $Label +
        "_" +
        [guid]::NewGuid().ToString("N")
    )
    $controllerDirectory = Join-Path (
        Join-Path $OutputRoot "_controllers"
    ) $runId
    $runDirectory = Join-Path $OutputRoot $runId
    $stdoutPath = Join-Path $OutputRoot "$runId.outer.stdout.log"
    $stderrPath = Join-Path $OutputRoot "$runId.outer.stderr.log"
    $ledgerBefore = Get-HelperInvocationCount -OutputRoot $OutputRoot
    $powershell = Join-Path $PSHOME "powershell.exe"
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    $savedUpdates = $env:SWSNN_CONTRACT_TEST_UPDATES
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE = "training_stall_custom"
        $env:SWSNN_CONTRACT_TEST_UPDATES = [string]$ExpectedUpdates
        $exitCode = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 15 `
            -Context "$Label resume preflight rejection" `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                $runId
                "-ExpectedUpdates"
                [string]$ExpectedUpdates
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-StartMode"
                "resume"
                "-ResumeCheckpoint"
                (Quote-ProcessArgument ([string]$selected.path))
                "-ResumeSha256"
                [string]$selected.sha256
                "-ResumeUpdate"
                [string]$selected.update
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
            )
    }
    finally {
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
        if ($null -eq $savedUpdates) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_UPDATES `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_UPDATES = $savedUpdates
        }
    }
    Assert-Contract `
        -Condition ($exitCode -eq 1) `
        -Message "$Label preflight rejection exit differs: $exitCode"
    Assert-Contains `
        -Source ([IO.File]::ReadAllText($stderrPath)) `
        -Needle $ExpectedErrorNeedle `
        -Message "$Label preflight rejection reason differs"
    Assert-Contract `
        -Condition (
            -not (Test-Path -LiteralPath $runDirectory) -and
            -not (Test-Path -LiteralPath $controllerDirectory)
        ) `
        -Message "$Label preflight rejection created child/controller artifacts"
    Assert-Contract `
        -Condition (
            (Get-HelperInvocationCount -OutputRoot $OutputRoot) -eq
                $ledgerBefore
        ) `
        -Message "$Label preflight rejection launched the helper"
    Write-Host "PASS resume preflight rejects $Label before child/artifacts"
}

function Invoke-DeepAncestorManifestMutationProbe {
    param(
        [Parameter(Mandatory = $true)]$ParentProbe,
        [Parameter(Mandatory = $true)]$DeepAncestorProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $manifestPath = Join-Path (
        $DeepAncestorProbe.run_directory
    ) "checkpoint_manifest.jsonl"
    $originalBytes = [IO.File]::ReadAllBytes($manifestPath)
    $originalSha256 = Get-Sha256 -Path $manifestPath
    try {
        [IO.File]::AppendAllText(
            $manifestPath,
            [Environment]::NewLine,
            [Text.UTF8Encoding]::new($false)
        )
        Invoke-ResumePreflightRejectionProbe `
            -ParentProbe $ParentProbe `
            -Label "deep_ancestor_manifest_mutation" `
            -ExpectedUpdates 30000 `
            -ExpectedErrorNeedle (
                "checkpoint manifest contains a blank line"
            ) `
            -HelperPath $HelperPath `
            -OutputRoot $OutputRoot `
            -CanonicalProject $CanonicalProject `
            -WorkingDirectory $WorkingDirectory
    }
    finally {
        [IO.File]::WriteAllBytes($manifestPath, $originalBytes)
    }
    Assert-Contract `
        -Condition ((Get-Sha256 -Path $manifestPath) -ceq $originalSha256) `
        -Message "deep ancestor manifest restoration differs"
}

function Invoke-AncestorForeignBindingProbe {
    param(
        [Parameter(Mandatory = $true)]$ParentProbe,
        [Parameter(Mandatory = $true)]$DeepAncestorProbe,
        [Parameter(Mandatory = $true)]$ForeignProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $configPath = Join-Path $DeepAncestorProbe.run_directory "config.json"
    $originalBytes = [IO.File]::ReadAllBytes($configPath)
    $originalSha256 = Get-Sha256 -Path $configPath
    try {
        $config = Get-Content -Raw -LiteralPath $configPath |
            ConvertFrom-Json
        $config.config.launch_binding.path = $ForeignProbe.launch_path
        $config.config.launch_binding.sha256 = $ForeignProbe.launch_sha256
        Write-Utf8NoBomText `
            -Path $configPath `
            -Value (
                (ConvertTo-Json -InputObject $config -Compress -Depth 100) +
                [Environment]::NewLine
            )
        Invoke-ResumePreflightRejectionProbe `
            -ParentProbe $ParentProbe `
            -Label "deep_ancestor_foreign_binding" `
            -ExpectedUpdates 30000 `
            -ExpectedErrorNeedle (
                "checkpoint lineage launch binding path differs"
            ) `
            -HelperPath $HelperPath `
            -OutputRoot $OutputRoot `
            -CanonicalProject $CanonicalProject `
            -WorkingDirectory $WorkingDirectory
    }
    finally {
        [IO.File]::WriteAllBytes($configPath, $originalBytes)
    }
    Assert-Contract `
        -Condition ((Get-Sha256 -Path $configPath) -ceq $originalSha256) `
        -Message "deep ancestor foreign-binding restoration differs"
}

function Invoke-RunningAncestorMutationProbe {
    param(
        [Parameter(Mandatory = $true)]$ParentProbe,
        [Parameter(Mandatory = $true)]$DeepAncestorProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [switch]$WithResults
    )

    $selected = $ParentProbe.recovery.recovery_checkpoint
    $expectedUpdates = [int]$selected.update + 10000
    $runId = (
        "contract_running_ancestor_mutation" +
        $(if ($WithResults) { "_results" } else { "" }) +
        "_" +
        [guid]::NewGuid().ToString("N")
    )
    $runDirectory = Join-Path $OutputRoot $runId
    $controllerDirectory = Join-Path (
        Join-Path $OutputRoot "_controllers"
    ) $runId
    $outerStdout = Join-Path $OutputRoot "$runId.outer.stdout.log"
    $outerStderr = Join-Path $OutputRoot "$runId.outer.stderr.log"
    $configPath = Join-Path $DeepAncestorProbe.run_directory "config.json"
    $originalConfigBytes = [IO.File]::ReadAllBytes($configPath)
    $originalConfigSha256 = Get-Sha256 -Path $configPath
    $powershell = Join-Path $PSHOME "powershell.exe"
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    $savedUpdates = $env:SWSNN_CONTRACT_TEST_UPDATES
    $controller = $null
    $pidState = $null
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE = if ($WithResults) {
            "training_stall_custom_results"
        }
        else {
            "training_stall_custom"
        }
        $env:SWSNN_CONTRACT_TEST_UPDATES = [string]$expectedUpdates
        $controller = Start-Process `
            -FilePath $powershell `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                $runId
                "-ExpectedUpdates"
                [string]$expectedUpdates
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-StartMode"
                "resume"
                "-ResumeCheckpoint"
                (Quote-ProcessArgument ([string]$selected.path))
                "-ResumeSha256"
                [string]$selected.sha256
                "-ResumeUpdate"
                [string]$selected.update
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
            ) `
            -RedirectStandardOutput $outerStdout `
            -RedirectStandardError $outerStderr `
            -PassThru `
            -WindowStyle Hidden
        $pidPath = Join-Path $runDirectory "contract_helper_pids.json"
        Wait-RequiredPath `
            -Path $pidPath `
            -TimeoutMilliseconds 10000 `
            -Context "running ancestor mutation helper PID"
        $pidState = Get-Content -Raw -LiteralPath $pidPath |
            ConvertFrom-Json

        [IO.File]::AppendAllText(
            $configPath,
            " ",
            [Text.UTF8Encoding]::new($false)
        )
        Assert-Contract `
            -Condition ($controller.WaitForExit(15000)) `
            -Message "running ancestor mutation controller did not exit"
        Assert-Contract `
            -Condition ([int]$controller.ExitCode -eq 1) `
            -Message "running ancestor mutation controller exit differs"
    }
    finally {
        [IO.File]::WriteAllBytes($configPath, $originalConfigBytes)
        if ($null -ne $controller -and -not $controller.HasExited) {
            Stop-Process -Id $controller.Id -Force `
                -ErrorAction SilentlyContinue
            $null = $controller.WaitForExit(5000)
        }
        if ($null -ne $controller) {
            $controller.Dispose()
        }
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
        if ($null -eq $savedUpdates) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_UPDATES `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_UPDATES = $savedUpdates
        }
    }
    Assert-Contract `
        -Condition ((Get-Sha256 -Path $configPath) -ceq $originalConfigSha256) `
        -Message "running ancestor config restoration differs"
    $status = Get-Content -Raw -LiteralPath (
        Join-Path $controllerDirectory "controller_status.json"
    ) | ConvertFrom-Json
    $recovery = Get-Content -Raw -LiteralPath (
        Join-Path $controllerDirectory "recovery.json"
    ) | ConvertFrom-Json
    Assert-Contract `
        -Condition (
            [string]$status.state -eq "failed_stalled" -and
            [string]$status.failure -match
                "pinned parent lineage artifact changed: config"
        ) `
        -Message "post-pin ancestor mutation failure evidence differs"
    Assert-Contract `
        -Condition (
            $null -eq $recovery.recovery_checkpoint -and
            $null -eq $recovery.suggested_recovery_run_id -and
            $null -eq $recovery.resume_command -and
            $null -eq $recovery.verification_retry_command
        ) `
        -Message "post-pin ancestor trust loss emitted a recovery command"
    Test-ProcessAbsent `
        -ProcessId ([int]$pidState.helper_pid) `
        -Location "post-pin ancestor mutation helper" `
        -ExpectedProcessName "swsnn_contract_hang_helper"
    Test-ProcessAbsent `
        -ProcessId ([int]$pidState.grandchild_pid) `
        -Location "post-pin ancestor mutation grandchild" `
        -ExpectedProcessName "PING"
    Write-Host (
        "PASS post-pin deep-ancestor mutation" +
        $(if ($WithResults) { " with current results" } else { "" }) +
        " latches trust loss and suppresses all recovery commands"
    )
}

function Invoke-VerifyOnlyPreflightRejectionProbe {
    param(
        [Parameter(Mandatory = $true)]$InitialProbe,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ExpectedErrorNeedle,
        [Parameter(Mandatory = $true)]
        [bool]$ExpectAuthenticatedStatusUpdate,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $launchSha256 = Get-Sha256 -Path $InitialProbe.launch_path
    $originalStatusSha256 = Get-Sha256 -Path $InitialProbe.status_path
    $retryRoot = Join-Path (
        $InitialProbe.controller_directory
    ) "verification_retries"
    $retryCountBefore = if (
        Test-Path -LiteralPath $retryRoot -PathType Container
    ) {
        @(Get-ChildItem -LiteralPath $retryRoot -Directory -Force).Count
    }
    else {
        0
    }
    $ledgerBefore = Get-HelperInvocationCount -OutputRoot $OutputRoot
    $token = [guid]::NewGuid().ToString("N")
    $stdoutPath = Join-Path $OutputRoot "$Label.$token.stdout.log"
    $stderrPath = Join-Path $OutputRoot "$Label.$token.stderr.log"
    $powershell = Join-Path $PSHOME "powershell.exe"
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE = "verifier_success"
        $exitCode = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 10 `
            -Context "$Label VerifyOnly preflight rejection" `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                [string]$InitialProbe.run_id
                "-ExpectedUpdates"
                [string]$InitialProbe.expected_updates
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-VerifyOnly"
                "-ExpectedLaunchManifestSha256"
                $launchSha256
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
                "-MutexName"
                (Quote-ProcessArgument ([string]$InitialProbe.mutex_name))
            )
    }
    finally {
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
    }
    Assert-Contract `
        -Condition ($exitCode -eq 1) `
        -Message "$Label VerifyOnly preflight rejection exit differs"
    Assert-Contains `
        -Source ([IO.File]::ReadAllText($stderrPath)) `
        -Needle $ExpectedErrorNeedle `
        -Message "$Label VerifyOnly rejection reason differs"
    Assert-Contract `
        -Condition (
            (Get-HelperInvocationCount -OutputRoot $OutputRoot) -eq
                $ledgerBefore
        ) `
        -Message "$Label VerifyOnly preflight launched verifier/training"
    $retryCountAfter = if (
        Test-Path -LiteralPath $retryRoot -PathType Container
    ) {
        @(Get-ChildItem -LiteralPath $retryRoot -Directory -Force).Count
    }
    else {
        0
    }
    Assert-Contract `
        -Condition ($retryCountAfter -eq $retryCountBefore) `
        -Message "$Label VerifyOnly rejection created a retry attempt"
    if ($ExpectAuthenticatedStatusUpdate) {
        $updatedStatus = Get-Content -Raw -LiteralPath (
            $InitialProbe.status_path
        ) | ConvertFrom-Json
        Assert-Contract `
            -Condition (
                (Get-Sha256 -Path $InitialProbe.status_path) -cne
                    $originalStatusSha256 -and
                $updatedStatus.latest_verification_retry.verified -eq
                    $false -and
                [string]$updatedStatus.latest_verification_retry.failure -match
                    [regex]::Escape($ExpectedErrorNeedle)
            ) `
            -Message (
                "$Label authenticated failure was not reflected in " +
                "original status"
            )
    }
    else {
        Assert-Contract `
            -Condition (
                (Get-Sha256 -Path $InitialProbe.status_path) -ceq
                    $originalStatusSha256
            ) `
            -Message "$Label unauthenticated rejection changed status"
    }
}

function Invoke-RequiredParentReparseProbes {
    param(
        [Parameter(Mandatory = $true)]$ParentProbe,
        [Parameter(Mandatory = $true)]$DeepAncestorProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $selectedPath =
        [string]$ParentProbe.recovery.recovery_checkpoint.path
    Invoke-WithRequiredFileReparse `
        -Path $selectedPath `
        -Context "selected parent checkpoint reparse" `
        -Action {
            Invoke-ResumePreflightRejectionProbe `
                -ParentProbe $ParentProbe `
                -Label "selected_parent_checkpoint_reparse" `
                -ExpectedUpdates 30000 `
                -ExpectedErrorNeedle "resume checkpoint is a reparse point" `
                -HelperPath $HelperPath `
                -OutputRoot $OutputRoot `
                -CanonicalProject $CanonicalProject `
                -WorkingDirectory $WorkingDirectory
        }

    $ancestorLaunchPath = $DeepAncestorProbe.launch_path
    Invoke-WithRequiredFileReparse `
        -Path $ancestorLaunchPath `
        -Context "ancestor launch manifest reparse" `
        -Action {
            Invoke-ResumePreflightRejectionProbe `
                -ParentProbe $ParentProbe `
                -Label "ancestor_launch_manifest_reparse" `
                -ExpectedUpdates 30000 `
                -ExpectedErrorNeedle "is a reparse point" `
                -HelperPath $HelperPath `
                -OutputRoot $OutputRoot `
                -CanonicalProject $CanonicalProject `
                -WorkingDirectory $WorkingDirectory
        }

    Invoke-WithRequiredDirectoryReparse `
        -Path $DeepAncestorProbe.run_directory `
        -Context "deep ancestor run directory reparse" `
        -Action {
            Invoke-ResumePreflightRejectionProbe `
                -ParentProbe $ParentProbe `
                -Label "deep_ancestor_run_reparse" `
                -ExpectedUpdates 30000 `
                -ExpectedErrorNeedle "is a reparse point" `
                -HelperPath $HelperPath `
                -OutputRoot $OutputRoot `
                -CanonicalProject $CanonicalProject `
                -WorkingDirectory $WorkingDirectory
        }
    Write-Host (
        "PASS required selected-parent/ancestor launch/ancestor-run " +
        "reparse rejection probes"
    )
}

function Invoke-VerifyOnlyLaunchReparseProbe {
    param(
        [Parameter(Mandatory = $true)]$InitialProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Invoke-WithRequiredFileReparse `
        -Path $InitialProbe.launch_path `
        -Context "VerifyOnly launch manifest reparse" `
        -Action {
            Invoke-VerifyOnlyPreflightRejectionProbe `
                -InitialProbe $InitialProbe `
                -Label "verifyonly_launch_manifest_reparse" `
                -ExpectedErrorNeedle (
                    "Verify-only rejects a reparse-point launch manifest"
                ) `
                -ExpectAuthenticatedStatusUpdate $false `
                -HelperPath $HelperPath `
                -OutputRoot $OutputRoot `
                -CanonicalProject $CanonicalProject `
                -WorkingDirectory $WorkingDirectory
        }
}

function Invoke-VerifyOnlyRetryRootReparseProbe {
    param(
        [Parameter(Mandatory = $true)]$InitialProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $retryRoot = Join-Path (
        $InitialProbe.controller_directory
    ) "verification_retries"
    Assert-Contract `
        -Condition (-not (Test-Path -LiteralPath $retryRoot)) `
        -Message "retry-root reparse fixture requires a fresh controller"
    $targetRoot = (
        $retryRoot +
        ".contract_reparse_target_" +
        [guid]::NewGuid().ToString("N")
    )
    [IO.Directory]::CreateDirectory($targetRoot) | Out-Null
    try {
        New-RequiredReparseLink `
            -LinkPath $retryRoot `
            -TargetPath $targetRoot `
            -Directory
        Invoke-VerifyOnlyPreflightRejectionProbe `
            -InitialProbe $InitialProbe `
            -Label "verifyonly_retry_root_reparse" `
            -ExpectedErrorNeedle "verification retry root" `
            -ExpectAuthenticatedStatusUpdate $true `
            -HelperPath $HelperPath `
            -OutputRoot $OutputRoot `
            -CanonicalProject $CanonicalProject `
            -WorkingDirectory $WorkingDirectory
    }
    finally {
        if (Test-Path -LiteralPath $retryRoot) {
            [IO.Directory]::Delete($retryRoot)
        }
        if (Test-Path -LiteralPath $targetRoot -PathType Container) {
            [IO.Directory]::Delete($targetRoot, $true)
        }
    }
    Write-Host "PASS required VerifyOnly retry-root reparse rejection"
}

function Invoke-ExternalParentRootProbe {
    param(
        [Parameter(Mandatory = $true)]$ExternalParentProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Invoke-ResumePreflightRejectionProbe `
        -ParentProbe $ExternalParentProbe `
        -Label "external_output_root_parent" `
        -ExpectedUpdates 10000 `
        -ExpectedErrorNeedle "checkpoint lineage output root differs" `
        -HelperPath $HelperPath `
        -OutputRoot $OutputRoot `
        -CanonicalProject $CanonicalProject `
        -WorkingDirectory $WorkingDirectory
}

function Get-CaseMutatedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $characters = $Path.ToCharArray()
    for ($index = 0; $index -lt $characters.Length; $index++) {
        if ([char]::IsLetter($characters[$index])) {
            $characters[$index] = if (
                [char]::IsUpper($characters[$index])
            ) {
                [char]::ToLowerInvariant($characters[$index])
            }
            else {
                [char]::ToUpperInvariant($characters[$index])
            }
            return -join $characters
        }
    }
    throw "could not construct a case-mutated path"
}

function Invoke-VerifyOnlyLaunchPathMutationProbe {
    param(
        [Parameter(Mandatory = $true)]$InitialProbe,
        [Parameter(Mandatory = $true)]
        [ValidateSet("relative", "external", "alias", "case")]
        [string]$Mutation,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $launchPath = $InitialProbe.launch_path
    $configPath = Join-Path $InitialProbe.run_directory "config.json"
    $launchBytes = [IO.File]::ReadAllBytes($launchPath)
    $configBytes = [IO.File]::ReadAllBytes($configPath)
    $launchSha256 = Get-Sha256 -Path $launchPath
    $configSha256 = Get-Sha256 -Path $configPath
    try {
        $launch = Get-Content -Raw -LiteralPath $launchPath |
            ConvertFrom-Json
        $canonicalRunDirectory =
            [IO.Path]::GetFullPath($InitialProbe.run_directory)
        $mutatedPath = switch ($Mutation) {
            "relative" {
                Resolve-Path `
                    -LiteralPath $canonicalRunDirectory `
                    -Relative
            }
            "external" {
                Join-Path (
                    Split-Path -Parent $OutputRoot
                ) (
                    "external_run_" +
                    [guid]::NewGuid().ToString("N")
                )
            }
            "alias" {
                Join-Path (
                    Join-Path $canonicalRunDirectory ".."
                ) ([IO.Path]::GetFileName($canonicalRunDirectory))
            }
            "case" {
                Get-CaseMutatedPath -Path $canonicalRunDirectory
            }
        }
        Assert-Contract `
            -Condition ([string]$mutatedPath -cne $canonicalRunDirectory) `
            -Message "$Mutation path mutation did not alter raw spelling"
        $launch.run_directory = [string]$mutatedPath
        Write-Utf8NoBomText `
            -Path $launchPath `
            -Value (
                (ConvertTo-Json -InputObject $launch -Compress -Depth 100) +
                [Environment]::NewLine
            )
        $mutatedLaunchSha256 = Get-Sha256 -Path $launchPath
        $config = Get-Content -Raw -LiteralPath $configPath |
            ConvertFrom-Json
        $config.config.launch_binding.sha256 = $mutatedLaunchSha256
        Write-Utf8NoBomText `
            -Path $configPath `
            -Value (
                (ConvertTo-Json -InputObject $config -Compress -Depth 100) +
                [Environment]::NewLine
            )
        Invoke-VerifyOnlyPreflightRejectionProbe `
            -InitialProbe $InitialProbe `
            -Label "verifyonly_run_path_$Mutation" `
            -ExpectedErrorNeedle (
                "Verify-only launch manifest run directory"
            ) `
            -ExpectAuthenticatedStatusUpdate $false `
            -HelperPath $HelperPath `
            -OutputRoot $OutputRoot `
            -CanonicalProject $CanonicalProject `
            -WorkingDirectory $WorkingDirectory
    }
    finally {
        [IO.File]::WriteAllBytes($launchPath, $launchBytes)
        [IO.File]::WriteAllBytes($configPath, $configBytes)
    }
    Assert-Contract `
        -Condition (
            (Get-Sha256 -Path $launchPath) -ceq $launchSha256 -and
            (Get-Sha256 -Path $configPath) -ceq $configSha256
        ) `
        -Message "$Mutation persisted-path fixture restoration differs"
    Write-Host "PASS VerifyOnly rejects $Mutation persisted run path"
}

function Invoke-CheckpointCadenceMatrix {
    param(
        [Parameter(Mandatory = $true)]$BoundaryParentProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $null = Invoke-LineageControllerProbe `
        -ParentProbe $BoundaryParentProbe `
        -Mode "training_stall_custom" `
        -ExpectedState "failed_stalled" `
        -ExpectedRecoveryKind "finalize-only" `
        -StartMode "resume" `
        -ExpectedUpdates 25000 `
        -FixtureUpdates "20000,25000" `
        -HelperPath $HelperPath `
        -OutputRoot $OutputRoot `
        -CanonicalProject $CanonicalProject `
        -WorkingDirectory $WorkingDirectory
    foreach (
        $invalidCase in @(
            [pscustomobject]@{
                name = "missing_boundary"
                updates = "25000"
            }
            [pscustomobject]@{
                name = "extra_parent_boundary"
                updates = "10000,20000,25000"
            }
            [pscustomobject]@{
                name = "noncadence"
                updates = "15000,20000,25000"
            }
        )
    ) {
        $null = Invoke-LineageControllerProbe `
            -ParentProbe $BoundaryParentProbe `
            -Mode "training_stall_custom" `
            -ExpectedState "failed_stalled" `
            -ExpectedRecoveryKind "none" `
            -StartMode "resume" `
            -ExpectedUpdates 25000 `
            -FixtureUpdates ([string]$invalidCase.updates) `
            -HelperPath $HelperPath `
            -OutputRoot $OutputRoot `
            -CanonicalProject $CanonicalProject `
            -WorkingDirectory $WorkingDirectory
        Write-Host (
            "PASS resume cadence rejects " +
            [string]$invalidCase.name
        )
    }
    $null = Invoke-ControllerProbe `
        -Mode "training_stall_custom" `
        -ExpectedState "failed_stalled" `
        -ExpectedRecoveryKind "none" `
        -ExpectedUpdates 25000 `
        -FixtureUpdates "0,10000,25000" `
        -HelperPath $HelperPath `
        -OutputRoot $OutputRoot `
        -CanonicalProject $CanonicalProject `
        -WorkingDirectory $WorkingDirectory
    Write-Host (
        "PASS checkpoint cadence positive resume boundary and " +
        "missing/extra/noncadence negative matrix"
    )
}

function Invoke-NoncanonicalProjectProbe {
    param(
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$NoncanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $runId = (
        "contract_probe_noncanonical_project_" +
        [guid]::NewGuid().ToString("N")
    )
    $powershell = Join-Path $PSHOME "powershell.exe"
    $stdoutPath = Join-Path $OutputRoot "$runId.stdout.log"
    $stderrPath = Join-Path $OutputRoot "$runId.stderr.log"
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE =
            "training_stall_resume"
        $controllerExit = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 10 `
            -Context "noncanonical project rejection probe" `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                $runId
                "-ExpectedUpdates"
                "1"
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $NoncanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
            )
    }
    finally {
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
    }
    Assert-Contract `
        -Condition ($controllerExit -eq 1) `
        -Message "noncanonical project probe must exit 1"
    Assert-Contains `
        -Source ([IO.File]::ReadAllText($stderrPath)) `
        -Needle "Production controller rejects a noncanonical project path" `
        -Message "noncanonical project rejection reason differs"
    Assert-Contract `
        -Condition (
            -not (Test-Path -LiteralPath (
                Join-Path $OutputRoot $runId
            ))
        ) `
        -Message "noncanonical project probe started a run directory"
    Assert-Contract `
        -Condition (
            -not (Test-Path -LiteralPath (
                Join-Path (
                    Join-Path $OutputRoot "_controllers"
                ) $runId
            ))
        ) `
        -Message "noncanonical project probe created controller artifacts"
    Write-Host "PASS controller rejects noncanonical project before child launch"
}

function Invoke-IdentifierAndMutexRejectionProbes {
    param(
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $powershell = Join-Path $PSHOME "powershell.exe"
    $ledgerCountBefore = @(
        Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter (
            "contract_helper_invocations.jsonl"
        ) -ErrorAction SilentlyContinue |
            ForEach-Object {
                Get-Content -LiteralPath $_.FullName
            }
    ).Count
    foreach (
        $invalidRunId in @(
            "."
            ".."
            ".hidden"
            "trailing."
            "mixed/name"
            "mixed\name"
            "name:stream"
            "CON"
            "con.txt"
            "PRN"
            "AUX.log"
            "NUL"
            "COM1.data"
            "LPT9"
        )
    ) {
        $probeToken = [guid]::NewGuid().ToString("N")
        $stdoutPath =
            Join-Path $OutputRoot "invalid_run_id_$probeToken.stdout.log"
        $stderrPath =
            Join-Path $OutputRoot "invalid_run_id_$probeToken.stderr.log"
        $invalidExit = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 10 `
            -Context "unsafe RunId rejection probe '$invalidRunId'" `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                (Quote-ProcessArgument $invalidRunId)
                "-ExpectedUpdates"
                "2"
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
            )
        Assert-Contract `
            -Condition ($invalidExit -eq 1) `
            -Message "controller accepted unsafe RunId '$invalidRunId'"
        $rejectionText = [IO.File]::ReadAllText($stderrPath)
        if ($invalidRunId -in @("mixed/name", "mixed\name", "name:stream")) {
            Assert-Contract `
                -Condition (
                    $rejectionText.Contains("Cannot validate argument") -and
                    $rejectionText.Contains("RunId")
                ) `
                -Message (
                    "RunId binder rejection reason differs for " +
                    "'$invalidRunId'"
                )
        }
        else {
            Assert-Contains `
                -Source $rejectionText `
                -Needle (
                    "RunId must be a canonical Windows-safe basename"
                ) `
                -Message (
                    "post-bind RunId rejection reason differs for " +
                    "'$invalidRunId'"
                )
        }
    }

    $wrongMutexRunId = (
        "contract_probe_wrong_mutex_" +
        [guid]::NewGuid().ToString("N")
    )
    $wrongMutexStdout =
        Join-Path $OutputRoot "$wrongMutexRunId.stdout.log"
    $wrongMutexStderr =
        Join-Path $OutputRoot "$wrongMutexRunId.stderr.log"
    $wrongMutexExit = Invoke-ProcessWithDeadline `
        -FilePath $powershell `
        -TimeoutSeconds 10 `
        -Context "arbitrary mutex rejection probe" `
        -StandardOutputPath $wrongMutexStdout `
        -StandardErrorPath $wrongMutexStderr `
        -ArgumentList @(
            "-NoProfile"
            "-NonInteractive"
            "-ExecutionPolicy"
            "Bypass"
            "-File"
            (Quote-ProcessArgument $ControllerPath)
            "-RunId"
            $wrongMutexRunId
            "-ExpectedUpdates"
            "2"
            "-JuliaThreads"
            "2"
            "-PollSeconds"
            "1"
            "-StallSeconds"
            "2"
            "-VerifierTimeoutSeconds"
            "2"
            "-JuliaExecutable"
            (Quote-ProcessArgument $HelperPath)
            "-ProjectPath"
            (Quote-ProcessArgument $CanonicalProject)
            "-DatasetPath"
            (Quote-ProcessArgument $PSScriptRoot)
            "-OutputRoot"
            (Quote-ProcessArgument $OutputRoot)
            "-WorkingDirectory"
            (Quote-ProcessArgument $WorkingDirectory)
            "-MutexName"
            (Quote-ProcessArgument "Local\OpenAI.Attacker.Bypass")
        )
    Assert-Contract `
        -Condition ($wrongMutexExit -eq 1) `
        -Message "controller accepted an arbitrary mutex identity"
    Assert-Contains `
        -Source ([IO.File]::ReadAllText($wrongMutexStderr)) `
        -Needle "MutexName is controller-derived and cannot be overridden" `
        -Message "arbitrary mutex rejection reason differs"
    Assert-Contract `
        -Condition (
            -not (Test-Path -LiteralPath (
                Join-Path $OutputRoot $wrongMutexRunId
            )) -and
            -not (Test-Path -LiteralPath (
                Join-Path (
                    Join-Path $OutputRoot "_controllers"
                ) $wrongMutexRunId
            ))
        ) `
        -Message "wrong-mutex rejection created controller/child artifacts"
    $ledgerCountAfter = @(
        Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter (
            "contract_helper_invocations.jsonl"
        ) -ErrorAction SilentlyContinue |
            ForEach-Object {
                Get-Content -LiteralPath $_.FullName
            }
    ).Count
    Assert-Contract `
        -Condition ($ledgerCountAfter -eq $ledgerCountBefore) `
        -Message "identifier/mutex rejection launched a helper process"
    Write-Host "PASS unsafe RunId/device aliases and arbitrary mutex rejection"
}

function Invoke-ForcedControllerExitProbe {
    param(
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $runId = (
        "contract_probe_forced_controller_exit_" +
        [guid]::NewGuid().ToString("N")
    )
    $powershell = Join-Path $PSHOME "powershell.exe"
    $argumentList = @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        ('"' + $ControllerPath + '"'),
        "-RunId",
        $runId,
        "-ExpectedUpdates",
        "2",
        "-JuliaThreads",
        "2",
        "-PollSeconds",
        "1",
        "-StallSeconds",
        "60",
        "-VerifierTimeoutSeconds",
        "60",
        "-JuliaExecutable",
        ('"' + $HelperPath + '"'),
        "-ProjectPath",
        ('"' + $CanonicalProject + '"'),
        "-DatasetPath",
        ('"' + $PSScriptRoot + '"'),
        "-OutputRoot",
        ('"' + $OutputRoot + '"'),
        "-WorkingDirectory",
        ('"' + $WorkingDirectory + '"')
    )
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    $controller = $null
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE =
            "training_stall_resume"
        $controller = Start-Process `
            -FilePath $powershell `
            -ArgumentList $argumentList `
            -PassThru `
            -WindowStyle Hidden
        $pidPath = Join-Path (
            Join-Path $OutputRoot $runId
        ) "contract_helper_pids.json"
        for ($attempt = 0; $attempt -lt 200; $attempt++) {
            if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
                break
            }
            if ($controller.HasExited) {
                throw (
                    "forced-exit probe controller exited before helper " +
                    "reported its PIDs"
                )
            }
            Start-Sleep -Milliseconds 50
        }
        Assert-Contract `
            -Condition (
                Test-Path -LiteralPath $pidPath -PathType Leaf
            ) `
            -Message "forced-exit probe helper PID artifact is missing"
        $pidState = Get-Content -Raw -LiteralPath $pidPath |
            ConvertFrom-Json

        $busyRunId = (
            "contract_probe_mutex_busy_" +
            [guid]::NewGuid().ToString("N")
        )
        $busyExit = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 10 `
            -Context "canonical mutex contention probe" `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                $busyRunId
                "-ExpectedUpdates"
                "2"
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
            )
        Assert-Contract `
            -Condition ($busyExit -eq 1) `
            -Message "second controller bypassed the canonical mutex"
        Assert-Contract `
            -Condition (
                -not (Test-Path -LiteralPath (
                    Join-Path $OutputRoot $busyRunId
                )) -and
                -not (Test-Path -LiteralPath (
                    Join-Path (
                        Join-Path $OutputRoot "_controllers"
                    ) $busyRunId
                ))
            ) `
            -Message "mutex-rejected controller created child artifacts"

        $heldLaunchPath = Join-Path (
            Join-Path (
                Join-Path $OutputRoot "_controllers"
            ) $runId
        ) "launch_manifest.json"
        $heldLaunch = Get-Content -Raw -LiteralPath $heldLaunchPath |
            ConvertFrom-Json
        $expectedMutexScope = (
            $CanonicalProject.ToLowerInvariant() +
            "`n" +
            ([IO.Path]::GetFullPath($OutputRoot)).ToLowerInvariant()
        )
        $expectedMutexScopeSha256 =
            Get-Utf8Sha256 -Value $expectedMutexScope
        $expectedMutexName = (
            "Local\OpenAI.SerialWorkspaceSNN.Arena100k.v1." +
            $expectedMutexScopeSha256.Substring(0, 32)
        )
        Assert-Contract `
            -Condition (
                [string]$heldLaunch.mutex_scope_sha256 -ceq
                    $expectedMutexScopeSha256 -and
                [string]$heldLaunch.mutex_name -ceq
                    $expectedMutexName
            ) `
            -Message (
                "launch mutex identity differs from independently derived " +
                "project/output scope"
            )
        Stop-Process -Id $controller.Id -Force
        Assert-Contract `
            -Condition ($controller.WaitForExit(5000)) `
            -Message "forced-exit controller survived forced termination"
        Test-ProcessAbsent `
            -ProcessId ([int]$pidState.helper_pid) `
            -Location "forced-exit helper" `
            -ExpectedProcessName "swsnn_contract_hang_helper"
        Test-ProcessAbsent `
            -ProcessId ([int]$pidState.grandchild_pid) `
            -Location "forced-exit grandchild" `
            -ExpectedProcessName "PING"
        $reacquireMutex = [Threading.Mutex]::new(
            $false,
            [string]$heldLaunch.mutex_name
        )
        $reacquired = $false
        try {
            try {
                $reacquired = $reacquireMutex.WaitOne(0)
            }
            catch [Threading.AbandonedMutexException] {
                $reacquired = $true
            }
            Assert-Contract `
                -Condition $reacquired `
                -Message "canonical mutex was not released after teardown"
            $reacquireMutex.ReleaseMutex()
            $reacquired = $false
        }
        finally {
            if ($reacquired) {
                $reacquireMutex.ReleaseMutex()
            }
            $reacquireMutex.Dispose()
        }
        Write-Host (
            "PASS canonical mutex contention/reacquire and forced controller " +
            "exit Job cleanup"
        )
    }
    finally {
        if (
            $null -ne $controller -and
            -not $controller.HasExited
        ) {
            Stop-Process -Id $controller.Id -Force `
                -ErrorAction SilentlyContinue
        }
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
    }
}

function Invoke-ManagedVerifyOnlyTimeoutProbe {
    param(
        [Parameter(Mandatory = $true)]$InitialProbe,
        [Parameter(Mandatory = $true)][string]$HelperPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot,
        [Parameter(Mandatory = $true)][string]$CanonicalProject,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $launchPath = Join-Path (
        $InitialProbe.controller_directory
    ) "launch_manifest.json"
    $launchSha256 = Get-Sha256 -Path $launchPath
    Assert-Contract `
        -Condition (
            $launchSha256 -eq
                [string]$InitialProbe.recovery.launch_manifest_sha256
        ) `
        -Message "initial verifier-timeout recovery launch SHA differs"
    $resultsPath = Join-Path $InitialProbe.run_directory "results.json"
    $manifestPath = Join-Path (
        $InitialProbe.run_directory
    ) "checkpoint_manifest.jsonl"
    $resultsSha256 = Get-Sha256 -Path $resultsPath
    $manifestSha256 = Get-Sha256 -Path $manifestPath
    $ledgerPath = Join-Path (
        $InitialProbe.run_directory
    ) "contract_helper_invocations.jsonl"
    $ledgerBefore = @(
        Get-Content -LiteralPath $ledgerPath |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    $retryRoot = Join-Path (
        $InitialProbe.controller_directory
    ) "verification_retries"
    $retryCountBefore = if (
        Test-Path -LiteralPath $retryRoot -PathType Container
    ) {
        @(Get-ChildItem -LiteralPath $retryRoot -Directory).Count
    }
    else {
        0
    }

    $powershell = Join-Path $PSHOME "powershell.exe"
    $savedMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    $verifyOnlyStartedAt = Get-Date
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE = "verifier_stall"
        $verifyOnlyExit = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 10 `
            -Context "managed verify-only timeout probe" `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                [string]$InitialProbe.run_id
                "-ExpectedUpdates"
                "2"
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "30"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-VerifyOnly"
                "-ExpectedLaunchManifestSha256"
                $launchSha256
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
                "-MutexName"
                (Quote-ProcessArgument ([string]$InitialProbe.mutex_name))
            )
    }
    finally {
        if ($null -eq $savedMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedMode
        }
    }
    Assert-Contract `
        -Condition ($verifyOnlyExit -eq 1) `
        -Message "managed verify-only timeout must exit 1"
    $verifyOnlyElapsed = ((Get-Date) - $verifyOnlyStartedAt).TotalSeconds
    Assert-Contract `
        -Condition ($verifyOnlyElapsed -lt 8.0) `
        -Message (
            "verifier timeout overshot its 2-second deadline: " +
            "$verifyOnlyElapsed seconds"
        )

    $retryDirectories = @(
        Get-ChildItem -LiteralPath $retryRoot -Directory |
            Sort-Object LastWriteTimeUtc
    )
    Assert-Contract `
        -Condition (
            $retryDirectories.Count -eq ($retryCountBefore + 1)
        ) `
        -Message "managed verify-only did not create one retry attempt"
    $attemptDirectory = $retryDirectories[-1].FullName
    $statusPath = Join-Path $attemptDirectory "controller_status.json"
    $recoveryPath = Join-Path $attemptDirectory "recovery.json"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $statusPath -PathType Leaf) `
        -Message "managed verify-only status is missing"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $recoveryPath -PathType Leaf) `
        -Message "managed verify-only recovery is missing"
    $status = Get-Content -Raw -LiteralPath $statusPath |
        ConvertFrom-Json
    $recovery = Get-Content -Raw -LiteralPath $recoveryPath |
        ConvertFrom-Json
    Assert-Contract `
        -Condition (
            [string]$status.controller_mode -eq "verify-only" -and
            [string]$status.state -eq "failed_verifier_stalled"
        ) `
        -Message "managed verify-only timeout status differs"
    Assert-Contract `
        -Condition (
            [string]$recovery.state -eq "failed_verifier_stalled" -and
            $recovery.verifier_terminated -eq $true -and
            $null -ne $recovery.verifier_exit_code
        ) `
        -Message "managed verify-only timeout recovery differs"
    $retryCommand = [string]$recovery.verification_retry_command
    Assert-Contract `
        -Condition ($retryCommand -match "(^| )-VerifyOnly( |$)") `
        -Message "managed retry did not remain controller-managed"
    $managedRetryParameters = [ordered]@{
        RunId = [string]$InitialProbe.run_id
        ExpectedUpdates = "2"
        JuliaThreads = "2"
        PollSeconds = "30"
        StallSeconds = "2"
        VerifierTimeoutSeconds = "2"
        JuliaExecutable = $HelperPath
        ProjectPath = $CanonicalProject
        TrainingScript = Join-Path $PSScriptRoot "train_arena_100k.jl"
        VerifierScript = Join-Path $PSScriptRoot "verify_arena_run.jl"
        DatasetPath = $PSScriptRoot
        OutputRoot = $OutputRoot
        WorkingDirectory = $WorkingDirectory
        MutexName = [string]$InitialProbe.mutex_name
        VerifyOnly = $true
        ExpectedLaunchManifestSha256 = $launchSha256
    }
    Assert-ManagedControllerCommandExact `
        -Command $retryCommand `
        -ExpectedExecutable $ControllerPath `
        -ExpectedParameters $managedRetryParameters `
        -Context "managed VerifyOnly timeout retry command"
    Assert-Contract `
        -Condition ((Get-Sha256 -Path $launchPath) -eq $launchSha256) `
        -Message "managed verify-only modified launch_manifest.json"
    Assert-Contract `
        -Condition ((Get-Sha256 -Path $resultsPath) -eq $resultsSha256) `
        -Message "managed verify-only modified results.json"
    Assert-Contract `
        -Condition ((Get-Sha256 -Path $manifestPath) -eq $manifestSha256) `
        -Message "managed verify-only modified checkpoint manifest"
    $ledgerAfter = @(
        Get-Content -LiteralPath $ledgerPath |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    Assert-Contract `
        -Condition ($ledgerAfter.Count -eq ($ledgerBefore.Count + 1)) `
        -Message "verify-only invocation ledger count differs"
    $newInvocation = $ledgerAfter[-1] | ConvertFrom-Json
    Assert-Contract `
        -Condition ([string]$newInvocation.phase -eq "verifier") `
        -Message "verify-only invocation ledger recorded a training launch"
    $originalStatusPath = Join-Path (
        $InitialProbe.controller_directory
    ) "controller_status.json"
    $originalStatus = Get-Content -Raw -LiteralPath $originalStatusPath |
        ConvertFrom-Json
    Assert-Contract `
        -Condition (
            [string]$originalStatus.state -eq
                "failed_verifier_stalled" -and
            [string]$originalStatus.controller_mode -eq "verify-only" -and
            $originalStatus.latest_verification_retry.verified -eq $false
        ) `
        -Message "verify-only failure was not reflected in original status"

    $pidPath = Join-Path (
        $InitialProbe.run_directory
    ) "contract_helper_pids.json"
    $pidState = Get-Content -Raw -LiteralPath $pidPath |
        ConvertFrom-Json
    Assert-Contract `
        -Condition ([string]$pidState.phase -eq "verifier") `
        -Message "verify-only probe unexpectedly launched training"
    Test-ProcessAbsent `
        -ProcessId ([int]$pidState.helper_pid) `
        -Location "managed verify-only helper" `
        -ExpectedProcessName "swsnn_contract_hang_helper"
    Test-ProcessAbsent `
        -ProcessId ([int]$pidState.grandchild_pid) `
        -Location "managed verify-only grandchild" `
        -ExpectedProcessName "PING"

    $retryCountBeforeWrongPin = $retryDirectories.Count
    $originalStatusShaBeforeWrongPin =
        Get-Sha256 -Path $originalStatusPath
    $ledgerCountBeforeWrongPin = $ledgerAfter.Count
    $wrongPinToken = [guid]::NewGuid().ToString("N")
    $wrongPinStdout =
        Join-Path $OutputRoot "wrong_pin_$wrongPinToken.stdout.log"
    $wrongPinStderr =
        Join-Path $OutputRoot "wrong_pin_$wrongPinToken.stderr.log"
    $wrongPinExit = Invoke-ProcessWithDeadline `
        -FilePath $powershell `
        -TimeoutSeconds 10 `
        -Context "wrong launch SHA rejection probe" `
        -StandardOutputPath $wrongPinStdout `
        -StandardErrorPath $wrongPinStderr `
        -ArgumentList @(
            "-NoProfile"
            "-NonInteractive"
            "-ExecutionPolicy"
            "Bypass"
            "-File"
            (Quote-ProcessArgument $ControllerPath)
            "-RunId"
            [string]$InitialProbe.run_id
            "-ExpectedUpdates"
            "2"
            "-JuliaThreads"
            "2"
            "-PollSeconds"
            "1"
            "-StallSeconds"
            "2"
            "-VerifierTimeoutSeconds"
            "2"
            "-VerifyOnly"
            "-ExpectedLaunchManifestSha256"
            ("0" * 64)
            "-JuliaExecutable"
            (Quote-ProcessArgument $HelperPath)
            "-ProjectPath"
            (Quote-ProcessArgument $CanonicalProject)
            "-DatasetPath"
            (Quote-ProcessArgument $PSScriptRoot)
            "-OutputRoot"
            (Quote-ProcessArgument $OutputRoot)
            "-WorkingDirectory"
            (Quote-ProcessArgument $WorkingDirectory)
            "-MutexName"
            (Quote-ProcessArgument ([string]$InitialProbe.mutex_name))
        )
    Assert-Contract `
        -Condition ($wrongPinExit -eq 1) `
        -Message "verify-only accepted a wrong launch SHA pin"
    Assert-Contains `
        -Source ([IO.File]::ReadAllText($wrongPinStderr)) `
        -Needle "Verify-only launch manifest SHA-256 differs" `
        -Message "wrong launch SHA rejection reason differs"
    $retryCountAfterWrongPin = @(
        Get-ChildItem -LiteralPath $retryRoot -Directory
    ).Count
    Assert-Contract `
        -Condition (
            $retryCountAfterWrongPin -eq $retryCountBeforeWrongPin
        ) `
        -Message "wrong launch SHA created a retry attempt or child"
    Assert-Contract `
        -Condition (
            (Get-Sha256 -Path $originalStatusPath) -eq
                $originalStatusShaBeforeWrongPin
        ) `
        -Message "wrong launch SHA modified original controller status"
    $ledgerCountAfterWrongPin = @(
        Get-Content -LiteralPath $ledgerPath |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    ).Count
    Assert-Contract `
        -Condition (
            $ledgerCountAfterWrongPin -eq $ledgerCountBeforeWrongPin
        ) `
        -Message "wrong launch SHA started a helper process"

    $savedSuccessMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
    try {
        $env:SWSNN_CONTRACT_TEST_HELPER_MODE = "verifier_success"
        $successExit = Invoke-ProcessWithDeadline `
            -FilePath $powershell `
            -TimeoutSeconds 10 `
            -Context "managed verify-only success probe" `
            -ArgumentList @(
                "-NoProfile"
                "-NonInteractive"
                "-ExecutionPolicy"
                "Bypass"
                "-File"
                (Quote-ProcessArgument $ControllerPath)
                "-RunId"
                [string]$InitialProbe.run_id
                "-ExpectedUpdates"
                "2"
                "-JuliaThreads"
                "2"
                "-PollSeconds"
                "1"
                "-StallSeconds"
                "2"
                "-VerifierTimeoutSeconds"
                "2"
                "-VerifyOnly"
                "-ExpectedLaunchManifestSha256"
                $launchSha256
                "-JuliaExecutable"
                (Quote-ProcessArgument $HelperPath)
                "-ProjectPath"
                (Quote-ProcessArgument $CanonicalProject)
                "-DatasetPath"
                (Quote-ProcessArgument $PSScriptRoot)
                "-OutputRoot"
                (Quote-ProcessArgument $OutputRoot)
                "-WorkingDirectory"
                (Quote-ProcessArgument $WorkingDirectory)
                "-MutexName"
                (Quote-ProcessArgument ([string]$InitialProbe.mutex_name))
            )
    }
    finally {
        if ($null -eq $savedSuccessMode) {
            Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedSuccessMode
        }
    }
    Assert-Contract `
        -Condition ($successExit -eq 0) `
        -Message "managed verify-only success replay did not exit zero"
    $retryCountAfterSuccess = @(
        Get-ChildItem -LiteralPath $retryRoot -Directory
    ).Count
    Assert-Contract `
        -Condition (
            $retryCountAfterSuccess -eq ($retryCountBeforeWrongPin + 1)
        ) `
        -Message "managed verify-only success did not create one attempt"
    $successLedger = @(
        Get-Content -LiteralPath $ledgerPath |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    Assert-Contract `
        -Condition (
            $successLedger.Count -eq ($ledgerCountBeforeWrongPin + 1) -and
            [string](($successLedger[-1] | ConvertFrom-Json).phase) -eq
                "verifier"
        ) `
        -Message "managed verify-only success launched training"
    $successfulOriginalStatus =
        Get-Content -Raw -LiteralPath $originalStatusPath |
        ConvertFrom-Json
    Assert-Contract `
        -Condition (
            [string]$successfulOriginalStatus.state -eq
                "verified_complete" -and
            $successfulOriginalStatus.verified -eq $true -and
            $successfulOriginalStatus.latest_verification_retry.verified -eq
                $true
        ) `
        -Message "verify-only success was not reflected in original status"
    Assert-Contract `
        -Condition (
            (Get-Sha256 -Path $launchPath) -eq $launchSha256 -and
            (Get-Sha256 -Path $resultsPath) -eq $resultsSha256 -and
            (Get-Sha256 -Path $manifestPath) -eq $manifestSha256
        ) `
        -Message "verify-only success changed immutable run evidence"

    $verificationPath = Join-Path (
        $InitialProbe.run_directory
    ) "verification.json"
    Assert-Contract `
        -Condition (Test-Path -LiteralPath $verificationPath -PathType Leaf) `
        -Message "successful VerifyOnly did not leave verification.json"
    foreach (
        $replayCase in @(
            [pscustomobject]@{
                mode = "verifier_success"
                expected_exit = 0
                expect_report = $true
                label = "fresh recreation over archived prior report"
            }
            [pscustomobject]@{
                mode = "verifier_zero_write_success"
                expected_exit = 1
                expect_report = $false
                label = "zero-write stale-report replay"
            }
        )
    ) {
        $preexistingVerificationSha256 =
            Get-Sha256 -Path $verificationPath
        $retryCountBeforeReplay = @(
            Get-ChildItem -LiteralPath $retryRoot -Directory
        ).Count
        $ledgerCountBeforeReplay = @(
            Get-Content -LiteralPath $ledgerPath |
                Where-Object {
                    -not [string]::IsNullOrWhiteSpace($_)
                }
        ).Count
        $savedReplayMode = $env:SWSNN_CONTRACT_TEST_HELPER_MODE
        try {
            $env:SWSNN_CONTRACT_TEST_HELPER_MODE =
                [string]$replayCase.mode
            $replayExit = Invoke-ProcessWithDeadline `
                -FilePath $powershell `
                -TimeoutSeconds 10 `
                -Context ([string]$replayCase.label) `
                -ArgumentList @(
                    "-NoProfile"
                    "-NonInteractive"
                    "-ExecutionPolicy"
                    "Bypass"
                    "-File"
                    (Quote-ProcessArgument $ControllerPath)
                    "-RunId"
                    [string]$InitialProbe.run_id
                    "-ExpectedUpdates"
                    "2"
                    "-JuliaThreads"
                    "2"
                    "-PollSeconds"
                    "1"
                    "-StallSeconds"
                    "2"
                    "-VerifierTimeoutSeconds"
                    "2"
                    "-VerifyOnly"
                    "-ExpectedLaunchManifestSha256"
                    $launchSha256
                    "-JuliaExecutable"
                    (Quote-ProcessArgument $HelperPath)
                    "-ProjectPath"
                    (Quote-ProcessArgument $CanonicalProject)
                    "-DatasetPath"
                    (Quote-ProcessArgument $PSScriptRoot)
                    "-OutputRoot"
                    (Quote-ProcessArgument $OutputRoot)
                    "-WorkingDirectory"
                    (Quote-ProcessArgument $WorkingDirectory)
                    "-MutexName"
                    (Quote-ProcessArgument (
                        [string]$InitialProbe.mutex_name
                    ))
                )
        }
        finally {
            if ($null -eq $savedReplayMode) {
                Remove-Item Env:SWSNN_CONTRACT_TEST_HELPER_MODE `
                    -ErrorAction SilentlyContinue
            }
            else {
                $env:SWSNN_CONTRACT_TEST_HELPER_MODE = $savedReplayMode
            }
        }
        Assert-Contract `
            -Condition (
                [int]$replayExit -eq [int]$replayCase.expected_exit
            ) `
            -Message (
                "$($replayCase.label) exit differs: $replayExit"
            )
        $replayDirectories = @(
            Get-ChildItem -LiteralPath $retryRoot -Directory |
                Sort-Object LastWriteTimeUtc
        )
        Assert-Contract `
            -Condition (
                $replayDirectories.Count -eq
                    ($retryCountBeforeReplay + 1)
            ) `
            -Message "$($replayCase.label) retry attempt count differs"
        $replayDirectory = $replayDirectories[-1].FullName
        $archivePath = Join-Path (
            $replayDirectory
        ) "preexisting_verification.json"
        Assert-Contract `
            -Condition (
                Test-Path -LiteralPath $archivePath -PathType Leaf
            ) `
            -Message "$($replayCase.label) did not archive prior report"
        Assert-Contract `
            -Condition (
                (Get-Sha256 -Path $archivePath) -ceq
                    $preexistingVerificationSha256
            ) `
            -Message "$($replayCase.label) changed archived report bytes"
        Assert-Contract `
            -Condition (
                [bool]$replayCase.expect_report -eq
                    (Test-Path -LiteralPath $verificationPath -PathType Leaf)
            ) `
            -Message "$($replayCase.label) fresh report presence differs"
        $replayLedger = @(
            Get-Content -LiteralPath $ledgerPath |
                Where-Object {
                    -not [string]::IsNullOrWhiteSpace($_)
                }
        )
        Assert-Contract `
            -Condition (
                $replayLedger.Count -eq ($ledgerCountBeforeReplay + 1) -and
                [string]((
                    $replayLedger[-1] | ConvertFrom-Json
                ).phase) -eq "verifier"
            ) `
            -Message "$($replayCase.label) launched training or no verifier"
        $replayOriginalStatus =
            Get-Content -Raw -LiteralPath $originalStatusPath |
            ConvertFrom-Json
        $previousArchive =
            $replayOriginalStatus.latest_verification_retry.
                previous_verification_archive
        Assert-Contract `
            -Condition (
                [string]$previousArchive.path -ceq $archivePath -and
                [string]$previousArchive.sha256 -ceq
                    $preexistingVerificationSha256 -and
                $replayOriginalStatus.latest_verification_retry.verified -eq
                    [bool]$replayCase.expect_report
            ) `
            -Message (
                "$($replayCase.label) original status archive/result differs"
            )
        if (-not [bool]$replayCase.expect_report) {
            Assert-Contract `
                -Condition (
                    [string]$replayOriginalStatus.state -ne
                        "verified_complete" -and
                    [string]$replayOriginalStatus.
                        latest_verification_retry.failure -match
                        "verification"
                ) `
                -Message (
                    "zero-write verifier reused a stale report or did not " +
                    "record failure"
                )
        }
    }
    Write-Host (
        "PASS managed VerifyOnly timeout/success, exact retry AST, " +
        "stale-report archive/recreation/zero-write rejection, " +
        "zero-training ledger, original status, and no-orphan contract"
    )
}

$resolvedController = (Resolve-Path -LiteralPath $ControllerPath).Path
$parseErrors = $null
$tokens = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $resolvedController,
    [ref]$tokens,
    [ref]$parseErrors
)
Assert-Contract `
    -Condition ($parseErrors.Count -eq 0) `
    -Message (
        "controller parse errors: " +
        (($parseErrors | ForEach-Object Message) -join "; ")
    )
Write-Host "PASS controller PowerShell parse"

if ($ParseOnly) {
    exit 0
}

$controllerSource = [IO.File]::ReadAllText($resolvedController)
$functionNames = @(
    $ast.FindAll(
        {
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst]
        },
        $true
    ) | ForEach-Object Name
)
foreach ($requiredFunction in @(
    "New-KillOnCloseJob",
    "Start-AssignedProcess",
    "Stop-ChildProcessTree",
    "Get-RecoveryCheckpoint",
    "New-VerificationRetryCommand"
)) {
    Assert-Contract `
        -Condition ($requiredFunction -in $functionNames) `
        -Message "controller is missing function $requiredFunction"
}

$retryCalls = @(
    $ast.FindAll(
        {
            param($node)
            $node -is [Management.Automation.Language.CommandAst] -and
            $node.GetCommandName() -eq "New-VerificationRetryCommand"
        },
        $true
    )
)
Assert-Contract `
    -Condition ($retryCalls.Count -gt 0) `
    -Message "controller has no managed verification-retry call site"
$requiredRetryArguments = @(
    "-ControllerScript",
    "-RunId",
    "-ExpectedUpdates",
    "-JuliaThreads",
    "-PollSeconds",
    "-StallSeconds",
    "-VerifierTimeoutSeconds",
    "-JuliaPath",
    "-ProjectPath",
    "-TrainingScript",
    "-VerifierScript",
    "-DatasetPath",
    "-OutputRoot",
    "-WorkingDirectory",
    "-MutexName",
    "-LaunchManifestSha256"
)
foreach ($retryCall in $retryCalls) {
    $elements = @(
        $retryCall.CommandElements |
            ForEach-Object { $_.Extent.Text }
    )
    foreach ($requiredArgument in $requiredRetryArguments) {
        Assert-Contract `
            -Condition ($requiredArgument -in $elements) `
            -Message (
                "managed verifier-retry call at line " +
                "$($retryCall.Extent.StartLineNumber) omits " +
                $requiredArgument
            )
    }
}

foreach ($requiredText in @(
    "--startup-file=no",
    "--history-file=no",
    "--project=`$resolvedProject",
    "--threads=`$JuliaThreads,0",
    "--threads=1,0",
    "-VerifyOnly",
    "-ExpectedLaunchManifestSha256",
    "Production controller rejects a noncanonical project path",
    "Production controller rejects a noncanonical training script",
    "Production controller rejects a noncanonical verifier script",
    "running_stalled_warning",
    "running_stalled_terminating",
    "failed_stalled",
    "failed_verifier_stalled",
    "CanonicalRunIdPattern",
    "Arena100k.v1.",
    "mutex_scope_sha256",
    "Assert-ExactJsonProperties",
    "payload_unverified",
    "resume_requires_fail_closed_payload_validation",
    "SWSNN_LAUNCH_MANIFEST_PATH",
    "SWSNN_LAUNCH_MANIFEST_SHA256",
    "parent_lineage",
    "Assert-PinnedLineageCurrent",
    "verification retry root",
    "latest_verification_retry",
    "remainingVerifierMilliseconds",
    "TerminateJobAndWaitForEmpty",
    "QueryInformationJobObject",
    "CREATE_SUSPENDED",
    "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
    "AssignProcessToJobObject",
    "ResumeThread"
)) {
    Assert-Contains `
        -Source $controllerSource `
        -Needle $requiredText `
        -Message "controller static contract is missing '$requiredText'"
}
Assert-Contract `
    -Condition (
        $controllerSource -match
            "(?s)CreateProcess(?:W)?.+AssignProcessToJobObject.+" +
            "ResumeThread"
    ) `
    -Message "child launch is not visibly create-suspended/assign/resume"
Assert-Contract `
    -Condition (
        $controllerSource -match
            "(?s)if \(\`$verifierElapsed -ge " +
            "\`$VerifierTimeoutSeconds\).+" +
            "verification_retry_command"
    ) `
    -Message (
        "verifier-timeout recovery must persist " +
        "verification_retry_command"
    )
Assert-Contract `
    -Condition (
        $controllerSource -match
            "(?s)failed_stalled.+Stop-ChildProcessTree.+?" +
            "WaitForExitAndGetCode.+?Get-RecoveryCheckpoint"
    ) `
    -Message "stalled recovery scans artifacts before child termination"
Assert-Contract `
    -Condition (
        $controllerSource -match
            "(?s)Stop-ChildProcessTree.+?WaitForExitAndGetCode.+?" +
            "TerminateJobAndWaitForEmpty.+?Get-RecoveryCheckpoint"
    ) `
    -Message "recovery scan does not wait for Job quiescence"
Assert-Contract `
    -Condition (
        $controllerSource -match
            "(?s)requestedOutputRoot.+?mutexScopeSha256.+?" +
            "canonicalMutexName.+?cannot be overridden"
    ) `
    -Message "mutex identity is not deterministically derived/fail-closed"
$numericOnlySha256 = "1234567890" * 6 + "1234"
$numericShaCommand = (
    "& " +
    (Quote-ManagedCommandLiteral $resolvedController) +
    " -VerifyOnly -ExpectedLaunchManifestSha256 " +
    (Quote-ManagedCommandLiteral $numericOnlySha256)
)
$parsedNumericShaCommand =
    Get-ParsedControllerCommand -Command $numericShaCommand
Assert-Contract `
    -Condition (
        $parsedNumericShaCommand.parameters[
            "ExpectedLaunchManifestSha256"
        ] -is [string] -and
        [string]$parsedNumericShaCommand.parameters[
            "ExpectedLaunchManifestSha256"
        ] -ceq $numericOnlySha256
    ) `
    -Message "numeric-only SHA command literal was coerced from string"
Write-Host "PASS controller static production contract"

if (-not $RunWatchdogProbes) {
    exit 0
}

if ([string]::IsNullOrWhiteSpace($ProbeRoot)) {
    $ProbeRoot = Join-Path (
        [IO.Path]::GetTempPath()
    ) (
        "swsnn_controller_contract_" +
        (Get-Date -Format "yyyyMMdd_HHmmss") +
        "_" +
        [guid]::NewGuid().ToString("N")
    )
}
$resolvedProbeRoot = [IO.Path]::GetFullPath($ProbeRoot)
Assert-Contract `
    -Condition (
        (Test-IsUnderRoot `
            -Path $resolvedProbeRoot `
            -Root ([IO.Path]::GetTempPath())) -or
        (Test-IsUnderRoot `
            -Path $resolvedProbeRoot `
            -Root "C:\tmp")
    ) `
    -Message "ProbeRoot must stay under the OS temp root or C:\tmp"
[IO.Directory]::CreateDirectory($resolvedProbeRoot) | Out-Null
$helperPath = Join-Path $resolvedProbeRoot "swsnn_contract_hang_helper.exe"
New-HangHelper -Destination $helperPath | Out-Null
Assert-Contract `
    -Condition (Test-Path -LiteralPath $helperPath -PathType Leaf) `
    -Message "contract hang helper was not compiled"

$canonicalProject = (
    Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
).Path
$workingDirectory = (
    Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")
).Path
$outputRoot = Join-Path $resolvedProbeRoot "output"
[IO.Directory]::CreateDirectory($outputRoot) | Out-Null

Invoke-NoncanonicalProjectProbe `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -NoncanonicalProject $resolvedProbeRoot `
    -WorkingDirectory $workingDirectory
Invoke-IdentifierAndMutexRejectionProbes `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-ForcedControllerExitProbe `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$lineageScratch = Invoke-ControllerProbe `
    -Mode "training_stall_resume" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "resume" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory `
    -ExpectedUpdates 20000
$lineageResume1 = Invoke-LineageControllerProbe `
    -ParentProbe $lineageScratch `
    -Mode "training_stall_boundary_10000" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "resume" `
    -StartMode "resume" `
    -ExpectedUpdates 20000 `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$lineageResume2 = Invoke-LineageControllerProbe `
    -ParentProbe $lineageResume1 `
    -Mode "training_stall_target" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "finalize-only" `
    -StartMode "resume" `
    -ExpectedUpdates 20000 `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$null = Invoke-LineageControllerProbe `
    -ParentProbe $lineageResume2 `
    -Mode "finalize_success_verifier_stall" `
    -ExpectedState "failed_verifier_stalled" `
    -ExpectedRecoveryKind "verification-retry" `
    -StartMode "finalize-only" `
    -ExpectedUpdates 20000 `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Write-Host (
    "PASS real scratch -> resume -> resume -> finalize-only controller " +
    "lineage with exact pre/post ancestor pins"
)
$foreignScratch = Invoke-ControllerProbe `
    -Mode "training_stall_resume" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "resume" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-DeepAncestorManifestMutationProbe `
    -ParentProbe $lineageResume2 `
    -DeepAncestorProbe $lineageScratch `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-AncestorForeignBindingProbe `
    -ParentProbe $lineageResume2 `
    -DeepAncestorProbe $lineageScratch `
    -ForeignProbe $foreignScratch `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-RequiredParentReparseProbes `
    -ParentProbe $lineageResume2 `
    -DeepAncestorProbe $lineageScratch `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-RunningAncestorMutationProbe `
    -ParentProbe $lineageResume2 `
    -DeepAncestorProbe $lineageScratch `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-RunningAncestorMutationProbe `
    -ParentProbe $lineageResume2 `
    -DeepAncestorProbe $lineageScratch `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory `
    -WithResults

$externalOutputRoot = Join-Path $resolvedProbeRoot "external_output"
[IO.Directory]::CreateDirectory($externalOutputRoot) | Out-Null
$externalScratch = Invoke-ControllerProbe `
    -Mode "training_stall_resume" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "resume" `
    -ExpectedUpdates 10000 `
    -HelperPath $helperPath `
    -OutputRoot $externalOutputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-ExternalParentRootProbe `
    -ExternalParentProbe $externalScratch `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory

foreach (
    $invalidRecoveryMode in @(
        "training_stall_invalid_blank"
        "training_stall_invalid_duplicate"
        "training_stall_invalid_float_token"
        "training_stall_invalid_duplicate_key"
        "training_stall_invalid_array_record"
        "training_stall_invalid_extra_file"
        "training_stall_invalid_foreign_binding"
        "training_stall_invalid_nested_duplicate"
        "training_stall_invalid_path_relative"
        "training_stall_invalid_path_external"
        "training_stall_invalid_path_alias"
        "training_stall_invalid_path_case"
        "training_stall_invalid_reparse_manifest"
        "training_stall_invalid_reparse_config"
        "training_stall_invalid_reparse_checkpoint"
    )
) {
    $null = Invoke-ControllerProbe `
        -Mode $invalidRecoveryMode `
        -ExpectedState "failed_stalled" `
        -ExpectedRecoveryKind "none" `
        -HelperPath $helperPath `
        -OutputRoot $outputRoot `
        -CanonicalProject $canonicalProject `
        -WorkingDirectory $workingDirectory
}
$null = Invoke-ControllerProbe `
    -Mode "training_stall_late_descendant" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "resume" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Write-Host (
    "PASS stop-before-scan with assigned late-writing descendant holding " +
    "the checkpoint manifest exclusively"
)
Invoke-CheckpointCadenceMatrix `
    -BoundaryParentProbe $lineageResume1 `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory

$null = Invoke-ControllerProbe `
    -Mode "training_stall_target" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "finalize-only" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$null = Invoke-ControllerProbe `
    -Mode "training_stall_target_results" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "verification-retry" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$verifyOnlyPathMutationBase = Invoke-ControllerProbe `
    -Mode "training_stall_target_results" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "verification-retry" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
foreach ($pathMutation in @("relative", "external", "alias", "case")) {
    Invoke-VerifyOnlyLaunchPathMutationProbe `
        -InitialProbe $verifyOnlyPathMutationBase `
        -Mutation $pathMutation `
        -HelperPath $helperPath `
        -OutputRoot $outputRoot `
        -CanonicalProject $canonicalProject `
        -WorkingDirectory $workingDirectory
}
$verifyOnlyLaunchReparseBase = Invoke-ControllerProbe `
    -Mode "training_stall_target_results" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "verification-retry" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-VerifyOnlyLaunchReparseProbe `
    -InitialProbe $verifyOnlyLaunchReparseBase `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$verifyOnlyRetryRootReparseBase = Invoke-ControllerProbe `
    -Mode "training_stall_target_results" `
    -ExpectedState "failed_stalled" `
    -ExpectedRecoveryKind "verification-retry" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-VerifyOnlyRetryRootReparseProbe `
    -InitialProbe $verifyOnlyRetryRootReparseBase `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
$verifierStallProbe = Invoke-ControllerProbe `
    -Mode "verifier_stall" `
    -ExpectedState "failed_verifier_stalled" `
    -ExpectedRecoveryKind "verification-retry" `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory
Invoke-ManagedVerifyOnlyTimeoutProbe `
    -InitialProbe $verifierStallProbe `
    -HelperPath $helperPath `
    -OutputRoot $outputRoot `
    -CanonicalProject $canonicalProject `
    -WorkingDirectory $workingDirectory

Write-Host (
    "PASS controller watchdog/recovery/orphan probes; " +
    "artifacts retained at $resolvedProbeRoot"
)
