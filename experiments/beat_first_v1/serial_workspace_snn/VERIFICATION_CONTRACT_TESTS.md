# Arena production controller/verifier contract tests

These tests target the fail-closed boundary around a production arena run.
They do not train a useful model and they do not replace the final 100k run.

The suite has three layers:

1. `test_verification_contract_units.jl` exercises verifier helpers with
   temporary files.
2. `test_controller_contract.ps1` parses the controller and can run disposable
   watchdog/recovery/orphan probes with a fake executable.
3. `test_verifier_fixture_mutations.ps1` reruns the real verifier against a
   disposable, already verified U2 run while temporarily tampering with one
   artifact at a time.

## 1. Cheap unit contract

Run the unit suite with the same hermetic runtime contract as the production
verifier:

```powershell
$julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$project = "C:\Users\fshuu\Documents\tetris\experiments\beat_first_v1"
$test = Join-Path $project "serial_workspace_snn\test_verification_contract_units.jl"

& $julia `
  --startup-file=no `
  --history-file=no `
  --project=$project `
  --threads=1,0 `
  $test
```

This covers:

- verifier startup/history options, `--threads=1,0`, canonical active Project,
  and BLAS thread count 1;
- exact checkpoint-directory membership, including rejection of alias files,
  wrong-update finalization files, and directories;
- blank and duplicate checkpoint-manifest lines;
- manifest-to-live-checkpoint path, byte-size, and SHA binding;
- generic artifact-reference path, byte-size, and SHA binding;
- exact v3 checkpoint payload schemas, finite/shape checks, deterministic
  sampler state, trainer/warmup/segment/dynamics state, and rejection of
  missing or extra payload fields;
- exact 36-field component-loss telemetry, its declared bit-exact alias
  contract, progress telemetry schema v3, and parameterized type/value
  mutations for every component field;
- exact completed-window denominator provenance via
  `completed_component_loss_window_updates`, including results/finalization
  binding, final trace `window_updates` equality, and rejection of zero-count
  windows carrying nonzero published or active sums;
- exact 92-column trace order and scalar types, canonical Boolean/alias text,
  component windows, full 42-field dynamics snapshots, and final
  trace-to-checkpoint telemetry equality;
- authoritative `checkpoint_kind` on every verification checkpoint report,
  with no parallel `kind` alias; the disposable scratch fixture additionally
  requires exactly one update-zero `training` entry for analyzer/benchmark
  compatibility;
- exact training-to-finalization equality for all 15 parameter arrays, both
  optimizer moment registries, every optimizer scalar, utility/counters,
  sampler/progress, initial state, config/runtime/dataset, trainer state,
  warmup, segment state, and last training dynamics;
- launch-manifest path/SHA binding, including rejection of a byte-identical
  foreign copy and a copied same-run-ID config/checkpoint pair;
- scratch parent checkpoint cadence (`0`, each 10k boundary, and target)
  and resume-segment cadence from the recorded parent update;
- residual finalization payload binding to the exact training parent,
  optimizer/parameter state, and canonical results/manifest paths;
- exact fixed-panel metric acceptance and rejection of a `1e-4` numeric
  tamper or a missing field.

Invoking this test without the required Julia flags is itself expected to
fail at `enforce_verifier_runtime!`.

## 2. Controller static and watchdog probes

The parse-only check does not start the controller:

```powershell
& .\experiments\beat_first_v1\serial_workspace_snn\test_controller_contract.ps1 `
  -ParseOnly
```

The full static check asserts that the controller source contains the
canonical project/script checks, hermetic Julia argv, verifier thread argv,
create-suspended/job-assignment/resume ordering, watchdog states, and verifier
retry recovery contract:

```powershell
& .\experiments\beat_first_v1\serial_workspace_snn\test_controller_contract.ps1
```

The optional runtime probes compile a disposable console EXE under the OS temp
directory. The controller accepts it through `-JuliaExecutable`; the helper
ignores Julia argv, creates contract-test-only checkpoint metadata, starts a
grandchild, and hangs. No Julia process or training run is started.

```powershell
& .\experiments\beat_first_v1\serial_workspace_snn\test_controller_contract.ps1 `
  -RunWatchdogProbes
```

Four cases are executed with `ExpectedUpdates=2`, `JuliaThreads=2`,
`PollSeconds=1`, `StallSeconds=2`, and `VerifierTimeoutSeconds=2`:

| Case | Required recovery |
| --- | --- |
| update 1, no results, training hangs | new-RunId `resume` command |
| update 2, no results, training hangs | new-RunId `finalize-only` command |
| update 2 plus results, training hangs | controller-managed `-VerifyOnly` retry |
| training exits and verifier hangs | controller-managed `-VerifyOnly` retry |

Before those cases, a noncanonical `-ProjectPath` probe must fail before a run
directory, controller directory, or child process is created.

Every case also requires a non-null killed-process exit code and verifies that
both the helper PID and its grandchild PID are gone after controller teardown.
Verifier-retry commands must contain `-VerifyOnly` and the exact pinned launch
manifest SHA-256. The managed retry is itself rerun with the hanging helper;
it must hit the verifier timeout again, persist another managed retry, and
leave no orphan.
Probe artifacts are intentionally retained and their path is printed.

The helper checkpoints are not JLD2 files. These probes prove watchdog timing,
termination, no-orphan behavior, recovery selection, and recovery command
shape. They do not prove that the emitted resume/finalize command can load the
checkpoint.

## 3. Disposable U2 happy fixture

Freeze controller, trainer, and verifier source before creating the fixture.
The launch manifest binds their byte counts and SHA-256 values, so any later
edit invalidates the fixture by design.

Use a unique temp output root and a run ID beginning with
`contract_fixture_`:

```powershell
$julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$project = "C:\Users\fshuu\Documents\tetris\experiments\beat_first_v1"
$controller = Join-Path $project "serial_workspace_snn\run_arena_100k_controller.ps1"
$dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
$output = Join-Path "C:\tmp" (
  "swsnn_contract_fixture_" + (Get-Date -Format "yyyyMMdd_HHmmss")
)
$runId = "contract_fixture_u2_" + [guid]::NewGuid().ToString("N")

& $controller `
  -RunId $runId `
  -ExpectedUpdates 2 `
  -JuliaThreads 20 `
  -PollSeconds 1 `
  -StallSeconds 600 `
  -VerifierTimeoutSeconds 3600 `
  -JuliaExecutable $julia `
  -ProjectPath $project `
  -DatasetPath $dataset `
  -OutputRoot $output `
  -WorkingDirectory "C:\Users\fshuu\Documents\tetris" `
  -MutexName "Local\OpenAI.SWSNN.ContractFixture.$runId"
if ($LASTEXITCODE -ne 0) {
  throw "U2 contract fixture did not reach verified_complete"
}

$run = Join-Path $output $runId
$launch = Join-Path (
  Join-Path $output "_controllers\$runId"
) "launch_manifest.json"
```

Before mutation testing, inspect `controller_status.json` and require
`state == "verified_complete"`, then keep the source tree unchanged. Mark this
specific temp fixture as disposable and bind the marker to the launch SHA:

```powershell
$stream = [IO.File]::OpenRead($launch)
$sha = [Security.Cryptography.SHA256]::Create()
try {
  $launchSha = (
    [BitConverter]::ToString($sha.ComputeHash($stream))
  ).Replace("-", "").ToLowerInvariant()
}
finally {
  $sha.Dispose()
  $stream.Dispose()
}
$marker = [ordered]@{
  format = "serial-workspace-snn-disposable-contract-fixture"
  version = 1
  run_id = $runId
  expected_updates = 2
  disposable = $true
  launch_manifest_sha256 = $launchSha
}
[IO.File]::WriteAllText(
  (Join-Path $run ".swsnn_disposable_fixture.json"),
  (($marker | ConvertTo-Json -Depth 4) + [Environment]::NewLine),
  [Text.UTF8Encoding]::new($false)
)
```

Run the mutation suite only against this disposable fixture:

```powershell
& .\experiments\beat_first_v1\serial_workspace_snn\test_verifier_fixture_mutations.ps1 `
  -RunDirectory $run `
  -LaunchManifestPath $launch `
  -ExpectedUpdates 2 `
  -RunId $runId `
  -StartMode scratch `
  -JuliaExecutable $julia `
  -ProjectPath $project `
  -AllowSacrificialMutation
```

The script refuses updates above 2, a run ID without the
`contract_fixture_` prefix, or a run outside the OS temp directory / `C:\tmp`.
It also rejects reparse-point path components and requires the exact disposable
marker above.
It writes every verifier report to a separate temp case directory, never to
the run's successful `verification.json`.

It first requires an unchanged baseline to pass, then requires failures for:

- missing `--startup-file=no` or `--history-file=no`;
- wrong default or interactive verifier thread counts;
- a noncanonical active verifier Project (with the canonical Project retained
  only as a package-resolution fallback);
- reordered launch runtime argv;
- a byte-identical launch manifest copied to a foreign path;
- an otherwise ignored live launch-manifest byte change, which must fail the
  checkpoint/config launch-binding SHA;
- noncanonical launch project;
- tampered top-level launch training thread count;
- tampered launch startup contract;
- a blank checkpoint-manifest line;
- an extra `checkpoint_latest.jld2`;
- a `1e-4` results metric tamper;
- a finalization training-checkpoint reference tamper.

The run artifact is restored after every in-place mutation and its original
SHA-256 is checked. A final unchanged baseline must pass again.

The results-only metric mutation is expected to fail at the finalization
artifact hash boundary, possibly before fixed-panel recomputation. The unit
test separately exercises `verify_metric_snapshot` directly. To prove the
same failure through the complete verifier after all upstream bindings pass,
a dedicated JLD2 rewrite tool would have to update the finalization payload
metric and all dependent results/finalization artifact SHA references
atomically. That deeper destructive fixture transformation is intentionally
not performed here.

## Finalize-only parent/live binding

The same mutation script has two additional cases when invoked with
`-StartMode finalize-only` and a complete parent triple:

- the parent checkpoint-manifest live path is redirected to a shadow path;
- one byte is appended to the live parent checkpoint.

Both must fail, both files are restored byte-exactly, and the final baseline
must pass.

A valid finalize-only fixture requires a real U2 training checkpoint whose
parent run has no `results.json`. The safest construction is a dedicated
disposable U2 scratch fixture stopped after its training checkpoint commit and
before finalization, followed by:

```powershell
& $controller `
  -RunId ("contract_fixture_finalize_u2_" + [guid]::NewGuid().ToString("N")) `
  -ExpectedUpdates 2 `
  -JuliaThreads 20 `
  -PollSeconds 1 `
  -StallSeconds 600 `
  -VerifierTimeoutSeconds 3600 `
  -StartMode finalize-only `
  -ResumeCheckpoint $parentCheckpoint `
  -ResumeSha256 $parentSha256 `
  -ResumeUpdate 2 `
  -JuliaExecutable $julia `
  -ProjectPath $project `
  -DatasetPath $dataset `
  -OutputRoot $output `
  -WorkingDirectory "C:\Users\fshuu\Documents\tetris" `
  -MutexName "Local\OpenAI.SWSNN.ContractFinalizeFixture"
```

There is currently no stock trainer failpoint that deterministically stops
after the target training-checkpoint commit but before finalization. Do not
derive this fixture by modifying a retained production run. If such a fixture
is not already available, record the two finalize-only mutation cases as
pending rather than weakening the temp/sacrificial guard.

For a finalize-only fixture, invoke:

```powershell
& .\experiments\beat_first_v1\serial_workspace_snn\test_verifier_fixture_mutations.ps1 `
  -RunDirectory $finalizeRun `
  -LaunchManifestPath $finalizeLaunch `
  -ExpectedUpdates 2 `
  -RunId $finalizeRunId `
  -StartMode finalize-only `
  -ParentCheckpoint $parentCheckpoint `
  -ParentSha256 $parentSha256 `
  -ParentUpdate 2 `
  -JuliaExecutable $julia `
  -ProjectPath $project `
  -AllowSacrificialMutation
```

## Remaining integration gaps

- The fake watchdog checkpoint proves recovery selection, not JLD2
  resume/finalize execution.
- A full foreign-parent and replaced-parent-JLD2 ancestry matrix requires two
  independent real U2 parent fixtures.
- A material metric tamper that reaches fixed-panel recomputation through all
  upstream SHA bindings requires coordinated JLD2 and JSON rewrites.
- The final 100k run still needs its own exact verifier pass and analysis; U2
  only validates the contract machinery.
