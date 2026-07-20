[CmdletBinding()]
param(
    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_margin1_teacher_signal_cpf_20000_v1",
    [string]$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# This entrypoint deliberately exposes no variant, update-count, objective, or
# resume option. The shared fail-closed controller independently enforces the
# same frozen geometry, split, seeds, gates, cadence, single-Julia lease, and
# 20-minute deadline as the parent k128 run while selecting only margin=1.0.
& (Join-Path $PSScriptRoot "run_three_layer_cpf_2500_signal.ps1") `
    -Dataset $Dataset `
    -OutputRoot $OutputRoot `
    -MaximumUpdates 20000 `
    -Variant k128 `
    -ObjectiveProfile margin100 `
    -Julia $Julia
