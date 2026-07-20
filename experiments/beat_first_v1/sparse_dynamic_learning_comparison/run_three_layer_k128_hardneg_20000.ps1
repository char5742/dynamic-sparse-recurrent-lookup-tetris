[CmdletBinding()]
param(
    [string]$Dataset = "D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    [string]$OutputRoot = "D:\tetris-paper-plus\runs\beat_first_v1\sparse_3l_k128_hardneg_teacher_signal_cpf_20000_v1",
    [string]$Julia = "C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# No variant/update/weight/mode/resume switches are exposed. The shared
# controller independently enforces the frozen k128 20k geometry, seeds,
# cadence, thresholds, single-Julia lease, parent SHA, and fresh output root.
& (Join-Path $PSScriptRoot "run_three_layer_cpf_2500_signal.ps1") `
    -Dataset $Dataset `
    -OutputRoot $OutputRoot `
    -MaximumUpdates 20000 `
    -Variant k128 `
    -ObjectiveProfile hardneg015 `
    -Julia $Julia
