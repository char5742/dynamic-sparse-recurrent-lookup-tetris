# TwinProp XOR/parity — exact sealed V2 status

## Canonical boundary

The only canonical parity implementation is:

- `TwinPropParityOfficialSealedV2Canonical.jl`
- `run_twinprop_parity_sealed_v2_canonical.jl`

It accepts only the nominal
`PaperELMTwinOfficialV2SealedReleaseV2.SealedOfficialELMRelease` type with
schema:

```text
hd_swsnn.paper_elm_v2.sealed_release.final.v2
```

The previous `PaperELMTwinOfficialV2SealedRelease` type is a negative control.
Its validation, training, hard-gate, and export methods are explicitly
overridden to fail.

## Bound mechanism

The parity gate requires and records all of the following:

- 1278 signed event inputs (639 excitatory, 639 inhibitory)
- 45 branches, 100 routed synapses per branch
- 1000 memory units and 2000 hidden units
- four regional NMDA-current outputs
- executable `:silu` activation
- `:twinprop_paper_reconstruction` compatibility profile
- exact frozen parameter, artifact, routing, normalizer, and payload hashes
- exact training-protocol hash
- teacher manifest, contract, source dataset, shard inventory, split, and
  morphology/catalog hashes
- independently pinned source hashes for the sealed evaluator, activation
  profile/hotfix, profiled loaders, Final model, differentiable path, core, and
  teacher-contract verifier

The raw held-out teacher evidence is recomputed before optimization and again
before detailed-NEURON export.

## What counts as a reproduction

Soft-twin and hard-projected-twin scores are projection diagnostics only.
Only a fresh detailed-NEURON run using the exported exact contacts and soma
spike decision window is compared with the paper.

The canonical runner supports only:

- dimension 2, full active detailed cell (XOR)
- dimension 4, full active detailed cell (parity)

It does not export `passive`, `no_nmda`, or `soma_only` by relabeling a full
run. Each ablation requires its own sealed twin, independently retrained
synapse/location optimization, and detailed-NEURON result lineage.

## Current measured status

As of 2026-07-29:

- exact V2 parity boundary tests: 15/15
- profiled SiLU numerical-path and input-gradient tests: 7/7
- immutable full-only runner tests: 16/16
- V2 caller-evidence forgery rejection: 14/14
- V2 primary sealed-release test: 30 checks passed, then one upstream
  development-fixture error because regional fit-split NMDA standard deviation
  was zero
- production/promotable V2 sealed artifact: absent
- actual detailed-NEURON XOR accuracy: unmeasured
- actual detailed-NEURON 4-bit parity accuracy: unmeasured

Therefore no paper-result reproduction is claimed yet, and the canonical
runner must not be started until the upstream V2 test is green and a
production/promotable artifact exists.

## Production invocation

```powershell
$env:TWINPROP_SEALED_ARTIFACT = '<exact V2 production artifact>'
$env:TWINPROP_TEACHER_MANIFEST = '<verified production manifest>'
$env:TWINPROP_TEACHER_SHARDS = '<verified production shard directory>'
julia --project=. experiments/beat_first_v1/paper_multicompartment_snn/run_twinprop_parity_sealed_v2_canonical.jl
```

The default dimensions are `2,4`; the runner fails closed when any required
evidence is absent.
