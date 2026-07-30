# TwinProp parity sealed release status

## Canonical entry

Use only:

```text
TwinPropParityOfficialSealedCanonical.jl
run_twinprop_parity_sealed_canonical.jl
```

The executable verifies the immutable runner source before loading it.  The
outer training/export API rejects legacy `FrozenTwin`, unsealed
`FrozenOfficialELMTwin`, and caller-attested `VerifiedOfficialELMTwin`.

## Required artifact

Parity optimization cannot start until the official ELM is available as:

```text
PaperELMTwinOfficialV2SealedRelease.SealedOfficialELMRelease
```

The sealed loader must recompute held-out voltage, soma-spike AUROC, and
regional NMDA-current metrics from the verified teacher manifest and shards.
Development-scale and caller-supplied metric records are rejected.

At the time this file was added, no production sealed artifact existed.
Therefore XOR and 4-bit parity accuracies are still unmeasured.

## Projection and export gates

The canonical path enforces:

- validation-only restart selection;
- absolute hard-twin accuracy/BCE thresholds;
- soft-to-hard accuracy/BCE gap thresholds;
- exact contacts per axon;
- Dale identity and non-negative conductance;
- independent E/I one-micrometre capacity;
- no soma or axon contacts;
- fresh hard projection and frozen-twin replay before export;
- run, mapping, dataset, catalog, morphology, teacher, sealed-attestation,
  and frozen-model hash bindings;
- event-count multiplicity in the NPZ;
- NPZ read-back and bit-exact reconstruction of the 1,278-channel hard input;
- final classification from detailed NEURON soma spikes only.

Soft and hard digital-twin scores are diagnostics.  Only the detailed NEURON
transfer result is an authoritative parity measurement.

## Public protocol ambiguity

The preprint simultaneously describes 4,000 excitatory plus 4,000 inhibitory
axons, 20 contacts per axon, 8,000 total contacts, and no more than one E plus
one I contact per dendritic micrometre.  The public Hay morphology has 12,263
one-micrometre slots per receptor kind, so the literal 80,000 contacts per kind
are infeasible.

The executable uses the disclosed constraint-consistent interpretation:

```text
8,000 total contacts
= 4,000 E + 4,000 I contacts
= 200 E + 200 I axons at 20 contacts per axon
```

This is a project reconstruction.  Author-code identity is not claimed.

## Current focused tests

```text
test_twinprop_parity_official_sealed_canonical_v2.jl
test_twinprop_parity_official_sealed_final.jl
test_run_twinprop_parity_sealed_canonical_entry.jl
test_twinprop_parity_official_attested_final_dev_fixed_v2.jl
test_twinprop_parity_official_v2_final.jl
test_twinprop_parity_official_canonical_final.jl
test_paper_elm_v2_sealed_release_forgery.jl
```

Do not use the earlier non-final hard-gate or attested-only files as release
entrypoints.  They are audit controls retained because Windows additive-file
ACLs prevent deletion.
