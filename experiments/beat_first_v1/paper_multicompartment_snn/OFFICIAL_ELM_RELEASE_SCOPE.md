# Official ELM model-level release control

`PaperELMTwinOfficialV2Release.jl` is a model-level immutable-attestation
control. It verifies that:

- the frozen official ELM model, parameters, source contract, and paper-scale
  metadata are unchanged;
- fixed model-quality gates are satisfied by the supplied metric record;
- the metric record, evaluator hash, and result hash are bound into the
  attestation;
- verified inference rejects a tampered artifact before entering the
  differentiable numerical kernel.

It is **not** a raw-evidence verifier and it does not promote a model from
caller-supplied arrays. The metric values passed to
`attest_official_elm_release` remain caller-supplied control-plane data.

Canonical production promotion must separately verify a sealed dataset root,
the evaluator executable/source hash, deterministic evaluation outputs, split
membership, trial count, and paper-duration evidence before invoking this
model-level control. Smoke, public held-out, or unsealed arrays cannot establish
production provenance.
