# Q1 frozen-old additive residual critic

This is the only experiment authorized by
`reports/clean_post_p1_parse_failure_strategy.md`. It is a new one-shot
scientific hypothesis; it does not repair, resume, or rerun the consumed
partial-tail experiment.

The immutable 20.8M old critic remains outside automatic differentiation and
is never modified. Its stored candidate scores are the base value. A
165,051-parameter `CompactCandidateQ(8,1,2)` initialized from the selected C13
checkpoint supplies an additive correction. Only the compact scalar Dense is
zeroed, so update 0 must produce bitwise `+0`, combined scores must equal the
stored old scores exactly, and top-1 agreement must be 1.0.

## Frozen scientific contract

- training role: aggregate rows 1:2160, episodes 1:12;
- original-old-policy rows: 1:1500; compact-DAgger rows: 1501:2160;
- reused development guard: rows 2161:2660, episodes 13:14;
- exact `t:t+3` same-episode/consecutive eligibility and no terminal at
  `t:t+2` (expected 2,124 training and 494 guard rows);
- all 2,000 batches of four are frozen before the start gate by
  `Xoshiro(0x5131_2026)` through deterministic shuffled epochs;
- 74 stored candidates, with every loss/max masked by `action_count`;
- target `r[t] + gamma*r[t+1] + gamma^2*r[t+2] + gamma^3*max(oldQ[t+3])`,
  `gamma=.997`, rewards stored and independently checked as score delta / 600;
- DAgger behavior actions are never used for bootstrap. The two rewards after
  the selected action follow compact behavior, so DAgger-role diagnostics
  explicitly disclose the resulting off-policy bias;
- selected Huber plus unit-weight all-valid-action zero-residual Huber anchor;
- one explicit Float64 L2 norm over every correction-gradient array leaf,
  followed by one shared `min(1,1/norm)` scale and then
  AdamW(3e-4,(.9,.999),1e-4), for exactly 2,000 Zygote updates;
- Julia 1.12.6, Lux 1.31.4, Zygote 0.7.11;
- first complete update <=60s, updates 6:25 individually <=1s and median
  <=.25s, projected and hard wall <=12 minutes, process tree <=4 GiB.

The rows 2161:2660 are not fresh held-out generalization data: the C13
initializer-selection process has already seen their aggregate role. Artifacts
therefore record `initializer_exposed_to_offline_rows=true` and
`offline_role="reused_development_guard"`. The gate is only a safety/regression
screen and gives no game-strength evidence.

## Guard promotion criteria

The fixed update-2000 correction is promoted only if all of the following hold:

1. eligible selected-action Huber improves at least 15% from zero correction;
2. correction/target-residual correlation >=.20 and sign agreement >=.60 on
   rows with absolute target residual >=.1;
3. combined/old top-1 agreement is >=.95 and <.995, with at least three changed
   states;
4. all-candidate correction RMS <=.25 and all outputs finite;
5. a fresh OpenVINO CPU correction graph matches the fresh Lux reference to
   `1e-4`.

A pass is named only `Q1-offline-promoted`. It does not authorize a game,
validation seeds 8001:8008, sealed test seeds 91001:91032, checkpoint selection,
or any claim that the old model has been beaten.

## Future composite inference contract

If a later independent review authorizes game evaluation, the old graph must
retain its historical candidate grouping: full chunks of 16 plus an actual-size
tail because LayerNorm includes candidate-batch dimensions. The correction may
use fixed-74 padding, but valid candidate indices must be aligned before adding
it to old scores. A monolithic 74-padded old/composite graph is forbidden unless
full-count equivalence is proved separately. Old and correction physical calls,
latency, and parameters must be reported independently.

`invoke_once.ps1 -ValidateOnly` performs source parsing, fresh Julia entrypoint
startup and synthetic harness checks without reading real checkpoints, model
weights, datasets, OpenVINO, game state, or the global Q1 marker. A real run is
not part of this implementation commit.
