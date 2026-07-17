# Frozen S1 development screen

Date: 2026-07-18 JST
Status: **failed and closed**

The clean post-C13 review authorized one frozen screen of the historical
checkpoint with ungated top-2 one-step Bellman reranking. The evaluator fixed
`top_k=2`, `blend=0.5`, `gamma=0.997`, and `q_margin_threshold=Inf` before any
new game. Commit `a4291206ced29348ccd63e145487e619ba0fe65a` and the freeze
manifest SHA-256
`df5054dd9b2a8577bad7a751149cd8e393ae26d2099e4655a7b6ea119670f9d9`
bound the source, environment, checkpoint, commands, seeds, budgets, and stop
rule.

## First and only executed pair

| Seed 5755, 100 pieces | Old argmax-Q | S1 lookahead | Difference |
|---|---:|---:|---:|
| Score | 5,900 | 5,500 | **-400** |
| Completed | yes | yes | no regression |
| Candidate evaluations | 4,457 | 13,345 (4,420 root + 8,925 successor) | 2.99x |
| Logical model passes | 100 | 300 | 3.00x |
| Physical backend requests | 344 | 1,013 | 2.94x |
| Inference seconds | 25.684 | 75.363 | 2.93x |
| Wall seconds | 41.066 | 95.204 | 2.32x |

The preregistered success rule required a strictly positive difference on all
three development seeds and immediate termination after any difference
`<= 0`. The first pair was `-400`, so seeds 5756 and 5757 were not run. This
exact S1 configuration is rejected. It is a system/search failure, not a
model-only result, and no validation or sealed test seed was touched.

## Provenance

- Freeze: `D:\tetris-paper-plus\runs\s1_screen_freeze_a429120.json`
- Result: `D:\tetris-paper-plus\runs\s1_screen_results_a429120.json`
- Baseline artifact SHA-256:
  `4a9def34b68b468ac26cab18afbfaddd306170dc4e576c7c30b6cba4e1c955f1`
- S1 artifact SHA-256:
  `f96c4472f417140beedc6fe8f47dd0d6d8a3e22f38f71a7f62ed6ec5ac2efedd`
- Historical checkpoint SHA-256:
  `7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1`

No rerun, hyperparameter adjustment, substitute seed, or post-result search was
performed.
