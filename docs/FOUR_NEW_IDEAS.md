# Four new ideas: API-free mechanism runner

This extension follows the State–Weight Reachability proposal rather than
continuing to tune merge A/B/C.

1. **State–Weight Reachability** fits a rank-`r` first-order update to a useful
   state correction and reports `rho`. The fitted component is the slow,
   parameter-reachable part; the residual is the fast KV part. It reports both
   rank scaling and context heterogeneity (`N=1,2,4,8,16,32`).
2. **KV-Conditioned LoRA Basis** learns continuous latent-coordinate weights
   over low-rank basis updates. Coordinates, not task IDs or categories, drive
   the mixture.
3. **Privileged-Future Latent Distillation** provides KL and hidden-state
   losses plus discounted backward targets from successful future states.
4. **Latent Population** keeps a bounded archive with score-based selection,
   mutation, and distance-weighted posterior retrieval.

The runner uses all 813 records in `data/MineExplorer-Benchmark/benchmark.jsonl`
and deliberately does not require Minecraft login, Azure, or an external API.
It currently generates deterministic activation-shaped tensors to test the
mathematics and data plumbing. These results are not causal model evidence;
real KV/hidden tensors from the VLM must replace `_make_contexts` before making
behavioral claims.

## Run

```bash
PYTHONPATH=. .venv/bin/python scripts/run_new_ideas.py \
  --scenarios data/MineExplorer-Benchmark/benchmark.jsonl \
  --output results/four_new_ideas.json

PYTHONPATH=. .venv/bin/python scripts/run_four_ideas_sweep.py \
  --output results/four_ideas_sweep.json
```

The 24GB model settings are in `configs/four_ideas_24gb.yaml`. Analysis is
offloaded to CPU and uses one BLAS thread to make hundreds of small SVDs fast;
this does not change Qwen inference or QLoRA settings.

## Current mechanism results

The full sweep completed over 813/813 scenarios (seed 42):

- Individual rank-8 reachability: `rho=1.000`; shared rank-8 reachability:
  `rho=0.809`.
- Shared rank curve: `0.189, 0.322, 0.542, 0.809, 0.9996, 0.9996,
  0.9996` for ranks `1,2,4,8,16,32,64`.
- Context heterogeneity shared `rho`: `1.000, 0.904, 0.823, 0.806, 0.808,
  0.809` for `N=1,2,4,8,16,32`; individual `rho` stays approximately `1`.
- Future-state loss falls from `0.0449` to `0.0112` at horizon 4 in the
  deterministic student/teacher check.
- Conditional basis coefficient entropy rises from `0` (one basis) to `0.673`
  (eight bases), confirming continuous rather than hard task routing.
- Population archives retain exactly the requested sizes 8/16/32/64 and return
  an 8-token posterior mixture.

See the raw machine-readable output at
`results/four_new_ideas.json` and `results/four_ideas_sweep.json`.

## Real-Qwen follow-up

The same APIs have now been run on two real layer-24 quality-KV causal effects.
The rank-8 individual/shared reachability values are `0.749/0.475`; at rank 32
they are `0.941/0.797`. Eighty paired Minecraft behavior trials show State at
`4/8` and `6/8` versus Weight-only at `0/8` and `1/8` for ranks 8 and 32,
respectively. Read `results/REAL_REACHABILITY_REPORT.md` for the necessary
cross-run and method-scope caveats.

The follow-up also tests scheduled and query-retrieved unreachable hidden
memory. Neither reproduces the native quality-KV rescue, which narrows the next
method step to preserving the attention K/V causal interface. A 24-trial
memory-free 1–4-hop MineExplorer pilot is stored in
`results/memory_free_hop_generalization_pilot.json`.
