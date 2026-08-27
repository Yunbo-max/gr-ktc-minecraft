# Real Qwen State–Weight Reachability Pilot

Date: 2026-08-27

## Protocol

- Qwen3-VL-8B-Instruct, BF16 inference on RTX 3090 24GB.
- Two real MineExplorer Gate-2 contexts (`0281`, `0299`).
- Teacher target: layer-24 output hidden shift causally induced by the matched
  quality-KV memory used in the earlier fast-rescue experiment.
- Reachability proxy: rank-constrained residual-stream update fitted from the
  no-memory layer input to the KV-induced output shift.
- Behavior conditions: Base, State (quality KV), Weight-individual,
  Weight-shared, and full-State + shared-Weight upper-bound control.
- Four sampling seeds per rank, with Minecraft reset before every condition.

This is a real-model/real-environment pilot. It is not an official PEAM result,
and the residual-stream adapter is a faithful test of the fitted linear proxy,
not a claim about every PEFT projection-specific LoRA.

## Offline real-effect reachability

| Rank | Mean individual rho | Joint rho |
|---:|---:|---:|
| 1 | 0.254 | 0.023 |
| 2 | 0.435 | 0.133 |
| 4 | 0.594 | 0.210 |
| 8 | 0.749 | 0.475 |
| 16 | 0.849 | 0.574 |
| 32 | 0.941 | 0.797 |
| 64 | 0.991 | 0.928 |

The individual–joint gap is present at every tested rank and is largest in the
low-rank regime. The reachable plus unreachable components reconstruct the
original KV effect with maximum relative error below `8.1e-9`.

## Minecraft behavior

| Condition | Rank 8 | Rank 32 |
|---|---:|---:|
| Base | 1/8 | 4/8 |
| State / quality KV | **4/8** | **6/8** |
| Weight individual | 0/8 | 1/8 |
| Weight shared | 0/8 | 1/8 |
| Full state + shared weight | 0/8 | 4/8 |

Within each paired run, State remains much stronger than Weight-only. Higher
rank substantially improves hidden-effect rho, but Weight-only behavior remains
far below State. Therefore linear hidden-state reachability is not sufficient
for behavioral consolidatability; generation-path stability and where the
update is realized also matter.

Base outcomes differ between the separately executed rank-8 and rank-32 runs,
despite identical nominal seeds. Minecraft reset/world dynamics and sampling
are therefore not reproducible enough to treat the cross-rank success counts as
a paired rank comparison. Only within-run condition comparisons are valid.

The full-State + Weight condition is an upper-bound interaction control and can
double-count reachable effects. It is not the proposed exact
`unreachable-state + reachable-weight` decomposition. Its degradation at rank 8
shows that naive state/weight addition can interfere.

An additional rank-8 pilot tested the exact offline split using a shared
residual LoRA for the reachable component and a token-phase hidden-state hook
for the unreachable component. Across two new seeds (four trials per
condition): Base `1/4`, State `3/4`, Weight-shared `1/4`, decomposed `0/4`.
Thus an algebraically exact teacher-forced decomposition is not sufficient when
replayed along a freely diverging generation path. Converting the unreachable
component back into a query-responsive KV memory, rather than a fixed phase
schedule, remains a required method step.

A query-responsive hidden-memory follow-up used the current decode state to
retrieve the nearest unreachable correction instead of replaying a fixed
phase. On two further seeds: Base `0/4`, State `4/4`, Weight-shared `0/4`,
query-retrieved decomposition `0/4`. Query retrieval alone therefore does not
recover the native KV intervention. The causal interface (attention K/V versus
post-layer hidden addition) is itself part of the mechanism.

## Other three ideas on real effects

- KV-conditioned basis, rank 8: one shared basis `rho=0.475`; two
  latent-conditioned bases `rho=0.727`.
- Future-privileged target distance grows with horizon: `0.219` (H=2), `0.355`
  (H=4), `0.441` (H=8), confirming that future information creates a
  non-trivial student target.
- A two-item latent population retrieves its matching causal trajectory with
  posterior weight effectively 1.0. This only validates routing on the two
  training contexts, not held-out retrieval.

## Evidence files

- `real_reachability_data.json`: extraction metadata.
- `real_reachability_analysis.json`: rank sweep and individual decomposition.
- `real_reachability_conditions_rank8_pilot.json`: 40 rank-8 trials.
- `real_reachability_conditions_rank32.json`: 40 rank-32 trials.
- `real_reachability_decomposed_rank8.json`: 16 exact-split behavior trials.
- `real_reachability_retrieved_rank8.json`: 16 query-retrieved split trials.

## MineExplorer hop generalization

A memory-free external pilot selected one 1/2/3/4-hop MineExplorer scenario and
two seeds (24 trials total):

| Method | Full success | Mean milestone score | Parser valid |
|---|---:|---:|---:|
| Base | 0/8 | 0.271 | 8/8 |
| Shared BC+DPO | 2/8 | 0.396 | 8/8 |
| GR-KTC QLoRA | 2/8 | 0.396 | 8/8 |

Both adapters improve the 2-hop crafting task from partial to full success.
They do not improve the selected 1-hop visual-location task or the 3/4-hop
tasks, where every method completes only part of the milestone chain. This is
a small stratified pilot, not evidence that reachability monotonically declines
with hop count.

Raw file: `memory_free_hop_generalization_pilot.json`.
- `real_four_ideas_analysis.json`: unified four-idea real-effect analysis.
