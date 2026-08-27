# GR-KTC experiment status — 2026-08-26

## Verified results

- Gate 0 local protocol: passed. Minecraft 1.19 dedicated server, Java 17,
  Mineflayer bridge, RTX 3090, Qwen3-VL-8B, MineExplorer 813+100 rows, parser,
  and local no-API execution are operational.
- Real local Qwen gather rollout: parser-valid and environment-successful for
  `Collect 4 oak logs`; final inventory was `oak_log: 4`.
- Deterministic prerequisite craft smoke: 10/10 parser-valid and 10/10
  environment success over T1–T5 and generation seeds 42/43. BF16 peak allocated
  GPU memory was 16.424 GiB. These are integration scenes, not PEAM held-out
  outcomes.
- Real K=4 acquisition group: four T1 rollouts, four successes, 17 immutable
  checksummed files. This group supplies BC data but no group-relative signal
  because its outcome variance is zero.
- QLoRA smoke: rank 32, alpha 64, seven PEAM-matched projection families,
  87,293,952 trainable parameters. One real-rollout BC update had loss 1.834,
  took 2.97 seconds, and peaked at 9.960 GiB.
- MineExplorer Pilot acquisition: 30 finalized composite groups / 120
  trajectories, including 12 mixed-outcome groups.
- Gate 1 passed on the preregistered Pilot size. Layer-24 direction cosine was
  33/42 = 78.57%, group-bootstrap 95% CI [0.571, 0.955], random-label
  one-sided p=0.0029. Layer 35 did not pass and is not selected downstream.
- Real all-36-layer KV prefix generation is operational at 16.56 GiB peak.
  One-context rescue pilot improved mean milestone score from 0.625 (no memory)
  to 0.875 (positive memory), but failed memory also reached 0.750; Gate 2
  remains open pending matched real-trajectory controls across contexts.
- Unit/integration tests: 51 passed.

## Synthetic mechanism gate

Across seeds 42/43/44 and 90 groups per seed, leave-group latent ranking AUROC
was 0.9985 / 0.9981 / 0.9958. Closed-form rank-4 LoRA reconstruction relative
error was approximately `1e-7`.

Merge alignment was:

| Seed | A | B | C | D-SB |
|---:|---:|---:|---:|---:|
| 42 | 0.810 | 0.903 | 0.987 | 0.065 |
| 43 | 0.940 | 0.230 | 0.957 | 0.343 |
| 44 | 0.922 | 0.305 | 0.964 | 0.608 |

This is a mechanism sanity check, not Minecraft success. Experimental
Schrödinger Bridge Method D is currently worse and unstable; it is not promoted
to the main method. A temporal-cost/epsilon sweep improved individual seeds but
did not yield one stable setting.

Method D now has a soft-control variant for K=4: effective-sample-size and
advantage-margin shrinkage toward the identity, plus an optional control-norm
cap. It remains an ablation until it beats A/B under the same real-data gates.

## Reproduction differences

- PEAM uses Azure GPT-4o as the slow-tier acquisition/fallback model. The current
  protocol uses local Qwen3-VL-8B for rollout sampling and therefore labels PEAM
  paper numbers as reference-only.
- Voyager pins Mineflayer 4.8.1. Its crafting transaction hung on the local 1.19
  dedicated server; Mineflayer 4.14.0 fixed the reference craft action and is
  pinned here.
- Headless hard reset disables the original `/kill` step because it causes an
  invalid movement packet race in Mineflayer. Inventory/equipment and setup
  commands remain deterministic.

## Not yet claimed

- No full 11-task × 3-seed PEAM result exists yet.
- A/B/C fast-memory rescue and 100-step GRTA+DPO memory-free improvements have
  not yet been measured in Minecraft.
- The public MineExplorer execution repository is unavailable; its dataset and
  verifier schema are implemented, but official 1,800-step parity is unverified.
