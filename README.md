# GR-KTC Minecraft

Implementation workspace for group-relative KV trajectory consolidation on the
Voyager/Mineflayer substrate.

## Current status

Implemented and tested:

- verifier-based group-relative advantages;
- incremental K/V recording compatible with legacy and modern Transformers caches;
- acquisition-only PCA whitening;
- signed group-relative covariance and positive/negative eigenspaces;
- merge A residual barycenter;
- merge B differentiable soft-DTW phase barycenter;
- merge C entropic Sinkhorn token-cloud phase barycenter;
- closed-form ridge/SVD LoRA initialization;
- bounded verifier-closed retry loop;
- RTX 3090 24 GB NF4 Qwen3-VL loader and configuration.

The runtime workspace can use the official Voyager repository, Qwen checkpoint,
MineExplorer dataset, and paper PDFs. These large/downloaded resources are not
committed; see `RESOURCE_MANIFEST.md` for their pinned sources.

## Test

```bash
python -m pytest -q
```

## Fast-loop semantics

Each matched context runs four rollouts. Only a mixed verifier-outcome group has
nonzero relative advantages and may update fast memory. The controller stops on
verified success, retry-budget exhaustion, or lack of improvement. Fast memory is
task-local and must be reset after task completion.

## 24 GB constraints

Use `configs/grktc_24gb.yaml`. The original checkpoint is loaded in NF4 with bf16
compute. Pilot runs record two text layers, cap generation at 512 tokens, and
offload incremental K/V to CPU. Slow consolidation uses QLoRA with microbatch 1
and gradient accumulation 16.

## API-free local stack

Minecraft 1.19, Java 17, the Voyager bridge, and Qwen3-VL-8B are installed. The
default experiment stack uses a local-only dedicated server in offline mode, so
neither Microsoft login nor a GPT/Azure API is required:

```bash
scripts/start_local_stack.sh
python scripts/run_local_qwen_action.py --task "Collect 4 oak logs" --execute
```

Offline mode is bound to `127.0.0.1` and must never be exposed to an untrusted
network. Qwen is both the fast policy and local rollout sampler. This is a valid
API-free GR-KTC protocol, but it is not an exact reproduction of PEAM's GPT-4o
slow-tier acquisition. PEAM's reported line therefore remains reference-only;
any direct statistical comparison must use the explicitly labelled local
acquisition control or a separately reproduced slow tier.
