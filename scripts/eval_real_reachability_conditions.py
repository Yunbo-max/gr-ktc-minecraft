#!/usr/bin/env python3
"""Minecraft behavior pilot for state/weight reachability conditions.

Conditions are paired on the same scene and sampling seed:
base, full matched quality-KV state, individual residual LoRA, shared residual
LoRA, and shared residual LoRA plus full state.  The last condition is a
conservative upper-bound control; exact unreachable-only state injection is
reported by the teacher-forced decomposition analysis, not claimed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.mineexplorer import load_mineexplorer
from evaluation.parser_validity import VoyagerActionParser
from evaluation.statistics import exact_mcnemar
from gr_ktc.generation import generate_with_kv_prefix
from gr_ktc.kv_prefix import KVPrefixMemory
from gr_ktc.model_loader import load_qwen3_vl_24gb
from gr_ktc.reachability import fit_individual_and_shared, fit_reachability
from gr_ktc.residual_adapter import ResidualLoRAHook, ScheduledResidualStateHook, text_layer
from gr_ktc.voyager_http import VoyagerHTTPClient, final_observation
from scripts.collect_mineexplorer_pilot import verifier_score_events
from scripts.run_local_qwen_action import compact_observation, primitive_programs
from scripts.run_local_smoke_suite import SYSTEM


CONDITIONS = (
    "base", "state", "weight_individual", "weight_shared",
    "state_plus_weight", "decomposed",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302, 303])
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/real_reachability_conditions.json",
    )
    args = parser.parse_args()
    if "base" not in args.conditions:
        raise ValueError("base condition is required for paired comparisons")

    source = json.loads((ROOT / "results/fast_kv_cross_context.json").read_text())
    scene_ids = source["scene_ids"]
    scenarios = {
        item.scene_id: item for item in load_mineexplorer(
            ROOT / "data/MineExplorer-Benchmark/benchmark.jsonl"
        ) if item.scene_id in scene_ids
    }
    data = load_file(ROOT / "results/real_reachability_data.safetensors")
    contexts = [(data[f"features_{scene}"], data[f"kv_effect_{scene}"]) for scene in scene_ids]
    individual = {
        scene: fit_reachability(x, y, rank=args.rank)
        for scene, (x, y) in zip(scene_ids, contexts, strict=True)
    }
    shared = fit_individual_and_shared(contexts, rank=args.rank)["shared"]
    shared_unreachable = {
        scene: target - features @ shared.delta_weight.T
        for scene, (features, target) in zip(scene_ids, contexts, strict=True)
    }

    model, processor = load_qwen3_vl_24gb(
        ROOT / "models/Qwen3-VL-8B-Instruct", precision="bf16"
    )
    model.eval()
    layer = text_layer(model, 24)
    config = model.config.text_config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    flat = load_file(ROOT / "results/fast_kv_cross_context_memories.safetensors")
    memories = {}
    for scene in scene_ids:
        advantages = torch.tensor(source["acquisitions"][scene]["advantages"])
        pos_count = int((advantages > 0).sum())
        neg_count = int((advantages < 0).sum())
        positive = {index: flat[f"scene_{scene}_positive_layer_{index}"] for index in range(config.num_hidden_layers)}
        failed = {index: flat[f"scene_{scene}_failed_layer_{index}"] for index in range(config.num_hidden_layers)}
        center = {
            index: (pos_count * positive[index] + neg_count * failed[index]) / (pos_count + neg_count)
            for index in range(config.num_hidden_layers)
        }
        center[24] = flat[f"scene_{scene}_contrastive_layer_24"]
        memories[scene] = KVPrefixMemory.from_flattened(
            center, kv_heads=config.num_key_value_heads, head_dim=head_dim,
            context_id=f"{scene}:matched", value_scale=0.25,
        )

    client = VoyagerHTTPClient(timeout_seconds=120)
    action_parser = VoyagerActionParser(ROOT / "third_party/voyager/voyager/env/mineflayer")
    programs = primitive_programs()
    system = SYSTEM.replace(
        "Prefer one direct helper call with the requested\ncount. mineBlock already searches and mines. Keep the answer under 100 tokens.",
        "Complete every subgoal in order using as many helper calls as needed. "
        "mineBlock already searches and mines. Keep the answer under 300 tokens.",
    )
    eos_ids = tuple(value for value in (
        processor.tokenizer.eos_token_id,
        getattr(model.generation_config, "eos_token_id", None),
    ) if isinstance(value, int))
    trials = []
    if args.resume and args.output.exists():
        trials = json.loads(args.output.read_text()).get("trials", [])
    completed = {(x["scene_id"], x["seed"], x["condition"]) for x in trials}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for scene in scene_ids:
        scenario = scenarios[scene]
        for seed in args.seeds:
            for condition in args.conditions:
                if (scene, seed, condition) in completed:
                    continue
                events = client.reset(
                    hard=True, kill_on_hard_reset=False, setup_commands=scenario.commands
                )
                observation = final_observation(events)
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "text", "text": (
                        f"Observation: {compact_observation(observation)}\nTask: {scenario.task_text}"
                    )}]},
                ]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                memory = memories[scene] if condition in ("state", "state_plus_weight") else None
                fit = individual[scene] if condition == "weight_individual" else shared
                with ExitStack() as hooks:
                    if condition in ("weight_individual", "weight_shared", "state_plus_weight", "decomposed"):
                        hooks.enter_context(ResidualLoRAHook(layer, fit.lora_a, fit.lora_b))
                    if condition == "decomposed":
                        hooks.enter_context(ScheduledResidualStateHook(layer, shared_unreachable[scene]))
                    ids = generate_with_kv_prefix(
                        model, inputs, memory,
                        context_id=f"{scene}:matched" if memory else None,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.9, top_p=0.95, eos_token_ids=eos_ids,
                        generator=torch.Generator(device=model.device).manual_seed(seed),
                    )
                text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
                try:
                    parsed = action_parser.parse(text)
                    parser_valid = True
                    try:
                        result_events = client.step(
                            code=parsed.exec_code,
                            programs=programs + "\n\n" + parsed.program_code,
                        )
                        score = verifier_score_events(scenario.milestones, result_events)
                        error = None
                    except Exception as exc:
                        score = 0.0
                        error = f"{type(exc).__name__}: {exc}"
                except ValueError as exc:
                    parser_valid, score, error = False, 0.0, str(exc)
                trials.append({
                    "scene_id": scene, "seed": seed, "condition": condition,
                    "score": score, "parser_valid": parser_valid,
                    "generated_tokens": int(ids.shape[-1]), "error": error, "text": text,
                })
                args.output.write_text(json.dumps({
                    "protocol": "real-reachability-conditions-v1-partial",
                    "rank": args.rank, "trials": trials,
                }, indent=2) + "\n")

    summary = {}
    for condition in args.conditions:
        subset = [x for x in trials if x["condition"] == condition]
        summary[condition] = {
            "full_successes": sum(x["score"] == 1 for x in subset),
            "mean_score": sum(x["score"] for x in subset) / max(len(subset), 1),
            "parser_valid": sum(x["parser_valid"] for x in subset),
            "trials": len(subset),
        }
    comparisons = {}
    indexed = {(x["scene_id"], x["seed"], x["condition"]): x for x in trials}
    for condition in args.conditions:
        if condition == "base":
            continue
        base = [indexed[(s, seed, "base")]["score"] == 1 for s in scene_ids for seed in args.seeds]
        other = [indexed[(s, seed, condition)]["score"] == 1 for s in scene_ids for seed in args.seeds]
        left, right, p = exact_mcnemar(other, base)
        comparisons[f"{condition}_vs_base"] = {"condition_only": left, "base_only": right, "p": p}
    report = {
        "protocol": "real-reachability-conditions-v1",
        "rank": args.rank, "scene_ids": scene_ids, "seeds": args.seeds,
        "conditions": args.conditions,
        "summary": summary, "comparisons": comparisons, "trials": trials,
        "rho_individual_mean": sum(x.rho for x in individual.values()) / len(individual),
        "rho_shared": shared.rho,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope_warning": "state_plus_weight uses full quality KV as an upper bound; decomposed uses a phase-scheduled hidden residual for the exact offline unreachable component, not a reconstructed KV cache.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
