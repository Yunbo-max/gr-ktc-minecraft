#!/usr/bin/env python3
"""Test whether native-KV subspace misalignment predicts LoRA interference."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr_ktc.grassmann import (
    bootstrap_spearman_interval,
    consensus_spectrum,
    exact_permutation_pvalue,
    grassmann_distance,
    latent_subspace,
    principal_angles,
    spearman_correlation,
)
from gr_ktc.reachability import fit_individual_and_shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memories", type=Path,
        default=ROOT / "results/fast_kv_four_context_memories.safetensors",
    )
    parser.add_argument(
        "--reachability", type=Path,
        default=ROOT / "results/real_reachability_four_contexts.safetensors",
    )
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--subspace-rank", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/kv_subspace_misalignment_four_contexts.json",
    )
    args = parser.parse_args()

    raw = load_file(args.memories)
    reachability = load_file(args.reachability)
    scenes = sorted(
        key.removeprefix("features_")
        for key in reachability if key.startswith("features_")
    )
    if len(scenes) < 3:
        raise ValueError("at least three contexts are required")
    width = raw[f"scene_{scenes[0]}_positive_layer_{args.layer}"].shape[1] // 2
    bases = {"key": {}, "value": {}, "joint": {}}
    corrections = {}
    contexts = {}
    for scene in scenes:
        positive = raw[f"scene_{scene}_positive_layer_{args.layer}"].float()
        failed = raw[f"scene_{scene}_failed_layer_{args.layer}"].float()
        correction = positive - failed
        corrections[scene] = correction
        matrices = {
            "key": correction[:, :width],
            "value": correction[:, width:],
            "joint": correction,
        }
        for channel, matrix in matrices.items():
            bases[channel][scene] = latent_subspace(matrix, args.subspace_rank)
        contexts[scene] = (
            reachability[f"features_{scene}"],
            reachability[f"kv_effect_{scene}"],
        )

    pairs = []
    for first, second in itertools.combinations(scenes, 2):
        pair_contexts = [contexts[first], contexts[second]]
        fit = fit_individual_and_shared(pair_contexts, rank=args.lora_rank)
        individual_mean = float(fit["rho_individual_mean"])
        shared = float(fit["rho_shared"])
        record = {
            "first": first,
            "second": second,
            "rho_individual_mean": individual_mean,
            "rho_shared": shared,
            "interference": individual_mean - shared,
        }
        channel_distances = []
        for channel in ("key", "value", "joint"):
            angles = principal_angles(
                bases[channel][first], bases[channel][second],
            )
            distance = grassmann_distance(
                bases[channel][first], bases[channel][second],
            )
            record[f"{channel}_angles_rad"] = angles.tolist()
            record[f"{channel}_grassmann"] = distance
            if channel in ("key", "value"):
                channel_distances.append(distance)
        record["kv_mean_grassmann"] = sum(channel_distances) / 2
        pairs.append(record)

    interference = torch.tensor([x["interference"] for x in pairs])
    correlations = {}
    for metric in (
        "key_grassmann", "value_grassmann", "joint_grassmann",
        "kv_mean_grassmann",
    ):
        distances = torch.tensor([x[metric] for x in pairs])
        rho = spearman_correlation(distances, interference)
        p_value, permutations = exact_permutation_pvalue(distances, interference)
        low, high = bootstrap_spearman_interval(
            distances, interference, samples=args.bootstrap_samples,
        )
        correlations[metric] = {
            "spearman": rho,
            "exact_two_sided_permutation_p": p_value,
            "permutations": permutations,
            "bootstrap_90ci_pair_resampling": [low, high],
        }

    spectra = {}
    for channel in ("key", "value", "joint"):
        spectrum = consensus_spectrum([bases[channel][scene] for scene in scenes])
        spectra[channel] = {
            "nonzero_eigenvalues": spectrum.tolist(),
            "top_r_mean": float(spectrum[:args.subspace_rank].mean()),
            "trace": float(spectrum.sum()),
        }

    primary = correlations["kv_mean_grassmann"]
    gate = (
        primary["spearman"] > 0.5
        and primary["exact_two_sided_permutation_p"] <= 0.10
        and primary["bootstrap_90ci_pair_resampling"][0] > 0
    )
    report = {
        "protocol": "cross-context-consolidation-misalignment-pilot-v1",
        "scenes": scenes,
        "layer": args.layer,
        "subspace_rank": args.subspace_rank,
        "lora_rank": args.lora_rank,
        "native_correction": "positive_KV_barycenter - failed_KV_barycenter",
        "basis_axis": "right singular vectors in latent feature space",
        "pairs": pairs,
        "correlations": correlations,
        "consensus_spectrum": spectra,
        "preregistered_primary": "mean of separate K and V Grassmann distances",
        "gate_rule": "Spearman > 0.5, exact permutation p <= 0.10, pair-bootstrap 90% lower bound > 0",
        "gate_pass": gate,
        "next_action": (
            "expand to 8 contexts before implementing transport/SMC"
            if gate else
            "stop Grassmann method branch; four-context pilot does not support misalignment hypothesis"
        ),
        "scope_warning": "Only four contexts/six dependent pairs and four-token merged supports. Pair bootstrap is descriptive; exact permutation is the primary finite-sample test. Raw per-rollout successful KV trajectories are required for a definitive study.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
