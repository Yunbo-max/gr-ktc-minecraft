import math

import torch

from gr_ktc.grassmann import (
    bootstrap_spearman_interval,
    consensus_spectrum,
    exact_permutation_pvalue,
    grassmann_distance,
    latent_subspace,
    principal_angles,
    spearman_correlation,
)


def test_principal_angles_identical_and_orthogonal():
    first = torch.eye(4)[:, :2]
    assert torch.allclose(principal_angles(first, first), torch.zeros(2))
    second = torch.eye(4)[:, 2:]
    angles = principal_angles(first, second)
    assert torch.allclose(angles, torch.full((2,), math.pi / 2))
    assert abs(grassmann_distance(first, second) - math.pi / math.sqrt(2)) < 1e-6


def test_latent_subspace_uses_feature_axis_and_consensus_bounds():
    matrix = torch.tensor([[2.0, 0, 0], [0, 1.0, 0]])
    basis = latent_subspace(matrix, rank=1)
    assert basis.shape == (3, 1)
    assert abs(float(basis[0, 0])) == 1
    spectrum = consensus_spectrum([basis, basis])
    assert torch.all((0 <= spectrum) & (spectrum <= 1))
    assert torch.allclose(spectrum[:1], torch.ones(1))


def test_spearman_permutation_and_bootstrap():
    x = torch.arange(6, dtype=torch.float64)
    y = x.square()
    assert spearman_correlation(x, y) == 1.0
    p, permutations = exact_permutation_pvalue(x, y)
    assert permutations == math.factorial(6)
    assert 0 < p <= 0.01
    low, high = bootstrap_spearman_interval(x, y, samples=200, seed=7)
    assert low > 0.9 and high <= 1
