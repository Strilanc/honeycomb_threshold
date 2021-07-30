import pytest

from experiment import RecordedExperimentData, total_error_to_per_round_error
import numpy as np

from probability_util import log_binomial


def test_total_error_to_per_round_error():
    np.testing.assert_allclose(
        total_error_to_per_round_error(5e-9, 50),
        1e-10,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        total_error_to_per_round_error(5e-4, 10**5),
        5e-9,
        rtol=1e-3,
    )


@pytest.mark.parametrize('p_round,n_iter', [
    (0.45, 0),
    (0.45, 1),
    (0.1, 0),
    (0.1, 1),
    (0.1, 2),
    (0.01, 0),
    (0.01, 1),
    (0.01, 2),
    (0.01, 3),
    (0.01, 4),
    (0.001, 5),
])
def test_total_error_to_per_round_error_vs_folding(p_round: float, n_iter: int):
    p_total = p_round
    n = 1
    for _ in range(n_iter):
        p_total = 2 * p_total * (1 - p_total)
        n *= 2
    p_round_recovered = total_error_to_per_round_error(p_total, n)
    np.testing.assert_allclose(
        p_round_recovered,
        p_round,
        rtol=1e-4,
    )


def test_likely_error_rate_bounds_shrink_towards_half():
    np.testing.assert_allclose(
        RecordedExperimentData(num_shots=10**5, num_correct=10**5 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.494122, 0.505878),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        RecordedExperimentData(num_shots=10**4, num_correct=10**4 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.481422, 0.518578),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        RecordedExperimentData(num_shots=10**4, num_correct=10**4 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-2),
        (0.48483, 0.51517),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        RecordedExperimentData(num_shots=1000, num_correct=500).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.44143, 0.55857),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        RecordedExperimentData(num_shots=100, num_correct=50).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.3204, 0.6796),
        rtol=1e-4,
    )


@pytest.mark.parametrize("n,c,ratio", [
    (100, 50, 1e-1),
    (100, 50, 1e-2),
    (100, 50, 1e-3),
    (1000, 500, 1e-3),
    (10**6, 100, 1e-3),
    (10**6, 100, 1e-2),
])
def test_likely_error_rate_bounds_vs_log_binomial(n: int, c: int, ratio: float):

    a, b = RecordedExperimentData(num_shots=n, num_correct=c).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=ratio)

    raw = log_binomial(p=(n - c) / n, n=n, hits=n - c)
    low = log_binomial(p=a, n=n, hits=n - c)
    high = log_binomial(p=b, n=n, hits=n - c)
    np.testing.assert_allclose(
        np.exp(low - raw),
        ratio,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        np.exp(high - raw),
        ratio,
        rtol=1e-2,
    )
