import pytest

import numpy as np

from plotting import total_error_to_per_piece_error


def test_total_error_to_per_piece_error():
    np.testing.assert_allclose(
        total_error_to_per_piece_error(5e-9, 50),
        1e-10,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        total_error_to_per_piece_error(5e-4, 10**5),
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
def test_total_error_to_per_piece_error_vs_folding(p_round: float, n_iter: int):
    p_total = p_round
    n = 1
    for _ in range(n_iter):
        p_total = 2 * p_total * (1 - p_total)
        n *= 2
    p_round_recovered = total_error_to_per_piece_error(p_total, n)
    np.testing.assert_allclose(
        p_round_recovered,
        p_round,
        rtol=1e-4,
    )
