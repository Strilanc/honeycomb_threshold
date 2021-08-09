import tempfile

import numpy as np
import pytest

from collect_data import collect_simulated_experiment_data, ShotData, read_recorded_data
from honeycomb_layout import HoneycombLayout
from plotting import plot_data
from probability_util import log_binomial


def test_collect_and_plot():
    with tempfile.TemporaryDirectory() as d:
        f = d + "/tmp.csv"
        collect_simulated_experiment_data(
            [
                HoneycombLayout(
                    noise=p,
                    data_width=2 * d,
                    data_height=6 * d,
                    sub_rounds=30,
                    style="SD6",
                    obs="V",
                ).as_decoder_problem("pymatching")
                for p in [1e-5, 1e-4]
                for d in [1, 2]
            ],
            out_path=f,
            discard_previous_data=True,
            min_shots=10,
            max_shots=10,
            max_sample_std_dev=1,
            min_seen_logical_errors=1,
        )

        plot_data(read_recorded_data(f), show=False, out_path=d + "/tmp.png", title="Test")




def test_likely_error_rate_bounds_shrink_towards_half():
    np.testing.assert_allclose(
        ShotData(num_shots=10 ** 5, num_correct=10 ** 5 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.494122, 0.505878),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        ShotData(num_shots=10 ** 4, num_correct=10 ** 4 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.481422, 0.518578),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        ShotData(num_shots=10 ** 4, num_correct=10 ** 4 / 2).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-2),
        (0.48483, 0.51517),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        ShotData(num_shots=1000, num_correct=500).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
        (0.44143, 0.55857),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        ShotData(num_shots=100, num_correct=50).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3),
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

    a, b = ShotData(num_shots=n, num_correct=c).likely_error_rate_bounds(desired_ratio_vs_max_likelihood=ratio)

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
