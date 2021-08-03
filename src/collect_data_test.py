import tempfile

from collect_data import collect_simulated_experiment_data
from honeycomb_layout import HoneycombLayout
from plotting import plot_data


def test_collect_and_plot():
    with tempfile.TemporaryDirectory() as d:
        f = d + "/tmp.csv"
        collect_simulated_experiment_data(
            *[
                HoneycombLayout(
                    noise=p,
                    tile_width=d,
                    tile_height=d,
                    sub_rounds=30,
                    style="SD6",
                    obs="V",
                )
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

        plot_data(f, show=False, out_path=d + "/tmp.png", title="Test")
