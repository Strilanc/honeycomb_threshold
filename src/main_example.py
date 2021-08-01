import numpy as np

from collect_data import collect_simulated_experiment_data
from honeycomb_layout import HoneycombLayout
from plotting import plot_data


def main():
    collect_simulated_experiment_data(
        *[
            HoneycombLayout(
                noise=p,
                tile_width=d,
                tile_height=d,
                sub_rounds=30,
                style="SD6",
                v_obs=True,
                h_obs=False,
            )
            for d in [2, 3, 4]
            for p in [0.001, 0.002, 0.003]
        ],
        out_path="test.csv",
        discard_previous_data=True,
        min_shots=10**3,
        max_shots=10**6,
        min_seen_logical_errors=10**2,
    )

    plot_data(
        "test.csv",
        title="LogLog per-sub-round error rates in periodic Honeycomb code under circuit noise")


if __name__ == '__main__':
    main()
