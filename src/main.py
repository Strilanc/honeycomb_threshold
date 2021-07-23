import numpy as np

from experiment import collect_simulated_experiment_data, plot_data
from honeycomb_layout import HoneycombLayout


def main():
    collect_simulated_experiment_data(
        *[
            HoneycombLayout(
                noise=p,
                tile_diam=d,
                sub_rounds=30,
            )
            for d in [2, 3, 4]
            for p in np.geomspace(start=5e-4, stop=3e-3, num=5)
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
