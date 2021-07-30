import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

import numpy as np

from experiment import collect_simulated_experiment_data
from honeycomb_layout import HoneycombLayout


def main():
    collect_simulated_experiment_data(
        *[
            HoneycombLayout(
                noise=p,
                tile_diam=d,
                sub_rounds=d * 6,
                style="3step_demolition",
            )
            for p in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
            for d in [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ],
        out_path="data/3step_demolition.csv",
        discard_previous_data=True,
        min_shots=10**3,
        max_shots=10**6,
        max_sample_std_dev=1e-2,
        min_seen_logical_errors=10**2,
        use_internal_decoder=True,
    )


if __name__ == '__main__':
    main()
