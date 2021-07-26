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
                sub_rounds=30,
                style="3step_inline",
            )
            for d in [1, 2, 3, 4, 5, 6, 7, 8, 9]
            for p in np.geomspace(start=1e-4, stop=5e-2, num=10)
        ],
        out_path="data/3step_inline.csv",
        discard_previous_data=True,
        min_shots=10**4,
        max_shots=10**7,
        min_seen_logical_errors=10**3,
        use_internal_decoder=True,
    )


if __name__ == '__main__':
    main()
