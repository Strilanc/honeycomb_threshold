import argparse
import math
import pathlib
import random
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import collect_simulated_experiment_data
from honeycomb_layout import HoneycombLayout


def main():
    parser = argparse.ArgumentParser(description="Sample honeycomb error rates.")
    parser.add_argument('--out_file', type=str, required=False, help="Write to a file in addition to stdout.")
    parser.add_argument('--job_id', type=int, required=False, help="The job id this process should handle running.")
    parser.add_argument('--jobs_count', type=int, required=False, help="The number of jobs the work is being split into, across machines.")
    args = vars(parser.parse_args())
    out_path = args.get('out_file', None)
    job_id = args.get('job_id', None)
    jobs_count = args.get('jobs_count', None)
    if (job_id is None) != (jobs_count is None):
        raise ValueError("Must specify both or neither of --job_id, --jobs_count")
    if job_id is not None:
        if not (0 < jobs_count and 0 <= job_id < jobs_count):
            raise ValueError("Need 0 < jobs_count and 0 <= job_id < jobs_count")

    cases = [
        HoneycombLayout(
            noise=p,
            tile_width=d,
            tile_height=d,
            sub_rounds=d * 6,
            style=style,
            obs=obs,
        )
        for p in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
        for d in [1, 2, 3, 4, 5]
        for style in ["SD6", "EM3", "PC3", "SI500"]
        for obs in ["H", "V"]
    ]
    if job_id is not None:
        cases = cases[job_id::jobs_count]

    collect_simulated_experiment_data(
        *cases,
        out_path=out_path,
        discard_previous_data=True,
        min_shots=20,
        max_batch=10**5,
        max_shots=10**5,
        max_sample_std_dev=1e-2,
        min_seen_logical_errors=10**1,
        use_internal_decoder=True,
    )


if __name__ == '__main__':
    main()
