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

    SPREAD = 5
    cases = [
        HoneycombLayout(
            noise=p,
            tile_width=u * 2,
            tile_height=u,
            sub_rounds=u * 18,  # 3 sub rounds, 2 distance, 3 cells.
            style=style,
            obs=obs,
        )
        for p in [
            0.0001,
            0.0002,
            0.0005,
            0.001,
            0.0015,
            0.002,
            0.0025,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.010,
            0.015,
        ]
        for u in [
            1,
            2,
            3,
            4,
            5,
        ]
        for style in [
            "SD6",
            "EM3",
            "PC3",
            "SI500",
        ]
        for obs in [
            "H",
            "V",
        ]
    ] * SPREAD
    if job_id is not None:
        cases = cases[job_id::jobs_count]

    collect_simulated_experiment_data(
        *cases,
        out_path=out_path,
        discard_previous_data=True,
        min_shots=25,
        max_batch=10**6 // SPREAD,
        max_shots=10**8 // SPREAD,
        max_sample_std_dev=1,
        min_seen_logical_errors=10**3 // SPREAD,
        use_internal_decoder=True,
    )


if __name__ == '__main__':
    main()
