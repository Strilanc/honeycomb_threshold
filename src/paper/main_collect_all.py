import argparse
import pathlib
import sys
from typing import List

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import collect_simulated_experiment_data, DecodingProblem
from honeycomb_layout import HoneycombLayout


def main():
    parser = argparse.ArgumentParser(description="Sample honeycomb error rates.")
    parser.add_argument('--surface_code_problems_directory', type=str, required=True, help="A directory of surface code problems to also do.")
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

    problems = honeycomb_problems()
    if job_id is not None:
        problems = problems[job_id::jobs_count]

    collect_simulated_experiment_data(
        problems,
        out_path=out_path,
        discard_previous_data=True,
        min_shots=25,
        max_batch=10**6,
        max_shots=10**8 * 0 + 10**5,
        max_sample_std_dev=1,
        min_seen_logical_errors=10**3 * 0 + 10,
    )


def honeycomb_problems() -> List[DecodingProblem]:
    layouts: List[HoneycombLayout] = [
        HoneycombLayout(
            noise=p,
            data_width=u * 4,
            data_height=u * 6,
            sub_rounds=u * 3 * 4 * 3,  # 3 sub rounds, 4 distance per u, 3 cells.
            style=style,
            obs=obs,
        )
        for p in [
            0.0001,
            0.0002,
            0.0003,
            # 0.0005,
            # 0.0007,
            0.0010,
            # 0.0015,
            0.0020,
            0.0030,
            # 0.0050,
            # 0.0070,
            # 0.0100,
            # 0.0150,
            # 0.0200,
            # 0.0300,
        ]
        for u in [
            1,
            2,
            3,
            # 4,
            # 5,
        ]
        for style in [
            "SI500",
            "SD6",
            "EM3_v2",
            "PC3",
        ]
        for obs in [
            "V",
            "H",
        ]
    ]
    return [
        lay.as_decoder_problem(decoder)
        for lay in layouts
        for decoder in [
            "internal",
            # "internal_correlated",
        ]
    ]


if __name__ == '__main__':
    main()
