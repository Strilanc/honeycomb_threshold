import argparse
import pathlib
import sys
from typing import List, Optional

import stim

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from noise import NoiseModel
from collect_data import collect_simulated_experiment_data, DecodingProblem, DecodingProblemDesc
from honeycomb_layout import HoneycombLayout


def main():
    parser = argparse.ArgumentParser(description="Sample honeycomb error rates.")
    parser.add_argument('--surface_code_problems_directory', type=str, required=False, help="A directory of surface code problems to also do.")
    parser.add_argument('--out_file', type=str, required=False, help="Write to a file in addition to stdout.")
    parser.add_argument('--job_spread', type=int, required=False, help="Splits up jobs.")
    parser.add_argument('--job_id', type=int, required=False, help="The job id this process should handle running.")
    parser.add_argument('--jobs_count', type=int, required=False, help="The number of jobs the work is being split into, across machines.")
    args = vars(parser.parse_args())
    out_path = args.get('out_file', None)
    job_id = args.get('job_id', None)
    jobs_count = args.get('jobs_count', None)
    surface_dir = args.get('surface_code_problems_directory', None)
    jobs_spread = args.get('job_spread', 1)
    if (job_id is None) != (jobs_count is None):
        raise ValueError("Must specify both or neither of --job_id, --jobs_count")
    if job_id is not None:
        if not (0 < jobs_count and 0 <= job_id < jobs_count):
            raise ValueError("Need 0 < jobs_count and 0 <= job_id < jobs_count")

    problems = honeycomb_problems() + surface_code_problems(surface_dir)
    print(f"Total problems (before spread): {len(problems)}", file=sys.stderr)
    problems *= jobs_spread
    print(f"Total problems (spread): {len(problems)}", file=sys.stderr)
    if job_id is not None:
        problems = problems[job_id::jobs_count]
    print(f"Problems being run: {len(problems)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("", file=sys.stderr)

    collect_simulated_experiment_data(
        problems,
        out_path=out_path,
        discard_previous_data=False,
        min_shots=25,
        max_batch=10**6,
        max_shots=10**8 // jobs_spread,
        max_sample_std_dev=1,
        min_seen_logical_errors=10**3 // jobs_spread,
    )


def surface_code_circuit(directory: str,
                         noise_name: str,
                         noise: float,
                         obs: str,
                         d: int) -> stim.Circuit:
    path = f"{directory}/{noise_name.lower()}_{obs}/d{d}_p0.0.stim"
    with open(path) as f:
        circuit = stim.Circuit(f.read())
    if noise_name == "SI500":
        noise_model = NoiseModel.SI500(noise)
    elif noise_name == "SD6":
        noise_model = NoiseModel.SD6(noise)
    else:
        raise NotImplementedError(noise_name)
    return noise_model.noisy_circuit(circuit)


def surface_code_problem(directory: str,
                         noise_name: str,
                         noise: float,
                         obs: str,
                         d: int,
                         decoder: str) -> DecodingProblem:
    def circuit_maker() -> stim.Circuit:
        return surface_code_circuit(directory, noise_name, noise, obs, d)

    return DecodingProblem(
        circuit_maker=circuit_maker,
        desc=DecodingProblemDesc(
            data_width=d,
            data_height=d,
            code_distance=d,
            num_qubits=2 * d * d - 1,
            rounds=3 * d,
            noise=noise,
            circuit_style=f"surface_{noise_name}",
            preserved_observable=obs,
            decoder=decoder
        ),
    )


def surface_code_problems(directory: Optional[str]) -> List[DecodingProblem]:
    if directory in ["-", "", None]:
        return []

    return [
        surface_code_problem(
            directory=directory,
            d=d,
            noise=p,
            noise_name=noise_name,
            obs=obs,
            decoder=decoder,
        )
        for decoder in ["internal", "internal_correlated"]
        for d in [3, 7, 11, 15, 19]
        for p in USED_NOISE_VALUES
        for noise_name in ["SD6", "SI500"]
        for obs in "XZ"
    ]

USED_NOISE_VALUES = [
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0007,
    0.0010,
    0.0015,
    0.0020,
    0.0030,
    0.0050,
    0.0070,
    0.0100,
    0.0150,
    0.0200,
    0.0300,
]

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
        for p in USED_NOISE_VALUES
        for u in [
            1,
            2,
            3,
            4,
            5,
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
        for decoder in [
            "internal",
            "internal_correlated",
        ]
        for lay in layouts
    ]


if __name__ == '__main__':
    main()
