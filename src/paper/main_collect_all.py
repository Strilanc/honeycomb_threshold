import argparse
import pathlib
import sys
from typing import List, Optional

import stim

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from noise import NoiseModel
from collect_data import collect_simulated_experiment_data, DecodingProblem, DecodingProblemDesc, \
    collect_detection_fraction_data
from honeycomb_layout import HoneycombLayout


def main():
    parser = argparse.ArgumentParser(description="Sample honeycomb error rates.")
    parser.add_argument('--surface_code_problems_directory', type=str, required=False, help="A directory of surface code problems to also do.")
    parser.add_argument('--out_file', type=str, required=False, help="Write to a file in addition to stdout.")
    parser.add_argument('--problem_id', type=int, required=False)
    parser.add_argument('--case_reduction', type=int, required=False)
    args = vars(parser.parse_args())
    out_path = args.get('out_file', None)
    problem_id = args.get('problem_id', None)
    surface_dir = args.get('surface_code_problems_directory')
    case_reduction = args.get('case_reduction') or 1
    collect_data(surface_dir=surface_dir,
                 problem_id=problem_id,
                 case_reduction=case_reduction,
                 out_path=out_path)


def collect_data(*,
                 surface_dir: Optional[str],
                 problem_id: Optional[int],
                 case_reduction: int,
                 out_path: Optional[str]):
    if surface_dir is None:
        surface_dir = f"{pathlib.Path(__file__).parent}/surface_code_circuits"
    problems = honeycomb_problems() + surface_code_problems(surface_dir)
    print(f"Problems: {len(problems)}", file=sys.stderr)
    if problem_id is not None:
        print(f"Running problem #: {problem_id}", file=sys.stderr)
        problems = [problems[problem_id]]

    # Hello jank my old fried, fancy seeing you here again.
    # Uncomment this, comment the next line, and add "EM3" to decoders to produce detection fraction data.
    # collect_detection_fraction_data(
    #     [p for p in problems if p.desc.decoder == DECODERS[0]],
    #     out_path=out_path,
    #     discard_previous_data=True,
    #     shots=1024,
    # )

    collect_simulated_experiment_data(
        problems,
        out_path=out_path,
        discard_previous_data=True,
        min_shots=25,
        max_batch=2**15,
        max_shots=10**8 // case_reduction,
        max_sample_std_dev=1,
        min_seen_logical_errors=10**3 // case_reduction,
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
        for decoder in DECODERS
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
DECODERS = [
    "internal",
    "internal_correlated",
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
        ]
        for obs in [
            "V",
            "H",
        ]
    ]
    return [
        lay.as_decoder_problem(decoder)
        for decoder in DECODERS
        for lay in layouts
    ]


if __name__ == '__main__':
    main()
