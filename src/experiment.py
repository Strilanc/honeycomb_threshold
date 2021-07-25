import csv
import math
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Sequence

import matplotlib.pyplot as plt

from decoding import sample_decode_count_correct
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout
from probability_util import log_binomial, binary_search


CSV_HEADER = "tile_diam,sub_rounds,physical_error_rate,circuit_style,num_shots,num_correct"


def collect_simulated_experiment_data(*cases: HoneycombLayout,
                                      min_shots: int,
                                      max_shots: int,
                                      min_seen_logical_errors: int,
                                      use_internal_decoder: bool = False,
                                      out_path: str,
                                      discard_previous_data: bool):
    print(CSV_HEADER)
    if discard_previous_data or not pathlib.Path(out_path).exists():
        with open(out_path, "w") as f:
            print(CSV_HEADER, file=f)

    for lay in cases:
        num_seen_errors = 0
        num_next_shots = min_shots
        total_shots = 0
        while True:
            circuit = generate_honeycomb_circuit(
                tile_diam=lay.tile_diam,
                sub_rounds=lay.sub_rounds,
                noise=lay.noise,
                style=lay.style,
            )
            num_correct = sample_decode_count_correct(
                num_shots=num_next_shots,
                circuit=circuit,
                use_internal_decoder=use_internal_decoder,
            )

            record = f"{lay.tile_diam},{lay.sub_rounds},{lay.noise},{lay.style},{num_next_shots},{num_correct}"
            with open(out_path, "a") as f:
                print(record, file=f)
            print(record)

            total_shots += num_next_shots
            num_seen_errors += num_next_shots - num_correct
            if num_seen_errors >= min_seen_logical_errors or total_shots >= max_shots:
                break
            num_next_shots = min(2 * num_next_shots, max_shots - total_shots)


def run_simulated_experiment(*,
                             physical_error_rates: List[float],
                             tile_diams: Sequence[int],
                             discard_previous_data: bool,
                             recorded_data_file_path: str,
                             other_data_paths: Sequence[str] = (),
                             shots: int,
                             sub_rounds: int,
                             use_internal_decoder: bool,
                             plot_title: str,
                             plot_after: bool):
    append = not discard_previous_data
    if not pathlib.Path(recorded_data_file_path).exists():
        append = False
    if shots > 0:
        with open(recorded_data_file_path, "a" if append else "w") as f:
            if not append:
                print(CSV_HEADER, file=f, flush=True)
            print("tile_diams", tile_diams)
            print("probabilities", physical_error_rates)
            print("num_shots", shots)
            for noise in physical_error_rates:
                s = f"physical error rate {noise}:"
                s = s.rjust(50)
                print(s , end="")
                for tile_diam in tile_diams:
                    circuit = generate_honeycomb_circuit(
                        tile_diam=tile_diam,
                        sub_rounds=sub_rounds,
                        noise=noise,
                    )
                    num_correct = sample_decode_count_correct(
                        num_shots=shots,
                        circuit=circuit,
                        use_internal_decoder=use_internal_decoder,
                    )
                    print(f" {shots - num_correct}", end="")
                    print(f"{tile_diam},{sub_rounds},{noise},{shots},{num_correct}", file=f, flush=True)
                print()

    if plot_after:
        plot_data(recorded_data_file_path, *other_data_paths, title=plot_title)


@dataclass
class RecordedExperimentData:
    num_shots: int = 0
    num_correct: int = 0

    def likely_error_rate_bounds(self, *, desired_ratio_vs_max_likelihood: float) -> Tuple[float, float]:
        """Compute relative-likelihood bounds.

        Returns the min/max error rates whose Bayes factors are within the given ratio of the maximum
        likelihood estimate.
        """
        actual_errors = self.num_shots - self.num_correct
        log_max_likelihood = log_binomial(p=actual_errors / self.num_shots, n=self.num_shots, hits=actual_errors)
        target_log_likelihood = log_max_likelihood + math.log(desired_ratio_vs_max_likelihood)
        acc = 100
        low = binary_search(
            func=lambda exp_err: log_binomial(p=exp_err / (acc * self.num_shots), n=self.num_shots, hits=actual_errors),
            target=target_log_likelihood,
            min_x=0,
            max_x=actual_errors * acc) / acc
        high = binary_search(
            func=lambda exp_err: -log_binomial(p=exp_err / (acc * self.num_shots), n=self.num_shots, hits=actual_errors),
            target=-target_log_likelihood,
            min_x=actual_errors * acc,
            max_x=self.num_shots * acc) / acc
        return low / self.num_shots, high / self.num_shots

    @property
    def logical_error_rate(self) -> float:
        if self.num_shots == 0:
            return 1
        return (self.num_shots - self.num_correct) / self.num_shots


def total_error_to_per_round_error(error_rate: float, rounds: int) -> float:
    """Convert from total error rate to per-round error rate."""
    randomize_rate = min(1.0, 2*error_rate)
    round_randomize_rate = 1 - (1 - randomize_rate)**(1 / rounds)
    round_error_rate = round_randomize_rate / 2
    return round_error_rate


def plot_data(*paths: str, title: str):
    lay_to_noise_to_results: Dict[Tuple[int, int], Dict[float, RecordedExperimentData]] = {}
    for path in paths:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                tile_diam = int(row["tile_diam"])
                physical_error_rate = float(row["physical_error_rate"])
                sub_rounds = int(row["sub_rounds"])
                d1 = lay_to_noise_to_results.setdefault((tile_diam, sub_rounds), {})
                d2 = d1.setdefault(physical_error_rate, RecordedExperimentData())
                d2.num_shots += int(row["num_shots"])
                d2.num_correct += int(row["num_correct"])

    cor = lambda p: total_error_to_per_round_error(p, rounds=sub_rounds)
    markers = "_ov*sp^<>8PhH+xXDd|"
    for tile_diam, sub_rounds in sorted(lay_to_noise_to_results.keys()):
        group = lay_to_noise_to_results[(tile_diam, sub_rounds)]
        xs = []
        ys = []
        x_bounds = []
        ys_low = []
        ys_high = []
        for physical_error_rate in sorted(group.keys()):
            datum = group[physical_error_rate]
            # Show curve going through max likelihood estimate, unless it's at zero error.
            if datum.num_shots > datum.num_correct:
                xs.append(physical_error_rate)
                ys.append(cor(datum.logical_error_rate))
            # Show relative-likelihood-above-1e-3 error bars as a colored region.
            low, high = datum.likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3)
            x_bounds.append(physical_error_rate)
            ys_low.append(cor(low))
            ys_high.append(cor(high))
        plt.plot(xs, ys, label=f"{tile_diam=},{sub_rounds=}", marker=markers[tile_diam], zorder=100 - tile_diam)
        plt.fill_between(x_bounds, ys_low, ys_high, alpha=0.3)

    plt.legend()
    plt.loglog()

    def format_tick(p: float) -> str:
        assert p > 0
        k = 0
        while p < 1:
            p *= 10
            k += 1
        assert p - int(p) < 1e-4
        return f"{int(p)}e-{k}"
    ticks_y = [k*10**-p for k in [1, 2, 5] for p in range(1, 10) if k*10**-p <= 0.5]
    ticks_x = [k*10**-p for k in [1, 2, 5] for p in range(1, 10) if k*10**-p <= 0.5]
    plt.xticks([x for x in ticks_x], labels=[format_tick(x) for x in ticks_x], rotation=45)
    plt.yticks([y for y in ticks_y], labels=[format_tick(y) for y in ticks_y])
    plt.xlim(0.0001, 0.5)
    plt.ylim(0.0000001, 0.5)
    plt.title(title)
    plt.ylabel("Logical Error Rate (Vertical Observable)")
    plt.xlabel("Physical Error Rate Parameter")
    plt.grid()
    plt.show()
