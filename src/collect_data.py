import csv
import dataclasses
import math
import pathlib
import time
from typing import Dict, Tuple, Optional

from decoding import sample_decode_count_correct
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout
from probability_util import log_binomial, binary_search

CSV_HEADER = ",".join([
    "tile_width",
    "tile_height",
    "sub_rounds",
    "physical_error_rate",
    "circuit_style",
    "preserved_observable",
    "num_shots",
    "num_correct",
    "total_processing_seconds",
])


def collect_simulated_experiment_data(*cases: HoneycombLayout,
                                      min_shots: int,
                                      max_shots: int,
                                      max_batch: Optional[int] = None,
                                      max_sample_std_dev: float = 1,
                                      min_seen_logical_errors: int,
                                      use_internal_decoder: bool = False,
                                      out_path: Optional[str],
                                      discard_previous_data: bool):
    """
    Args:
        cases: The layouts to sample from.
        min_shots: The minimum number of samples to take from each case.

            This property effectively controls the quality of estimates of error rates when the true
            error is close to 50%

        max_shots: The maximum cutoff number of samples to take from each case.

            This property effectively controls the "noise floor" below which error rates cannot
            be estimated well. For example, setting this to 1e6 means that error rates below
            5e-5 will have estimates with large similar-relative-likelihood regions.

            This property overrides all the properties that ask for more samples until some
            statistical requirement is met.
        max_sample_std_dev: Defaults to irrelevant. When this is set, more samples will be taken
            until sqrt(p * (1-p) / n) is at most this value, where n is the number of shots sampled
            and p is the number of logical errors seen divided by n. Note that p is not round
            adjusted; you may need to set max_sample_std_dev lower to account for the total-to-round
            conversion.

            This property is intended for controlling the quality of estimates of error rates when
            the true error rate is close to 50%.
        min_seen_logical_errors: More samples will be taken until the number of logical errors seen
            is at least this large. Set to 10 or 100 for fast estimates. Set to 1000 or 10000 for
            good statistical estimates of low probability errors.
        use_internal_decoder: Defaults to False. Switches the decoder from pymatching to an internal
            decoder, which must be present as a binary in the source directory.
        out_path: Where to write the CSV sample statistic data. Setting this to none doesn't write
            to file; only writes to stdout.
        max_batch: Defaults to unused. If set, then at most this many shots are collected at one
            time.
        discard_previous_data: If set, `out_path` is overwritten. If not set, `out_path` will be
            appended to (or created if needed).
    """
    print(CSV_HEADER, flush=True)
    if out_path is not None:
        if discard_previous_data or not pathlib.Path(out_path).exists():
            with open(out_path, "w") as f:
                print(CSV_HEADER, file=f)

    if max_batch is None:
        max_batch = max_shots

    for lay in cases:
        circuit = generate_honeycomb_circuit(lay)
        num_seen_errors = 0
        num_next_shots = min_shots
        total_shots = 0
        while True:
            t0 = time.monotonic()
            num_correct = sample_decode_count_correct(
                num_shots=num_next_shots,
                circuit=circuit,
                use_internal_decoder=use_internal_decoder,
            )
            t1 = time.monotonic()
            record = ",".join(str(e) for e in [
                lay.tile_width,
                lay.tile_height,
                lay.sub_rounds,
                lay.noise,
                lay.style,
                lay.obs,
                num_next_shots,
                num_correct,
                t1 - t0,
            ])
            if out_path is not None:
                with open(out_path, "a") as f:
                    print(record, file=f)
            print(record, flush=True)

            total_shots += num_next_shots
            num_seen_errors += num_next_shots - num_correct
            p = num_seen_errors / total_shots
            cur_sample_std_dev = math.sqrt(p * (1 - p) / total_shots)
            if total_shots >= max_shots:
                break
            if num_seen_errors >= min_seen_logical_errors and cur_sample_std_dev <= max_sample_std_dev:
                break
            if num_seen_errors >= total_shots * 0.48 and num_seen_errors >= 10:
                break
            num_next_shots = min(max_batch, min(2 * num_next_shots, max_shots - total_shots))


@dataclasses.dataclass
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


GROUPED_RECORDED_DATA = Dict[HoneycombLayout, Dict[HoneycombLayout, Dict[float, RecordedExperimentData]]]

def read_recorded_data(*paths: str) -> GROUPED_RECORDED_DATA:
    result: GROUPED_RECORDED_DATA = {}
    for path in paths:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                lay1 = HoneycombLayout(
                    noise=0,
                    tile_width=1,
                    tile_height=1,
                    sub_rounds=1,
                    style=row["circuit_style"],
                    obs=row["preserved_observable"],
                )
                g1 = result.setdefault(lay1, {})

                lay2 = HoneycombLayout(
                    noise=0,
                    tile_width=int(row["tile_width"]),
                    tile_height=int(row["tile_height"]),
                    sub_rounds=int(row["sub_rounds"]),
                    style=row["circuit_style"],
                    obs=row["preserved_observable"],
                )
                g2 = g1.setdefault(lay2, {})

                physical_error_rate = float(row["physical_error_rate"])
                g3 = g2.setdefault(physical_error_rate, RecordedExperimentData())

                g3.num_shots += int(row["num_shots"])
                g3.num_correct += int(row["num_correct"])
    return result
