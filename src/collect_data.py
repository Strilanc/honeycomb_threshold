import csv
import dataclasses
import math
import pathlib
import time
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Any

import stim

from decoding import sample_decode_count_correct
from probability_util import log_binomial, binary_search

CSV_HEADER = ",".join([
    "data_width",
    "data_height",
    "rounds",
    "noise",
    "circuit_style",
    "preserved_observable",
    "code_distance",
    "num_qubits",
    "num_shots",
    "num_correct",
    "total_processing_seconds",
    "decoder",
    "version",
])
CSV_HEADER_VERSION = 2


@dataclasses.dataclass(frozen=True, unsafe_hash=True, order=True)
class DecodingProblemDesc:
    # noinspection PyUnresolvedReferences
    """Succinct data summarizing a decoding problem.

    Attributes:
        data_width: The width of the grid of data qubits.
        data_height: The height of the grid of data qubits.
        code_distance: int
        num_qubits: int
        rounds: Identifying information about the problem. The width of the grid of data qubits.
        noise: int
        circuit_style: str
        preserved_observable: str
        decoder: The name of the decoder that was used.
    """
    data_width: int
    data_height: int
    code_distance: int
    num_qubits: int
    rounds: int
    noise: float
    circuit_style: str
    preserved_observable: str
    decoder: str

    def with_changes(self,
                     *,
                     data_width: Optional[int] = None,
                     data_height: Optional[int] = None,
                     code_distance: Optional[int] = None,
                     num_qubits: Optional[int] = None,
                     rounds: Optional[int] = None,
                     noise: Optional[float] = None,
                     circuit_style: Optional[str] = None,
                     preserved_observable: Optional[str] = None,
                     decoder: Optional[str] = None,
    ) -> 'DecodingProblemDesc':
        return DecodingProblemDesc(
            data_width=self.data_width if data_width is None else data_width,
            data_height=self.data_height if data_height is None else data_height,
            code_distance=self.code_distance if code_distance is None else code_distance,
            num_qubits=self.num_qubits if num_qubits is None else num_qubits,
            rounds=self.rounds if rounds is None else rounds,
            noise=self.noise if noise is None else noise,
            circuit_style=self.circuit_style if circuit_style is None else circuit_style,
            preserved_observable=self.preserved_observable if preserved_observable is None else preserved_observable,
            decoder=self.decoder if decoder is None else decoder,
        )


@dataclasses.dataclass
class DecodingProblem:
    # noinspection PyUnresolvedReferences
    """Defines a decoding problem to sample from.

    Attributes:
        desc: Identifying information about the problem.
        circuit_maker: Produces a stim circuit with annotated noise and detectors.
    """
    desc: DecodingProblemDesc
    circuit_maker: Callable[[], stim.Circuit]


def collect_simulated_experiment_data(problems: List[DecodingProblem],
                                      *,
                                      min_shots: int,
                                      max_shots: int,
                                      max_batch: Optional[int] = None,
                                      max_sample_std_dev: float = 1,
                                      min_seen_logical_errors: int,
                                      out_path: Optional[str],
                                      discard_previous_data: bool):
    """
    Args:
        problems: The decoding problems to collect sample data from.
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

    for problem in problems:
        num_seen_errors = 0
        num_next_shots = min_shots
        total_shots = 0
        while True:
            t0 = time.monotonic()
            num_correct = sample_decode_count_correct(
                num_shots=num_next_shots,
                circuit=problem.circuit_maker(),
                decoder=problem.desc.decoder,
            )
            t1 = time.monotonic()
            record = ",".join(str(e) for e in [
                problem.desc.data_width,
                problem.desc.data_height,
                problem.desc.rounds,
                problem.desc.noise,
                problem.desc.circuit_style,
                problem.desc.preserved_observable,
                problem.desc.code_distance,
                problem.desc.num_qubits,
                num_next_shots,
                num_correct,
                t1 - t0,
                problem.desc.decoder,
                CSV_HEADER_VERSION,
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


def collect_detection_fraction_data(problems: List[DecodingProblem],
                                    *,
                                    shots: int,
                                    out_path: Optional[str],
                                    discard_previous_data: bool):
    print(CSV_HEADER, flush=True)
    if out_path is not None:
        if discard_previous_data or not pathlib.Path(out_path).exists():
            with open(out_path, "w") as f:
                print(CSV_HEADER, file=f)

    for problem in problems:
        t0 = time.monotonic()
        samples = problem.circuit_maker().compile_detector_sampler().sample(shots)
        num_detections = np.count_nonzero(samples)
        num_samples = math.prod(samples.shape)
        t1 = time.monotonic()
        record = ",".join(str(e) for e in [
            problem.desc.data_width,
            problem.desc.data_height,
            problem.desc.rounds,
            problem.desc.noise,
            problem.desc.circuit_style,
            "-",
            problem.desc.code_distance,
            problem.desc.num_qubits,
            num_samples,
            num_samples - num_detections,
            t1 - t0,
            "detection_fraction",
            CSV_HEADER_VERSION,
        ])
        if out_path is not None:
            with open(out_path, "a") as f:
                print(record, file=f)
        print(record, flush=True)


@dataclasses.dataclass
class RemainingWork:
    shot_data: 'ShotData'
    max_shots: int
    max_errors: int
    threshold_circuit_breaker: float

    @property
    def finished(self) -> bool:
        if self.shot_data.num_shots >= self.max_shots:
            return True
        if self.shot_data.num_errors >= self.max_errors:
            return True
        if self.shot_data.logical_error_rate >= self.threshold_circuit_breaker and self.shot_data.num_shots >= 10:
            return True
        return False

    @property
    def remaining_shots(self) -> int:
        if self.finished:
            return 0
        return self.max_shots - self.shot_data.num_shots

    @property
    def remaining_errors(self) -> int:
        if self.finished:
            return 0
        return self.max_errors - self.shot_data.num_errors

    @property
    def remaining_time(self) -> float:
        if self.finished:
            return 0
        times = [float('inf')]
        if self.shot_data.num_shots:
            times.append(self.remaining_shots * self.shot_data.total_processing_seconds / self.shot_data.num_shots)
        if self.shot_data.num_errors:
            times.append(self.remaining_errors * self.shot_data.total_processing_seconds / self.shot_data.num_errors)
        return min(times)



@dataclasses.dataclass
class ShotData:
    num_shots: int = 0
    num_correct: int = 0
    total_processing_seconds: float = 0

    @property
    def num_errors(self) -> int:
        return self.num_shots - self.num_correct

    def remaining_work(self, max_shots: int, max_errors: int, threshold_circuit_breaker: float) -> RemainingWork:
        return RemainingWork(shot_data=self, max_shots=max_shots, max_errors=max_errors, threshold_circuit_breaker=threshold_circuit_breaker)

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


@dataclasses.dataclass
class ProblemShotData:
    data: Dict[DecodingProblemDesc, ShotData]

    def grouped_by(self, key: Callable[[DecodingProblemDesc], Any]) -> Dict[Any, 'ProblemShotData']:
        result = {}
        for k, v in self.data.items():
            result.setdefault(key(k), ProblemShotData({})).data[k] = v
        return {k: result[k] for k in sorted(result.keys())}

    def merged_by(self, key: Callable[[DecodingProblemDesc], Any]) -> Dict[Any, 'ShotData']:
        result: Dict[Any, 'ShotData'] = {}
        for k, v in self.data.items():
            d = result.setdefault(key(k), ShotData())
            d.num_shots += v.num_shots
            d.num_correct += v.num_correct
            d.total_processing_seconds += v.total_processing_seconds
        return {k: result[k] for k in sorted(result.keys())}

    def filter(self, predicate: Callable[[DecodingProblemDesc], bool]) -> 'ProblemShotData':
        result = ProblemShotData({})
        for k, v in self.data.items():
            if predicate(k):
                result.data[k] = v
        return result


def read_recorded_data(*paths: str) -> ProblemShotData:
    result = ProblemShotData({})
    for path in paths:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                key = DecodingProblemDesc(
                    code_distance=int(row["code_distance"]),
                    num_qubits=int(row["num_qubits"]),
                    data_width=int(row["data_width"]),
                    data_height=int(row["data_height"]),
                    rounds=int(row["rounds"]),
                    noise=float(row["noise"]),
                    circuit_style=row["circuit_style"],
                    preserved_observable=row["preserved_observable"],
                    decoder=row["decoder"],
                )
                val = result.data.setdefault(key, ShotData())
                val.num_shots += int(row["num_shots"])
                val.num_correct += int(row["num_correct"])
                val.total_processing_seconds += float(row["total_processing_seconds"])
    return result
