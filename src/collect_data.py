import math
import pathlib
import time

from decoding import sample_decode_count_correct
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout

CSV_HEADER = "tile_width,tile_height,sub_rounds,physical_error_rate,circuit_style,num_shots,num_correct,total_processing_seconds"


def collect_simulated_experiment_data(*cases: HoneycombLayout,
                                      min_shots: int,
                                      max_shots: int,
                                      max_sample_std_dev: float = 1,
                                      min_seen_logical_errors: int,
                                      use_internal_decoder: bool = False,
                                      out_path: str,
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
        out_path: Where to write the CSV sample statistic data.
        discard_previous_data: If set, `out_path` is overwritten. If not set, `out_path` will be
            appended to (or created if needed).
    """
    print(CSV_HEADER)
    if discard_previous_data or not pathlib.Path(out_path).exists():
        with open(out_path, "w") as f:
            print(CSV_HEADER, file=f)

    for lay in cases:
        num_seen_errors = 0
        num_next_shots = min_shots
        total_shots = 0
        while True:
            t0 = time.monotonic()
            circuit = generate_honeycomb_circuit(lay)
            num_correct = sample_decode_count_correct(
                num_shots=num_next_shots,
                circuit=circuit,
                use_internal_decoder=use_internal_decoder,
            )
            t1 = time.monotonic()
            record = f"{lay.tile_width},{lay.tile_height},{lay.sub_rounds},{lay.noise},{lay.style},{num_next_shots},{num_correct},{t1 - t0}"
            with open(out_path, "a") as f:
                print(record, file=f)
            print(record)

            total_shots += num_next_shots
            num_seen_errors += num_next_shots - num_correct
            p = num_seen_errors / total_shots
            cur_sample_std_dev = math.sqrt(p * (1 - p) / total_shots)
            if total_shots >= max_shots:
                break
            if num_seen_errors >= min_seen_logical_errors and cur_sample_std_dev <= max_sample_std_dev:
                break
            num_next_shots = min(2 * num_next_shots, max_shots - total_shots)
