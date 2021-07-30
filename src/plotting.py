import csv
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt

from probability_util import log_binomial, binary_search


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
    if error_rate > 0.5:
        return 1 - total_error_to_per_round_error(1 - error_rate, rounds)
    assert 0 <= error_rate <= 0.5
    randomize_rate = 2*error_rate
    round_randomize_rate = 1 - (1 - randomize_rate)**(1 / rounds)
    round_error_rate = round_randomize_rate / 2
    return round_error_rate


def plot_data(*paths: str, title: str, out_path: Optional[str] = None, show: bool = None, fig: plt.Figure = None, ax: plt.Axes = None):
    if out_path is None and show is None:
        show = True

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

    assert fig is None == ax is None
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

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
            if datum.num_correct < datum.num_shots:
                xs.append(physical_error_rate)
                ys.append(cor(datum.logical_error_rate))
            # Show relative-likelihood-above-1e-3 error bars as a colored region.
            low, high = datum.likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3)
            x_bounds.append(physical_error_rate)
            ys_low.append(cor(low))
            ys_high.append(cor(high))
        ax.plot(xs, ys, label=f"{tile_diam=},{sub_rounds=}", marker=markers[tile_diam], zorder=100 - tile_diam)
        ax.fill_between(x_bounds, ys_low, ys_high, alpha=0.3)

    ax.legend()
    ax.loglog()

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
    ax.set_xticks([x for x in ticks_x])
    ax.set_yticks([y for y in ticks_y])
    ax.set_xticklabels([format_tick(x) for x in ticks_x], rotation=45)
    ax.set_yticklabels([format_tick(y) for y in ticks_y])
    ax.set_xlim(1e-4, 0.5)
    ax.set_ylim(1e-8, 0.5)
    ax.set_title(title)
    ax.set_ylabel("Per-Sub-Round Logical Error Rate (Vertical Observable)")
    ax.set_xlabel("Physical Error Rate Parameter (p)")
    ax.grid()
    if out_path is not None:
        fig.tight_layout()
        fig.savefig(out_path)
    if show:
        plt.show()

    return fig, ax
