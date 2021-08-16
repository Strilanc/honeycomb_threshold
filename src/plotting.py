import math
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

from collect_data import ProblemShotData, DecodingProblemDesc


def total_error_to_per_piece_error(error_rate: float, pieces: int) -> float:
    """Convert from total error rate to per-round error rate."""
    if error_rate > 0.5:
        return 1 - total_error_to_per_piece_error(1 - error_rate, pieces)
    assert 0 <= error_rate <= 0.5
    randomize_rate = 2*error_rate
    round_randomize_rate = 1 - (1 - randomize_rate)**(1 / pieces)
    round_error_rate = round_randomize_rate / 2
    return round_error_rate


def plot_data(data: ProblemShotData,
              *,
              title: str,
              out_path: Optional[str] = None,
              show: bool = False,
              fig: plt.Figure = None,
              ax: plt.Axes = None,
              legend: bool = True,
              marker_offset=0,
              focus_on_threshold: bool = True):
    if out_path is None and show is None and ax is None:
        show = True

    assert (fig is None) == (ax is None)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    markers = "ov*sp^<>8PhH+xXDd|" * 100
    colors = list(mcolors.TABLEAU_COLORS) * 3
    order = 0
    d: DecodingProblemDesc
    for key, vals in data.grouped_by(lambda e: e.with_changes(noise=0)).items():
        cells = int(math.ceil(key.rounds / key.code_distance))
        def cor(p: float) -> float:
            return total_error_to_per_piece_error(p, pieces=cells)

        xs = []
        ys = []
        x_bounds = []
        ys_low = []
        ys_high = []

        v2 = {k.noise: v for k, v in vals.data.items()}
        for noise in sorted(v2.keys()):
            shot_data = v2[noise]
            # Show curve going through max likelihood estimate, unless it's at zero error.
            if shot_data.num_correct < shot_data.num_shots:
                xs.append(noise)
                ys.append(cor(shot_data.logical_error_rate))
            # Show relative-likelihood-above-1e-3 error bars as a colored region.
            low, high = shot_data.likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3)
            x_bounds.append(noise)
            ys_low.append(cor(low))
            ys_high.append(cor(high))
        label = f"{key.rounds} rounds, {key.data_width}x{key.data_height} data"
        ax.plot(xs, ys, label=label, marker=markers[order + marker_offset], zorder=100 - order, color=colors[order + marker_offset])
        ax.fill_between(x_bounds, ys_low, ys_high, alpha=0.3, color=colors[order + marker_offset])
        order += 1

    if data and legend:
        ax.legend(loc="lower right")
    ax.loglog()

    def format_tick(p: float) -> str:
        assert p > 0
        k = 0
        while p < 1:
            p *= 10
            k += 1
        assert p - int(p) < 1e-4
        return f"{int(p)}e-{k}"
    if focus_on_threshold:
        x_min, x_max = 5e-4, 1e-2
        y_min, y_max = 1e-2, 1e-1
    else:
        x_min, x_max = 1e-4, 3e-2
        y_min, y_max = 1e-8, 0.5
    ticks_x = [0.5] * (x_max >= 0.5) + [k*10**-p for k in [1] for p in range(1, 10) if x_min <= k*10**-p <= x_max]
    ticks_y = [0.5] * (y_max >= 0.5) + [k*10**-p for k in [1] for p in range(1, 10) if y_min <= k*10**-p <= y_max]
    ax.set_ylabel("Per-round Logical Error Rate")
    ax.set_xlabel("Physical Error Rate")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.grid()
    ax.set_xticks([x for x in ticks_x])
    ax.set_yticks([y for y in ticks_y])
    ax.set_xticklabels([format_tick(x) for x in ticks_x], rotation=70)
    ax.set_yticklabels([format_tick(y) for y in ticks_y])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *args: ""))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *args: ""))
    if out_path is not None:
        fig.tight_layout()
        fig.savefig(out_path)
    if show:
        plt.show()

    return fig, ax
