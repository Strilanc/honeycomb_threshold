from typing import Optional

import matplotlib.pyplot as plt

from collect_data import read_recorded_data, GROUPED_RECORDED_DATA


def total_error_to_per_round_error(error_rate: float, rounds: int) -> float:
    """Convert from total error rate to per-round error rate."""
    if error_rate > 0.5:
        return 1 - total_error_to_per_round_error(1 - error_rate, rounds)
    assert 0 <= error_rate <= 0.5
    randomize_rate = 2*error_rate
    round_randomize_rate = 1 - (1 - randomize_rate)**(1 / rounds)
    round_error_rate = round_randomize_rate / 2
    return round_error_rate


def plot_data(data: GROUPED_RECORDED_DATA,
              title: str,
              out_path: Optional[str] = None,
              show: bool = False,
              fig: plt.Figure = None,
              ax: plt.Axes = None):
    if out_path is None and show is None and ax is None:
        show = True

    assert (fig is None) == (ax is None)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    markers = "ov*sp^<>8PhH+xXDd|"
    order = 0
    for k1 in sorted(data.keys()):
        g1 = data[k1]
        for k2 in sorted(g1.keys()):
            g2 = g1[k2]
            cor = lambda p: total_error_to_per_round_error(p, rounds=k2.sub_rounds // 3)
            xs = []
            ys = []
            x_bounds = []
            ys_low = []
            ys_high = []
            for physical_error_rate in sorted(g2.keys()):
                datum = g2[physical_error_rate]
                # Show curve going through max likelihood estimate, unless it's at zero error.
                if datum.num_correct < datum.num_shots:
                    xs.append(physical_error_rate)
                    ys.append(cor(datum.logical_error_rate))
                # Show relative-likelihood-above-1e-3 error bars as a colored region.
                low, high = datum.likely_error_rate_bounds(desired_ratio_vs_max_likelihood=1e-3)
                x_bounds.append(physical_error_rate)
                ys_low.append(cor(low))
                ys_high.append(cor(high))
            ax.plot(xs, ys, label=k2.legend_label(), marker=markers[order], zorder=100 - order)
            ax.fill_between(x_bounds, ys_low, ys_high, alpha=0.3)
            order += 1

    if data:
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
    ticks_y = [k*10**-p for k in [1, 2, 5] for p in range(1, 10) if k*10**-p <= 0.5]
    ticks_x = [k*10**-p for k in [1, 2, 5] for p in range(1, 10) if k*10**-p <= 0.5]
    ax.set_xticks([x for x in ticks_x])
    ax.set_yticks([y for y in ticks_y])
    ax.set_xticklabels([format_tick(x) for x in ticks_x], rotation=70)
    ax.set_xlim(1e-4, 0.5)
    ax.set_ylim(1e-8, 0.5)
    ax.set_yticklabels([format_tick(y) for y in ticks_y])
    ax.set_ylabel("Per-round Logical Error Rate")
    ax.set_title(title)
    ax.set_xlabel("Noise (p)")
    ax.grid()
    if out_path is not None:
        fig.tight_layout()
        fig.savefig(out_path)
    if show:
        plt.show()

    return fig, ax
