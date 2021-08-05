import collections
import dataclasses
import functools
import math
import pathlib
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import matplotlib.colors as mcolors

from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import read_recorded_data, GROUPED_RECORDED_DATA, RecordedExperimentData
from honeycomb_layout import HoneycombLayout

from plotting import plot_data, total_error_to_per_round_error

import matplotlib.pyplot as plt

def main():
    if len(sys.argv) == 1:
        csvs = [str(f.absolute()) for f in pathlib.Path("data/run4_4qx6q").glob("*.csv")]
        #raise ValueError("Specify csv files to include as command line arguments.")
    else:
        csvs = sys.argv[1:]

    all_data = read_recorded_data(*csvs)

    make_lambda_combo_plot(all_data, 0)
    make_lambda_combo_plot(all_data, 1)
    make_lambda_combo_plot(all_data, 2)

    plt.show()


def make_lambda_combo_plot(all_data: GROUPED_RECORDED_DATA, t: int):
    keys = [
        HoneycombLayout(
            tile_width=1,
            tile_height=1,
            sub_rounds=1,
            style=style,
            obs=obs,
            noise=0,
        )
        for style in ["SD6", "EM3", "PC3", "SI500"]
        for obs in ["H", "V"]
    ]
    seen_probabilities = {
        p
        for x in all_data.values()
        for y in x.values()
        for p in y.keys()
    }
    p2i = {p: i for i, p in enumerate(sorted(seen_probabilities))}
    if not all(k in keys for k in all_data.keys()):
        raise NotImplementedError(repr(all_data.keys()))

    fig = plt.figure()
    gs = fig.add_gridspec(1, 4, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    for i in range(0, len(keys), 2):
        k1 = keys[i]
        k2 = keys[i + 1]
        v = {}
        if k1 in all_data:
            v[k1] = all_data[k1]
        if k2 in all_data:
            v[k2] = all_data[k2]
        ax: plt.Axes = axs[i // 2]

        if t == 0:
            plot_lambda_line_fits(
                v,
                ax=ax,
                fig=fig,
                p2i=p2i)
        elif t == 1:
            plot_lambda(v, ax=ax, fig=fig)
        else:
            plot_quop_regions(v, style=k1.style, ax=ax, fig=fig)

        ax.set_title(k1.style)
        if i == 6 and t == 0:
            ax.legend(loc="upper right")
        if i == 6 and t == 2:
            ax.legend(loc="lower right")
    for ax in fig.get_axes():
        ax.label_outer()
    return fig, axs


@dataclasses.dataclass
class LambdaGroup:
    noise: float
    distance_error_pairs_h: Dict[int, RecordedExperimentData]
    distance_error_pairs_v: Dict[int, RecordedExperimentData]

    @functools.cached_property
    def combo_data(self) -> Tuple[List[int], List[float]]:
        distances = sorted(set(self.distance_error_pairs_h.keys()) | set(self.distance_error_pairs_v.keys()))
        xs = []
        ys = []
        for d in distances:
            d1 = self.distance_error_pairs_h.get(d)
            d2 = self.distance_error_pairs_v.get(d)
            p1 = d1.logical_error_rate if d1 is not None else None
            p2 = d2.logical_error_rate if d2 is not None else None

            if p2 is None:
                p2 = p1
                print("WARNING FILLING IN FAKE DATA REMOVE")
            if p1 is None:
                p1 = p2
                print("WARNING FILLING IN FAKE DATA REMOVE")

            if p1 and p2:
                p1 = total_error_to_per_round_error(p1, 3)
                p2 = total_error_to_per_round_error(p2, 3)
                p_either = 1 - (1 - p1) * (1 - p2)
                xs.append(d)
                ys.append(p_either)
        return xs, ys

    @staticmethod
    def groups_from_data(data: GROUPED_RECORDED_DATA) -> Dict[float, 'LambdaGroup']:
        groups: Dict[float, LambdaGroup] = {}
        for k1 in data.keys():
            g1 = data[k1]
            for k2 in g1.keys():
                g2 = g1[k2]
                for physical_error_rate in g2.keys():
                    if physical_error_rate not in groups:
                        groups[physical_error_rate] = LambdaGroup(physical_error_rate, {}, {})
                    g = groups[physical_error_rate]
                    d = g.distance_error_pairs_h if k2.obs == "H" else g.distance_error_pairs_v
                    assert k2.tile_height == k2.tile_width // 2
                    assert k2.code_distance_1qdep == k2.tile_height * 4
                    d[k2.code_distance_1qdep] = g2[physical_error_rate]
        return groups

    @functools.cached_property
    def linear_fit_d_to_log_err(self) -> LinregressResult:
        xs, ys = self.combo_data
        if not xs or len(xs) <= 1:
            return linregress([0, 1], [0, 0])
        return linregress(xs, [math.log(y) for y in ys])

    def projected_error(self, distance: float) -> float:
        r = self.linear_fit_d_to_log_err
        return math.exp(r.slope * distance + r.intercept)

    def projected_distance(self, target_error: float) -> float:
        r = self.linear_fit_d_to_log_err
        return (math.log(target_error) - r.intercept) / r.slope

    def projected_required_qubit_count(self, target_error: float, style: str) -> int:
        d = int(math.ceil(self.projected_distance(target_error)))
        u = int(math.ceil(d / 4))
        lay = HoneycombLayout(tile_width=u * 2, tile_height=u, sub_rounds=1, style=style, obs="H", noise=0)
        assert lay.code_distance_1qdep in [d, d + 1, d + 2, d + 3]
        return lay.num_qubits


def plot_quop_regions(data: GROUPED_RECORDED_DATA,
                      *,
                      style: str,
                      fig: plt.Figure = None,
                      ax: plt.Axes = None):
    assert (fig is None) == (ax is None)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    groups = LambdaGroup.groups_from_data(data)
    ps = [1e-6, 1e-9, 1e-12]
    labels = ["megaquop regime", "gigaquop regime", "teraquop+ regime"]
    curves = [([], []) for _ in range(4)]
    y_max = 1e5

    for noise in sorted(groups.keys()):
        group = groups[noise]
        r = group.linear_fit_d_to_log_err
        if r.slope < -0.02:
            curves[3][0].append(noise)
            curves[3][1].append(y_max)
            for k, p in enumerate(ps):
                q = group.projected_required_qubit_count(p, style=style)
                curves[k][0].append(noise)
                curves[k][1].append(q)
    for k in range(3):
        ax.plot(curves[k][0], curves[k][1], label=labels[k])
        ax.fill_between(curves[k][0], curves[k][1], curves[k + 1][1])
    ax.loglog()
    ax.set_xlim(1e-4, 1e-2)
    ax.set_ylim(1e2, y_max)
    ax.set_xlabel("Noise")
    ax.set_ylabel("Physical qubits per logical qubit")
    ax.grid()


def plot_lambda_line_fits(data: GROUPED_RECORDED_DATA,
                *,
                p2i: Dict[float, int],
                fig: plt.Figure = None,
                ax: plt.Axes = None):
    assert (fig is None) == (ax is None)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    groups = LambdaGroup.groups_from_data(data)
    markers = "ov*sp^<>8PhH+xXDd|"
    colors = list(mcolors.TABLEAU_COLORS) * 3
    lambda_xs = []
    lambda_ys = []
    for noise in sorted(groups.keys()):
        group = groups[noise]
        r = group.linear_fit_d_to_log_err
        xs, ys = group.combo_data
        order = p2i[noise]
        if r.slope < -0.02:
            ys2 = [1e0, 1e-13]
            xs2 = [group.projected_distance(y) for y in ys2]
            ax.plot(xs2, ys2, '--', color=colors[order])
            lambda_xs.append(noise)
            lambda_ys.append(1 / math.exp(r.slope))
        ax.scatter(xs, ys, color=colors[order], marker=markers[order], label=f"{noise=}")
    ax.semilogy()
    ax.set_xlim(0, 50)
    ax.set_ylim(1e-12, 1e0)
    ax.set_xlabel("1QDep Code Distance")
    ax.set_ylabel("Code Cell Error Rate (Either Observable)")
    ax.grid()


def plot_lambda(data: GROUPED_RECORDED_DATA,
                *,
                fig: plt.Figure = None,
                ax: plt.Axes = None):
    assert (fig is None) == (ax is None)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    groups = LambdaGroup.groups_from_data(data)
    lambda_xs = []
    lambda_ys = []
    for noise in sorted(groups.keys()):
        group = groups[noise]
        r = group.linear_fit_d_to_log_err
        print(noise, r.slope)
        if r.slope != 0:
            lambda_xs.append(noise)
            lambda_ys.append(1 / math.exp(r.slope)**2)

    ax.set_xlabel("Noise")
    ax.set_ylabel("Suppression per Code Step (λ)")
    ax.semilogx()
    ax.plot(lambda_xs, lambda_ys)
    ax.set_xlim(1e-4, 1e-2)
    ax.set_ylim(1, 40)
    ax.grid()


if __name__ == '__main__':
    main()
