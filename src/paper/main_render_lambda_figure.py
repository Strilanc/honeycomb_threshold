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

from collect_data import read_recorded_data, ProblemShotData, ShotData, DecodingProblemDesc
from honeycomb_layout import HoneycombLayout

from plotting import total_error_to_per_piece_error

import matplotlib.pyplot as plt

def main():
    if len(sys.argv) == 1:
        raise ValueError("Specify csv files to include as command line arguments.")

    csvs = []
    for path in sys.argv[1:]:
        p = pathlib.Path(path)
        if p.is_dir():
            csvs.extend(p.glob("*.csv"))
        else:
            csvs.append(p)

    all_data = read_recorded_data(*csvs)

    make_lambda_combo_plot(all_data, 0)
    make_lambda_combo_plot(all_data, 1)
    make_lambda_combo_plot(all_data, 2)

    plt.show()


def make_lambda_combo_plot(all_data: ProblemShotData, magic_plot_type_integer: int):
    expected_styles = ["honeycomb_SD6", "honeycomb_EM3_v2", "honeycomb_PC3", "honeycomb_SI500"]
    seen_probabilities = {
        k.noise
        for k in all_data.data
    }
    p2i = {p: i for i, p in enumerate(sorted(seen_probabilities))}
    groups = all_data.grouped_by(lambda e: e.circuit_style)
    for k in groups.keys():
        if k not in expected_styles:
            raise NotImplementedError()

    fig = plt.figure()
    gs = fig.add_gridspec(1, 4, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    for i, style in enumerate(expected_styles):
        style_data = groups.get(style, ProblemShotData({}))
        ax: plt.Axes = axs[i]

        if magic_plot_type_integer == 0:
            plot_lambda_line_fits(
                style_data,
                ax=ax,
                fig=fig,
                p2i=p2i)
        elif magic_plot_type_integer == 1:
            plot_lambda(style_data, ax=ax, fig=fig)
        else:
            plot_quop_regions(style_data, style=style, ax=ax, fig=fig)

        if style.endswith("_v2"):
            ax.set_title(style[:-3])
        else:
            ax.set_title(style)
        if i == 6 and magic_plot_type_integer == 0:
            ax.legend(loc="upper right")
        if i == 6 and magic_plot_type_integer == 2:
            ax.legend(loc="lower right")
    for ax in fig.get_axes():
        ax.label_outer()
    return fig, axs


@dataclasses.dataclass
class LambdaGroup:
    noise: float
    rep: DecodingProblemDesc
    distance_error_pairs_h: Dict[int, ShotData]
    distance_error_pairs_v: Dict[int, ShotData]

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
                cells = int(math.ceil(self.rep.rounds / self.rep.code_distance))
                p1 = total_error_to_per_piece_error(p1, cells)
                p2 = total_error_to_per_piece_error(p2, cells)
                p_either = 1 - (1 - p1) * (1 - p2)
                xs.append(d)
                ys.append(p_either)
        return xs, ys

    @staticmethod
    def groups_from_data(data: ProblemShotData) -> Dict[float, 'LambdaGroup']:
        groups: Dict[float, LambdaGroup] = {}
        for key, val in data.data.items():
            if key.noise not in groups:
                groups[key.noise] = LambdaGroup(key.noise, key, {}, {})
            g = groups[key.noise]
            d = g.distance_error_pairs_h if key.preserved_observable in "HX" else g.distance_error_pairs_v
            assert key.code_distance not in d
            d[key.code_distance] = val
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
        lay = HoneycombLayout(data_width=u * 4, data_height=u * 6, sub_rounds=1, style=style, obs="H", noise=0)
        assert lay.code_distance_1qdep in [d, d + 1, d + 2, d + 3]
        return lay.num_qubits


def plot_quop_regions(data: ProblemShotData,
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


def plot_lambda_line_fits(data: ProblemShotData,
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


def plot_lambda(data: ProblemShotData,
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
