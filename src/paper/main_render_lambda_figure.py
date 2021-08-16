import dataclasses
import functools
import math
import pathlib
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.colors as mcolors

from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import read_recorded_data, ProblemShotData, ShotData, DecodingProblemDesc
from plotting import total_error_to_per_piece_error


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

    fig0, _ = plot_lambda_line_fits_combo(all_data, focus=False)
    fig0.set_size_inches(13, 10)
    fig0.savefig("gen/linefits_all.pdf", bbox_inches='tight')
    fig0.savefig("gen/linefits_all.png", bbox_inches='tight')

    fig1, _ = plot_lambda_line_fits_combo(all_data, focus=True)
    fig1.set_size_inches(13, 10)
    fig1.savefig("gen/linefits.pdf", bbox_inches='tight')
    fig1.savefig("gen/linefits.png", bbox_inches='tight')

    fig2, _ = plot_lambda_combo(all_data)
    fig2.set_size_inches(13, 5)
    fig2.savefig("gen/lambda.pdf", bbox_inches='tight')
    fig2.savefig("gen/lambda.png", bbox_inches='tight')

    fig3, _ = plot_teraquop_combo(all_data)
    fig3.set_size_inches(13, 5)
    fig3.savefig("gen/teraquop.pdf", bbox_inches='tight')
    fig3.savefig("gen/teraquop.png", bbox_inches='tight')

    plt.show()


def plot_lambda_line_fits_combo(all_data: ProblemShotData, focus: bool) -> Tuple[plt.Figure, plt.Axes]:
    styles = {
        "SD6": [
            ("honeycomb_SD6", "internal"),
            ("honeycomb_SD6", "internal_correlated"),
            ("surface_SD6", "internal"),
            ("surface_SD6", "internal_correlated"),
        ],
        "SI500": [
            ("honeycomb_SI500", "internal"),
            ("honeycomb_SI500", "internal_correlated"),
            ("surface_SI500", "internal"),
            ("surface_SI500", "internal_correlated"),
        ],
        "EM3": [
            None,
            None,
            ("honeycomb_EM3_v2", "internal"),
            ("honeycomb_EM3_v2", "internal_correlated"),
        ],
        # "PC3": [
        #     ("honeycomb_PC3", "internal"),
        #     ("honeycomb_PC3", "internal_correlated"),
        #     None,
        #     None,
        # ],
    }
    if focus:
        styles = {
            "SD6": [
                ("honeycomb_SD6", "internal_correlated"),
                ("surface_SD6", "internal_correlated"),
            ],
            "SI500": [
                ("honeycomb_SI500", "internal_correlated"),
                ("surface_SI500", "internal_correlated"),
            ],
            "EM3": [
                None,
                ("honeycomb_EM3_v2", "internal_correlated"),
            ],
        }

    seen_probabilities = {
        k.noise
        for k in all_data.data
    }
    p2i = {p: i for i, p in enumerate(sorted(seen_probabilities))}
    all_groups = all_data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    fig = plt.figure()
    ncols = len(styles)
    nrows = len(styles["SD6"])
    gs = fig.add_gridspec(ncols=ncols, nrows=nrows, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    used = set()
    for col, (name, cases) in enumerate(styles.items()):
        for row, style_decoder in enumerate(cases):
            ax: plt.Axes = axs[row][col]
            if style_decoder is None:
                ax.axis('off')
                continue
            used.add((row, col))
            style_data = all_groups.get(style_decoder, ProblemShotData({}))
            axs[row][col].set_title(name)

            groups = LambdaGroup.groups_from_data(style_data)
            markers = "ov*sp^<>8P+xXDd|"
            colors = list(mcolors.TABLEAU_COLORS) * 3
            lambda_xs = []
            lambda_ys = []
            for noise in sorted(groups.keys()):
                group = groups[noise]
                r = group.linear_fit_d_to_log_err
                xs, ys = group.combo_data
                order = p2i[noise]
                if group.appears_to_be_suppressing_errors:
                    ys2 = [1e0, 1e-13]
                    xs2 = [group.projected_distance(y) for y in ys2]
                    ax.plot(xs2, ys2, '--', color=colors[order])
                    lambda_xs.append(noise)
                    lambda_ys.append(1 / math.exp(r.slope))
                ax.scatter(xs, ys, color=colors[order], marker=markers[order], label=f"{noise}")
            ax.semilogy()
            ax.set_xlim(0, 30)
            ax.set_ylim(1e-12, 1e0)
            ax.grid()

    a, b = axs[0][0].get_legend_handles_labels()
    axs[0][-1].legend(a[::-1], b[::-1], loc="upper center", title="Physical Error Rates")

    for row in range(nrows):
        for col in range(ncols):
            if (row + 1, col) in used:
                axs[row][col].set_xlabel("")
            if (row - 1, col) in used:
                axs[row][col].set_title("")
            if (row, col - 1) in used:
                axs[row][col].set_ylabel("")

    for k in range(len(styles)):
        axs[-1][k].set_xlabel("Code Distance")
    for k in range(nrows):
        style_decoder = styles["SD6"][k]
        title = style_decoder[0].split("_")[0]
        title = title.capitalize()
        title += " (Correlated)" if "correlated" in style_decoder[1] else " (Standard)"
        axs[k][0].set_ylabel(f"{title}\nCode Cell Error Rate")
    for ax_row in axs:
        for ax in ax_row:
            ax.yaxis.set_ticks_position('both')
    return fig, axs


def plot_lambda_combo(all_data: ProblemShotData) -> Tuple[plt.Figure, plt.Axes]:
    styles = {
        "SD6": [
            ("honeycomb_SD6", "internal"),
            ("honeycomb_SD6", "internal_correlated"),
            ("surface_SD6", "internal"),
            ("surface_SD6", "internal_correlated"),
        ],
        "SI500": [
            ("honeycomb_SI500", "internal"),
            ("honeycomb_SI500", "internal_correlated"),
            ("surface_SI500", "internal"),
            ("surface_SI500", "internal_correlated"),
        ],
        "EM3": [
            ("honeycomb_EM3_v2", "internal"),
            ("honeycomb_EM3_v2", "internal_correlated"),
        ],
        # "PC3": [
        #     ("honeycomb_PC3", "internal"),
        #     ("honeycomb_PC3", "internal_correlated"),
        # ],
    }

    all_groups = all_data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    fig = plt.figure()
    gs = fig.add_gridspec(ncols=len(styles), nrows=1, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    have_any_poor = False
    for i, (name, cases) in enumerate(styles.items()):
        ax: plt.Axes = axs[i]

        poor_xs = []
        poor_ys = []
        for case_i, case in enumerate(cases):
            case_data = all_groups.get(case, ProblemShotData({}))
            groups = LambdaGroup.groups_from_data(case_data)
            lambda_xs = []
            lambda_ys = []
            noises = sorted(groups.keys())
            if not noises:
                continue
            for k, noise in enumerate(noises):
                group = groups[noise]
                if group.appears_to_be_suppressing_errors:
                    y = 1 / math.exp(group.linear_fit_d_to_log_err.slope) ** 2
                    lambda_xs.append(noise)
                    lambda_ys.append(y)
                    if group.has_low_confidence_extrapolation:
                        poor_xs.append(noise)
                        poor_ys.append(y)
            rep = groups[noises[0]].rep
            label = rep.circuit_style
            if label.endswith("_v2"):
                label = label[:-3]
            if label.endswith("_" + name):
                label = label[:-len(name) - 1]
            label = label.capitalize()
            if "correlated" in rep.decoder:
                label += " (Correlated)"
            else:
                label += " (Standard)"
            ax.plot(lambda_xs, lambda_ys, marker="ov*s"[case_i], label=label)

        have_any_poor |= bool(poor_ys)
        if have_any_poor:
            ax.scatter(poor_xs, poor_ys, s=200, color="red", zorder=100, alpha=0.3)

        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Lambda Factor")
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(1, 100)
        ax.loglog()
        ax.grid(which='minor')
        ax.grid(which='major', color='black')

        ax.set_title(name)
    axs[1].legend(loc="upper right")
    for ax in fig.get_axes():
        ax.label_outer()
    for ax in axs:
        ax.yaxis.set_ticks_position('both')
    return fig, axs


def plot_teraquop_combo(all_data: ProblemShotData) -> Tuple[plt.Figure, plt.Axes]:
    styles = {
        "SD6": [
            ("honeycomb_SD6", "internal"),
            ("honeycomb_SD6", "internal_correlated"),
            ("surface_SD6", "internal"),
            ("surface_SD6", "internal_correlated"),
        ],
        "SI500": [
            ("honeycomb_SI500", "internal"),
            ("honeycomb_SI500", "internal_correlated"),
            ("surface_SI500", "internal"),
            ("surface_SI500", "internal_correlated"),
        ],
        "EM3": [
            ("honeycomb_EM3_v2", "internal"),
            ("honeycomb_EM3_v2", "internal_correlated"),
        ],
        # "PC3": [
        #     ("honeycomb_PC3", "internal"),
        #     ("honeycomb_PC3", "internal_correlated"),
        # ],
    }

    all_groups = all_data.grouped_by(lambda e: (e.circuit_style, e.decoder))

    fig = plt.figure()
    gs = fig.add_gridspec(ncols=len(styles), nrows=1, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    have_any_poor = False
    for i, (name, cases) in enumerate(styles.items()):
        ax: plt.Axes = axs[i]

        poor_xs = []
        poor_ys = []
        for case_i, (style, decoder) in enumerate(cases):
            case_data = all_groups.get((style, decoder), ProblemShotData({}))
            groups = LambdaGroup.groups_from_data(case_data)
            lambda_xs = []
            lambda_ys = []
            noises = sorted(groups.keys())
            if not noises:
                continue
            for k, noise in enumerate(noises):
                group = groups[noise]
                if group.appears_to_be_suppressing_errors:
                    y = group.projected_required_qubit_count(1e-12, style)
                    lambda_xs.append(noise)
                    lambda_ys.append(y)
                    if group.has_low_confidence_extrapolation:
                        poor_xs.append(noise)
                        poor_ys.append(y)
            rep = groups[noises[0]].rep
            label = rep.circuit_style
            if label.endswith("_v2"):
                label = label[:-3]
            if label.endswith("_" + name):
                label = label[:-len(name) - 1]
            label = label.capitalize()
            if "correlated" in rep.decoder:
                label += " (Correlated)"
            else:
                label += " (Standard)"
            ax.plot(lambda_xs, lambda_ys, marker="ov*s"[case_i], label=label)

        have_any_poor |= bool(poor_ys)
        if have_any_poor:
            ax.scatter(poor_xs, poor_ys, s=200, color="red", zorder=100, alpha=0.3)

        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Teraquop Qubit Count")
        ax.set_xlim(1e-4, 2e-2)
        ax.set_ylim(1e2, 1e5)
        ax.loglog()
        ax.grid(which='minor')
        ax.grid(which='major', color='black')

        ax.set_title(name)
    axs[1].legend(loc="lower right")
    for ax in axs:
        ax.yaxis.set_ticks_position('both')
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

            if p2 is None or p1 is None:
                print(f"WARNING EXTRAPOLATING SECOND OBSERVABLE DATA POINT FOR {self.rep}")
            if p2 is None:
                p2 = p1
            if p1 is None:
                p1 = p2

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
            assert key.code_distance not in d, key
            d[key.code_distance] = val
        return groups

    @property
    def appears_to_be_suppressing_errors(self) -> bool:
        r = self.linear_fit_d_to_log_err
        return (r.slope < -0.02 and r.intercept < -0.1) or r.slope < -1

    @property
    def has_low_confidence_extrapolation(self) -> bool:
        return min(len(self.distance_error_pairs_v), len(self.distance_error_pairs_h)) < 3

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
        q = self.rep.num_qubits
        if "surface" in style:
            q += 1
            q *= (d / self.rep.code_distance) ** 2
            q -= 1
        elif "honeycomb" in style:
            d = int(math.ceil(d / 4)) * 4
            q *= (d / self.rep.code_distance) ** 2
        else:
            raise NotImplementedError()
        assert abs(q - math.floor(q + 0.5)) < 1e-5
        return q


if __name__ == '__main__':
    main()
