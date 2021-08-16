import pathlib
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.
from collect_data import read_recorded_data, ProblemShotData


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

    fig = plot_detection_fraction(all_data)
    fig.set_size_inches(15, 5)
    fig.savefig("gen/detectionfraction.pdf", bbox_inches='tight')
    fig.savefig("gen/detectionfraction.png", bbox_inches='tight')


    plt.show()


def plot_detection_fraction(all_data: ProblemShotData) -> plt.Figure:
    styles = {
        "surface_SD6": "SD6\nSurface Code",
        "surface_SI500": "SI500\nSurface Code",
        "honeycomb_SD6": "SD6\nHoneycomb Code",
        "honeycomb_SI500": "SI500\nHoneycomb Code",
        "honeycomb_EM3_v2": "EM3\nHoneycomb Code",
        "honeycomb_EM3": "Tweaked EM3\nHoneycomb Code",
    }

    p2i = {p: i for i, p in enumerate(sorted(set(e.noise for e in all_data.data.keys())))}
    all_groups = all_data.grouped_by(lambda e: e.circuit_style)

    fig = plt.figure()
    ncols = len(styles)
    nrows = 1
    gs = fig.add_gridspec(ncols=ncols + 1, nrows=nrows, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    markers = "ov*sp^<>8P+xXDd|"
    colors = list(mcolors.TABLEAU_COLORS) * 3
    for col, style in enumerate(styles):
        ax: plt.Axes = axs[col]
        style_data = all_groups.get(style, ProblemShotData({})).grouped_by(lambda e: e.noise)
        for noise, case_data in style_data.items():
            xs = []
            ys = []
            for k, v in case_data.data.items():
                xs.append(k.code_distance)
                ys.append(v.logical_error_rate)
            order = p2i[noise]
            ax.plot(xs, ys, label=str(noise), marker=markers[order], color=colors[order])
        ax.set_title(styles[style])
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(["", "5", "10", "15", "20"], rotation=90)
    axs[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axs[0].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    axs[-1].axis('off')
    a, b = axs[-2].get_legend_handles_labels()
    axs[-1].legend(a[::-1], b[::-1], loc='upper left', title="Physical Error Rate")
    axs[0].set_ylabel("Detection Fraction")
    for ax in axs:
        ax.set_xlabel("Code Distance")
        ax.grid()

    #     ax: plt.Axes = axs[col]
    #     plt.show(style_data.)
    #     v2 = {k.noise: v for k, v in vals.data.items()}
    #     plot_data(
    #         style_data,
    #         title=name,
    #         ax=ax,
    #         fig=fig,
    #         legend=False,
    #         marker_offset=4 if style_obs_decoder[1] in "XZ" else 0,
    #         focus_on_threshold=False)
    #
    # a1, b1 = axs[0][0].get_legend_handles_labels()
    # a2, b2 = axs[-1][0].get_legend_handles_labels()
    # axs[0][-1].legend(
    #     [
    #         mpatches.Patch(color='white', label='Surface Code Sizes:'),
    #         *a1,
    #         mpatches.Patch(color='white', label='Honeycomb Code Sizes:'),
    #         *a2,
    #     ],
    #     [
    #         "Surface Code Sizes:",
    #         *b1,
    #         "Honeycomb Code Sizes:",
    #         *b2,
    #     ],
    #     loc="upper left",
    # )
    # for k in range(nrows):
    #     style, obs, decoder = styles["SD6"][k]
    #     if obs == "H":
    #         obs = "Horizontal"
    #     if obs == "V":
    #         obs = "Vertical"
    #     style = style.split("_")[0]
    #     if "correlated" in decoder:
    #         style += " (correlated)"
    #     axs[k][0].set_ylabel(f"{style}\n{obs} observable\nCode cell error rate")
    # for row in range(nrows):
    #     for col in range(ncols):
    #         if (row + 1, col) in used:
    #             axs[row][col].set_xlabel("")
    #         if (row - 1, col) in used:
    #             axs[row][col].set_title("")
    #         if (row, col - 1) in used:
    #             axs[row][col].set_ylabel("")

    return fig


if __name__ == '__main__':
    main()
