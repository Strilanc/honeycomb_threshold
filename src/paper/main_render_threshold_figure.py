import pathlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import read_recorded_data, ProblemShotData

from plotting import plot_data

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
    fig = plot_thresholds(all_data, zoom_in=False)
    fig.set_size_inches(13, 20)
    fig.savefig("gen/threshold.pdf", bbox_inches='tight')
    plt.show()


def plot_thresholds(all_data: ProblemShotData, zoom_in: bool) -> plt.Figure:
    styles = {
        "SD6": [
            ("surface_SD6", "X", "internal"),
            ("surface_SD6", "Z", "internal"),
            ("surface_SD6", "X", "internal_correlated"),
            ("surface_SD6", "Z", "internal_correlated"),
            ("honeycomb_SD6", "H", "internal"),
            ("honeycomb_SD6", "V", "internal"),
            ("honeycomb_SD6", "H", "internal_correlated"),
            ("honeycomb_SD6", "V", "internal_correlated"),
        ],
        "SI500": [
            ("surface_SI500", "X", "internal"),
            ("surface_SI500", "Z", "internal"),
            ("surface_SI500", "X", "internal_correlated"),
            ("surface_SI500", "Z", "internal_correlated"),
            ("honeycomb_SI500", "H", "internal"),
            ("honeycomb_SI500", "V", "internal"),
            ("honeycomb_SI500", "H", "internal_correlated"),
            ("honeycomb_SI500", "V", "internal_correlated"),
        ],
        "EM3": [
            None,
            None,
            None,
            None,
            ("honeycomb_EM3_v2", "H", "internal"),
            ("honeycomb_EM3_v2", "V", "internal"),
            ("honeycomb_EM3_v2", "H", "internal_correlated"),
            ("honeycomb_EM3_v2", "V", "internal_correlated"),
        ],
        # "PC3": [
        #     None,
        #     None,
        #     None,
        #     None,
        #     ("honeycomb_PC3", "H", "internal"),
        #     ("honeycomb_PC3", "V", "internal"),
        #     ("honeycomb_PC3", "H", "internal_correlated"),
        #     ("honeycomb_PC3", "V", "internal_correlated"),
        # ],
    }
    all_groups = all_data.grouped_by(lambda e: (e.circuit_style, e.preserved_observable, e.decoder))

    fig = plt.figure()
    gs = fig.add_gridspec(ncols=len(styles), nrows=8, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    used = set()
    for col, (name, cases) in enumerate(styles.items()):
        for row, style_obs_decoder in enumerate(cases):
            ax: plt.Axes = axs[row][col]
            if style_obs_decoder is None:
                ax.axis('off')
                continue
            used.add((row, col))
            style_data = all_groups.get(style_obs_decoder, ProblemShotData({}))
            ax: plt.Axes = axs[row][col]
            plot_data(
                style_data,
                title=name,
                ax=ax,
                fig=fig,
                legend=False,
                marker_offset=4 if style_obs_decoder[1] in "XZ" else 0,
                focus_on_threshold=zoom_in)

    axs[0][2].legend(*axs[0][0].get_legend_handles_labels(), loc="upper left", title="Surface Code Sizes")
    axs[2][2].legend(*axs[7][2].get_legend_handles_labels(), loc="upper left", title="Honeycomb Code Sizes")
    for k in range(8):
        style, obs, decoder = styles["SD6"][k]
        if obs == "H":
            obs = "Horizontal"
        if obs == "V":
            obs = "Vertical"
        style = style.split("_")[0]
        if "correlated" in decoder:
            style += " (correlated)"
        axs[k][0].set_ylabel(f"{style}\n{obs} observable\nCode cell error rate")
    for row in range(8):
        for col in range(len(styles)):
            if (row - 1, col) in used:
                axs[row][col].set_xlabel("")
                axs[row][col].set_title("")
            if (row, col - 1) in used:
                axs[row][col].set_ylabel("")

    return fig


if __name__ == '__main__':
    main()
