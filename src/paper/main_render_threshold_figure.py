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
    for zoom in [False, True]:
        plot_thresholds(all_data, zoom)
    plt.show()


def plot_thresholds(all_data: ProblemShotData, zoom_in: bool):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 4, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    expected_obs_styles = [
        (obs, style)
        for obs in ["H", "V"]
        for style in ["honeycomb_SD6", "honeycomb_EM3_v2", "honeycomb_PC3", "honeycomb_SI500"]
    ]
    groups = all_data.grouped_by(lambda desc: (desc.preserved_observable, desc.circuit_style))
    for k in groups.keys():
        if k not in expected_obs_styles:
            raise NotImplementedError()

    for i, k in enumerate(expected_obs_styles):
        v = groups.get(k, ProblemShotData({}))
        ax: plt.Axes = axs[i // 4][i % 4]
        plot_data(
            v,
            title="",
            ax=ax,
            fig=fig,
            legend=i == 7,
            focus_on_threshold=zoom_in)
        if i < 4:
            if k[1].endswith("_v2"):
                ax.set_title(k[1][:-3])
            else:
                ax.set_title(k[1])
        if k[1] == "V":
            ax.set_ylabel("VERTICAL OBSERVABLE\nError Rate per Code Distance Cell")
        else:
            ax.set_ylabel("HORIZONTAL OBSERVABLE\nError Rate per Code Distance Cell")
    for ax in fig.get_axes():
        ax.label_outer()


if __name__ == '__main__':
    main()
