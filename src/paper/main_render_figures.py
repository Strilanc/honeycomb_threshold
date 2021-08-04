import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import read_recorded_data
from honeycomb_layout import HoneycombLayout

from plotting import plot_data

import matplotlib.pyplot as plt

def main():
    fig = plt.figure()
    gs = fig.add_gridspec(2, 4, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=True)
    if len(sys.argv) == 1:
        raise ValueError("Specify csv files to include as command line arguments.")

    all_data = read_recorded_data(*sys.argv[1:])
    keys = [
        HoneycombLayout(
            tile_width=1,
            tile_height=1,
            sub_rounds=1,
            style=style,
            obs=obs,
            noise=0,
        )
        for obs in ["H", "V"]
        for style in ["SD6", "EM3", "PC3", "SI500"]
    ]
    if not all(k in keys for k in all_data.keys()):
        raise NotImplementedError(repr(all_data.keys()))
    for i, k in enumerate(keys):
        if k in all_data:
            v = {k: all_data[k]}
        else:
            v = {}
        ax = axs[i // 4][i % 4]
        plot_data(
            v,
            title="",
            ax=ax,
            fig=fig)
        if i < 4:
            ax.set_title(k.style)
        if k.obs == "V":
            ax.set_ylabel("HORIZONTAL Obs Per-round Error")
        else:
            ax.set_ylabel("VERTICAL Obs Per-round Error")
    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()


if __name__ == '__main__':
    main()
