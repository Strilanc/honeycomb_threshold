import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from plotting import plot_data


def main():
    plot_data(
        "data/EM3.csv",
        title="Toric Honeycomb round errors (EM3)",
        out_path="gen/EM3.png",
        show=False)

    plot_data(
        "data/SD6.csv",
        title="Toric Honeycomb round errors (SD6)",
        out_path="gen/SD6.png",
        show=False)

    plot_data(
        "data/PC3.csv",
        title="Toric Honeycomb round errors (PC3)",
        out_path="gen/PC3.png",
        show=False)


if __name__ == '__main__':
    main()
