import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from experiment import plot_data


def main():
    plot_data(
        "data/3step_demolition.csv",
        title="Toric Honeycomb round errors (3step_demolition circuit)",
        out_path="gen/3step_demolition.png",
        show=False)

    plot_data(
        "data/6step_cnot.csv",
        title="Toric Honeycomb round errors (6step_cnot circuit)",
        out_path="gen/6step_cnot.png",
        show=False)


if __name__ == '__main__':
    main()
