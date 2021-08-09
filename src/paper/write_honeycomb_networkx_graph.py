import pathlib
import sys

import networkx as nx

from paper.main_collect_all import surface_code_circuit

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from decoding import detector_error_model_to_nx_graph
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout


def main():
    c = generate_honeycomb_circuit(HoneycombLayout(
        noise=0.001,
        data_width=4,
        data_height=6,
        sub_rounds=30,
        style="SD6",
        obs="H",
    ))
    # c = surface_code_circuit("surface_code_circuits", "SD6", 0.001, "X", 7)
    g = detector_error_model_to_nx_graph(c.detector_error_model(decompose_errors=True))
    nx.readwrite.gpickle.write_gpickle(g, path="test.graph")


if __name__ == '__main__':
    main()
