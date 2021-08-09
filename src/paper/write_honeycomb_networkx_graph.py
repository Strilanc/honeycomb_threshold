import networkx as nx

from decoding import detector_error_model_to_nx_graph
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout


def main():
    c = generate_honeycomb_circuit(HoneycombLayout(
        noise=0.001,
        data_width=8,
        data_height=12,
        sub_rounds=30,
        style="SD6",
        obs="V",
    ))
    g = detector_error_model_to_nx_graph(c.detector_error_model(decompose_errors=True))
    nx.readwrite.gpickle.write_gpickle(g, path="test.graph")


if __name__ == '__main__':
    main()
