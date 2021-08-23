import itertools
import networkx as nx

import pytest

from decoding import sample_decode_count_correct, internal_decoder_path, detector_error_model_to_nx_graph
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout


@pytest.mark.parametrize('tile_diam,sub_rounds,style,obs,decoder', itertools.product(
    range(1, 3),
    range(5, 10),
    ["PC3", "SD6", "EM3"],
    ["H", "V"],
    ["internal", "internal_correlated"],
) if internal_decoder_path() is not None else [])
def test_internal_decoder_runs(tile_diam: int, sub_rounds: int, style: str, obs: str, decoder: str):
    sample_decode_count_correct(
        num_shots=100,
        circuit=generate_honeycomb_circuit(HoneycombLayout(
            data_width=2 * tile_diam,
            data_height=6 * tile_diam,
            sub_rounds=sub_rounds,
            noise=0.001,
            style=style,
            obs=obs,
        )),
        decoder=decoder,
    )


@pytest.mark.parametrize('tile_diam,sub_rounds,style,obs', itertools.product(
    range(1, 3),
    range(5, 10),
    ["PC3", "SD6", "EM3"],
    ["H", "V"]
))
def test_pymatching_runs(tile_diam: int, sub_rounds: int, style: str, obs: str):
    sample_decode_count_correct(
        num_shots=100,
        circuit=generate_honeycomb_circuit(HoneycombLayout(
            data_width=2 * tile_diam,
            data_height=6 * tile_diam,
            sub_rounds=sub_rounds,
            noise=0.001,
            style=style,
            obs=obs,
        )),
        decoder="pymatching",
    )


@pytest.mark.parametrize('style', ["PC3", "EM3", "EM3_v2", "SI1000", "SD6"])
def test_graph_has_two_connected_components(style: str):
    error_graph = detector_error_model_to_nx_graph(
        generate_honeycomb_circuit(HoneycombLayout(
            data_width=8,
            data_height=12,
            sub_rounds=99,
            noise=0.001,
            style=style,
            obs='V',
        )).detector_error_model(decompose_errors=True),
    )

    components = list(nx.connected_components(error_graph))
    assert len(components) == 2

    # The components should be roughly the same size.
    a, b = components
    assert 0.7 < len(a) / len(b) < 1.3

    for n, d in sorted(error_graph.nodes(data=True), key=lambda key: key[0]):
        assert 'coords' in d
        assert len(d['coords']) == 3

    degree = max(
        len(list(error_graph.neighbors(n)))
        for n, data in error_graph.nodes(data=True)
        if not data.get('is_boundary'))
    assert degree == (18 if style in ['EM3', 'EM3_v2'] else 12)
