import itertools
import networkx as nx

import pytest

from decoding import sample_decode_count_correct, internal_decoder_path, detector_error_model_to_nx_graph
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout


@pytest.mark.parametrize('tile_diam,sub_rounds,style,obs', itertools.product(
    range(1, 5),
    range(1, 24),
    ["PC3", "SD6", "EM3"],
    ["H", "V"]
) if internal_decoder_path() is not None else [])
def test_internal_decoder_actually_runs(tile_diam: int, sub_rounds: int, style: str, obs: str):
    sample_decode_count_correct(
        num_shots=100,
        circuit=generate_honeycomb_circuit(HoneycombLayout(
            tile_width=tile_diam,
            tile_height=tile_diam,
            sub_rounds=sub_rounds,
            noise=0.001,
            style=style,
            obs=obs,
        )),
        use_internal_decoder=True,
    )


@pytest.mark.parametrize('tile_diam,sub_rounds,style,obs', itertools.product(
    range(1, 3),
    range(5, 10),
    ["PC3", "SD6", "EM3"],
    ["H", "V"]
) if internal_decoder_path() is not None else [])
def test_pymatching_runs(tile_diam: int, sub_rounds: int, style: str, obs: str):
    sample_decode_count_correct(
        num_shots=100,
        circuit=generate_honeycomb_circuit(HoneycombLayout(
            tile_width=tile_diam,
            tile_height=tile_diam,
            sub_rounds=sub_rounds,
            noise=0.001,
            style=style,
            obs=obs,
        )),
        use_internal_decoder=False,
    )


@pytest.mark.parametrize('style', ["PC3", "EM3", "EM3_CORR", "SI500", "SD6"])
def test_graph_has_two_connected_components(style: str):
    error_graph = detector_error_model_to_nx_graph(
        generate_honeycomb_circuit(HoneycombLayout(
            tile_width=2,
            tile_height=1,
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
