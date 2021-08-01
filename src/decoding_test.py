import itertools

import pytest

from decoding import sample_decode_count_correct, internal_decoder_path
from honeycomb_circuit import generate_honeycomb_circuit


@pytest.mark.parametrize('tile_diam,sub_rounds,style', itertools.product(
    range(1, 5),
    range(1, 24),
    ["PC3", "SD6", "EM3"],
) if internal_decoder_path() is not None else [])
def test_internal_decoder_actually_runs(tile_diam: int, sub_rounds: int, style: str):
    sample_decode_count_correct(
        num_shots=100,
        circuit=generate_honeycomb_circuit(
            tile_width=tile_diam,
            tile_height=tile_diam,
            sub_rounds=sub_rounds,
            noise=0.001,
            style=style,
        ),
        use_internal_decoder=True,
    )
