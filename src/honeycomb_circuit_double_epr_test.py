"""
This is a hacked together test for confirming that the honeycomb code can store 2 logical qubits.

For logical qubits 1, 2 and ancilla qubits a, b it measures

    X1*Xa
    Z1*Zb
    X2*Xb
    Z2*Zb

before and after doing the honeycomb memory circuit while injecting noise.

This proves there are two logical qubits as long as you trust that the ancilla qubits are qubits
with X and Z observables that anticommute.
"""

from typing import List

from decoding import sample_decode_count_correct
from hack_pycharm_pybind_pytest_workaround import stim
from honeycomb_layout import HoneycombLayout
from measure_tracker import MeasurementTracker, Prev

BASIS_TO_PAULI_TARGET = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}


def measure_obs_anc_parities(*, lay: HoneycombLayout, sub_round: int, observable_ids: List[int]) -> stim.Circuit:
    anc_a = lay.num_qubits
    anc_b = anc_a + 1
    ancilla_observables = [
        stim.target_x(anc_a),
        stim.target_z(anc_a),
        stim.target_x(anc_b),
        stim.target_z(anc_b),
    ]
    logical_observables = [
        lay.obs_h_before_sub_round(sub_round),
        lay.obs_v_before_sub_round(sub_round),
        lay.obs_h_before_sub_round(sub_round + 3),
        lay.obs_v_before_sub_round(sub_round + 3),
    ]
    c = stim.Circuit()
    for obs_id in observable_ids:
        targets = [ancilla_observables[obs_id]]
        obs_basis, obs_qubits = logical_observables[obs_id]
        p = BASIS_TO_PAULI_TARGET[obs_basis]
        for q in obs_qubits:
            targets.append(stim.target_combiner())
            targets.append(p(lay.q2i[q]))
        c.append_operation("MPP", targets)
        c.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], obs_id)
    return c


def generate_double_epr_honeycomb_circuit() -> stim.Circuit:
    lay = HoneycombLayout(data_width=4,
                          data_height=6,
                          sub_rounds=100,
                          style='EM3',
                          obs='--- X1 Z1 X2 Z2 all entangled with ancillae ----',
                          noise=1e-5)
    mtrack = MeasurementTracker()
    qs = set(range(lay.num_qubits))
    observable_ids = [0, 1, 2, 3]

    mem = stim.Circuit()
    for sub_round in range(lay.sub_rounds):
        c = stim.Circuit()
        edges = lay.round_edges(sub_round)
        basis = lay.sub_round_edge_basis(sub_round)

        p = BASIS_TO_PAULI_TARGET[basis]
        c.append_operation("MPP", [
            t
            for e in edges
            for t in [
                p(lay.q2i[e.left]),
                stim.target_combiner(),
                p(lay.q2i[e.right]),
            ]
        ])

        mtrack.add_measurements(*edges)
        for obs_id in observable_ids:
            e = lay.obs_h_edges if obs_id % 2 == 0 else lay.obs_v_edges
            c.append_operation(
                "OBSERVABLE_INCLUDE",
                mtrack.get_record_targets(*(set(e) & set(edges))),
                obs_id,
            )
        for h in lay.round_hex_centers((sub_round - 1) % 3):
            mtrack.add_group(*lay.first_edges_around_hex(h), group_key=('1/2', h))
        for h in lay.round_hex_centers((sub_round - 2) % 3):
            if sub_round >= 3:
                mtrack.add_group(*lay.second_edges_around_hex(h), ('1/2', h), group_key=h)
            if sub_round >= 6:
                mtrack.append_detector(h, Prev(h), out_circuit=c)
        c.append_operation("TICK", [])
        if 3 <= sub_round <= lay.sub_rounds:
            c = lay.noise_model.noisy_circuit(c, qs=qs)
        mem += c

    return (
        measure_obs_anc_parities(sub_round=0, lay=lay, observable_ids=observable_ids)
        + mem
        + measure_obs_anc_parities(sub_round=lay.sub_rounds, lay=lay, observable_ids=observable_ids)
    )


def test_circuit():
    circuit = generate_double_epr_honeycomb_circuit()
    assert sample_decode_count_correct(
        circuit=circuit,
        num_shots=10000,
        decoder='pymatching',
    ) > 9950
