from typing import List, Sequence

import stim

from honeycomb_layout import HoneycombLayout
from measure_tracker import MeasurementTracker, Prev


def generate_honeycomb_circuit(tile_diam: int, sub_rounds: int, noise: float) -> stim.Circuit:
    """Generates a honeycomb code circuit performing a fault tolerant memory experiment.

    Performs fault tolerant initialization, idling, and measurement.

    Reference:
        "Dynamically Generated Logical Qubits"
        Matthew B. Hastings, Jeongwan Haah
        https://arxiv.org/abs/2107.02194
    """
    lay = HoneycombLayout(
        tile_diam=tile_diam,
        sub_rounds=sub_rounds,
        noise=noise,
    )
    mtrack = MeasurementTracker()
    next_sub_round = 0
    circuit = stim.Circuit()

    # Annotate the locations of qubits used by the circuit.
    for q, i in lay.q2i.items():
        circuit.append_operation("QUBIT_COORDS", [i], [q.real, q.imag])

    # Initialize data.
    circuit.append_operation("R", lay.data_qubit_indices)
    circuit.append_operation("X_ERROR", lay.data_qubit_indices, lay.noise)  # Reset error.
    circuit.append_operation("DEPOLARIZE1", lay.qubit_indices_except(lay.data_qubit_indices), lay.noise)  # Idle error.
    circuit.append_operation("TICK", [])

    # Create dummy stabilizer records to compare against during initial rounds.
    # Then run enough rounds to ensure later measurements aren't comparing to dummies anymore.
    for r in range(3):
        # Z edges start in a known state.
        mtrack.add_dummies(*lay.round_edges(r), obstacle=r != 2)
        # Z stabilizers start in a known state.
        mtrack.add_dummies(*lay.round_hex_centers(r), obstacle=r != 2)
        # Y stabilizers start half formed (have Z part but not X part).
        mtrack.add_dummies(*[('1/2', h) for h in lay.round_hex_centers(r)], obstacle=r != 1)
    while next_sub_round < sub_rounds and next_sub_round < 4:
        circuit += _generate_honeycomb_sub_round(lay, mtrack, next_sub_round)
        next_sub_round += 1

    # Use a loop to advance the steady state as close as possible to the end.
    iterations = (sub_rounds - next_sub_round) // 3
    if iterations > 0:
        loop_body = stim.Circuit()
        for k in range(3):
            loop_body += _generate_honeycomb_sub_round(lay, mtrack, next_sub_round + k)
        circuit += loop_body * iterations
        next_sub_round += iterations * 3

    # Run a few more sub rounds, if needed, to get to the final state.
    while next_sub_round < sub_rounds:
        circuit += _generate_honeycomb_sub_round(lay, mtrack, next_sub_round)
        next_sub_round += 1

    # Prepare data qubit basis for fault tolerant data measurement.
    obs_basis, obs_qubits = lay.obs_1_before_sub_round(sub_rounds)
    append_basis_switch_vs_z(circuit, lay.data_qubit_indices, obs_basis)
    circuit.append_operation("DEPOLARIZE1", lay.data_qubit_indices, lay.noise)  # Clifford error.
    circuit.append_operation("DEPOLARIZE1", lay.qubit_indices_except(lay.data_qubit_indices), lay.noise)  # Idle error.
    circuit.append_operation("TICK", [])

    # Perform the fault tolerant data measurement.
    circuit.append_operation("X_ERROR", lay.data_qubit_indices, lay.noise)  # Measurement error.
    circuit.append_operation("DEPOLARIZE1", lay.qubit_indices_except(lay.data_qubit_indices), lay.noise)  # Idle error.
    circuit.append_operation("M", lay.data_qubit_indices)
    mtrack.add_measurements(*lay.data_qubit_coords)

    last_measure_basis = "XYZ"[(sub_rounds - 1) % 3]

    assert last_measure_basis == obs_basis if sub_rounds % 2 == 0 else obs_basis == "XYZ"[sub_rounds % 3]
    if last_measure_basis == obs_basis:
        # Compare data measurements to same-basis edge measurements from last round.
        for e in lay.round_edges("XYZ".index(obs_basis)):
            mtrack.add_group(e.left, e.right, group_key=e)
            mtrack.append_detector(
                e, Prev(e),
                out_circuit=circuit,
                coords=[e.center.real, e.center.imag, 0],
            )
    else:
        # Synthesize other-basis stabilizers by completing the cycles half-finished last round.
        other_basis, = set("XYZ") - {last_measure_basis, obs_basis}
        for h in lay.round_hex_centers("XYZ".index(other_basis)):
            mtrack.add_group(*lay.qubits_around_hex(h), ('1/2', h), group_key=h)
            mtrack.append_detector(
                h, Prev(h),
                out_circuit=circuit,
                coords=[h.real, h.imag, 0],
            )

    # Synthesize stabilizers from the six data measurements around hexes of the same basis.
    for h in lay.round_hex_centers("XYZ".index(obs_basis)):
        mtrack.add_group(*lay.qubits_around_hex(h), group_key=h)
        mtrack.append_detector(
            h, Prev(h),
            out_circuit=circuit,
            coords=[h.real, h.imag, 0],
        )

    circuit.append_operation(
        "OBSERVABLE_INCLUDE",
        mtrack.get_record_targets(*obs_qubits),
        0)

    return circuit


def _generate_honeycomb_sub_round(lay: HoneycombLayout,
                                  mtrack: MeasurementTracker,
                                  sub_round: int) -> stim.Circuit:
    """Returns a circuit performing one layer of edge measurements from the honeycomb code."""

    circuit = stim.Circuit()

    # Compute edge operations.
    cnot_targets_1st: List[int] = []
    cnot_targets_2nd: List[int] = []
    measure_targets: List[int] = []
    for edge in lay.round_edges(sub_round):
        qa, qb, qc = lay.q2i[edge.left], lay.q2i[edge.right], lay.q2i[edge.center]
        cnot_targets_1st.append(qa)
        cnot_targets_1st.append(qc)
        cnot_targets_2nd.append(qb)
        cnot_targets_2nd.append(qc)
        measure_targets.append(qc)
        mtrack.add_measurements(edge)

    # Begin measurements by reseting measurement qubits and switching data qubit basis.
    idle = lay.qubit_indices_except([*measure_targets, *lay.data_qubit_indices])
    circuit.append_operation("R", measure_targets)
    append_basis_switch_vs_z(circuit, lay.data_qubit_indices, lay.sub_round_edge_basis(sub_round))
    circuit.append_operation("X_ERROR", measure_targets, lay.noise)  # Reset error.
    circuit.append_operation("DEPOLARIZE1", lay.data_qubit_indices, lay.noise)  # Clifford error.
    circuit.append_operation("DEPOLARIZE1", idle, lay.noise)  # Idle error.
    circuit.append_operation("TICK", [])

    # Perform first layer of CNOTs.
    idle = lay.qubit_indices_except(cnot_targets_1st)
    circuit.append_operation("CNOT", cnot_targets_1st)
    circuit.append_operation("DEPOLARIZE2", cnot_targets_1st, lay.noise)  # Clifford error.
    circuit.append_operation("DEPOLARIZE1", idle, lay.noise)  # Idle error.
    circuit.append_operation("TICK", [])

    # Perform second layer of CNOTs.
    idle = lay.qubit_indices_except(cnot_targets_2nd)
    circuit.append_operation("CNOT", cnot_targets_2nd)
    circuit.append_operation("DEPOLARIZE2", cnot_targets_2nd, lay.noise)  # Clifford error.
    circuit.append_operation("DEPOLARIZE1", idle, lay.noise)  # Idle error.
    circuit.append_operation("TICK", [])

    # Finish measurements and restore data qubit basis.
    idle = lay.qubit_indices_except([*measure_targets, *lay.data_qubit_indices])
    circuit.append_operation("X_ERROR", measure_targets, lay.noise)  # Measure error.
    circuit.append_operation("DEPOLARIZE1", lay.data_qubit_indices, lay.noise)  # Clifford error.
    circuit.append_operation("DEPOLARIZE1", idle, lay.noise)  # Idle error.
    circuit.append_operation("M", measure_targets)
    append_basis_switch_vs_z(circuit, lay.data_qubit_indices, lay.sub_round_edge_basis(sub_round))

    # Multiply edge measurements along the observable's path into the observable.
    circuit.append_operation(
        "OBSERVABLE_INCLUDE",
        mtrack.get_record_targets(*(
                set(lay.obs_1_edges) & set(lay.round_edges(sub_round))
        )),
        0,
    )

    # Edges from this round form half of the edges for the spoke-center hexes from last round.
    for h in lay.round_hex_centers((sub_round - 1) % 3):
        mtrack.add_group(*lay.first_edges_around_hex(h), group_key=('1/2', h))
    # Edges from this round complete the stabilizer for the spoke-center hexes from two rounds ago.
    for h in lay.round_hex_centers((sub_round - 2) % 3):
        mtrack.add_group(('1/2', h), *lay.second_edges_around_hex(h), group_key=h)
        mtrack.append_detector(h, Prev(h), out_circuit=circuit, coords=[h.real, h.imag, 0])
    circuit.append_operation("SHIFT_COORDS", [], [0, 0, 1])
    circuit.append_operation("TICK", [])

    return circuit


def append_basis_switch_vs_z(out_circuit: stim.Circuit, targets: Sequence[int], new_basis: str):
    if new_basis == 'X':
        out_circuit.append_operation("H", targets)
    if new_basis == 'Y':
        out_circuit.append_operation("H_YZ", targets)
