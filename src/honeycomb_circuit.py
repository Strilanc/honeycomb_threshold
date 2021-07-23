from typing import List, Sequence, Tuple

import stim

from honeycomb_layout import HoneycombLayout
from measure_tracker import MeasurementTracker, Prev


def fuse_moments(moments: List[stim.Circuit]) -> stim.Circuit:
    c = stim.Circuit()
    for m in moments:
        c += m
        c.append_operation("TICK", [])
    return c


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

    # Annotate the locations of qubits used by the circuit.
    moments = [stim.Circuit()]
    for q, i in lay.q2i.items():
        moments[-1].append_operation("QUBIT_COORDS", [i], [q.real, q.imag])

    # Initialize data.
    moments[-1].append_operation("R", lay.data_qubit_indices)

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
        moments += _generate_honeycomb_sub_round(lay, mtrack, next_sub_round)
        next_sub_round += 1

    # Use a loop to advance the steady state as close as possible to the end.
    iterations = (sub_rounds - next_sub_round) // 3
    a = stim.Circuit()
    b = stim.Circuit()
    ab = stim.Circuit()
    a.append_operation("C_XYZ", lay.data_qubit_indices_1st)
    b.append_operation("C_XYZ", lay.data_qubit_indices_2nd)
    ab.append_operation("C_XYZ", lay.data_qubit_indices)
    if iterations > 0:
        r1 = _sub_round_resets(lay, next_sub_round)
        r2 = _sub_round_resets(lay, next_sub_round + 1)
        a1, _ = _sub_round_2q_ops(lay, next_sub_round)
        moments += [
            r1 + a,
            a1,
            r2 + ab,
        ]
        for x in range((iterations - 1) * 3 + 1):
            m1 = _sub_round_measurements_and_detectors(lay, mtrack, next_sub_round + x)
            _, b1 = _sub_round_2q_ops(lay, next_sub_round + x)
            a2, b2 = _sub_round_2q_ops(lay, next_sub_round + x + 1)
            r3 = _sub_round_resets(lay, next_sub_round + x + 2)
            moments += [
                a2 + b1,
                r3 + ab + m1,
            ]
        m2 = _sub_round_measurements_and_detectors(lay, mtrack, next_sub_round + 1)
        m3 = _sub_round_measurements_and_detectors(lay, mtrack, next_sub_round + 2)
        _, b2 = _sub_round_2q_ops(lay, next_sub_round + 1)
        a3, b3 = _sub_round_2q_ops(lay, next_sub_round + 2)
        moments += [
            a3 + b2,
            m2,
            b,
            b3,
            m3
        ]
        next_sub_round += iterations * 3

    # Run a few more sub rounds, if needed, to get to the final state.
    while next_sub_round < sub_rounds:
        moments += _generate_honeycomb_sub_round(lay, mtrack, next_sub_round)
        next_sub_round += 1

    # Prepare data qubit basis for fault tolerant data measurement.
    obs_basis, obs_qubits = lay.obs_1_before_sub_round(sub_rounds)
    if obs_basis != lay.sub_round_edge_basis(sub_rounds - 1):
        moments.append(stim.Circuit())
        moments[-1].append_operation("C_XYZ", lay.data_qubit_indices)

    # Perform the fault tolerant data measurement.
    moments.append(stim.Circuit())
    moments[-1].append_operation("M", lay.data_qubit_indices)
    mtrack.add_measurements(*lay.data_qubit_coords)

    last_measure_basis = "XYZ"[(sub_rounds - 1) % 3]

    assert last_measure_basis == obs_basis if sub_rounds % 2 == 0 else obs_basis == "XYZ"[sub_rounds % 3]
    if last_measure_basis == obs_basis:
        # Compare data measurements to same-basis edge measurements from last round.
        for e in lay.round_edges("XYZ".index(obs_basis)):
            mtrack.add_group(e.left, e.right, group_key=e)
            mtrack.append_detector(
                e, Prev(e),
                out_circuit=moments[-1],
                coords=[e.center.real, e.center.imag, 0],
            )
    else:
        # Synthesize other-basis stabilizers by completing the cycles half-finished last round.
        other_basis, = set("XYZ") - {last_measure_basis, obs_basis}
        for h in lay.round_hex_centers("XYZ".index(other_basis)):
            mtrack.add_group(*lay.qubits_around_hex(h), ('1/2', h), group_key=h)
            mtrack.append_detector(
                h, Prev(h),
                out_circuit=moments[-1],
                coords=[h.real, h.imag, 0],
            )

    # Synthesize stabilizers from the six data measurements around hexes of the same basis.
    for h in lay.round_hex_centers("XYZ".index(obs_basis)):
        mtrack.add_group(*lay.qubits_around_hex(h), group_key=h)
        mtrack.append_detector(
            h, Prev(h),
            out_circuit=moments[-1],
            coords=[h.real, h.imag, 0],
        )

    moments[-1].append_operation(
        "OBSERVABLE_INCLUDE",
        mtrack.get_record_targets(*obs_qubits),
        0)

    return fuse_moments(moments)


def _sub_round_2q_ops(lay: HoneycombLayout, sub_round: int) -> Tuple[stim.Circuit, stim.Circuit]:
    cnot_targets_1st: List[int] = []
    cnot_targets_2nd: List[int] = []
    for edge in lay.round_edges(sub_round):
        qa, qb, qc = lay.q2i[edge.left], lay.q2i[edge.right], lay.q2i[edge.center]
        cnot_targets_1st.append(qa)
        cnot_targets_1st.append(qc)
        cnot_targets_2nd.append(qb)
        cnot_targets_2nd.append(qc)
    result1 = stim.Circuit()
    result2 = stim.Circuit()
    result1.append_operation("CNOT", cnot_targets_1st)
    result2.append_operation("CNOT", cnot_targets_2nd)
    return result1, result2


def _generate_honeycomb_sub_round(lay: HoneycombLayout,
                                  mtrack: MeasurementTracker,
                                  sub_round: int) -> List[stim.Circuit]:
    """Returns a circuit performing one layer of edge measurements from the honeycomb code."""


    moments = [stim.Circuit() for _ in range(6)]

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
    moments[0].append_operation("R", measure_targets)
    moments[1].append_operation("C_XYZ", lay.data_qubit_indices_1st)

    # Perform first layer of CNOTs.
    moments[2].append_operation("C_XYZ", lay.data_qubit_indices_2nd)
    moments[3].append_operation("CNOT", cnot_targets_1st)

    # Perform second layer of CNOTs.
    moments[4].append_operation("CNOT", cnot_targets_2nd)

    # Finish measurements and restore data qubit basis.
    moments[5].append_operation("M", measure_targets)

    # Multiply edge measurements along the observable's path into the observable.
    moments[5].append_operation(
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
        mtrack.append_detector(h, Prev(h), out_circuit=moments[-1], coords=[h.real, h.imag, 0])
    moments[5].append_operation("SHIFT_COORDS", [], [0, 0, 1])

    return moments


def _sub_round_resets(
        lay: HoneycombLayout,
        sub_round: int) -> List[stim.Circuit]:
    round_edges = lay.round_edges(sub_round)
    moment = stim.Circuit()
    moment.append_operation("R", [lay.q2i[edge.center] for edge in round_edges])
    return moment


def _sub_round_measurements_and_detectors(
        lay: HoneycombLayout,
        mtrack: MeasurementTracker,
        sub_round: int) -> List[stim.Circuit]:
    """Returns a circuit performing one layer of edge measurements from the honeycomb code."""

    round_edges = lay.round_edges(sub_round)
    moment = stim.Circuit()
    moment.append_operation("M", [lay.q2i[edge.center] for edge in round_edges])
    mtrack.add_measurements(*round_edges)

    # Multiply edge measurements along the observable's path into the observable.
    moment.append_operation(
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
        mtrack.append_detector(h, Prev(h), out_circuit=moment, coords=[h.real, h.imag, 0])
    moment.append_operation("SHIFT_COORDS", [], [0, 0, 1])

    return moment
