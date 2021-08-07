from typing import List, Tuple

import stim

from honeycomb_layout import HoneycombLayout
from measure_tracker import MeasurementTracker, Prev


def generate_honeycomb_circuit(lay: HoneycombLayout) -> stim.Circuit:
    """Generates a honeycomb code circuit performing a fault tolerant memory experiment.

    Performs fault tolerant initialization, idling, and measurement.

    Args:
        lay: Configuration information for the circuit.

    Reference:
        "Dynamically Generated Logical Qubits"
        Matthew B. Hastings, Jeongwan Haah
        https://arxiv.org/abs/2107.02194
    """
    mtrack = MeasurementTracker()

    # Annotate the locations of qubits used by the circuit.
    result = stim.Circuit()
    for q in lay.used_qubit_coords:
        i = lay.q2i[q]
        result.append_operation("QUBIT_COORDS", [i], [q.real, q.imag])

    result += fault_tolerant_init(lay, mtrack)
    if lay.style == "SD6":
        result += generate_rounds_sd6(lay, mtrack)
    elif lay.style == "PC3":
        result += generate_rounds_pc3(lay, mtrack)
    elif lay.style == "EM3":
        result += generate_rounds_em3(lay, mtrack)
    elif lay.style == "EM3_CORR":
        result += generate_rounds_pc3(lay, mtrack)
    elif lay.style == "SI500":
        result += generate_rounds_si500(lay, mtrack)
    else:
        raise NotImplementedError(lay.style)
    result += fault_tolerant_measurement(lay, mtrack)

    return lay.noise_model.noisy_circuit(result)


def fault_tolerant_init(lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    result = stim.Circuit()
    result.append_operation("R", lay.used_qubit_indices)
    result.append_operation("TICK", [])

    init_basis = lay.obs_before_sub_round(0)[0]
    if init_basis != "Z":
        result.append_operation(f"H_{init_basis}Z", lay.data_qubit_indices)
        result.append_operation("TICK", [])

    # Create dummy stabilizer records to compare against during initial rounds.
    # Then run enough rounds to ensure later measurements aren't comparing to dummies anymore.
    for sub_round in range(3):
        edge_init = "XYZ".index(init_basis)
        half_init = (edge_init - 1) % 3
        # Z edges start in a known state.
        mtrack.add_dummies(*lay.round_edges(sub_round), obstacle=sub_round != edge_init)
        mtrack.add_dummies(*[('1/2', e) for e in lay.round_edges(sub_round)])
        # Z stabilizers start in a known state.
        mtrack.add_dummies(*lay.round_hex_centers(sub_round), obstacle=sub_round != edge_init)
        # Y stabilizers start half formed (have Z part but not X part).
        mtrack.add_dummies(*[('1/2', h) for h in lay.round_hex_centers(sub_round)], obstacle=sub_round != half_init)

    return  result


def fault_tolerant_measurement(lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    result = stim.Circuit()

    # Perform the fault tolerant data measurement.
    obs_basis, obs_qubits = lay.obs_before_sub_round(lay.sub_rounds)
    if obs_basis != "Z":
        result.append_operation(f"H_{obs_basis}Z", lay.data_qubit_indices)
        result.append_operation("TICK", [])
    result.append_operation("M", lay.data_qubit_indices)
    mtrack.add_measurements(*lay.data_qubit_coords)

    last_measure_basis = "XYZ"[(lay.sub_rounds - 1) % 3]
    if last_measure_basis == obs_basis:
        # Compare data measurements to same-basis edge measurements from last round.
        for e in lay.round_edges("XYZ".index(obs_basis)):
            mtrack.add_group(e.left, e.right, group_key=e)
            mtrack.append_detector(
                e, Prev(e),
                out_circuit=result,
                coords=[e.center.real, e.center.imag, 0],
            )
    else:
        # Synthesize other-basis stabilizers by completing the cycles half-finished last round.
        other_basis, = set("XYZ") - {last_measure_basis, obs_basis}
        for h in lay.round_hex_centers("XYZ".index(other_basis)):
            mtrack.add_group(*lay.qubits_around_hex(h), ('1/2', h), group_key=h)
            mtrack.append_detector(
                h, Prev(h),
                out_circuit=result,
                coords=[h.real, h.imag, 0],
            )

    # Synthesize stabilizers from the six data measurements around hexes of the same basis.
    for h in lay.round_hex_centers("XYZ".index(obs_basis)):
        mtrack.add_group(*lay.qubits_around_hex(h), group_key=h)
        mtrack.append_detector(
            h, Prev(h),
            out_circuit=result,
            coords=[h.real, h.imag, 0],
        )

    result.append_operation(
        "OBSERVABLE_INCLUDE",
        mtrack.get_record_targets(*obs_qubits),
        lay.obs_index)

    return result


def generate_rounds_sd6(lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    n = lay.sub_rounds
    result = stim.Circuit()
    if n == 0:
        raise NotImplementedError("Zero rounds.")
    elif n == 1:
        # For a single sub round there's no time to pipeline.
        result += _sub_round_resets(lay, 0)
        result += _cycle_data(lay, True, True)
        result.append_operation("TICK", [])
        result += _sub_round_2q_ops(lay, 0)[0]
        result.append_operation("TICK", [])
        result += _sub_round_2q_ops(lay, 0)[1]
        result.append_operation("TICK", [])
        result += _sub_round_measurements_and_detectors(lay=lay, mtrack=mtrack, sub_round=0)
    else:
        # Get the pipeline started.
        result += _sub_round_resets(lay, 0)
        result += _cycle_data(lay, True, False)
        result.append_operation("TICK", [])
        result += _sub_round_2q_ops(lay, 0)[0]
        result.append_operation("TICK", [])
        result += _sub_round_resets(lay, 1)
        result += _cycle_data(lay, True, True)
        result.append_operation("TICK", [])

        # Run the sub rounds in a pipelined fashion.
        next_sub_round = 0
        while next_sub_round < 4 and next_sub_round < n - 2:
            result += _sd6_pipeline_step(lay, next_sub_round, mtrack)
            next_sub_round += 1
        iterations = max(0, n - next_sub_round - 2) // 3
        if iterations > 1:
            loop_body = stim.Circuit()
            loop_body += _sd6_pipeline_step(lay, next_sub_round, mtrack)
            loop_body += _sd6_pipeline_step(lay, next_sub_round + 1, mtrack)
            loop_body += _sd6_pipeline_step(lay, next_sub_round + 2, mtrack)
            result += loop_body * iterations
            next_sub_round += iterations * 3
        for sub_round in range(next_sub_round, n - 2):
            result += _sd6_pipeline_step(lay, sub_round, mtrack)

        # End the pipeline.
        _, b2 = _sub_round_2q_ops(lay, n - 2)
        a3, b3 = _sub_round_2q_ops(lay, n - 1)
        result += a3 + b2
        result.append_operation("TICK", [])
        result += _sub_round_measurements_and_detectors(lay=lay, mtrack=mtrack, sub_round=n - 2)
        result.append_operation("TICK", [])
        result += _cycle_data(lay, False, True)
        result.append_operation("TICK", [])
        result += b3
        result.append_operation("TICK", [])
        result += _sub_round_measurements_and_detectors(lay=lay, mtrack=mtrack, sub_round=n - 1)

    if (lay.sub_rounds + 1) % 3 == 0:
        result.append_operation("C_ZYX", lay.data_qubit_indices)
    elif (lay.sub_rounds + 2) % 3 == 0:
        # C_XYZ because C_ZYX**2 == C_XYZ
        result.append_operation("C_XYZ", lay.data_qubit_indices)
    result.append_operation("TICK", [])

    return result


def generate_rounds_pc3(lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    n = lay.sub_rounds
    moments = []
    if n == 0:
        raise NotImplementedError("Zero rounds.")
    k = 0
    while k < n + 2:
        circuit = stim.Circuit()
        if 0 <= k < n:
            circuit.append_operation(lay.sub_round_edge_basis(k) + "CX",
                                     [lay.q2i[q] for e in lay.round_edges(k) for q in [e.left, e.center]])
        if 0 <= k - 1 < n:
            circuit.append_operation(lay.sub_round_edge_basis(k - 1) + "CX",
                                     [lay.q2i[q] for e in lay.round_edges(k - 1) for q in [e.right, e.center]])
        if 0 <= k - 2 < n:
            circuit += _sub_round_measurements_and_detectors(lay=lay, mtrack=mtrack, sub_round=k - 2)
        elif k - 2 <= 0:
            circuit += _sub_round_resets(lay, k - 2)
        circuit.append_operation("TICK", [])
        moments.append(circuit)
        k += 1

        # Stabilizers are established by round 6.
        # The edge comparisons across rounds, used to get the stabilizers, are established by 9.
        # The repeating state has period 3; by round 12 we've got the whole loop.
        if k == 12:
            iterations = (n - k + 3) // 3
            if iterations > 1:
                moments[-3:] = [(moments[-3] + moments[-2] + moments[-1]) * iterations]
                k += iterations * 3 - 3

    result = stim.Circuit()
    for m in moments:
        result += m

    for k in range(3):
        edges = mtrack.get_record_targets(*(
            ('1/2', e)
            for e in (set(lay.obs_edges) & set(lay.round_edges(k)))
        ))
        if edges is not None:
            result.append_operation(
                "OBSERVABLE_INCLUDE",
                edges,
                lay.obs_index,
            )

    return result


def generate_rounds_em3(lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    n = lay.sub_rounds
    moments = []
    if n == 0:
        raise NotImplementedError("Zero rounds.")

    k = 0
    while k < n:
        edges = lay.round_edges(k)
        basis = lay.sub_round_edge_basis(k)
        tp = [stim.target_x, stim.target_y, stim.target_z]["XYZ".index(basis)]
        circuit = stim.Circuit()
        circuit.append_operation("MPP", [t for e in edges for t in [tp(lay.q2i[e.left]), stim.target_combiner(), tp(lay.q2i[e.right])]])
        circuit += _sub_round_measurements_and_detectors(lay=lay, mtrack=mtrack, sub_round=k)
        circuit.append_operation("TICK", [])
        moments.append(circuit)
        k += 1

        # Steady state starts on round 4 and has period 3. Once we get to 7 we've got the whole loop.
        if k == 7:
            iterations = (n - k + 3) // 3
            if iterations > 1:
                moments[-3:] = [(moments[-3] + moments[-2] + moments[-1]) * iterations]
                k += iterations * 3 - 3

    result = stim.Circuit()
    for m in moments:
        result += m
    return result


def generate_cycle_si500(subround: int, lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    circuit = stim.Circuit()
    if subround > 0:
        circuit.append_operation("R", lay.measure_qubit_indices)
        circuit.append_operation("TICK", [])

    circuit.append_operation("C_ZYX", lay.data_qubit_indices_1st)
    circuit.append_operation("H", lay.measure_qubit_indices)
    circuit.append_operation("TICK", [])

    for k in range(3):
        circuit.append_operation("C_ZYX", lay.data_qubit_indices_2nd)
        circuit.append_operation("CZ", [lay.q2i[q] for e in lay.round_edges(k) for q in [e.left, e.center]])
        circuit.append_operation("TICK", [])

        if k < 2:
            circuit.append_operation("C_ZYX", lay.data_qubit_indices_1st)
        circuit.append_operation("CZ", [lay.q2i[q] for e in lay.round_edges(k) for q in [e.right, e.center]])
        circuit.append_operation("TICK", [])

    circuit.append_operation("H", lay.measure_qubit_indices)
    circuit.append_operation("TICK", [])

    for k in range(3):
        circuit += _sub_round_measurements_and_detectors(lay=lay, mtrack=mtrack, sub_round=subround + k)
    circuit.append_operation("TICK", [])

    return circuit


def generate_rounds_si500(lay: HoneycombLayout, mtrack: MeasurementTracker) -> stim.Circuit:
    n = lay.sub_rounds
    if n == 0:
        raise NotImplementedError("Zero rounds.")
    if n % 3 != 0:
        raise NotImplementedError(n)

    circuit = stim.Circuit()
    circuit += generate_cycle_si500(0, lay, mtrack)
    if n > 3:
        circuit += generate_cycle_si500(3, lay, mtrack)
    if n > 6:
        circuit += generate_cycle_si500(6, lay, mtrack) * ((n - 6) // 3)

    return circuit


def _sd6_pipeline_step(lay: HoneycombLayout, sub_round: int, mtrack: MeasurementTracker) -> stim.Circuit:
    _, b1 = _sub_round_2q_ops(lay, sub_round)
    a2, _ = _sub_round_2q_ops(lay, sub_round + 1)
    r3 = _sub_round_resets(lay, sub_round + 2)
    c = _cycle_data(lay, True, True)
    result = stim.Circuit()
    result += a2
    result += b1
    result.append_operation("TICK", [])
    result += r3
    result += c
    result += _sub_round_measurements_and_detectors(
        lay=lay,
        mtrack=mtrack,
        sub_round=sub_round,
    )
    result.append_operation("TICK", [])
    return result


def _cycle_data(lay: HoneycombLayout, first: bool, second: bool) -> stim.Circuit:
    result = stim.Circuit()
    if first and second:
        result.append_operation("C_ZYX", lay.data_qubit_indices)
    elif first:
        result.append_operation("C_ZYX", lay.data_qubit_indices_1st)
    elif second:
        result.append_operation("C_ZYX", lay.data_qubit_indices_2nd)
    return result


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


def _sub_round_resets(
        lay: HoneycombLayout,
        sub_round: int) -> stim.Circuit:
    round_edges = lay.round_edges(sub_round)
    moment = stim.Circuit()
    moment.append_operation("R", [lay.q2i[edge.center] for edge in round_edges])
    return moment


def MPP_CORR(targets: List[int], p: float) -> stim.Circuit:

    # Generate all possible combinations of (non-identity) channels.  Assumes triple of targets.
    circuit = stim.Circuit()

    first_targets = ["I", stim.target_x(targets[0]), stim.target_y(targets[0]), stim.target_z(targets[0])]
    second_targets = ["I", stim.target_x(targets[1]), stim.target_y(targets[1]), stim.target_z(targets[1])]
    measure_targets = ["I", stim.target_x(targets[2])]

    errors = []
    for first_target in first_targets:
        for second_target in second_targets:
            for measure_target in measure_targets:
                error = []
                if first_target != "I":
                    error.append(first_target)
                if second_target != "I":
                    error.append(second_target)
                if measure_target != "I":
                    error.append(measure_target)

                if len(error) > 0:
                    errors.append(error)

    for error in errors:
        circuit.append_operation("CORRELATED_ERROR", error, p)

    return circuit


def _sub_round_measurements_and_detectors(
        *,
        lay: HoneycombLayout,
        mtrack: MeasurementTracker,
        sub_round: int,
) -> stim.Circuit:
    """Returns a circuit performing one layer of edge measurements from the honeycomb code.

    Handles annotating detectors and observables within the bulk of the circuit.
    """

    xor_vs_previous = lay.style == "PC3" or lay.style == "EM3_CORR"
    do_measurement = lay.style != "EM3"

    round_edges = lay.round_edges(sub_round)
    moment = stim.Circuit()

    # Measure the ancillae.
    if do_measurement:
        if False and lay.style == "EM3_CORR":
            triples = []
            for e in round_edges:
                triples.append([lay.q2i[q] for q in [e.left, e.right, e.center]])
            for triple in triples:
                MPP_CORR(triple, lay.noise_model.noisy_gates["MPP_CORR"])
        moment.append_operation("M", [lay.q2i[edge.center] for edge in round_edges])
    mtrack.add_measurements(*(('1/2', e) for e in round_edges))
    # Reconstruct edge measurements using previous round if needed due to non-demo measurement.
    for e in round_edges:
        recover_value_set = [0, 1] if xor_vs_previous else [0]
        mtrack.add_group(*[Prev(('1/2', e), offset=t) for t in recover_value_set], group_key=e)

    # Multiply edge measurements along the observable's path into the observable.
    manually_accumulating_obs = lay.style != "PC3" and lay.style != "EM3_CORR"
    if manually_accumulating_obs:
        moment.append_operation(
            "OBSERVABLE_INCLUDE",
            mtrack.get_record_targets(*(
                    set(lay.obs_edges) & set(lay.round_edges(sub_round))
            )),
            lay.obs_index,
        )

    # When initializing in the X basis, the first subround edge measurements are deterministic.
    if sub_round == 0 and lay.obs_before_sub_round(0)[0] == 'X':
        for e in lay.round_edges(0):
            mtrack.append_detector(e, out_circuit=moment, coords=(e.center.real, e.center.imag, 0))

    # Edges from this round form half of the edges for the spoke-center hexes from last round.
    for h in lay.round_hex_centers((sub_round - 1) % 3):
        mtrack.add_group(*lay.first_edges_around_hex(h), group_key=('1/2', h))
    # Edges from this round complete the stabilizer for the spoke-center hexes from two rounds ago.
    for h in lay.round_hex_centers((sub_round - 2) % 3):
        mtrack.add_group(*lay.second_edges_around_hex(h), ('1/2', h), group_key=h)
        mtrack.append_detector(h, Prev(h), out_circuit=moment, coords=[h.real, h.imag, 0])
    moment.append_operation("SHIFT_COORDS", [], [0, 0, 1])

    return moment
