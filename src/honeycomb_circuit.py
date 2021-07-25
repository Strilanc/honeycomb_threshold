from typing import List, Sequence, Tuple

import stim

from honeycomb_layout import HoneycombLayout
from measure_tracker import MeasurementTracker, Prev


TWO_QUBIT_OPS = {"CX", "XCX", "YCX"}
ONE_QUBIT_OPS = {"C_XYZ", "H", "H_YZ"}
RESET_OPS = {"R", "MR"}
MEASURE_OPS = {"M", "MR"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS"}
STABILIZER_OPS = RESET_OPS | MEASURE_OPS | ONE_QUBIT_OPS | TWO_QUBIT_OPS

def fuse_moments(*, lay: HoneycombLayout, moments: List[stim.Circuit]) -> stim.Circuit:
    result = stim.Circuit()
    skip_tick = True
    for moment_circuit in moments:
        if skip_tick:
            skip_tick = False
        else:
            result.append_operation("TICK", [])

        if len(moment_circuit) == 1 and isinstance(moment_circuit[0], stim.CircuitRepeatBlock):
            # HACK: Repeated blocks should already be annotated.
            result += moment_circuit
            skip_tick = True
            continue

        idle = set(lay.q2i.values())
        pre = stim.Circuit()
        post = stim.Circuit()
        for op in moment_circuit:
            if not isinstance(op, stim.CircuitInstruction):
                raise NotImplementedError(str(op))
            targets = []
            handled = False
            if op.name in STABILIZER_OPS:
                targets = [t.value for t in op.targets_copy()]
                for t in targets:
                    idle.remove(t)
                handled = True
            if op.name in RESET_OPS:
                post.append_operation("X_ERROR", targets, lay.noise)
                handled = True
            if op.name in MEASURE_OPS:
                pre.append_operation("X_ERROR", targets, lay.noise)
                handled = True
            if op.name in ONE_QUBIT_OPS:
                post.append_operation("DEPOLARIZE1", targets, lay.noise)
                handled = True
            if op.name in TWO_QUBIT_OPS:
                post.append_operation("DEPOLARIZE2", targets, lay.noise)
                handled = True
            if op.name in ANNOTATION_OPS:
                handled = True
            if not handled:
                raise NotImplementedError(op.name)
        result += pre
        result += moment_circuit
        result += post
        if idle:
            result.append_operation("DEPOLARIZE1", sorted(idle), lay.noise)
    return result


def generate_honeycomb_circuit(tile_diam: int,
                               sub_rounds: int,
                               noise: float,
                               style: str = "6",
                               ) -> stim.Circuit:
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
        style=style,
    )
    mtrack = MeasurementTracker()

    # Annotate the locations of qubits used by the circuit.
    moments = [stim.Circuit()]
    for q, i in lay.q2i.items():
        moments[-1].append_operation("QUBIT_COORDS", [i], [q.real, q.imag])

    # Initialize data.
    moments[-1].append_operation("R", sorted(lay.q2i.values()))

    # Create dummy stabilizer records to compare against during initial rounds.
    # Then run enough rounds to ensure later measurements aren't comparing to dummies anymore.
    for sub_round in range(3):
        # Z edges start in a known state.
        mtrack.add_dummies(*lay.round_edges(sub_round), obstacle=sub_round != 2)
        # Z stabilizers start in a known state.
        mtrack.add_dummies(*lay.round_hex_centers(sub_round), obstacle=sub_round != 2)
        # Y stabilizers start half formed (have Z part but not X part).
        mtrack.add_dummies(*[('1/2', h) for h in lay.round_hex_centers(sub_round)], obstacle=sub_round != 1)

    # Use a loop to advance the steady state as close as possible to the end.
    if lay.style == "6":
        moments += generate_rounds_six_cycle(lay, mtrack)
    elif lay.style == "3":
        moments += generate_rounds_three_cycle(lay, mtrack)
    else:
        raise NotImplementedError(lay.style)


    # Perform the fault tolerant data measurement.
    moments[-1].append_operation("M", lay.data_qubit_indices)
    mtrack.add_measurements(*lay.data_qubit_coords)

    obs_basis, obs_qubits = lay.obs_1_before_sub_round(sub_rounds)
    last_measure_basis = "XYZ"[(sub_rounds - 1) % 3]
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

    return fuse_moments(lay=lay, moments=moments)


def generate_rounds_six_cycle(lay: HoneycombLayout, mtrack: MeasurementTracker) -> List[stim.Circuit]:
    n = lay.sub_rounds
    moments = []
    if n == 0:
        raise NotImplementedError("Zero rounds.")
    elif n == 1:
        # For a single sub round there's no time to pipeline.
        moments += [
            _sub_round_resets(lay, 0) + _cycle_data(lay, True, True),
            *_sub_round_2q_ops(lay, 0),
            _sub_round_measurements_and_detectors(lay, mtrack, 0),
        ]
    else:
        # Get the pipeline started.
        moments += [
            _sub_round_resets(lay, 0) + _cycle_data(lay, True, False),
            _sub_round_2q_ops(lay, 0)[0],
            _sub_round_resets(lay, 1) + _cycle_data(lay, True, True),
        ]

        # Run the sub rounds in a pipelined fashion.
        next_sub_round = 0
        while next_sub_round < 4 and next_sub_round < n - 2:
            moments += _pipeline_step(lay, next_sub_round, mtrack)
            next_sub_round += 1
        iterations = max(0, n - next_sub_round - 2) // 3
        if iterations > 1:
            loop_body = fuse_moments(lay=lay, moments=[
                *_pipeline_step(lay, 1, mtrack),
                *_pipeline_step(lay, 2, mtrack),
                *_pipeline_step(lay, 0, mtrack),
            ])
            loop_body.append_operation("TICK", [])
            moments += [loop_body * iterations]
            next_sub_round += iterations * 3
        for sub_round in range(next_sub_round, n - 2):
            moments += _pipeline_step(lay, sub_round, mtrack)

        # End the pipeline.
        _, b2 = _sub_round_2q_ops(lay, n - 2)
        a3, b3 = _sub_round_2q_ops(lay, n - 1)
        moments += [
            a3 + b2,
            _sub_round_measurements_and_detectors(lay, mtrack, n - 2),
            _cycle_data(lay, False, True),
            b3,
            _sub_round_measurements_and_detectors(lay, mtrack, n - 1)
        ]

    obs_basis, obs_qubits = lay.obs_1_before_sub_round(lay.sub_rounds)
    if obs_basis != lay.sub_round_edge_basis(lay.sub_rounds - 1):
        moments[-1].append_operation("C_XYZ", lay.data_qubit_indices)
        moments.append(stim.Circuit())

    return moments


def generate_rounds_three_cycle(lay: HoneycombLayout, mtrack: MeasurementTracker) -> List[stim.Circuit]:
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
            circuit += _sub_round_measurements_and_detectors(lay, mtrack, k - 2, demolition=True)
        elif k - 2 <= 0:
            circuit += _sub_round_resets(lay, k - 2)
        moments.append(circuit)
        k += 1

        # Steady state starts on round 6 and has period 3. Once we get to 9 we've got the whole loop.
        if k == 9:
            iterations = (n - k + 3) // 3
            if iterations > 1:
                moments[-3:] = [fuse_moments(lay=lay, moments=moments[-3:]) * iterations]
                k += iterations * 3 - 3

    obs_basis, _ = lay.obs_1_before_sub_round(lay.sub_rounds)
    if obs_basis == "X":
        circuit = stim.Circuit()
        circuit.append_operation("H", lay.data_qubit_indices)
        moments.append(circuit)
    elif obs_basis == "Y":
        circuit = stim.Circuit()
        circuit.append_operation("H_YZ", lay.data_qubit_indices)
        moments.append(circuit)

    moments.append(stim.Circuit())

    return moments


def _pipeline_step(lay: HoneycombLayout, sub_round: int, mtrack: MeasurementTracker):
    _, b1 = _sub_round_2q_ops(lay, sub_round)
    a2, _ = _sub_round_2q_ops(lay, sub_round + 1)
    r3 = _sub_round_resets(lay, sub_round + 2)
    c = _cycle_data(lay, True, True)
    return [
        a2 + b1,
        r3 + c + _sub_round_measurements_and_detectors(lay, mtrack, sub_round),
    ]


def _cycle_data(lay: HoneycombLayout, first: bool, second: bool) -> stim.Circuit:
    result = stim.Circuit()
    if first and second:
        result.append_operation("C_XYZ", lay.data_qubit_indices)
    elif first:
        result.append_operation("C_XYZ", lay.data_qubit_indices_1st)
    elif second:
        result.append_operation("C_XYZ", lay.data_qubit_indices_2nd)
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
        sub_round: int) -> List[stim.Circuit]:
    round_edges = lay.round_edges(sub_round)
    moment = stim.Circuit()
    moment.append_operation("R", [lay.q2i[edge.center] for edge in round_edges])
    return moment


def _sub_round_measurements_and_detectors(
        lay: HoneycombLayout,
        mtrack: MeasurementTracker,
        sub_round: int,
        demolition: bool = False) -> List[stim.Circuit]:
    """Returns a circuit performing one layer of edge measurements from the honeycomb code."""

    round_edges = lay.round_edges(sub_round)
    moment = stim.Circuit()
    op = "MR" if demolition else "M"
    moment.append_operation(op, [lay.q2i[edge.center] for edge in round_edges])
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
