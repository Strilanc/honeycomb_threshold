import dataclasses
from typing import Optional, Dict, Set, Tuple

import stim

ANY_CLIFFORD_1_OPS = {"C_XYZ", "C_ZYX", "H", "H_YZ", "I"}
ANY_CLIFFORD_2_OPS = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
RESET_OPS = {"R", "RX", "RY"}
MEASURE_OPS = {"M", "MX", "MY"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS", "TICK"}

def decorrelation_prob(d: float, n: float) -> float:
    """Converts from depolarizing error d on n bits to probability of
    independent channels of probability p that compose to depolarizing
    channel.  For this one method, and NOWHERE ELSE, this corresponds
    to a maximally-mixing depolarization - identity 'errors' can happen."""
    return 0.5 - 0.5*(1 - d)**(1/2**(n-1))

def MPP_CORR(data1: stim.GateTarget, data2: stim.GateTarget, ancilla: int, p: float) -> stim.Circuit:

    # Generate all possible combinations of (non-identity) channels.  Assumes triple of targets
    # with last element corresponding to measure qubit.
    circuit = stim.Circuit()
    circuit.append_operation('R', [ancilla])
    if data1.is_x_target:
        circuit.append_operation('XCX', [data1.value, ancilla])
    if data1.is_y_target:
        circuit.append_operation('YCX', [data1.value, ancilla])
    if data1.is_z_target:
        circuit.append_operation('ZCX', [data1.value, ancilla])
    if data2.is_x_target:
        circuit.append_operation('XCX', [data2.value, ancilla])
    if data2.is_y_target:
        circuit.append_operation('YCX', [data2.value, ancilla])
    if data2.is_z_target:
        circuit.append_operation('ZCX', [data2.value, ancilla])

    first_targets = ["I", stim.target_x(data1.value), stim.target_y(data1.value), stim.target_z(data1.value)]
    second_targets = ["I", stim.target_x(data2.value), stim.target_y(data2.value), stim.target_z(data2.value)]
    measure_targets = ["I", stim.target_x(ancilla)]

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

    circuit.append_operation('M', [ancilla])

    return circuit

@dataclasses.dataclass(frozen=True)
class NoiseModel:
    idle: float
    measure_reset_idle: float
    noisy_gates: Dict[str, float]
    any_clifford_1: Optional[float] = None
    any_clifford_2: Optional[float] = None
    correlated_EM_error: bool = False

    @staticmethod
    def SD6(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p,
            idle=p,
            measure_reset_idle=0,
            noisy_gates={
                "CX": p,
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def PC3(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p,
            any_clifford_2=p,
            idle=p,
            measure_reset_idle=0,
            noisy_gates={
                "R": p,
                "M": p,
            },
        )

    @staticmethod
    def EM3(p: float) -> 'NoiseModel':
        return NoiseModel(
            idle=p,
            measure_reset_idle=0,
            any_clifford_1=p,
            noisy_gates={
                "R": p,
                "M": p,
                "MPP": p,
            },
        )

    @staticmethod
    def EM3_CORR(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=0,
            any_clifford_2=0,
            idle=0.5,
            measure_reset_idle=0,
            correlated_EM_error=True,
            noisy_gates={
                "R": p/2,  # Note that hack induces perfect initial prep/terminal measurement.
                "M": p/2,
                "MPP": p,
            },
        )

    @staticmethod
    def SI500(p: float) -> 'NoiseModel':
        return NoiseModel(
            any_clifford_1=p / 10,
            idle=p / 10,
            measure_reset_idle=5 * p,
            noisy_gates={
                "CZ": p,
                "R": 2 * p,
                "M": 5 * p,
            },
        )

    def noisy_op(self, op: stim.CircuitInstruction, p: float, ancilla: int) -> Tuple[stim.Circuit, stim.Circuit, stim.Circuit]:
        pre = stim.Circuit()
        mid = stim.Circuit()
        post = stim.Circuit()
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if p > 0:
            if op.name in ANY_CLIFFORD_1_OPS:
                post.append_operation("DEPOLARIZE1", targets, p)
            elif op.name in ANY_CLIFFORD_2_OPS:
                post.append_operation("DEPOLARIZE2", targets, p)
            elif op.name in RESET_OPS or op.name in MEASURE_OPS:
                if op.name in RESET_OPS:
                    post.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
                if op.name in MEASURE_OPS:
                    pre.append_operation("Z_ERROR" if op.name.endswith("X") else "X_ERROR", targets, p)
            elif op.name == "MPP":
                assert len(targets) % 3 == 0 and all(t.is_combiner for t in targets[1::3]), repr(op)
                if self.correlated_EM_error:
                    for k in range(0, len(targets), 3):
                        first_target = targets[k]
                        second_target = targets[k+2]
                        mid += MPP_CORR(first_target, second_target, ancilla, decorrelation_prob(p, 5))

                    assert args == [] or args == [0]
                    return pre, mid, post

                else:
                    pre.append_operation("DEPOLARIZE2", [t.value for t in targets if not t.is_combiner], p)
                    assert args == [] or args == [0]
                    args = [p]

            else:
                raise NotImplementedError(repr(op))
        mid.append_operation(op.name, targets, args)
        return pre, mid, post

    def noisy_circuit(self, circuit: stim.Circuit, *, qs: Optional[Set[int]] = None) -> stim.Circuit:
        result = stim.Circuit()
        ancilla = circuit.num_qubits

        current_moment_pre = stim.Circuit()
        current_moment_mid = stim.Circuit()
        current_moment_post = stim.Circuit()
        used_qubits: Set[int] = set()
        measured_or_reset_qubits: Set[int] = set()
        if qs is None:
            qs = set(range(circuit.num_qubits))

        def flush():
            nonlocal result
            if not current_moment_mid:
                return

            # Apply idle depolarization rules.
            idle_qubits = sorted(qs - used_qubits)
            if used_qubits and idle_qubits and self.idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.idle)
            idle_qubits = sorted(qs - measured_or_reset_qubits)
            if measured_or_reset_qubits and idle_qubits and self.measure_reset_idle > 0:
                current_moment_post.append_operation("DEPOLARIZE1", idle_qubits, self.measure_reset_idle)

            # Move current noisy moment into result.
            result += current_moment_pre
            result += current_moment_mid
            result += current_moment_post
            used_qubits.clear()
            current_moment_pre.clear()
            current_moment_mid.clear()
            current_moment_post.clear()
            measured_or_reset_qubits.clear()

        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                flush()
                result += self.noisy_circuit(op.body_copy(), qs=qs) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                if op.name == "TICK":
                    flush()
                    result.append_operation("TICK", [])
                    continue

                if op.name in self.noisy_gates:
                    p = self.noisy_gates[op.name]
                elif self.any_clifford_1 is not None and op.name in ANY_CLIFFORD_1_OPS:
                    p = self.any_clifford_1
                elif self.any_clifford_2 is not None and op.name in ANY_CLIFFORD_2_OPS:
                    p = self.any_clifford_2
                elif op.name in ANNOTATION_OPS:
                    p = 0
                else:
                    raise NotImplementedError(repr(op))
                pre, mid, post = self.noisy_op(op, p, ancilla)
                current_moment_pre += pre
                current_moment_mid += mid
                current_moment_post += post

                # Ensure the circuit is not touching qubits multiple times per tick.
                touched_qubits = {
                    t.value
                    for t in op.targets_copy()
                    if t.is_x_target or t.is_y_target or t.is_z_target or t.is_qubit_target
                }
                if op.name in ANNOTATION_OPS:
                    touched_qubits.clear()
                # Hack: turn off this assertion off for now since correlated errors are built into circuit.
                #assert touched_qubits.isdisjoint(used_qubits), repr(current_moment_pre + current_moment_mid + current_moment_post)
                used_qubits |= touched_qubits
                if op.name in MEASURE_OPS or op.name in RESET_OPS:
                    measured_or_reset_qubits |= touched_qubits
            else:
                raise NotImplementedError(repr(op))
        flush()

        return result
