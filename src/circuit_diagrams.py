import math
from typing import Iterator, List, Dict, Any, Tuple

import stim


def _iter_circuit(c: stim.Circuit, reps: int, nested: int) -> Iterator[stim.CircuitInstruction]:
    for _ in range(reps):
        for op in c:
            if isinstance(op, stim.CircuitRepeatBlock):
                yield from _iter_circuit(op.body_copy(), op.repeat_count if nested <= 0 else 1, nested - 1)
            elif isinstance(op, stim.CircuitInstruction):
                if nested <= 0 or op.name == "QUBIT_COORDS":
                    yield op
            else:
                raise NotImplementedError()


def diagram_2d(values: Dict[complex, Any]):
    assert all(v.real == int(v.real) for v in values)
    assert all(v.imag == int(v.imag) for v in values)
    assert all(v.real >= 0 and v.imag >= 0 for v in values)
    w = int(max((v.real for v in values), default=0) + 1)
    h = int(max((v.imag for v in values), default=0) + 1)
    s = ""
    for y in range(h):
        line = ""
        for x in range(w):
            c = str(values.get(x + y * 1j, " "))
            for k in range(len(c) - 1):
                assert line.endswith(" ")
                line = line[:-1]
            line += c
        s += line.rstrip() + "\n"
    return s.rstrip()


def diagram_3d(values: List[Dict[complex, Any]], connector_levels: List[List[Tuple[complex, complex]]]):
    min_r = min(c.real for lvl in values for c in lvl.keys())
    max_r = max(c.real for lvl in values for c in lvl.keys())
    min_i = min(c.imag for lvl in values for c in lvl.keys())
    max_i = max(c.imag for lvl in values for c in lvl.keys())
    x_span = 1j * (len(values) + 2)
    y_span = 1 * (len(values) + 3)
    z_span = (1 + 1j) * 1

    def round_complex(c: complex) -> complex:
        return math.ceil(c.real) + math.ceil(c.imag) * 1j

    def plot(xy: complex, z: int) -> complex:
        return round_complex((xy.real + 1) * x_span + (xy.imag + 2) * y_span + z * z_span)

    plotted_connectors = []
    plotted_values = {
        plot(xy, z): c
        for z, vs in enumerate(values)
        for xy, c in vs.items()
    }
    for z, (lvl, vs) in enumerate(zip(connector_levels, values)):
        for a, b in lvl:
            v1 = min(a.real, b.real)
            v2 = max(b.real, a.real)
            w1 = min(a.imag, b.imag)
            w2 = max(b.imag, a.imag)
            if v1 == min_r and v2 == max_r:
                v1 += a.imag * 1j
                v2 += b.imag * 1j
                plotted_connectors.append((plot(v1, z), plot(v1 - 0.5, z)))
                plotted_connectors.append((plot(v2, z), plot(v2 + 0.5, z)))
                plotted_values[plot(v1 - 0.5, z)] = '~'
                plotted_values[plot(v2 + 0.5, z)] = '~'
            elif w1 == min_i and w2 == max_i:
                w1 *= 1j
                w2 *= 1j
                w1 += a.real
                w2 += b.real
                plotted_connectors.append((plot(w1, z), plot(w1 - 0.5j, z)))
                plotted_connectors.append((plot(w2, z), plot(w2 + 0.5j, z)))
                plotted_values[plot(w1 - 0.5j, z)] = '~'
                plotted_values[plot(w2 + 0.5j, z)] = '~'
            else:
                plotted_connectors.append((plot(a, z), plot(b, z)))

    projected = {}
    for p1, p2 in plotted_connectors:
        if p1.real == p2.real:
            r = p1.real
            i1, i2 = int(p1.imag), int(p2.imag)
            i1, i2 = min(i1, i2), max(i1, i2) + 1
            for k in range(i1, i2):
                key = r + k*1j
                if projected.get(key, " ") in "-+":
                    projected[key] = "+"
                else:
                    projected[key] = "|"
        if p1.imag == p2.imag:
            r = p1.imag
            i1, i2 = int(p1.real), int(p2.real)
            i1, i2 = min(i1, i2), max(i1, i2) + 1
            for k in range(i1, i2):
                key = r*1j + k
                if projected.get(key, " ") in "|+":
                    projected[key] = "+"
                elif projected.get(key, " ") in "-":
                    projected[key] = "="
                else:
                    projected[key] = "-"

    for p, c in plotted_values.items():
        projected[p] = c

    return diagram_2d(projected)


def plot_circuit(c: stim.Circuit, only_repeat_block: bool):
    i2q: Dict[int, complex] = {}
    seen_coords = set()
    levels: List[Dict[complex, str]] = []
    connectors: List[List[Tuple[complex, complex]]] = []
    cur_level: Dict[complex, str] = {}
    cur_connectors: List[Tuple[complex, complex]] = []
    for op in _iter_circuit(c, 1, only_repeat_block):
        if op.name == "QUBIT_COORDS":
            for t in op.targets_copy():
                q = t.value
                r, i = op.gate_args_copy()
                p = r + i * 2j
                assert q not in i2q
                i2q[q] = p
                assert p not in seen_coords
                seen_coords.add(p)
        elif op.name in ["M", "R", "H", "H_YZ", "C_XYZ", "MR"]:
            s = op.name
            if op.name == "H_YZ":
                s = "G"
            if op.name == "C_XYZ":
                s = "C"
            if op.name == "MR":
                s = "D"
            for t in op.targets_copy():
                q = i2q[t.value]
                assert q not in cur_level
                cur_level[q] = s
        elif op.name in ["CX", "XCX", "YCX"]:
            qs = [t.value for t in op.targets_copy()]
            for k in range(0, len(qs), 2):
                q1 = i2q[qs[k]]
                q2 = i2q[qs[k+1]]
                assert q1 not in cur_level
                cur_level[q1] = "@" if len(op.name) == 2 else op.name[0]
                assert q2 not in cur_level
                cur_level[q2] = "X"
                cur_connectors.append((q1, q2))
        elif op.name == "TICK":
            levels.append(cur_level)
            connectors.append(cur_connectors)
            cur_level = {}
            cur_connectors = []
        elif op.name in ["DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"]:
            pass
        elif op.name in ["X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"]:
            pass
        else:
            raise NotImplementedError(op.name)
    if cur_level:
        levels.append(cur_level)
        connectors.append(cur_connectors)
    for level in levels:
        for q in i2q.values():
            if q not in level:
                level[q] = "\\"
    return diagram_3d(levels, connectors)
