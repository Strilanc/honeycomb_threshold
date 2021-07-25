from typing import Dict, Any

from honeycomb_layout import HoneycombLayout, Edge


def diagram_2d(values: Dict[complex, Any]):
    assert all(v.real == int(v.real) for v in values)
    assert all(v.imag == int(v.imag) for v in values)
    assert all(v.real >= 0 and v.imag >= 0 for v in values)
    w = int(max((v.real for v in values), default=0) + 1)
    h = int(max((v.imag for v in values), default=0) + 1)
    s = ""
    for y in range(h):
        line = "    "
        for x in range(w):
            c = str(values.get(x + y * 1j, " "))
            for k in range(len(c) - 1):
                assert line.endswith(" ")
                line = line[:-1]
            line += c
        s += line.rstrip() + "\n"
    return s.rstrip()


def test_ctx():
    ctx1 = HoneycombLayout(tile_diam=1, sub_rounds=18, noise=0.001, style="6step_cnot")
    ctx2 = HoneycombLayout(tile_diam=2, sub_rounds=18, noise=0.001, style="6step_cnot")
    ctx3 = HoneycombLayout(tile_diam=3, sub_rounds=18, noise=0.001, style="6step_cnot")

    assert ctx1.wrap(-1) == 3
    assert ctx1.wrap(-1j) == 5j
    assert ctx3.wrap(-1) == 11
    assert ctx3.wrap(-1j) == 17j
    assert ctx1.first_edges_around_hex(0) == (
        Edge(left=1, right=1 + 5j, center=1 + 5.5j),
        Edge(left=1 + 1j, right=3 + 1j, center=1j),
        Edge(left=3, right=3 + 5j, center=3 + 5.5j),
    )
    assert ctx1.second_edges_around_hex(0) == (
        Edge(left=3 + 5j, right=1 + 5j, center=5j),
        Edge(left=1, right=1 + 1j, center=1 + 0.5j),
        Edge(left=3, right=3 + 1j, center=3 + 0.5j),
    )
    assert ctx1.all_edges_around_hex(0) == ctx1.first_edges_around_hex(0) + ctx1.second_edges_around_hex(0)
    assert ctx1.qubits_around_hex(0) == (
        1,
        1 + 1j,
        1 + 5j,
        3,
        3 + 1j,
        3 + 5j,
    )
    assert ctx1.data_qubit_coords == (
        1,
        1 + 1j,
        1 + 2j,
        1 + 3j,
        1 + 4j,
        1 + 5j,
        3,
        3 + 1j,
        3 + 2j,
        3 + 3j,
        3 + 4j,
        3 + 5j,
    )
    assert ctx1.measure_qubit_coords == (
        1j,
        3j,
        5j,
        1 + 0.5j,
        1 + 1.5j,
        1 + 2.5j,
        1 + 3.5j,
        1 + 4.5j,
        1 + 5.5j,
        2 + 0j,
        2 + 2j,
        2 + 4j,
        3 + 0.5j,
        3 + 1.5j,
        3 + 2.5j,
        3 + 3.5j,
        3 + 4.5j,
        3 + 5.5j,
    )
    assert len(ctx3.data_qubit_coords) == len(ctx1.data_qubit_coords) * 3 * 3
    assert len(ctx3.measure_qubit_coords) == len(ctx1.measure_qubit_coords) * 3 * 3
    assert len(ctx1.data_qubit_indices) == len(ctx1.data_qubit_coords)
    assert len(ctx1.q2i) == 12 * 2.5
    assert ctx1.qubit_indices_except([2, 3, 4]) == [0, 1, *range(5, int(12 * 2.5))]
    assert ctx1.coord_width == 4
    assert ctx1.coord_height == 6
    assert ctx3.coord_width == 12
    assert ctx3.coord_height == 18
    assert ctx1.round_hex_centers(0) == (0, 2 + 3j)
    assert ctx1.round_hex_centers(1) == (4j, 2 + 1j)
    assert ctx1.round_hex_centers(2) == (2j, 2 + 5j)
    assert ctx2.round_hex_centers(0) == (0, 0 + 6j, 2 + 3j, 2 + 9j, 4, 4 + 6j, 6 + 3j, 6 + 9j)
    assert ctx1.obs_1_qubits == (1, 1 + 1j, 1 + 2j, 1 + 3j, 1 + 4j, 1 + 5j)
    assert ctx1.obs_1_edges == (
        Edge(left=1 + 0j, right=1 + 1j, center=1 + 0.5j),
        Edge(left=1 + 1j, right=1 + 2j, center=1 + 1.5j),
        Edge(left=1 + 2j, right=1 + 3j, center=1 + 2.5j),
        Edge(left=1 + 3j, right=1 + 4j, center=1 + 3.5j),
        Edge(left=1 + 4j, right=1 + 5j, center=1 + 4.5j),
        Edge(left=1 + 5j, right=1 + 0j, center=1 + 5.5j),
    )
    assert ctx1.round_edges(0) == (
        Edge(left=1 + 3j, right=3 + 3j, center=0 + 3j),
        Edge(left=1 + 1j, right=1 + 2j, center=1 + 1.5j),
        Edge(left=1 + 4j, right=1 + 5j, center=1 + 4.5j),
        Edge(left=1, right=3, center=2),
        Edge(left=3 + 1j, right=3 + 2j, center=3 + 1.5j),
        Edge(left=3 + 4j, right=3 + 5j, center=3 + 4.5j),
    )
    assert ctx1.round_edges(1) == (
        Edge(left=1 + 1j, right=3 + 1j, center=0 + 1j),
        Edge(left=1 + 2j, right=1 + 3j, center=1 + 2.5j),
        Edge(left=1 + 5j, right=1 + 0j, center=1 + 5.5j),
        Edge(left=1 + 4j, right=3 + 4j, center=2 + 4j),
        Edge(left=3 + 2j, right=3 + 3j, center=3 + 2.5j),
        Edge(left=3 + 5j, right=3 + 0j, center=3 + 5.5j),
    )
    assert len(ctx1.all_edges) == len(ctx1.round_edges(0)) * 3 == len(ctx1.round_edges(2)) * 3


def test_ctx_layout():
    ctx = HoneycombLayout(tile_diam=2, sub_rounds=20, noise=0.001, style="6step_cnot")
    def scale(pt: complex) -> complex:
        return pt.real * 8 + pt.imag * 2j
    d = {}
    for q in ctx.data_qubit_coords:
        d[scale(q)] = "q"
    for q in ctx.measure_qubit_coords:
        d[scale(q)] = "m"
    for k, v in ctx._hex_center_categories.items():
        d[scale(k)] = "XYZ"[v]
    for r in range(6):
        p, qs = ctx.obs_1_before_sub_round(r)
        for q in qs:
            d[scale(q) + r + 1] = p
    assert diagram_2d(d).strip() == """
    X       q  YXXZ m       q       X       q       m       q
            m               m               m               m
    m       qZYYX   Y       q       m       q       Y       q
            m               m               m               m
    Z       qZY  XZ m       q       Z       q       m       q
            m               m               m               m
    m       q  YXXZ X       q       m       q       X       q
            m               m               m               m
    Y       qZYYX   m       q       Y       q       m       q
            m               m               m               m
    m       qZY  XZ Z       q       m       q       Z       q
            m               m               m               m
    X       q  YXXZ m       q       X       q       m       q
            m               m               m               m
    m       qZYYX   Y       q       m       q       Y       q
            m               m               m               m
    Z       qZY  XZ m       q       Z       q       m       q
            m               m               m               m
    m       q  YXXZ X       q       m       q       X       q
            m               m               m               m
    Y       qZYYX   m       q       Y       q       m       q
            m               m               m               m
    m       qZY  XZ Z       q       m       q       Z       q
            m               m               m               m
      """.strip()


def test_ctx_indexing():
    ctx = HoneycombLayout(tile_diam=1, sub_rounds=20, noise=0.001, style="6step_cnot")
    def scale(pt: complex) -> complex:
        return pt.real * 4 + pt.imag * 2j
    d = {}
    for q, i in ctx.q2i.items():
        d[scale(q)] = str(i)
    for k, v in ctx._hex_center_categories.items():
        d[scale(k)] = "XYZ"[v]
    assert diagram_2d(d).strip() == """
    X   0  21   6
       15      24
   12   1   Y   7
       16      25
    Z   2  22   8
       17      26
   13   3   X   9
       18      27
    Y   4  23  10
       19      28
   14   5   Z  11
       20      29
        """.strip()
