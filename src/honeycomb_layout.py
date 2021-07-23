"""This file contains data and utility methods for working with the honeycomb layout."""

import functools
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple


def _data_qubit_parity(q: complex) -> bool:
    """To optimally interleave operations, it's useful to split into a checkerboard pattern."""
    return (q.real // 2 + q.imag) % 2 != 0


class Edge:
    def __init__(self, *, left: complex, right: complex, center: complex):
        if (_data_qubit_parity(left), left.real, left.imag) > (_data_qubit_parity(right), right.real, right.imag):
            left, right = right, left
        self.left = left
        self.right = right
        self.center = center

    def __repr__(self):
        return f"Edge(left={self.left!r}, right={self.right!r}, center={self.center!r})"

    def _key(self):
        return self.left, self.right, self.center

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self._key() == other._key()

    def __ne__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self._key() != other._key()

    def __hash__(self):
        return hash(self._key())


@dataclass
class EdgeType:
    hex_to_hex_delta: complex
    hex_to_qubit_delta: complex

    def hex_to_edge(self, center: complex, config: 'HoneycombLayout') -> Edge:
        a = center + self.hex_to_qubit_delta
        b = center + self.hex_to_hex_delta - self.hex_to_qubit_delta
        return Edge(left=config.wrap(a),
                    right=config.wrap(b),
                    center=config.wrap((a + b) / 2))



EDGE_TYPES = [
    EdgeType(hex_to_hex_delta=2 - 3j, hex_to_qubit_delta=1 - 1j),
    EdgeType(hex_to_hex_delta=2 + 3j, hex_to_qubit_delta=1 + 1j),
    EdgeType(hex_to_hex_delta=4, hex_to_qubit_delta=1),
]

FIRST_EDGES_AROUND_HEX: List[Tuple[complex, complex]] = [
    (+1 - 1j, +1),  # Top right.
    (+1 + 1j, -1 + 1j),  # Bottom.
    (-1, -1 - 1j),  # Top left.
]

SECOND_EDGES_AROUND_HEX: List[Tuple[complex, complex]] = [
    (-1 - 1j, +1 - 1j),  # Top.
    (+1, +1 + 1j),  # Bottom right.
    (-1 + 1j, -1),  # Bottom left.
]


class HoneycombLayout:
    def __init__(self, tile_diam: int, sub_rounds: int, noise: float):
        self.tile_diam = tile_diam
        self.sub_rounds = sub_rounds
        self.noise = noise

    def wrap(self, c: complex) -> complex:
        r = c.real % self.coord_width
        i = c.imag % self.coord_height
        return r + i * 1j

    def first_edges_around_hex(self, center: complex) -> Tuple[Edge, ...]:
        offsets = [
            (+1 - 1j, +1),  # Top right.
            (+1 + 1j, -1 + 1j),  # Bottom.
            (-1, -1 - 1j),  # Top left.
        ]
        return tuple(
            Edge(
                left=self.wrap(center + a),
                right=self.wrap(center + b),
                center=self.wrap(center + (a + b) / 2),
            )
            for a, b in offsets
        )

    def sub_round_edge_basis(self, sub_round: int) -> str:
        return "XYZ"[sub_round % 3]

    def second_edges_around_hex(self, center: complex) -> Tuple[Edge, ...]:
        offsets = [
            (-1 - 1j, +1 - 1j),  # Top.
            (+1, +1 + 1j),  # Bottom right.
            (-1 + 1j, -1),  # Bottom left.
        ]
        return tuple(
            Edge(
                left=self.wrap(center + a),
                right=self.wrap(center + b),
                center=self.wrap(center + (a + b) / 2),
            )
            for a, b in offsets
        )

    def all_edges_around_hex(self, center: complex) -> Tuple[Edge, ...]:
        return self.first_edges_around_hex(center) + self.second_edges_around_hex(center)

    def qubits_around_hex(self, center: complex) -> Tuple[complex, ...]:
        return tuple(sorted_complex(
            self.wrap(center + edge_type.hex_to_qubit_delta * sign)
            for edge_type in EDGE_TYPES
            for sign in [-1, +1]
        ))

    def obs_1_before_sub_round(self, sub_round: int) -> Tuple[str, List[complex]]:
        case = sub_round % 6
        if case == 0:
            obs_pattern = "_ZZ"
        elif case == 1:
            obs_pattern = "_YY"
        elif case == 2:
            obs_pattern = "YY_"
        elif case == 3:
            obs_pattern = "XX_"
        elif case == 4:
            obs_pattern = "X_X"
        else:
            obs_pattern = "Z_Z"
        c, = set(obs_pattern) - {'_'}
        return c, [
            q
            for c, q in zip(obs_pattern * self.tile_diam * 2, self.obs_1_qubits)
            if c != "_"
        ]

    @functools.cached_property
    def data_qubit_indices_1st(self) -> Tuple[int, ...]:
        return tuple(self.q2i[q] for q in self.data_qubit_coords if not _data_qubit_parity(q))

    @functools.cached_property
    def data_qubit_indices_2nd(self) -> Tuple[int, ...]:
        return tuple(self.q2i[q] for q in self.data_qubit_coords if _data_qubit_parity(q))

    @functools.cached_property
    def data_qubit_coords(self) -> Tuple[complex, ...]:
        return tuple(sorted_complex({
            q
            for e in self.all_edges
            for q in [e.left, e.right]
        }))

    @functools.cached_property
    def measure_qubit_coords(self) -> Tuple[complex, ...]:
        """Find all the qubit positions around the hexes."""
        return tuple(sorted_complex({
            e.center
            for e in self.all_edges
        }))

    @functools.cached_property
    def data_qubit_indices(self) -> Tuple[int, ...]:
        return tuple(self.q2i[q] for q in self.data_qubit_coords)

    @functools.cached_property
    def q2i(self) -> Dict[complex, int]:
        return {
            q: i
            for i, q in enumerate(sorted_complex(
                self.data_qubit_coords + self.measure_qubit_coords))
        }

    def qubit_indices_except(self, indices: Iterable[int]) -> List[int]:
        return sorted(set(self.q2i.values()) - set(indices))

    @functools.cached_property
    def coord_width(self) -> float:
        return 4.0 * self.tile_diam

    @functools.cached_property
    def coord_height(self) -> float:
        return 6.0 * self.tile_diam

    @functools.lru_cache(maxsize=3)
    def round_hex_centers(self, r: int) -> Tuple[complex, ...]:
        assert 0 <= r < 3
        return tuple(sorted_complex(
            h
            for h, category in self._hex_center_categories.items()
            if category == r
        ))

    @functools.cached_property
    def obs_1_edges(self) -> Tuple[Edge]:
        return tuple(sorted([
            e
            for e in self.all_edges
            if e.left.real == e.right.real == 1
        ], key=lambda e: (e.center.real, e.center.imag)))

    @functools.cached_property
    def obs_1_qubits(self) -> Tuple[complex]:
        return tuple(sorted_complex(
            q
            for q in self.data_qubit_coords
            if q.real == 1
        ))

    @functools.cached_property
    def all_edges(self) -> Tuple[Edge, ...]:
        return self.round_edges(0) + self.round_edges(1) + self.round_edges(2)

    @functools.lru_cache(maxsize=3)
    def round_edges(self, r: int) -> Tuple[Edge, ...]:
        r %= 3
        return tuple(sorted([
            edge_type.hex_to_edge(h, self)
            for h in self.round_hex_centers(r)
            for edge_type in EDGE_TYPES
        ], key=lambda e: (e.center.real, e.center.imag)))

    @functools.cached_property
    def _hex_center_categories(self) -> Dict[complex, int]:
        """Generate and categorize the hexes defining the circuit."""
        result: Dict[complex, int] = {}
        for row in range(3 * self.tile_diam):
            for col in range(2 * self.tile_diam):
                center = row * 2j + 2 * col - 1j * (col % 2)
                category = (-row - col % 2) % 3
                result[self.wrap(center)] = category
        return result


def sorted_complex(xs: Iterable[complex]) -> List[complex]:
    return sorted(xs, key=lambda v: (v.real, v.imag))
