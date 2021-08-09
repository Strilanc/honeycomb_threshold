import itertools

import pytest

from honeycomb_circuit import generate_honeycomb_circuit
from hack_pycharm_pybind_pytest_workaround import stim
from honeycomb_layout import HoneycombLayout


@pytest.mark.parametrize('tile_width,tile_height_extra,sub_rounds,obs,style', itertools.product(
    range(1, 5),
    [-1, 0, +1],
    range(1, 24),
    ["H", "V"],
    ["PC3", "SD6", "EM3", "SI500"],
))
def test_circuit_has_decomposing_error_model(
        tile_width: int,
        tile_height_extra: int,
        sub_rounds: int,
        obs: str,
        style: str):
    if style == "SI500" and sub_rounds % 3 != 0:
        return
    circuit = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2 * tile_width,
        data_height=6 * max(1, tile_width + tile_height_extra),
        sub_rounds=sub_rounds,
        noise=0.001,
        style=style,
        obs=obs,
    ))
    _ = circuit.detector_error_model(decompose_errors=True)


def test_circuit_details_SD6():
    actual = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2,
        data_height=6,
        sub_rounds=1003,
        noise=0.001,
        style="SD6",
        obs="V",
    ))
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(1, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(1, 2) 2
        QUBIT_COORDS(1, 3) 3
        QUBIT_COORDS(1, 4) 4
        QUBIT_COORDS(1, 5) 5
        QUBIT_COORDS(3, 0) 6
        QUBIT_COORDS(3, 1) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(3, 3) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11
        QUBIT_COORDS(0, 1) 12
        QUBIT_COORDS(0, 3) 13
        QUBIT_COORDS(0, 5) 14
        QUBIT_COORDS(1, 0.5) 15
        QUBIT_COORDS(1, 1.5) 16
        QUBIT_COORDS(1, 2.5) 17
        QUBIT_COORDS(1, 3.5) 18
        QUBIT_COORDS(1, 4.5) 19
        QUBIT_COORDS(1, 5.5) 20
        QUBIT_COORDS(2, 0) 21
        QUBIT_COORDS(2, 2) 22
        QUBIT_COORDS(2, 4) 23
        QUBIT_COORDS(3, 0.5) 24
        QUBIT_COORDS(3, 1.5) 25
        QUBIT_COORDS(3, 2.5) 26
        QUBIT_COORDS(3, 3.5) 27
        QUBIT_COORDS(3, 4.5) 28
        QUBIT_COORDS(3, 5.5) 29
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        TICK

        R 13 16 19 21 25 28
        C_ZYX 0 2 4 7 9 11
        X_ERROR(0.001) 13 16 19 21 25 28
        DEPOLARIZE1(0.001) 0 2 4 7 9 11 1 3 5 6 8 10 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        CX 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE1(0.001) 1 3 5 6 8 10 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        R 12 17 20 23 26 29
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        X_ERROR(0.001) 12 17 20 23 26 29
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 13 14 15 16 18 19 21 22 24 25 27 28
        TICK

        CX 7 12 2 17 0 20 4 23 9 26 11 29
        CX 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.001) 14 15 18 22 24 27
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        R 14 15 18 22 24 27
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        M 13 16 19 21 25 28
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 14 15 18 22 24 27
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 17 20 23 26 29
        TICK

        CX 11 14 0 15 4 18 2 22 7 24 9 27
        CX 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE1(0.001) 13 16 19 21 25 28
        TICK

        X_ERROR(0.001) 12 17 20 23 26 29
        R 13 16 19 21 25 28
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        M 12 17 20 23 26 29
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 13 16 19 21 25 28
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 14 15 18 22 24 27
        TICK

        CX 9 13 2 16 4 19 0 21 7 25 11 28
        CX 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE1(0.001) 12 17 20 23 26 29
        TICK

        X_ERROR(0.001) 14 15 18 22 24 27
        R 12 17 20 23 26 29
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        M 14 15 18 22 24 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 12 17 20 23 26 29
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 13 16 19 21 25 28
        TICK

        CX 7 12 2 17 0 20 4 23 9 26 11 29
        CX 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.001) 14 15 18 22 24 27
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        R 14 15 18 22 24 27
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        M 13 16 19 21 25 28
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 14 15 18 22 24 27
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 17 20 23 26 29
        TICK

        REPEAT 332 {
            CX 11 14 0 15 4 18 2 22 7 24 9 27
            CX 1 12 3 17 5 20 10 23 8 26 6 29
            DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
            DEPOLARIZE1(0.001) 13 16 19 21 25 28
            TICK

            X_ERROR(0.001) 12 17 20 23 26 29
            R 13 16 19 21 25 28
            C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
            M 12 17 20 23 26 29
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            X_ERROR(0.001) 13 16 19 21 25 28
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 14 15 18 22 24 27
            TICK

            CX 9 13 2 16 4 19 0 21 7 25 11 28
            CX 5 14 1 15 3 18 8 22 6 24 10 27
            DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
            DEPOLARIZE1(0.001) 12 17 20 23 26 29
            TICK

            X_ERROR(0.001) 14 15 18 22 24 27
            R 12 17 20 23 26 29
            C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
            M 14 15 18 22 24 27
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            X_ERROR(0.001) 12 17 20 23 26 29
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 13 16 19 21 25 28
            TICK

            CX 7 12 2 17 0 20 4 23 9 26 11 29
            CX 3 13 1 16 5 19 6 21 8 25 10 28
            DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
            DEPOLARIZE1(0.001) 14 15 18 22 24 27
            TICK

            X_ERROR(0.001) 13 16 19 21 25 28
            R 14 15 18 22 24 27
            C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
            M 13 16 19 21 25 28
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            X_ERROR(0.001) 14 15 18 22 24 27
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 17 20 23 26 29
            TICK
        }

        CX 11 14 0 15 4 18 2 22 7 24 9 27
        CX 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE1(0.001) 13 16 19 21 25 28
        TICK

        X_ERROR(0.001) 12 17 20 23 26 29
        R 13 16 19 21 25 28
        C_ZYX 0 1 2 3 4 5 6 7 8 9 10 11
        M 12 17 20 23 26 29
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 13 16 19 21 25 28
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 14 15 18 22 24 27
        TICK

        CX 9 13 2 16 4 19 0 21 7 25 11 28
        CX 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE1(0.001) 12 17 20 23 26 29
        TICK

        X_ERROR(0.001) 14 15 18 22 24 27
        M 14 15 18 22 24 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 16 17 19 20 21 23 25 26 28 29
        TICK

        C_ZYX 1 3 5 6 8 10
        DEPOLARIZE1(0.001) 1 3 5 6 8 10 0 2 4 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        TICK

        CX 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE2(0.001) 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.001) 0 2 4 7 9 11 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        M 13 16 19 21 25 28
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        H_YZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        TICK

        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 2, 0) rec[-36] rec[-35] rec[-32] rec[-30] rec[-29] rec[-26] rec[-18] rec[-17] rec[-14] rec[-11] rec[-10] rec[-9] rec[-5] rec[-4] rec[-3]
        DETECTOR(2, 5, 0) rec[-34] rec[-33] rec[-31] rec[-28] rec[-27] rec[-25] rec[-16] rec[-15] rec[-13] rec[-12] rec[-8] rec[-7] rec[-6] rec[-2] rec[-1]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-18] rec[-16] rec[-13] rec[-9] rec[-8] rec[-7] rec[-3] rec[-2] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-17] rec[-15] rec[-14] rec[-12] rec[-11] rec[-10] rec[-6] rec[-5] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-8] rec[-7]
        DEPOLARIZE1(0.001) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
""")


def test_circuit_details_PC3():
    actual = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2,
        data_height=6,
        sub_rounds=1003,
        noise=0.001,
        style="PC3",
        obs="V",
    ))
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(1, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(1, 2) 2
        QUBIT_COORDS(1, 3) 3
        QUBIT_COORDS(1, 4) 4
        QUBIT_COORDS(1, 5) 5
        QUBIT_COORDS(3, 0) 6
        QUBIT_COORDS(3, 1) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(3, 3) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11
        QUBIT_COORDS(0, 1) 12
        QUBIT_COORDS(0, 3) 13
        QUBIT_COORDS(0, 5) 14
        QUBIT_COORDS(1, 0.5) 15
        QUBIT_COORDS(1, 1.5) 16
        QUBIT_COORDS(1, 2.5) 17
        QUBIT_COORDS(1, 3.5) 18
        QUBIT_COORDS(1, 4.5) 19
        QUBIT_COORDS(1, 5.5) 20
        QUBIT_COORDS(2, 0) 21
        QUBIT_COORDS(2, 2) 22
        QUBIT_COORDS(2, 4) 23
        QUBIT_COORDS(3, 0.5) 24
        QUBIT_COORDS(3, 1.5) 25
        QUBIT_COORDS(3, 2.5) 26
        QUBIT_COORDS(3, 3.5) 27
        QUBIT_COORDS(3, 4.5) 28
        QUBIT_COORDS(3, 5.5) 29
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        TICK

        XCX 9 13 2 16 4 19 0 21 7 25 11 28
        R 12 17 20 23 26 29
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28
        X_ERROR(0.001) 12 17 20 23 26 29
        DEPOLARIZE1(0.001) 1 3 5 6 8 10 14 15 18 22 24 27
        TICK

        YCX 7 12 2 17 0 20 4 23 9 26 11 29
        XCX 3 13 1 16 5 19 6 21 8 25 10 28
        R 14 15 18 22 24 27
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
        X_ERROR(0.001) 14 15 18 22 24 27
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        CX 11 14 0 15 4 18 2 22 7 24 9 27
        YCX 1 12 3 17 5 20 10 23 8 26 6 29
        M 13 16 19 21 25 28
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
        TICK

        X_ERROR(0.001) 12 17 20 23 26 29
        XCX 9 13 2 16 4 19 0 21 7 25 11 28
        CX 5 14 1 15 3 18 8 22 6 24 10 27
        M 12 17 20 23 26 29
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
        TICK

        X_ERROR(0.001) 14 15 18 22 24 27
        YCX 7 12 2 17 0 20 4 23 9 26 11 29
        XCX 3 13 1 16 5 19 6 21 8 25 10 28
        M 14 15 18 22 24 27
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        CX 11 14 0 15 4 18 2 22 7 24 9 27
        YCX 1 12 3 17 5 20 10 23 8 26 6 29
        M 13 16 19 21 25 28
        DETECTOR(0, 4, 0) rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
        TICK

        # === stabilizers are now all established, but not all edge flip flops are established ===

        X_ERROR(0.001) 12 17 20 23 26 29
        XCX 9 13 2 16 4 19 0 21 7 25 11 28
        CX 5 14 1 15 3 18 8 22 6 24 10 27
        M 12 17 20 23 26 29
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
        TICK

        X_ERROR(0.001) 14 15 18 22 24 27
        YCX 7 12 2 17 0 20 4 23 9 26 11 29
        XCX 3 13 1 16 5 19 6 21 8 25 10 28
        M 14 15 18 22 24 27
        DETECTOR(0, 0, 0) rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 3, 0) rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        CX 11 14 0 15 4 18 2 22 7 24 9 27
        YCX 1 12 3 17 5 20 10 23 8 26 6 29
        M 13 16 19 21 25 28
        DETECTOR(0, 4, 0) rec[-42] rec[-40] rec[-37] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-41] rec[-39] rec[-38] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
        TICK

        # === stabilizers and edge flip flops now all established ===

        REPEAT 331 {
            X_ERROR(0.001) 12 17 20 23 26 29
            XCX 9 13 2 16 4 19 0 21 7 25 11 28
            CX 5 14 1 15 3 18 8 22 6 24 10 27
            M 12 17 20 23 26 29
            DETECTOR(0, 2, 0) rec[-48] rec[-47] rec[-44] rec[-42] rec[-41] rec[-38] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-46] rec[-45] rec[-43] rec[-40] rec[-39] rec[-37] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
            TICK

            X_ERROR(0.001) 14 15 18 22 24 27
            YCX 7 12 2 17 0 20 4 23 9 26 11 29
            XCX 3 13 1 16 5 19 6 21 8 25 10 28
            M 14 15 18 22 24 27
            DETECTOR(0, 0, 0) rec[-48] rec[-46] rec[-43] rec[-42] rec[-41] rec[-38] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-47] rec[-45] rec[-44] rec[-40] rec[-39] rec[-37] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29 3 13 1 16 5 19 6 21 8 25 10 28
            TICK

            X_ERROR(0.001) 13 16 19 21 25 28
            CX 11 14 0 15 4 18 2 22 7 24 9 27
            YCX 1 12 3 17 5 20 10 23 8 26 6 29
            M 13 16 19 21 25 28
            DETECTOR(0, 4, 0) rec[-48] rec[-46] rec[-43] rec[-42] rec[-40] rec[-37] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-47] rec[-45] rec[-44] rec[-41] rec[-39] rec[-38] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27 1 12 3 17 5 20 10 23 8 26 6 29
            TICK
        }

        X_ERROR(0.001) 12 17 20 23 26 29
        XCX 9 13 2 16 4 19 0 21 7 25 11 28
        CX 5 14 1 15 3 18 8 22 6 24 10 27
        M 12 17 20 23 26 29
        DETECTOR(0, 2, 0) rec[-48] rec[-47] rec[-44] rec[-42] rec[-41] rec[-38] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-46] rec[-45] rec[-43] rec[-40] rec[-39] rec[-37] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28 5 14 1 15 3 18 8 22 6 24 10 27
        TICK

        X_ERROR(0.001) 14 15 18 22 24 27
        XCX 3 13 1 16 5 19 6 21 8 25 10 28
        M 14 15 18 22 24 27
        DETECTOR(0, 0, 0) rec[-48] rec[-46] rec[-43] rec[-42] rec[-41] rec[-38] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 3, 0) rec[-47] rec[-45] rec[-44] rec[-40] rec[-39] rec[-37] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE2(0.001) 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.001) 0 2 4 7 9 11 12 17 20 23 26 29
        TICK

        X_ERROR(0.001) 13 16 19 21 25 28
        M 13 16 19 21 25 28
        DETECTOR(0, 4, 0) rec[-48] rec[-46] rec[-43] rec[-42] rec[-40] rec[-37] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-47] rec[-45] rec[-44] rec[-41] rec[-39] rec[-38] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-17] rec[-16]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10]
        H_YZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        TICK
        
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 2, 0) rec[-54] rec[-53] rec[-50] rec[-48] rec[-47] rec[-44] rec[-30] rec[-29] rec[-26] rec[-18] rec[-17] rec[-14] rec[-11] rec[-10] rec[-9] rec[-5] rec[-4] rec[-3]
        DETECTOR(2, 5, 0) rec[-52] rec[-51] rec[-49] rec[-46] rec[-45] rec[-43] rec[-28] rec[-27] rec[-25] rec[-16] rec[-15] rec[-13] rec[-12] rec[-8] rec[-7] rec[-6] rec[-2] rec[-1]
        DETECTOR(0, 4, 0) rec[-42] rec[-40] rec[-37] rec[-36] rec[-34] rec[-31] rec[-24] rec[-22] rec[-19] rec[-18] rec[-16] rec[-13] rec[-9] rec[-8] rec[-7] rec[-3] rec[-2] rec[-1]
        DETECTOR(2, 1, 0) rec[-41] rec[-39] rec[-38] rec[-35] rec[-33] rec[-32] rec[-23] rec[-21] rec[-20] rec[-17] rec[-15] rec[-14] rec[-12] rec[-11] rec[-10] rec[-6] rec[-5] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-8] rec[-7]
        DEPOLARIZE1(0.001) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    """)


def test_circuit_details_EM3():
    actual = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2,
        data_height=6,
        sub_rounds=1003,
        noise=0.001,
        style="EM3",
        obs="V",
    ))
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(1, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(1, 2) 2
        QUBIT_COORDS(1, 3) 3
        QUBIT_COORDS(1, 4) 4
        QUBIT_COORDS(1, 5) 5
        QUBIT_COORDS(3, 0) 6
        QUBIT_COORDS(3, 1) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(3, 3) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11
        R 0 1 2 3 4 5 6 7 8 9 10 11
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        DEPOLARIZE2(0.001) 9 3 2 1 4 5 0 6 7 8 11 10
        MPP(0.001) X9*X3 X2*X1 X4*X5 X0*X6 X7*X8 X11*X10
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        TICK

        DEPOLARIZE2(0.001) 7 1 2 3 0 5 4 10 9 8 11 6
        MPP(0.001) Y7*Y1 Y2*Y3 Y0*Y5 Y4*Y10 Y9*Y8 Y11*Y6
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK

        DEPOLARIZE2(0.001) 11 5 0 1 4 3 2 8 7 6 9 10
        MPP(0.001) Z11*Z5 Z0*Z1 Z4*Z3 Z2*Z8 Z7*Z6 Z9*Z10
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        TICK

        DEPOLARIZE2(0.001) 9 3 2 1 4 5 0 6 7 8 11 10
        MPP(0.001) X9*X3 X2*X1 X4*X5 X0*X6 X7*X8 X11*X10
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        TICK

        REPEAT 333 {
            DEPOLARIZE2(0.001) 7 1 2 3 0 5 4 10 9 8 11 6
            MPP(0.001) Y7*Y1 Y2*Y3 Y0*Y5 Y4*Y10 Y9*Y8 Y11*Y6
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK

            DEPOLARIZE2(0.001) 11 5 0 1 4 3 2 8 7 6 9 10
            MPP(0.001) Z11*Z5 Z0*Z1 Z4*Z3 Z2*Z8 Z7*Z6 Z9*Z10
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK

            DEPOLARIZE2(0.001) 9 3 2 1 4 5 0 6 7 8 11 10
            MPP(0.001) X9*X3 X2*X1 X4*X5 X0*X6 X7*X8 X11*X10
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }

        H_YZ 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 2, 0) rec[-36] rec[-35] rec[-32] rec[-30] rec[-29] rec[-26] rec[-18] rec[-17] rec[-14] rec[-11] rec[-10] rec[-9] rec[-5] rec[-4] rec[-3]
        DETECTOR(2, 5, 0) rec[-34] rec[-33] rec[-31] rec[-28] rec[-27] rec[-25] rec[-16] rec[-15] rec[-13] rec[-12] rec[-8] rec[-7] rec[-6] rec[-2] rec[-1]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-18] rec[-16] rec[-13] rec[-9] rec[-8] rec[-7] rec[-3] rec[-2] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-17] rec[-15] rec[-14] rec[-12] rec[-11] rec[-10] rec[-6] rec[-5] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-8] rec[-7]
    """)


def test_circuit_details_EM3_v2():
    actual = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2,
        data_height=6,
        sub_rounds=1003,
        noise=0.001,
        style="EM3_v2",
        obs="V",
    ))
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(1, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(1, 2) 2
        QUBIT_COORDS(1, 3) 3
        QUBIT_COORDS(1, 4) 4
        QUBIT_COORDS(1, 5) 5
        QUBIT_COORDS(3, 0) 6
        QUBIT_COORDS(3, 1) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(3, 3) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11
        R 0 1 2 3 4 5 6 7 8 9 10 11
        X_ERROR(0.0005) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        R 12
        XCX 9 12 3 12
        E(3.12647e-05) X12
        E(3.12647e-05) X3
        E(3.12647e-05) X3 X12
        E(3.12647e-05) Y3
        E(3.12647e-05) Y3 X12
        E(3.12647e-05) Z3
        E(3.12647e-05) Z3 X12
        E(3.12647e-05) X9
        E(3.12647e-05) X9 X12
        E(3.12647e-05) X9 X3
        E(3.12647e-05) X9 X3 X12
        E(3.12647e-05) X9 Y3
        E(3.12647e-05) X9 Y3 X12
        E(3.12647e-05) X9 Z3
        E(3.12647e-05) X9 Z3 X12
        E(3.12647e-05) Y9
        E(3.12647e-05) Y9 X12
        E(3.12647e-05) Y9 X3
        E(3.12647e-05) Y9 X3 X12
        E(3.12647e-05) Y9 Y3
        E(3.12647e-05) Y9 Y3 X12
        E(3.12647e-05) Y9 Z3
        E(3.12647e-05) Y9 Z3 X12
        E(3.12647e-05) Z9
        E(3.12647e-05) Z9 X12
        E(3.12647e-05) Z9 X3
        E(3.12647e-05) Z9 X3 X12
        E(3.12647e-05) Z9 Y3
        E(3.12647e-05) Z9 Y3 X12
        E(3.12647e-05) Z9 Z3
        E(3.12647e-05) Z9 Z3 X12
        M 12
        R 12
        XCX 2 12 1 12
        E(3.12647e-05) X12
        E(3.12647e-05) X1
        E(3.12647e-05) X1 X12
        E(3.12647e-05) Y1
        E(3.12647e-05) Y1 X12
        E(3.12647e-05) Z1
        E(3.12647e-05) Z1 X12
        E(3.12647e-05) X2
        E(3.12647e-05) X2 X12
        E(3.12647e-05) X2 X1
        E(3.12647e-05) X2 X1 X12
        E(3.12647e-05) X2 Y1
        E(3.12647e-05) X2 Y1 X12
        E(3.12647e-05) X2 Z1
        E(3.12647e-05) X2 Z1 X12
        E(3.12647e-05) Y2
        E(3.12647e-05) Y2 X12
        E(3.12647e-05) Y2 X1
        E(3.12647e-05) Y2 X1 X12
        E(3.12647e-05) Y2 Y1
        E(3.12647e-05) Y2 Y1 X12
        E(3.12647e-05) Y2 Z1
        E(3.12647e-05) Y2 Z1 X12
        E(3.12647e-05) Z2
        E(3.12647e-05) Z2 X12
        E(3.12647e-05) Z2 X1
        E(3.12647e-05) Z2 X1 X12
        E(3.12647e-05) Z2 Y1
        E(3.12647e-05) Z2 Y1 X12
        E(3.12647e-05) Z2 Z1
        E(3.12647e-05) Z2 Z1 X12
        M 12
        R 12
        XCX 4 12 5 12
        E(3.12647e-05) X12
        E(3.12647e-05) X5
        E(3.12647e-05) X5 X12
        E(3.12647e-05) Y5
        E(3.12647e-05) Y5 X12
        E(3.12647e-05) Z5
        E(3.12647e-05) Z5 X12
        E(3.12647e-05) X4
        E(3.12647e-05) X4 X12
        E(3.12647e-05) X4 X5
        E(3.12647e-05) X4 X5 X12
        E(3.12647e-05) X4 Y5
        E(3.12647e-05) X4 Y5 X12
        E(3.12647e-05) X4 Z5
        E(3.12647e-05) X4 Z5 X12
        E(3.12647e-05) Y4
        E(3.12647e-05) Y4 X12
        E(3.12647e-05) Y4 X5
        E(3.12647e-05) Y4 X5 X12
        E(3.12647e-05) Y4 Y5
        E(3.12647e-05) Y4 Y5 X12
        E(3.12647e-05) Y4 Z5
        E(3.12647e-05) Y4 Z5 X12
        E(3.12647e-05) Z4
        E(3.12647e-05) Z4 X12
        E(3.12647e-05) Z4 X5
        E(3.12647e-05) Z4 X5 X12
        E(3.12647e-05) Z4 Y5
        E(3.12647e-05) Z4 Y5 X12
        E(3.12647e-05) Z4 Z5
        E(3.12647e-05) Z4 Z5 X12
        M 12
        R 12
        XCX 0 12 6 12
        E(3.12647e-05) X12
        E(3.12647e-05) X6
        E(3.12647e-05) X6 X12
        E(3.12647e-05) Y6
        E(3.12647e-05) Y6 X12
        E(3.12647e-05) Z6
        E(3.12647e-05) Z6 X12
        E(3.12647e-05) X0
        E(3.12647e-05) X0 X12
        E(3.12647e-05) X0 X6
        E(3.12647e-05) X0 X6 X12
        E(3.12647e-05) X0 Y6
        E(3.12647e-05) X0 Y6 X12
        E(3.12647e-05) X0 Z6
        E(3.12647e-05) X0 Z6 X12
        E(3.12647e-05) Y0
        E(3.12647e-05) Y0 X12
        E(3.12647e-05) Y0 X6
        E(3.12647e-05) Y0 X6 X12
        E(3.12647e-05) Y0 Y6
        E(3.12647e-05) Y0 Y6 X12
        E(3.12647e-05) Y0 Z6
        E(3.12647e-05) Y0 Z6 X12
        E(3.12647e-05) Z0
        E(3.12647e-05) Z0 X12
        E(3.12647e-05) Z0 X6
        E(3.12647e-05) Z0 X6 X12
        E(3.12647e-05) Z0 Y6
        E(3.12647e-05) Z0 Y6 X12
        E(3.12647e-05) Z0 Z6
        E(3.12647e-05) Z0 Z6 X12
        M 12
        R 12
        XCX 7 12 8 12
        E(3.12647e-05) X12
        E(3.12647e-05) X8
        E(3.12647e-05) X8 X12
        E(3.12647e-05) Y8
        E(3.12647e-05) Y8 X12
        E(3.12647e-05) Z8
        E(3.12647e-05) Z8 X12
        E(3.12647e-05) X7
        E(3.12647e-05) X7 X12
        E(3.12647e-05) X7 X8
        E(3.12647e-05) X7 X8 X12
        E(3.12647e-05) X7 Y8
        E(3.12647e-05) X7 Y8 X12
        E(3.12647e-05) X7 Z8
        E(3.12647e-05) X7 Z8 X12
        E(3.12647e-05) Y7
        E(3.12647e-05) Y7 X12
        E(3.12647e-05) Y7 X8
        E(3.12647e-05) Y7 X8 X12
        E(3.12647e-05) Y7 Y8
        E(3.12647e-05) Y7 Y8 X12
        E(3.12647e-05) Y7 Z8
        E(3.12647e-05) Y7 Z8 X12
        E(3.12647e-05) Z7
        E(3.12647e-05) Z7 X12
        E(3.12647e-05) Z7 X8
        E(3.12647e-05) Z7 X8 X12
        E(3.12647e-05) Z7 Y8
        E(3.12647e-05) Z7 Y8 X12
        E(3.12647e-05) Z7 Z8
        E(3.12647e-05) Z7 Z8 X12
        M 12
        R 12
        XCX 11 12 10 12
        E(3.12647e-05) X12
        E(3.12647e-05) X10
        E(3.12647e-05) X10 X12
        E(3.12647e-05) Y10
        E(3.12647e-05) Y10 X12
        E(3.12647e-05) Z10
        E(3.12647e-05) Z10 X12
        E(3.12647e-05) X11
        E(3.12647e-05) X11 X12
        E(3.12647e-05) X11 X10
        E(3.12647e-05) X11 X10 X12
        E(3.12647e-05) X11 Y10
        E(3.12647e-05) X11 Y10 X12
        E(3.12647e-05) X11 Z10
        E(3.12647e-05) X11 Z10 X12
        E(3.12647e-05) Y11
        E(3.12647e-05) Y11 X12
        E(3.12647e-05) Y11 X10
        E(3.12647e-05) Y11 X10 X12
        E(3.12647e-05) Y11 Y10
        E(3.12647e-05) Y11 Y10 X12
        E(3.12647e-05) Y11 Z10
        E(3.12647e-05) Y11 Z10 X12
        E(3.12647e-05) Z11
        E(3.12647e-05) Z11 X12
        E(3.12647e-05) Z11 X10
        E(3.12647e-05) Z11 X10 X12
        E(3.12647e-05) Z11 Y10
        E(3.12647e-05) Z11 Y10 X12
        E(3.12647e-05) Z11 Z10
        E(3.12647e-05) Z11 Z10 X12
        M 12
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        TICK
        R 12
        YCX 7 12 1 12
        E(3.12647e-05) X12
        E(3.12647e-05) X1
        E(3.12647e-05) X1 X12
        E(3.12647e-05) Y1
        E(3.12647e-05) Y1 X12
        E(3.12647e-05) Z1
        E(3.12647e-05) Z1 X12
        E(3.12647e-05) X7
        E(3.12647e-05) X7 X12
        E(3.12647e-05) X7 X1
        E(3.12647e-05) X7 X1 X12
        E(3.12647e-05) X7 Y1
        E(3.12647e-05) X7 Y1 X12
        E(3.12647e-05) X7 Z1
        E(3.12647e-05) X7 Z1 X12
        E(3.12647e-05) Y7
        E(3.12647e-05) Y7 X12
        E(3.12647e-05) Y7 X1
        E(3.12647e-05) Y7 X1 X12
        E(3.12647e-05) Y7 Y1
        E(3.12647e-05) Y7 Y1 X12
        E(3.12647e-05) Y7 Z1
        E(3.12647e-05) Y7 Z1 X12
        E(3.12647e-05) Z7
        E(3.12647e-05) Z7 X12
        E(3.12647e-05) Z7 X1
        E(3.12647e-05) Z7 X1 X12
        E(3.12647e-05) Z7 Y1
        E(3.12647e-05) Z7 Y1 X12
        E(3.12647e-05) Z7 Z1
        E(3.12647e-05) Z7 Z1 X12
        M 12
        R 12
        YCX 2 12 3 12
        E(3.12647e-05) X12
        E(3.12647e-05) X3
        E(3.12647e-05) X3 X12
        E(3.12647e-05) Y3
        E(3.12647e-05) Y3 X12
        E(3.12647e-05) Z3
        E(3.12647e-05) Z3 X12
        E(3.12647e-05) X2
        E(3.12647e-05) X2 X12
        E(3.12647e-05) X2 X3
        E(3.12647e-05) X2 X3 X12
        E(3.12647e-05) X2 Y3
        E(3.12647e-05) X2 Y3 X12
        E(3.12647e-05) X2 Z3
        E(3.12647e-05) X2 Z3 X12
        E(3.12647e-05) Y2
        E(3.12647e-05) Y2 X12
        E(3.12647e-05) Y2 X3
        E(3.12647e-05) Y2 X3 X12
        E(3.12647e-05) Y2 Y3
        E(3.12647e-05) Y2 Y3 X12
        E(3.12647e-05) Y2 Z3
        E(3.12647e-05) Y2 Z3 X12
        E(3.12647e-05) Z2
        E(3.12647e-05) Z2 X12
        E(3.12647e-05) Z2 X3
        E(3.12647e-05) Z2 X3 X12
        E(3.12647e-05) Z2 Y3
        E(3.12647e-05) Z2 Y3 X12
        E(3.12647e-05) Z2 Z3
        E(3.12647e-05) Z2 Z3 X12
        M 12
        R 12
        YCX 0 12 5 12
        E(3.12647e-05) X12
        E(3.12647e-05) X5
        E(3.12647e-05) X5 X12
        E(3.12647e-05) Y5
        E(3.12647e-05) Y5 X12
        E(3.12647e-05) Z5
        E(3.12647e-05) Z5 X12
        E(3.12647e-05) X0
        E(3.12647e-05) X0 X12
        E(3.12647e-05) X0 X5
        E(3.12647e-05) X0 X5 X12
        E(3.12647e-05) X0 Y5
        E(3.12647e-05) X0 Y5 X12
        E(3.12647e-05) X0 Z5
        E(3.12647e-05) X0 Z5 X12
        E(3.12647e-05) Y0
        E(3.12647e-05) Y0 X12
        E(3.12647e-05) Y0 X5
        E(3.12647e-05) Y0 X5 X12
        E(3.12647e-05) Y0 Y5
        E(3.12647e-05) Y0 Y5 X12
        E(3.12647e-05) Y0 Z5
        E(3.12647e-05) Y0 Z5 X12
        E(3.12647e-05) Z0
        E(3.12647e-05) Z0 X12
        E(3.12647e-05) Z0 X5
        E(3.12647e-05) Z0 X5 X12
        E(3.12647e-05) Z0 Y5
        E(3.12647e-05) Z0 Y5 X12
        E(3.12647e-05) Z0 Z5
        E(3.12647e-05) Z0 Z5 X12
        M 12
        R 12
        YCX 4 12 10 12
        E(3.12647e-05) X12
        E(3.12647e-05) X10
        E(3.12647e-05) X10 X12
        E(3.12647e-05) Y10
        E(3.12647e-05) Y10 X12
        E(3.12647e-05) Z10
        E(3.12647e-05) Z10 X12
        E(3.12647e-05) X4
        E(3.12647e-05) X4 X12
        E(3.12647e-05) X4 X10
        E(3.12647e-05) X4 X10 X12
        E(3.12647e-05) X4 Y10
        E(3.12647e-05) X4 Y10 X12
        E(3.12647e-05) X4 Z10
        E(3.12647e-05) X4 Z10 X12
        E(3.12647e-05) Y4
        E(3.12647e-05) Y4 X12
        E(3.12647e-05) Y4 X10
        E(3.12647e-05) Y4 X10 X12
        E(3.12647e-05) Y4 Y10
        E(3.12647e-05) Y4 Y10 X12
        E(3.12647e-05) Y4 Z10
        E(3.12647e-05) Y4 Z10 X12
        E(3.12647e-05) Z4
        E(3.12647e-05) Z4 X12
        E(3.12647e-05) Z4 X10
        E(3.12647e-05) Z4 X10 X12
        E(3.12647e-05) Z4 Y10
        E(3.12647e-05) Z4 Y10 X12
        E(3.12647e-05) Z4 Z10
        E(3.12647e-05) Z4 Z10 X12
        M 12
        R 12
        YCX 9 12 8 12
        E(3.12647e-05) X12
        E(3.12647e-05) X8
        E(3.12647e-05) X8 X12
        E(3.12647e-05) Y8
        E(3.12647e-05) Y8 X12
        E(3.12647e-05) Z8
        E(3.12647e-05) Z8 X12
        E(3.12647e-05) X9
        E(3.12647e-05) X9 X12
        E(3.12647e-05) X9 X8
        E(3.12647e-05) X9 X8 X12
        E(3.12647e-05) X9 Y8
        E(3.12647e-05) X9 Y8 X12
        E(3.12647e-05) X9 Z8
        E(3.12647e-05) X9 Z8 X12
        E(3.12647e-05) Y9
        E(3.12647e-05) Y9 X12
        E(3.12647e-05) Y9 X8
        E(3.12647e-05) Y9 X8 X12
        E(3.12647e-05) Y9 Y8
        E(3.12647e-05) Y9 Y8 X12
        E(3.12647e-05) Y9 Z8
        E(3.12647e-05) Y9 Z8 X12
        E(3.12647e-05) Z9
        E(3.12647e-05) Z9 X12
        E(3.12647e-05) Z9 X8
        E(3.12647e-05) Z9 X8 X12
        E(3.12647e-05) Z9 Y8
        E(3.12647e-05) Z9 Y8 X12
        E(3.12647e-05) Z9 Z8
        E(3.12647e-05) Z9 Z8 X12
        M 12
        R 12
        YCX 11 12 6 12
        E(3.12647e-05) X12
        E(3.12647e-05) X6
        E(3.12647e-05) X6 X12
        E(3.12647e-05) Y6
        E(3.12647e-05) Y6 X12
        E(3.12647e-05) Z6
        E(3.12647e-05) Z6 X12
        E(3.12647e-05) X11
        E(3.12647e-05) X11 X12
        E(3.12647e-05) X11 X6
        E(3.12647e-05) X11 X6 X12
        E(3.12647e-05) X11 Y6
        E(3.12647e-05) X11 Y6 X12
        E(3.12647e-05) X11 Z6
        E(3.12647e-05) X11 Z6 X12
        E(3.12647e-05) Y11
        E(3.12647e-05) Y11 X12
        E(3.12647e-05) Y11 X6
        E(3.12647e-05) Y11 X6 X12
        E(3.12647e-05) Y11 Y6
        E(3.12647e-05) Y11 Y6 X12
        E(3.12647e-05) Y11 Z6
        E(3.12647e-05) Y11 Z6 X12
        E(3.12647e-05) Z11
        E(3.12647e-05) Z11 X12
        E(3.12647e-05) Z11 X6
        E(3.12647e-05) Z11 X6 X12
        E(3.12647e-05) Z11 Y6
        E(3.12647e-05) Z11 Y6 X12
        E(3.12647e-05) Z11 Z6
        E(3.12647e-05) Z11 Z6 X12
        M 12
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        R 12
        CX 11 12 5 12
        E(3.12647e-05) X12
        E(3.12647e-05) X5
        E(3.12647e-05) X5 X12
        E(3.12647e-05) Y5
        E(3.12647e-05) Y5 X12
        E(3.12647e-05) Z5
        E(3.12647e-05) Z5 X12
        E(3.12647e-05) X11
        E(3.12647e-05) X11 X12
        E(3.12647e-05) X11 X5
        E(3.12647e-05) X11 X5 X12
        E(3.12647e-05) X11 Y5
        E(3.12647e-05) X11 Y5 X12
        E(3.12647e-05) X11 Z5
        E(3.12647e-05) X11 Z5 X12
        E(3.12647e-05) Y11
        E(3.12647e-05) Y11 X12
        E(3.12647e-05) Y11 X5
        E(3.12647e-05) Y11 X5 X12
        E(3.12647e-05) Y11 Y5
        E(3.12647e-05) Y11 Y5 X12
        E(3.12647e-05) Y11 Z5
        E(3.12647e-05) Y11 Z5 X12
        E(3.12647e-05) Z11
        E(3.12647e-05) Z11 X12
        E(3.12647e-05) Z11 X5
        E(3.12647e-05) Z11 X5 X12
        E(3.12647e-05) Z11 Y5
        E(3.12647e-05) Z11 Y5 X12
        E(3.12647e-05) Z11 Z5
        E(3.12647e-05) Z11 Z5 X12
        M 12
        R 12
        CX 0 12 1 12
        E(3.12647e-05) X12
        E(3.12647e-05) X1
        E(3.12647e-05) X1 X12
        E(3.12647e-05) Y1
        E(3.12647e-05) Y1 X12
        E(3.12647e-05) Z1
        E(3.12647e-05) Z1 X12
        E(3.12647e-05) X0
        E(3.12647e-05) X0 X12
        E(3.12647e-05) X0 X1
        E(3.12647e-05) X0 X1 X12
        E(3.12647e-05) X0 Y1
        E(3.12647e-05) X0 Y1 X12
        E(3.12647e-05) X0 Z1
        E(3.12647e-05) X0 Z1 X12
        E(3.12647e-05) Y0
        E(3.12647e-05) Y0 X12
        E(3.12647e-05) Y0 X1
        E(3.12647e-05) Y0 X1 X12
        E(3.12647e-05) Y0 Y1
        E(3.12647e-05) Y0 Y1 X12
        E(3.12647e-05) Y0 Z1
        E(3.12647e-05) Y0 Z1 X12
        E(3.12647e-05) Z0
        E(3.12647e-05) Z0 X12
        E(3.12647e-05) Z0 X1
        E(3.12647e-05) Z0 X1 X12
        E(3.12647e-05) Z0 Y1
        E(3.12647e-05) Z0 Y1 X12
        E(3.12647e-05) Z0 Z1
        E(3.12647e-05) Z0 Z1 X12
        M 12
        R 12
        CX 4 12 3 12
        E(3.12647e-05) X12
        E(3.12647e-05) X3
        E(3.12647e-05) X3 X12
        E(3.12647e-05) Y3
        E(3.12647e-05) Y3 X12
        E(3.12647e-05) Z3
        E(3.12647e-05) Z3 X12
        E(3.12647e-05) X4
        E(3.12647e-05) X4 X12
        E(3.12647e-05) X4 X3
        E(3.12647e-05) X4 X3 X12
        E(3.12647e-05) X4 Y3
        E(3.12647e-05) X4 Y3 X12
        E(3.12647e-05) X4 Z3
        E(3.12647e-05) X4 Z3 X12
        E(3.12647e-05) Y4
        E(3.12647e-05) Y4 X12
        E(3.12647e-05) Y4 X3
        E(3.12647e-05) Y4 X3 X12
        E(3.12647e-05) Y4 Y3
        E(3.12647e-05) Y4 Y3 X12
        E(3.12647e-05) Y4 Z3
        E(3.12647e-05) Y4 Z3 X12
        E(3.12647e-05) Z4
        E(3.12647e-05) Z4 X12
        E(3.12647e-05) Z4 X3
        E(3.12647e-05) Z4 X3 X12
        E(3.12647e-05) Z4 Y3
        E(3.12647e-05) Z4 Y3 X12
        E(3.12647e-05) Z4 Z3
        E(3.12647e-05) Z4 Z3 X12
        M 12
        R 12
        CX 2 12 8 12
        E(3.12647e-05) X12
        E(3.12647e-05) X8
        E(3.12647e-05) X8 X12
        E(3.12647e-05) Y8
        E(3.12647e-05) Y8 X12
        E(3.12647e-05) Z8
        E(3.12647e-05) Z8 X12
        E(3.12647e-05) X2
        E(3.12647e-05) X2 X12
        E(3.12647e-05) X2 X8
        E(3.12647e-05) X2 X8 X12
        E(3.12647e-05) X2 Y8
        E(3.12647e-05) X2 Y8 X12
        E(3.12647e-05) X2 Z8
        E(3.12647e-05) X2 Z8 X12
        E(3.12647e-05) Y2
        E(3.12647e-05) Y2 X12
        E(3.12647e-05) Y2 X8
        E(3.12647e-05) Y2 X8 X12
        E(3.12647e-05) Y2 Y8
        E(3.12647e-05) Y2 Y8 X12
        E(3.12647e-05) Y2 Z8
        E(3.12647e-05) Y2 Z8 X12
        E(3.12647e-05) Z2
        E(3.12647e-05) Z2 X12
        E(3.12647e-05) Z2 X8
        E(3.12647e-05) Z2 X8 X12
        E(3.12647e-05) Z2 Y8
        E(3.12647e-05) Z2 Y8 X12
        E(3.12647e-05) Z2 Z8
        E(3.12647e-05) Z2 Z8 X12
        M 12
        R 12
        CX 7 12 6 12
        E(3.12647e-05) X12
        E(3.12647e-05) X6
        E(3.12647e-05) X6 X12
        E(3.12647e-05) Y6
        E(3.12647e-05) Y6 X12
        E(3.12647e-05) Z6
        E(3.12647e-05) Z6 X12
        E(3.12647e-05) X7
        E(3.12647e-05) X7 X12
        E(3.12647e-05) X7 X6
        E(3.12647e-05) X7 X6 X12
        E(3.12647e-05) X7 Y6
        E(3.12647e-05) X7 Y6 X12
        E(3.12647e-05) X7 Z6
        E(3.12647e-05) X7 Z6 X12
        E(3.12647e-05) Y7
        E(3.12647e-05) Y7 X12
        E(3.12647e-05) Y7 X6
        E(3.12647e-05) Y7 X6 X12
        E(3.12647e-05) Y7 Y6
        E(3.12647e-05) Y7 Y6 X12
        E(3.12647e-05) Y7 Z6
        E(3.12647e-05) Y7 Z6 X12
        E(3.12647e-05) Z7
        E(3.12647e-05) Z7 X12
        E(3.12647e-05) Z7 X6
        E(3.12647e-05) Z7 X6 X12
        E(3.12647e-05) Z7 Y6
        E(3.12647e-05) Z7 Y6 X12
        E(3.12647e-05) Z7 Z6
        E(3.12647e-05) Z7 Z6 X12
        M 12
        R 12
        CX 9 12 10 12
        E(3.12647e-05) X12
        E(3.12647e-05) X10
        E(3.12647e-05) X10 X12
        E(3.12647e-05) Y10
        E(3.12647e-05) Y10 X12
        E(3.12647e-05) Z10
        E(3.12647e-05) Z10 X12
        E(3.12647e-05) X9
        E(3.12647e-05) X9 X12
        E(3.12647e-05) X9 X10
        E(3.12647e-05) X9 X10 X12
        E(3.12647e-05) X9 Y10
        E(3.12647e-05) X9 Y10 X12
        E(3.12647e-05) X9 Z10
        E(3.12647e-05) X9 Z10 X12
        E(3.12647e-05) Y9
        E(3.12647e-05) Y9 X12
        E(3.12647e-05) Y9 X10
        E(3.12647e-05) Y9 X10 X12
        E(3.12647e-05) Y9 Y10
        E(3.12647e-05) Y9 Y10 X12
        E(3.12647e-05) Y9 Z10
        E(3.12647e-05) Y9 Z10 X12
        E(3.12647e-05) Z9
        E(3.12647e-05) Z9 X12
        E(3.12647e-05) Z9 X10
        E(3.12647e-05) Z9 X10 X12
        E(3.12647e-05) Z9 Y10
        E(3.12647e-05) Z9 Y10 X12
        E(3.12647e-05) Z9 Z10
        E(3.12647e-05) Z9 Z10 X12
        M 12
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        TICK
        R 12
        XCX 9 12 3 12
        E(3.12647e-05) X12
        E(3.12647e-05) X3
        E(3.12647e-05) X3 X12
        E(3.12647e-05) Y3
        E(3.12647e-05) Y3 X12
        E(3.12647e-05) Z3
        E(3.12647e-05) Z3 X12
        E(3.12647e-05) X9
        E(3.12647e-05) X9 X12
        E(3.12647e-05) X9 X3
        E(3.12647e-05) X9 X3 X12
        E(3.12647e-05) X9 Y3
        E(3.12647e-05) X9 Y3 X12
        E(3.12647e-05) X9 Z3
        E(3.12647e-05) X9 Z3 X12
        E(3.12647e-05) Y9
        E(3.12647e-05) Y9 X12
        E(3.12647e-05) Y9 X3
        E(3.12647e-05) Y9 X3 X12
        E(3.12647e-05) Y9 Y3
        E(3.12647e-05) Y9 Y3 X12
        E(3.12647e-05) Y9 Z3
        E(3.12647e-05) Y9 Z3 X12
        E(3.12647e-05) Z9
        E(3.12647e-05) Z9 X12
        E(3.12647e-05) Z9 X3
        E(3.12647e-05) Z9 X3 X12
        E(3.12647e-05) Z9 Y3
        E(3.12647e-05) Z9 Y3 X12
        E(3.12647e-05) Z9 Z3
        E(3.12647e-05) Z9 Z3 X12
        M 12
        R 12
        XCX 2 12 1 12
        E(3.12647e-05) X12
        E(3.12647e-05) X1
        E(3.12647e-05) X1 X12
        E(3.12647e-05) Y1
        E(3.12647e-05) Y1 X12
        E(3.12647e-05) Z1
        E(3.12647e-05) Z1 X12
        E(3.12647e-05) X2
        E(3.12647e-05) X2 X12
        E(3.12647e-05) X2 X1
        E(3.12647e-05) X2 X1 X12
        E(3.12647e-05) X2 Y1
        E(3.12647e-05) X2 Y1 X12
        E(3.12647e-05) X2 Z1
        E(3.12647e-05) X2 Z1 X12
        E(3.12647e-05) Y2
        E(3.12647e-05) Y2 X12
        E(3.12647e-05) Y2 X1
        E(3.12647e-05) Y2 X1 X12
        E(3.12647e-05) Y2 Y1
        E(3.12647e-05) Y2 Y1 X12
        E(3.12647e-05) Y2 Z1
        E(3.12647e-05) Y2 Z1 X12
        E(3.12647e-05) Z2
        E(3.12647e-05) Z2 X12
        E(3.12647e-05) Z2 X1
        E(3.12647e-05) Z2 X1 X12
        E(3.12647e-05) Z2 Y1
        E(3.12647e-05) Z2 Y1 X12
        E(3.12647e-05) Z2 Z1
        E(3.12647e-05) Z2 Z1 X12
        M 12
        R 12
        XCX 4 12 5 12
        E(3.12647e-05) X12
        E(3.12647e-05) X5
        E(3.12647e-05) X5 X12
        E(3.12647e-05) Y5
        E(3.12647e-05) Y5 X12
        E(3.12647e-05) Z5
        E(3.12647e-05) Z5 X12
        E(3.12647e-05) X4
        E(3.12647e-05) X4 X12
        E(3.12647e-05) X4 X5
        E(3.12647e-05) X4 X5 X12
        E(3.12647e-05) X4 Y5
        E(3.12647e-05) X4 Y5 X12
        E(3.12647e-05) X4 Z5
        E(3.12647e-05) X4 Z5 X12
        E(3.12647e-05) Y4
        E(3.12647e-05) Y4 X12
        E(3.12647e-05) Y4 X5
        E(3.12647e-05) Y4 X5 X12
        E(3.12647e-05) Y4 Y5
        E(3.12647e-05) Y4 Y5 X12
        E(3.12647e-05) Y4 Z5
        E(3.12647e-05) Y4 Z5 X12
        E(3.12647e-05) Z4
        E(3.12647e-05) Z4 X12
        E(3.12647e-05) Z4 X5
        E(3.12647e-05) Z4 X5 X12
        E(3.12647e-05) Z4 Y5
        E(3.12647e-05) Z4 Y5 X12
        E(3.12647e-05) Z4 Z5
        E(3.12647e-05) Z4 Z5 X12
        M 12
        R 12
        XCX 0 12 6 12
        E(3.12647e-05) X12
        E(3.12647e-05) X6
        E(3.12647e-05) X6 X12
        E(3.12647e-05) Y6
        E(3.12647e-05) Y6 X12
        E(3.12647e-05) Z6
        E(3.12647e-05) Z6 X12
        E(3.12647e-05) X0
        E(3.12647e-05) X0 X12
        E(3.12647e-05) X0 X6
        E(3.12647e-05) X0 X6 X12
        E(3.12647e-05) X0 Y6
        E(3.12647e-05) X0 Y6 X12
        E(3.12647e-05) X0 Z6
        E(3.12647e-05) X0 Z6 X12
        E(3.12647e-05) Y0
        E(3.12647e-05) Y0 X12
        E(3.12647e-05) Y0 X6
        E(3.12647e-05) Y0 X6 X12
        E(3.12647e-05) Y0 Y6
        E(3.12647e-05) Y0 Y6 X12
        E(3.12647e-05) Y0 Z6
        E(3.12647e-05) Y0 Z6 X12
        E(3.12647e-05) Z0
        E(3.12647e-05) Z0 X12
        E(3.12647e-05) Z0 X6
        E(3.12647e-05) Z0 X6 X12
        E(3.12647e-05) Z0 Y6
        E(3.12647e-05) Z0 Y6 X12
        E(3.12647e-05) Z0 Z6
        E(3.12647e-05) Z0 Z6 X12
        M 12
        R 12
        XCX 7 12 8 12
        E(3.12647e-05) X12
        E(3.12647e-05) X8
        E(3.12647e-05) X8 X12
        E(3.12647e-05) Y8
        E(3.12647e-05) Y8 X12
        E(3.12647e-05) Z8
        E(3.12647e-05) Z8 X12
        E(3.12647e-05) X7
        E(3.12647e-05) X7 X12
        E(3.12647e-05) X7 X8
        E(3.12647e-05) X7 X8 X12
        E(3.12647e-05) X7 Y8
        E(3.12647e-05) X7 Y8 X12
        E(3.12647e-05) X7 Z8
        E(3.12647e-05) X7 Z8 X12
        E(3.12647e-05) Y7
        E(3.12647e-05) Y7 X12
        E(3.12647e-05) Y7 X8
        E(3.12647e-05) Y7 X8 X12
        E(3.12647e-05) Y7 Y8
        E(3.12647e-05) Y7 Y8 X12
        E(3.12647e-05) Y7 Z8
        E(3.12647e-05) Y7 Z8 X12
        E(3.12647e-05) Z7
        E(3.12647e-05) Z7 X12
        E(3.12647e-05) Z7 X8
        E(3.12647e-05) Z7 X8 X12
        E(3.12647e-05) Z7 Y8
        E(3.12647e-05) Z7 Y8 X12
        E(3.12647e-05) Z7 Z8
        E(3.12647e-05) Z7 Z8 X12
        M 12
        R 12
        XCX 11 12 10 12
        E(3.12647e-05) X12
        E(3.12647e-05) X10
        E(3.12647e-05) X10 X12
        E(3.12647e-05) Y10
        E(3.12647e-05) Y10 X12
        E(3.12647e-05) Z10
        E(3.12647e-05) Z10 X12
        E(3.12647e-05) X11
        E(3.12647e-05) X11 X12
        E(3.12647e-05) X11 X10
        E(3.12647e-05) X11 X10 X12
        E(3.12647e-05) X11 Y10
        E(3.12647e-05) X11 Y10 X12
        E(3.12647e-05) X11 Z10
        E(3.12647e-05) X11 Z10 X12
        E(3.12647e-05) Y11
        E(3.12647e-05) Y11 X12
        E(3.12647e-05) Y11 X10
        E(3.12647e-05) Y11 X10 X12
        E(3.12647e-05) Y11 Y10
        E(3.12647e-05) Y11 Y10 X12
        E(3.12647e-05) Y11 Z10
        E(3.12647e-05) Y11 Z10 X12
        E(3.12647e-05) Z11
        E(3.12647e-05) Z11 X12
        E(3.12647e-05) Z11 X10
        E(3.12647e-05) Z11 X10 X12
        E(3.12647e-05) Z11 Y10
        E(3.12647e-05) Z11 Y10 X12
        E(3.12647e-05) Z11 Z10
        E(3.12647e-05) Z11 Z10 X12
        M 12
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        TICK
        REPEAT 333 {
            R 12
            YCX 7 12 1 12
            E(3.12647e-05) X12
            E(3.12647e-05) X1
            E(3.12647e-05) X1 X12
            E(3.12647e-05) Y1
            E(3.12647e-05) Y1 X12
            E(3.12647e-05) Z1
            E(3.12647e-05) Z1 X12
            E(3.12647e-05) X7
            E(3.12647e-05) X7 X12
            E(3.12647e-05) X7 X1
            E(3.12647e-05) X7 X1 X12
            E(3.12647e-05) X7 Y1
            E(3.12647e-05) X7 Y1 X12
            E(3.12647e-05) X7 Z1
            E(3.12647e-05) X7 Z1 X12
            E(3.12647e-05) Y7
            E(3.12647e-05) Y7 X12
            E(3.12647e-05) Y7 X1
            E(3.12647e-05) Y7 X1 X12
            E(3.12647e-05) Y7 Y1
            E(3.12647e-05) Y7 Y1 X12
            E(3.12647e-05) Y7 Z1
            E(3.12647e-05) Y7 Z1 X12
            E(3.12647e-05) Z7
            E(3.12647e-05) Z7 X12
            E(3.12647e-05) Z7 X1
            E(3.12647e-05) Z7 X1 X12
            E(3.12647e-05) Z7 Y1
            E(3.12647e-05) Z7 Y1 X12
            E(3.12647e-05) Z7 Z1
            E(3.12647e-05) Z7 Z1 X12
            M 12
            R 12
            YCX 2 12 3 12
            E(3.12647e-05) X12
            E(3.12647e-05) X3
            E(3.12647e-05) X3 X12
            E(3.12647e-05) Y3
            E(3.12647e-05) Y3 X12
            E(3.12647e-05) Z3
            E(3.12647e-05) Z3 X12
            E(3.12647e-05) X2
            E(3.12647e-05) X2 X12
            E(3.12647e-05) X2 X3
            E(3.12647e-05) X2 X3 X12
            E(3.12647e-05) X2 Y3
            E(3.12647e-05) X2 Y3 X12
            E(3.12647e-05) X2 Z3
            E(3.12647e-05) X2 Z3 X12
            E(3.12647e-05) Y2
            E(3.12647e-05) Y2 X12
            E(3.12647e-05) Y2 X3
            E(3.12647e-05) Y2 X3 X12
            E(3.12647e-05) Y2 Y3
            E(3.12647e-05) Y2 Y3 X12
            E(3.12647e-05) Y2 Z3
            E(3.12647e-05) Y2 Z3 X12
            E(3.12647e-05) Z2
            E(3.12647e-05) Z2 X12
            E(3.12647e-05) Z2 X3
            E(3.12647e-05) Z2 X3 X12
            E(3.12647e-05) Z2 Y3
            E(3.12647e-05) Z2 Y3 X12
            E(3.12647e-05) Z2 Z3
            E(3.12647e-05) Z2 Z3 X12
            M 12
            R 12
            YCX 0 12 5 12
            E(3.12647e-05) X12
            E(3.12647e-05) X5
            E(3.12647e-05) X5 X12
            E(3.12647e-05) Y5
            E(3.12647e-05) Y5 X12
            E(3.12647e-05) Z5
            E(3.12647e-05) Z5 X12
            E(3.12647e-05) X0
            E(3.12647e-05) X0 X12
            E(3.12647e-05) X0 X5
            E(3.12647e-05) X0 X5 X12
            E(3.12647e-05) X0 Y5
            E(3.12647e-05) X0 Y5 X12
            E(3.12647e-05) X0 Z5
            E(3.12647e-05) X0 Z5 X12
            E(3.12647e-05) Y0
            E(3.12647e-05) Y0 X12
            E(3.12647e-05) Y0 X5
            E(3.12647e-05) Y0 X5 X12
            E(3.12647e-05) Y0 Y5
            E(3.12647e-05) Y0 Y5 X12
            E(3.12647e-05) Y0 Z5
            E(3.12647e-05) Y0 Z5 X12
            E(3.12647e-05) Z0
            E(3.12647e-05) Z0 X12
            E(3.12647e-05) Z0 X5
            E(3.12647e-05) Z0 X5 X12
            E(3.12647e-05) Z0 Y5
            E(3.12647e-05) Z0 Y5 X12
            E(3.12647e-05) Z0 Z5
            E(3.12647e-05) Z0 Z5 X12
            M 12
            R 12
            YCX 4 12 10 12
            E(3.12647e-05) X12
            E(3.12647e-05) X10
            E(3.12647e-05) X10 X12
            E(3.12647e-05) Y10
            E(3.12647e-05) Y10 X12
            E(3.12647e-05) Z10
            E(3.12647e-05) Z10 X12
            E(3.12647e-05) X4
            E(3.12647e-05) X4 X12
            E(3.12647e-05) X4 X10
            E(3.12647e-05) X4 X10 X12
            E(3.12647e-05) X4 Y10
            E(3.12647e-05) X4 Y10 X12
            E(3.12647e-05) X4 Z10
            E(3.12647e-05) X4 Z10 X12
            E(3.12647e-05) Y4
            E(3.12647e-05) Y4 X12
            E(3.12647e-05) Y4 X10
            E(3.12647e-05) Y4 X10 X12
            E(3.12647e-05) Y4 Y10
            E(3.12647e-05) Y4 Y10 X12
            E(3.12647e-05) Y4 Z10
            E(3.12647e-05) Y4 Z10 X12
            E(3.12647e-05) Z4
            E(3.12647e-05) Z4 X12
            E(3.12647e-05) Z4 X10
            E(3.12647e-05) Z4 X10 X12
            E(3.12647e-05) Z4 Y10
            E(3.12647e-05) Z4 Y10 X12
            E(3.12647e-05) Z4 Z10
            E(3.12647e-05) Z4 Z10 X12
            M 12
            R 12
            YCX 9 12 8 12
            E(3.12647e-05) X12
            E(3.12647e-05) X8
            E(3.12647e-05) X8 X12
            E(3.12647e-05) Y8
            E(3.12647e-05) Y8 X12
            E(3.12647e-05) Z8
            E(3.12647e-05) Z8 X12
            E(3.12647e-05) X9
            E(3.12647e-05) X9 X12
            E(3.12647e-05) X9 X8
            E(3.12647e-05) X9 X8 X12
            E(3.12647e-05) X9 Y8
            E(3.12647e-05) X9 Y8 X12
            E(3.12647e-05) X9 Z8
            E(3.12647e-05) X9 Z8 X12
            E(3.12647e-05) Y9
            E(3.12647e-05) Y9 X12
            E(3.12647e-05) Y9 X8
            E(3.12647e-05) Y9 X8 X12
            E(3.12647e-05) Y9 Y8
            E(3.12647e-05) Y9 Y8 X12
            E(3.12647e-05) Y9 Z8
            E(3.12647e-05) Y9 Z8 X12
            E(3.12647e-05) Z9
            E(3.12647e-05) Z9 X12
            E(3.12647e-05) Z9 X8
            E(3.12647e-05) Z9 X8 X12
            E(3.12647e-05) Z9 Y8
            E(3.12647e-05) Z9 Y8 X12
            E(3.12647e-05) Z9 Z8
            E(3.12647e-05) Z9 Z8 X12
            M 12
            R 12
            YCX 11 12 6 12
            E(3.12647e-05) X12
            E(3.12647e-05) X6
            E(3.12647e-05) X6 X12
            E(3.12647e-05) Y6
            E(3.12647e-05) Y6 X12
            E(3.12647e-05) Z6
            E(3.12647e-05) Z6 X12
            E(3.12647e-05) X11
            E(3.12647e-05) X11 X12
            E(3.12647e-05) X11 X6
            E(3.12647e-05) X11 X6 X12
            E(3.12647e-05) X11 Y6
            E(3.12647e-05) X11 Y6 X12
            E(3.12647e-05) X11 Z6
            E(3.12647e-05) X11 Z6 X12
            E(3.12647e-05) Y11
            E(3.12647e-05) Y11 X12
            E(3.12647e-05) Y11 X6
            E(3.12647e-05) Y11 X6 X12
            E(3.12647e-05) Y11 Y6
            E(3.12647e-05) Y11 Y6 X12
            E(3.12647e-05) Y11 Z6
            E(3.12647e-05) Y11 Z6 X12
            E(3.12647e-05) Z11
            E(3.12647e-05) Z11 X12
            E(3.12647e-05) Z11 X6
            E(3.12647e-05) Z11 X6 X12
            E(3.12647e-05) Z11 Y6
            E(3.12647e-05) Z11 Y6 X12
            E(3.12647e-05) Z11 Z6
            E(3.12647e-05) Z11 Z6 X12
            M 12
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
            R 12
            CX 11 12 5 12
            E(3.12647e-05) X12
            E(3.12647e-05) X5
            E(3.12647e-05) X5 X12
            E(3.12647e-05) Y5
            E(3.12647e-05) Y5 X12
            E(3.12647e-05) Z5
            E(3.12647e-05) Z5 X12
            E(3.12647e-05) X11
            E(3.12647e-05) X11 X12
            E(3.12647e-05) X11 X5
            E(3.12647e-05) X11 X5 X12
            E(3.12647e-05) X11 Y5
            E(3.12647e-05) X11 Y5 X12
            E(3.12647e-05) X11 Z5
            E(3.12647e-05) X11 Z5 X12
            E(3.12647e-05) Y11
            E(3.12647e-05) Y11 X12
            E(3.12647e-05) Y11 X5
            E(3.12647e-05) Y11 X5 X12
            E(3.12647e-05) Y11 Y5
            E(3.12647e-05) Y11 Y5 X12
            E(3.12647e-05) Y11 Z5
            E(3.12647e-05) Y11 Z5 X12
            E(3.12647e-05) Z11
            E(3.12647e-05) Z11 X12
            E(3.12647e-05) Z11 X5
            E(3.12647e-05) Z11 X5 X12
            E(3.12647e-05) Z11 Y5
            E(3.12647e-05) Z11 Y5 X12
            E(3.12647e-05) Z11 Z5
            E(3.12647e-05) Z11 Z5 X12
            M 12
            R 12
            CX 0 12 1 12
            E(3.12647e-05) X12
            E(3.12647e-05) X1
            E(3.12647e-05) X1 X12
            E(3.12647e-05) Y1
            E(3.12647e-05) Y1 X12
            E(3.12647e-05) Z1
            E(3.12647e-05) Z1 X12
            E(3.12647e-05) X0
            E(3.12647e-05) X0 X12
            E(3.12647e-05) X0 X1
            E(3.12647e-05) X0 X1 X12
            E(3.12647e-05) X0 Y1
            E(3.12647e-05) X0 Y1 X12
            E(3.12647e-05) X0 Z1
            E(3.12647e-05) X0 Z1 X12
            E(3.12647e-05) Y0
            E(3.12647e-05) Y0 X12
            E(3.12647e-05) Y0 X1
            E(3.12647e-05) Y0 X1 X12
            E(3.12647e-05) Y0 Y1
            E(3.12647e-05) Y0 Y1 X12
            E(3.12647e-05) Y0 Z1
            E(3.12647e-05) Y0 Z1 X12
            E(3.12647e-05) Z0
            E(3.12647e-05) Z0 X12
            E(3.12647e-05) Z0 X1
            E(3.12647e-05) Z0 X1 X12
            E(3.12647e-05) Z0 Y1
            E(3.12647e-05) Z0 Y1 X12
            E(3.12647e-05) Z0 Z1
            E(3.12647e-05) Z0 Z1 X12
            M 12
            R 12
            CX 4 12 3 12
            E(3.12647e-05) X12
            E(3.12647e-05) X3
            E(3.12647e-05) X3 X12
            E(3.12647e-05) Y3
            E(3.12647e-05) Y3 X12
            E(3.12647e-05) Z3
            E(3.12647e-05) Z3 X12
            E(3.12647e-05) X4
            E(3.12647e-05) X4 X12
            E(3.12647e-05) X4 X3
            E(3.12647e-05) X4 X3 X12
            E(3.12647e-05) X4 Y3
            E(3.12647e-05) X4 Y3 X12
            E(3.12647e-05) X4 Z3
            E(3.12647e-05) X4 Z3 X12
            E(3.12647e-05) Y4
            E(3.12647e-05) Y4 X12
            E(3.12647e-05) Y4 X3
            E(3.12647e-05) Y4 X3 X12
            E(3.12647e-05) Y4 Y3
            E(3.12647e-05) Y4 Y3 X12
            E(3.12647e-05) Y4 Z3
            E(3.12647e-05) Y4 Z3 X12
            E(3.12647e-05) Z4
            E(3.12647e-05) Z4 X12
            E(3.12647e-05) Z4 X3
            E(3.12647e-05) Z4 X3 X12
            E(3.12647e-05) Z4 Y3
            E(3.12647e-05) Z4 Y3 X12
            E(3.12647e-05) Z4 Z3
            E(3.12647e-05) Z4 Z3 X12
            M 12
            R 12
            CX 2 12 8 12
            E(3.12647e-05) X12
            E(3.12647e-05) X8
            E(3.12647e-05) X8 X12
            E(3.12647e-05) Y8
            E(3.12647e-05) Y8 X12
            E(3.12647e-05) Z8
            E(3.12647e-05) Z8 X12
            E(3.12647e-05) X2
            E(3.12647e-05) X2 X12
            E(3.12647e-05) X2 X8
            E(3.12647e-05) X2 X8 X12
            E(3.12647e-05) X2 Y8
            E(3.12647e-05) X2 Y8 X12
            E(3.12647e-05) X2 Z8
            E(3.12647e-05) X2 Z8 X12
            E(3.12647e-05) Y2
            E(3.12647e-05) Y2 X12
            E(3.12647e-05) Y2 X8
            E(3.12647e-05) Y2 X8 X12
            E(3.12647e-05) Y2 Y8
            E(3.12647e-05) Y2 Y8 X12
            E(3.12647e-05) Y2 Z8
            E(3.12647e-05) Y2 Z8 X12
            E(3.12647e-05) Z2
            E(3.12647e-05) Z2 X12
            E(3.12647e-05) Z2 X8
            E(3.12647e-05) Z2 X8 X12
            E(3.12647e-05) Z2 Y8
            E(3.12647e-05) Z2 Y8 X12
            E(3.12647e-05) Z2 Z8
            E(3.12647e-05) Z2 Z8 X12
            M 12
            R 12
            CX 7 12 6 12
            E(3.12647e-05) X12
            E(3.12647e-05) X6
            E(3.12647e-05) X6 X12
            E(3.12647e-05) Y6
            E(3.12647e-05) Y6 X12
            E(3.12647e-05) Z6
            E(3.12647e-05) Z6 X12
            E(3.12647e-05) X7
            E(3.12647e-05) X7 X12
            E(3.12647e-05) X7 X6
            E(3.12647e-05) X7 X6 X12
            E(3.12647e-05) X7 Y6
            E(3.12647e-05) X7 Y6 X12
            E(3.12647e-05) X7 Z6
            E(3.12647e-05) X7 Z6 X12
            E(3.12647e-05) Y7
            E(3.12647e-05) Y7 X12
            E(3.12647e-05) Y7 X6
            E(3.12647e-05) Y7 X6 X12
            E(3.12647e-05) Y7 Y6
            E(3.12647e-05) Y7 Y6 X12
            E(3.12647e-05) Y7 Z6
            E(3.12647e-05) Y7 Z6 X12
            E(3.12647e-05) Z7
            E(3.12647e-05) Z7 X12
            E(3.12647e-05) Z7 X6
            E(3.12647e-05) Z7 X6 X12
            E(3.12647e-05) Z7 Y6
            E(3.12647e-05) Z7 Y6 X12
            E(3.12647e-05) Z7 Z6
            E(3.12647e-05) Z7 Z6 X12
            M 12
            R 12
            CX 9 12 10 12
            E(3.12647e-05) X12
            E(3.12647e-05) X10
            E(3.12647e-05) X10 X12
            E(3.12647e-05) Y10
            E(3.12647e-05) Y10 X12
            E(3.12647e-05) Z10
            E(3.12647e-05) Z10 X12
            E(3.12647e-05) X9
            E(3.12647e-05) X9 X12
            E(3.12647e-05) X9 X10
            E(3.12647e-05) X9 X10 X12
            E(3.12647e-05) X9 Y10
            E(3.12647e-05) X9 Y10 X12
            E(3.12647e-05) X9 Z10
            E(3.12647e-05) X9 Z10 X12
            E(3.12647e-05) Y9
            E(3.12647e-05) Y9 X12
            E(3.12647e-05) Y9 X10
            E(3.12647e-05) Y9 X10 X12
            E(3.12647e-05) Y9 Y10
            E(3.12647e-05) Y9 Y10 X12
            E(3.12647e-05) Y9 Z10
            E(3.12647e-05) Y9 Z10 X12
            E(3.12647e-05) Z9
            E(3.12647e-05) Z9 X12
            E(3.12647e-05) Z9 X10
            E(3.12647e-05) Z9 X10 X12
            E(3.12647e-05) Z9 Y10
            E(3.12647e-05) Z9 Y10 X12
            E(3.12647e-05) Z9 Z10
            E(3.12647e-05) Z9 Z10 X12
            M 12
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
            R 12
            XCX 9 12 3 12
            E(3.12647e-05) X12
            E(3.12647e-05) X3
            E(3.12647e-05) X3 X12
            E(3.12647e-05) Y3
            E(3.12647e-05) Y3 X12
            E(3.12647e-05) Z3
            E(3.12647e-05) Z3 X12
            E(3.12647e-05) X9
            E(3.12647e-05) X9 X12
            E(3.12647e-05) X9 X3
            E(3.12647e-05) X9 X3 X12
            E(3.12647e-05) X9 Y3
            E(3.12647e-05) X9 Y3 X12
            E(3.12647e-05) X9 Z3
            E(3.12647e-05) X9 Z3 X12
            E(3.12647e-05) Y9
            E(3.12647e-05) Y9 X12
            E(3.12647e-05) Y9 X3
            E(3.12647e-05) Y9 X3 X12
            E(3.12647e-05) Y9 Y3
            E(3.12647e-05) Y9 Y3 X12
            E(3.12647e-05) Y9 Z3
            E(3.12647e-05) Y9 Z3 X12
            E(3.12647e-05) Z9
            E(3.12647e-05) Z9 X12
            E(3.12647e-05) Z9 X3
            E(3.12647e-05) Z9 X3 X12
            E(3.12647e-05) Z9 Y3
            E(3.12647e-05) Z9 Y3 X12
            E(3.12647e-05) Z9 Z3
            E(3.12647e-05) Z9 Z3 X12
            M 12
            R 12
            XCX 2 12 1 12
            E(3.12647e-05) X12
            E(3.12647e-05) X1
            E(3.12647e-05) X1 X12
            E(3.12647e-05) Y1
            E(3.12647e-05) Y1 X12
            E(3.12647e-05) Z1
            E(3.12647e-05) Z1 X12
            E(3.12647e-05) X2
            E(3.12647e-05) X2 X12
            E(3.12647e-05) X2 X1
            E(3.12647e-05) X2 X1 X12
            E(3.12647e-05) X2 Y1
            E(3.12647e-05) X2 Y1 X12
            E(3.12647e-05) X2 Z1
            E(3.12647e-05) X2 Z1 X12
            E(3.12647e-05) Y2
            E(3.12647e-05) Y2 X12
            E(3.12647e-05) Y2 X1
            E(3.12647e-05) Y2 X1 X12
            E(3.12647e-05) Y2 Y1
            E(3.12647e-05) Y2 Y1 X12
            E(3.12647e-05) Y2 Z1
            E(3.12647e-05) Y2 Z1 X12
            E(3.12647e-05) Z2
            E(3.12647e-05) Z2 X12
            E(3.12647e-05) Z2 X1
            E(3.12647e-05) Z2 X1 X12
            E(3.12647e-05) Z2 Y1
            E(3.12647e-05) Z2 Y1 X12
            E(3.12647e-05) Z2 Z1
            E(3.12647e-05) Z2 Z1 X12
            M 12
            R 12
            XCX 4 12 5 12
            E(3.12647e-05) X12
            E(3.12647e-05) X5
            E(3.12647e-05) X5 X12
            E(3.12647e-05) Y5
            E(3.12647e-05) Y5 X12
            E(3.12647e-05) Z5
            E(3.12647e-05) Z5 X12
            E(3.12647e-05) X4
            E(3.12647e-05) X4 X12
            E(3.12647e-05) X4 X5
            E(3.12647e-05) X4 X5 X12
            E(3.12647e-05) X4 Y5
            E(3.12647e-05) X4 Y5 X12
            E(3.12647e-05) X4 Z5
            E(3.12647e-05) X4 Z5 X12
            E(3.12647e-05) Y4
            E(3.12647e-05) Y4 X12
            E(3.12647e-05) Y4 X5
            E(3.12647e-05) Y4 X5 X12
            E(3.12647e-05) Y4 Y5
            E(3.12647e-05) Y4 Y5 X12
            E(3.12647e-05) Y4 Z5
            E(3.12647e-05) Y4 Z5 X12
            E(3.12647e-05) Z4
            E(3.12647e-05) Z4 X12
            E(3.12647e-05) Z4 X5
            E(3.12647e-05) Z4 X5 X12
            E(3.12647e-05) Z4 Y5
            E(3.12647e-05) Z4 Y5 X12
            E(3.12647e-05) Z4 Z5
            E(3.12647e-05) Z4 Z5 X12
            M 12
            R 12
            XCX 0 12 6 12
            E(3.12647e-05) X12
            E(3.12647e-05) X6
            E(3.12647e-05) X6 X12
            E(3.12647e-05) Y6
            E(3.12647e-05) Y6 X12
            E(3.12647e-05) Z6
            E(3.12647e-05) Z6 X12
            E(3.12647e-05) X0
            E(3.12647e-05) X0 X12
            E(3.12647e-05) X0 X6
            E(3.12647e-05) X0 X6 X12
            E(3.12647e-05) X0 Y6
            E(3.12647e-05) X0 Y6 X12
            E(3.12647e-05) X0 Z6
            E(3.12647e-05) X0 Z6 X12
            E(3.12647e-05) Y0
            E(3.12647e-05) Y0 X12
            E(3.12647e-05) Y0 X6
            E(3.12647e-05) Y0 X6 X12
            E(3.12647e-05) Y0 Y6
            E(3.12647e-05) Y0 Y6 X12
            E(3.12647e-05) Y0 Z6
            E(3.12647e-05) Y0 Z6 X12
            E(3.12647e-05) Z0
            E(3.12647e-05) Z0 X12
            E(3.12647e-05) Z0 X6
            E(3.12647e-05) Z0 X6 X12
            E(3.12647e-05) Z0 Y6
            E(3.12647e-05) Z0 Y6 X12
            E(3.12647e-05) Z0 Z6
            E(3.12647e-05) Z0 Z6 X12
            M 12
            R 12
            XCX 7 12 8 12
            E(3.12647e-05) X12
            E(3.12647e-05) X8
            E(3.12647e-05) X8 X12
            E(3.12647e-05) Y8
            E(3.12647e-05) Y8 X12
            E(3.12647e-05) Z8
            E(3.12647e-05) Z8 X12
            E(3.12647e-05) X7
            E(3.12647e-05) X7 X12
            E(3.12647e-05) X7 X8
            E(3.12647e-05) X7 X8 X12
            E(3.12647e-05) X7 Y8
            E(3.12647e-05) X7 Y8 X12
            E(3.12647e-05) X7 Z8
            E(3.12647e-05) X7 Z8 X12
            E(3.12647e-05) Y7
            E(3.12647e-05) Y7 X12
            E(3.12647e-05) Y7 X8
            E(3.12647e-05) Y7 X8 X12
            E(3.12647e-05) Y7 Y8
            E(3.12647e-05) Y7 Y8 X12
            E(3.12647e-05) Y7 Z8
            E(3.12647e-05) Y7 Z8 X12
            E(3.12647e-05) Z7
            E(3.12647e-05) Z7 X12
            E(3.12647e-05) Z7 X8
            E(3.12647e-05) Z7 X8 X12
            E(3.12647e-05) Z7 Y8
            E(3.12647e-05) Z7 Y8 X12
            E(3.12647e-05) Z7 Z8
            E(3.12647e-05) Z7 Z8 X12
            M 12
            R 12
            XCX 11 12 10 12
            E(3.12647e-05) X12
            E(3.12647e-05) X10
            E(3.12647e-05) X10 X12
            E(3.12647e-05) Y10
            E(3.12647e-05) Y10 X12
            E(3.12647e-05) Z10
            E(3.12647e-05) Z10 X12
            E(3.12647e-05) X11
            E(3.12647e-05) X11 X12
            E(3.12647e-05) X11 X10
            E(3.12647e-05) X11 X10 X12
            E(3.12647e-05) X11 Y10
            E(3.12647e-05) X11 Y10 X12
            E(3.12647e-05) X11 Z10
            E(3.12647e-05) X11 Z10 X12
            E(3.12647e-05) Y11
            E(3.12647e-05) Y11 X12
            E(3.12647e-05) Y11 X10
            E(3.12647e-05) Y11 X10 X12
            E(3.12647e-05) Y11 Y10
            E(3.12647e-05) Y11 Y10 X12
            E(3.12647e-05) Y11 Z10
            E(3.12647e-05) Y11 Z10 X12
            E(3.12647e-05) Z11
            E(3.12647e-05) Z11 X12
            E(3.12647e-05) Z11 X10
            E(3.12647e-05) Z11 X10 X12
            E(3.12647e-05) Z11 Y10
            E(3.12647e-05) Z11 Y10 X12
            E(3.12647e-05) Z11 Z10
            E(3.12647e-05) Z11 Z10 X12
            M 12
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        H_YZ 0 1 2 3 4 5 6 7 8 9 10 11
        TICK
        X_ERROR(0.0005) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 2, 0) rec[-36] rec[-35] rec[-32] rec[-30] rec[-29] rec[-26] rec[-18] rec[-17] rec[-14] rec[-11] rec[-10] rec[-9] rec[-5] rec[-4] rec[-3]
        DETECTOR(2, 5, 0) rec[-34] rec[-33] rec[-31] rec[-28] rec[-27] rec[-25] rec[-16] rec[-15] rec[-13] rec[-12] rec[-8] rec[-7] rec[-6] rec[-2] rec[-1]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-18] rec[-16] rec[-13] rec[-9] rec[-8] rec[-7] rec[-3] rec[-2] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-17] rec[-15] rec[-14] rec[-12] rec[-11] rec[-10] rec[-6] rec[-5] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-8] rec[-7]
    """)


def test_circuit_details_EM3_h_obs():
    actual = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2,
        data_height=6,
        sub_rounds=1003,
        noise=0.001,
        style="EM3",
        obs="H",
    ))
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(1, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(1, 2) 2
        QUBIT_COORDS(1, 3) 3
        QUBIT_COORDS(1, 4) 4
        QUBIT_COORDS(1, 5) 5
        QUBIT_COORDS(3, 0) 6
        QUBIT_COORDS(3, 1) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(3, 3) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11

        R 0 1 2 3 4 5 6 7 8 9 10 11
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        H 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        # X subround. Compare X parities to X initializations.
        DEPOLARIZE2(0.001) 9 3 2 1 4 5 0 6 7 8 11 10
        MPP(0.001) X9*X3 X2*X1 X4*X5 X0*X6 X7*X8 X11*X10
        OBSERVABLE_INCLUDE(1) rec[-3]
        DETECTOR(0, 3, 0) rec[-6]
        DETECTOR(1, 1.5, 0) rec[-5]
        DETECTOR(1, 4.5, 0) rec[-4]
        DETECTOR(2, 0, 0) rec[-3]
        DETECTOR(3, 1.5, 0) rec[-2]
        DETECTOR(3, 4.5, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK

        # Y subround. Get X*Y=Z stabilizers for first time.
        DEPOLARIZE2(0.001) 7 1 2 3 0 5 4 10 9 8 11 6
        MPP(0.001) Y7*Y1 Y2*Y3 Y0*Y5 Y4*Y10 Y9*Y8 Y11*Y6
        OBSERVABLE_INCLUDE(1) rec[-6]
        SHIFT_COORDS(0, 0, 1)
        TICK

        # Z subround. Get Y*Z=X stabilizers to compare against initialization.
        DEPOLARIZE2(0.001) 11 5 0 1 4 3 2 8 7 6 9 10
        MPP(0.001) Z11*Z5 Z0*Z1 Z4*Z3 Z2*Z8 Z7*Z6 Z9*Z10
        OBSERVABLE_INCLUDE(1) rec[-5] rec[-2]
        DETECTOR(0, 0, 0) rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 3, 0) rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK

        # X subround. Get Z*X=Y stabilizers for the first time.
        DEPOLARIZE2(0.001) 9 3 2 1 4 5 0 6 7 8 11 10
        MPP(0.001) X9*X3 X2*X1 X4*X5 X0*X6 X7*X8 X11*X10
        OBSERVABLE_INCLUDE(1) rec[-3]
        SHIFT_COORDS(0, 0, 1)
        TICK

        REPEAT 333 {
            # Y subround. Get X*Y = Z stabilizers to compare against last time.
            DEPOLARIZE2(0.001) 7 1 2 3 0 5 4 10 9 8 11 6
            MPP(0.001) Y7*Y1 Y2*Y3 Y0*Y5 Y4*Y10 Y9*Y8 Y11*Y6
            OBSERVABLE_INCLUDE(1) rec[-6]
            DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK

            # Z subround. Get Y*Z = X stabilizers to compare against last time.
            DEPOLARIZE2(0.001) 11 5 0 1 4 3 2 8 7 6 9 10
            MPP(0.001) Z11*Z5 Z0*Z1 Z4*Z3 Z2*Z8 Z7*Z6 Z9*Z10
            OBSERVABLE_INCLUDE(1) rec[-5] rec[-2]
            DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK

            # X subround. Get Z*X = Y stabilizers to compare against last time.
            DEPOLARIZE2(0.001) 9 3 2 1 4 5 0 6 7 8 11 10
            MPP(0.001) X9*X3 X2*X1 X4*X5 X0*X6 X7*X8 X11*X10
            OBSERVABLE_INCLUDE(1) rec[-3]
            DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }

        H 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        # Compare X data measurements to X parity measurements from last subround.
        DETECTOR(0, 3, 0) rec[-18] rec[-9] rec[-3]
        DETECTOR(1, 1.5, 0) rec[-17] rec[-11] rec[-10]
        DETECTOR(1, 4.5, 0) rec[-16] rec[-8] rec[-7]
        DETECTOR(2, 0, 0) rec[-15] rec[-12] rec[-6]
        DETECTOR(3, 1.5, 0) rec[-14] rec[-5] rec[-4]
        DETECTOR(3, 4.5, 0) rec[-13] rec[-2] rec[-1]
        # Compare X data measurements to previous X stabilizer reconstruction.
        DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-7] rec[-6] rec[-5] rec[-1]
        DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-8] rec[-4] rec[-3] rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-11] rec[-5]
    """)


def test_circuit_details_si500():
    actual = generate_honeycomb_circuit(HoneycombLayout(
        data_width=2,
        data_height=6,
        sub_rounds=3 * 300,
        noise=0.001,
        style="SI500",
        obs="V",
    ))
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(1, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(1, 2) 2
        QUBIT_COORDS(1, 3) 3
        QUBIT_COORDS(1, 4) 4
        QUBIT_COORDS(1, 5) 5
        QUBIT_COORDS(3, 0) 6
        QUBIT_COORDS(3, 1) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(3, 3) 9
        QUBIT_COORDS(3, 4) 10
        QUBIT_COORDS(3, 5) 11
        QUBIT_COORDS(0, 1) 12
        QUBIT_COORDS(0, 3) 13
        QUBIT_COORDS(0, 5) 14
        QUBIT_COORDS(1, 0.5) 15
        QUBIT_COORDS(1, 1.5) 16
        QUBIT_COORDS(1, 2.5) 17
        QUBIT_COORDS(1, 3.5) 18
        QUBIT_COORDS(1, 4.5) 19
        QUBIT_COORDS(1, 5.5) 20
        QUBIT_COORDS(2, 0) 21
        QUBIT_COORDS(2, 2) 22
        QUBIT_COORDS(2, 4) 23
        QUBIT_COORDS(3, 0.5) 24
        QUBIT_COORDS(3, 1.5) 25
        QUBIT_COORDS(3, 2.5) 26
        QUBIT_COORDS(3, 3.5) 27
        QUBIT_COORDS(3, 4.5) 28
        QUBIT_COORDS(3, 5.5) 29

        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        X_ERROR(0.002) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        TICK

        C_ZYX 0 2 4 7 9 11
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 1 3 5 6 8 10
        TICK

        # X sub-round part 1
        C_ZYX 1 3 5 6 8 10
        CZ 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE1(0.0001) 1 3 5 6 8 10
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE1(0.0001) 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        # X sub-round part 2
        C_ZYX 0 2 4 7 9 11
        CZ 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11
        DEPOLARIZE2(0.001) 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.0001) 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        # Y sub-round part 1
        C_ZYX 1 3 5 6 8 10
        CZ 7 12 2 17 0 20 4 23 9 26 11 29
        DEPOLARIZE1(0.0001) 1 3 5 6 8 10
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29
        DEPOLARIZE1(0.0001) 13 14 15 16 18 19 21 22 24 25 27 28
        TICK

        # Y sub-round part 2
        C_ZYX 0 2 4 7 9 11
        CZ 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11
        DEPOLARIZE2(0.001) 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE1(0.0001) 13 14 15 16 18 19 21 22 24 25 27 28
        TICK

        # Z sub-round part 1
        C_ZYX 1 3 5 6 8 10
        CZ 11 14 0 15 4 18 2 22 7 24 9 27
        DEPOLARIZE1(0.0001) 1 3 5 6 8 10
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27
        DEPOLARIZE1(0.0001) 12 13 16 17 19 20 21 23 25 26 28 29
        TICK

        # Z sub-round part 2
        CZ 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE2(0.001) 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11 12 13 16 17 19 20 21 23 25 26 28 29
        TICK

        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        DEPOLARIZE1(0.0001) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        # Finish first round.
        X_ERROR(0.005) 13 16 19 21 25 28 12 17 20 23 26 29 14 15 18 22 24 27
        M 13 16 19 21 25 28
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        M 12 17 20 23 26 29
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        M 14 15 18 22 24 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.0001) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.002) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        X_ERROR(0.002) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        DEPOLARIZE1(0.0001) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.002) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        C_ZYX 0 2 4 7 9 11
        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 1 3 5 6 8 10
        TICK

        # X sub-round part 1
        C_ZYX 1 3 5 6 8 10
        CZ 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE1(0.0001) 1 3 5 6 8 10
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE1(0.0001) 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        # X sub-round part 2
        C_ZYX 0 2 4 7 9 11
        CZ 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11
        DEPOLARIZE2(0.001) 3 13 1 16 5 19 6 21 8 25 10 28
        DEPOLARIZE1(0.0001) 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        # Y sub-round part 1
        C_ZYX 1 3 5 6 8 10
        CZ 7 12 2 17 0 20 4 23 9 26 11 29
        DEPOLARIZE1(0.0001) 1 3 5 6 8 10
        DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29
        DEPOLARIZE1(0.0001) 13 14 15 16 18 19 21 22 24 25 27 28
        TICK

        # Y sub-round part 2
        C_ZYX 0 2 4 7 9 11
        CZ 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11
        DEPOLARIZE2(0.001) 1 12 3 17 5 20 10 23 8 26 6 29
        DEPOLARIZE1(0.0001) 13 14 15 16 18 19 21 22 24 25 27 28
        TICK

        # Z sub-round part 1
        C_ZYX 1 3 5 6 8 10
        CZ 11 14 0 15 4 18 2 22 7 24 9 27
        DEPOLARIZE1(0.0001) 1 3 5 6 8 10
        DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27
        DEPOLARIZE1(0.0001) 12 13 16 17 19 20 21 23 25 26 28 29
        TICK

        # Z sub-round part 2
        CZ 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE2(0.001) 5 14 1 15 3 18 8 22 6 24 10 27
        DEPOLARIZE1(0.0001) 0 2 4 7 9 11 12 13 16 17 19 20 21 23 25 26 28 29
        TICK

        H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        DEPOLARIZE1(0.0001) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        # Finish second round.
        X_ERROR(0.005) 13 16 19 21 25 28 12 17 20 23 26 29 14 15 18 22 24 27
        M 13 16 19 21 25 28
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        M 12 17 20 23 26 29
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        M 14 15 18 22 24 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.0001) 0 1 2 3 4 5 6 7 8 9 10 11
        DEPOLARIZE1(0.002) 0 1 2 3 4 5 6 7 8 9 10 11
        TICK

        # Now in stable state for cross-round comparisons. Use a loop.
        REPEAT 298 {
            R 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
            X_ERROR(0.002) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
            DEPOLARIZE1(0.0001) 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.002) 0 1 2 3 4 5 6 7 8 9 10 11
            TICK

            C_ZYX 0 2 4 7 9 11
            H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
            DEPOLARIZE1(0.0001) 0 2 4 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 1 3 5 6 8 10
            TICK

            C_ZYX 1 3 5 6 8 10
            CZ 9 13 2 16 4 19 0 21 7 25 11 28
            DEPOLARIZE1(0.0001) 1 3 5 6 8 10
            DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28
            DEPOLARIZE1(0.0001) 12 14 15 17 18 20 22 23 24 26 27 29
            TICK

            C_ZYX 0 2 4 7 9 11
            CZ 3 13 1 16 5 19 6 21 8 25 10 28
            DEPOLARIZE1(0.0001) 0 2 4 7 9 11
            DEPOLARIZE2(0.001) 3 13 1 16 5 19 6 21 8 25 10 28
            DEPOLARIZE1(0.0001) 12 14 15 17 18 20 22 23 24 26 27 29
            TICK

            C_ZYX 1 3 5 6 8 10
            CZ 7 12 2 17 0 20 4 23 9 26 11 29
            DEPOLARIZE1(0.0001) 1 3 5 6 8 10
            DEPOLARIZE2(0.001) 7 12 2 17 0 20 4 23 9 26 11 29
            DEPOLARIZE1(0.0001) 13 14 15 16 18 19 21 22 24 25 27 28
            TICK

            C_ZYX 0 2 4 7 9 11
            CZ 1 12 3 17 5 20 10 23 8 26 6 29
            DEPOLARIZE1(0.0001) 0 2 4 7 9 11
            DEPOLARIZE2(0.001) 1 12 3 17 5 20 10 23 8 26 6 29
            DEPOLARIZE1(0.0001) 13 14 15 16 18 19 21 22 24 25 27 28
            TICK

            C_ZYX 1 3 5 6 8 10
            CZ 11 14 0 15 4 18 2 22 7 24 9 27
            DEPOLARIZE1(0.0001) 1 3 5 6 8 10
            DEPOLARIZE2(0.001) 11 14 0 15 4 18 2 22 7 24 9 27
            DEPOLARIZE1(0.0001) 12 13 16 17 19 20 21 23 25 26 28 29
            TICK

            CZ 5 14 1 15 3 18 8 22 6 24 10 27
            DEPOLARIZE2(0.001) 5 14 1 15 3 18 8 22 6 24 10 27
            DEPOLARIZE1(0.0001) 0 2 4 7 9 11 12 13 16 17 19 20 21 23 25 26 28 29
            TICK

            H 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
            DEPOLARIZE1(0.0001) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 0 1 2 3 4 5 6 7 8 9 10 11
            TICK

            X_ERROR(0.005) 13 16 19 21 25 28 12 17 20 23 26 29 14 15 18 22 24 27
            M 13 16 19 21 25 28
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            M 12 17 20 23 26 29
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            M 14 15 18 22 24 27
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.0001) 0 1 2 3 4 5 6 7 8 9 10 11
            DEPOLARIZE1(0.002) 0 1 2 3 4 5 6 7 8 9 10 11
            TICK
        }

        # Data measurement.
        X_ERROR(0.005) 0 1 2 3 4 5 6 7 8 9 10 11
        M 0 1 2 3 4 5 6 7 8 9 10 11
        DETECTOR(0, 5, 0) rec[-18] rec[-7] rec[-1]
        DETECTOR(1, 0.5, 0) rec[-17] rec[-12] rec[-11]
        DETECTOR(1, 3.5, 0) rec[-16] rec[-9] rec[-8]
        DETECTOR(2, 2, 0) rec[-15] rec[-10] rec[-4]
        DETECTOR(3, 0.5, 0) rec[-14] rec[-6] rec[-5]
        DETECTOR(3, 3.5, 0) rec[-13] rec[-3] rec[-2]
        DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-11] rec[-10] rec[-9] rec[-5] rec[-4] rec[-3]
        DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-12] rec[-8] rec[-7] rec[-6] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-8] rec[-7]
        DEPOLARIZE1(0.0001) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
        DEPOLARIZE1(0.002) 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    """)
