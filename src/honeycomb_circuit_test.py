import itertools

import pytest

from honeycomb_circuit import generate_honeycomb_circuit
from hack_pycharm_pybind_pytest_workaround import stim


@pytest.mark.parametrize('tile_diam,sub_rounds,style', itertools.product(
    range(1, 5),
    range(1, 24),
    ["PC3", "SD6", "EM3"],
))
def test_circuit_has_decomposing_error_model(tile_diam: int, sub_rounds: int, style: str):
    circuit = generate_honeycomb_circuit(
        tile_diam=tile_diam,
        sub_rounds=sub_rounds,
        noise=0.001,
        style=style,
    )
    _ = circuit.detector_error_model(decompose_errors=True)


def test_circuit_details_SD6():
    actual = generate_honeycomb_circuit(
        tile_diam=1,
        sub_rounds=1003,
        noise=0.001,
        style="SD6",
    )
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
        C_XYZ 0 2 4 7 9 11
        X_ERROR(0.001) 13 16 19 21 25 28
        DEPOLARIZE1(0.001) 0 2 4 7 9 11 1 3 5 6 8 10 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        CX 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE2(0.001) 9 13 2 16 4 19 0 21 7 25 11 28
        DEPOLARIZE1(0.001) 1 3 5 6 8 10 12 14 15 17 18 20 22 23 24 26 27 29
        TICK

        R 12 17 20 23 26 29
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
            C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
            C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
            C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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
        C_XYZ 0 1 2 3 4 5 6 7 8 9 10 11
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

        C_XYZ 1 3 5 6 8 10
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
    actual = generate_honeycomb_circuit(
        tile_diam=1,
        sub_rounds=1003,
        noise=0.001,
        style="PC3",
    )
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

        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11
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
    actual = generate_honeycomb_circuit(
        tile_diam=1,
        sub_rounds=1003,
        noise=0.001,
        style="EM3",
    )
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
