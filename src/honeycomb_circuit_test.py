import itertools

import pytest
import stim

from honeycomb_circuit import generate_honeycomb_circuit


@pytest.mark.parametrize('tile_diam,sub_rounds', itertools.product(
    range(1, 5),
    range(1, 24),
))
def test_circuit_has_decomposing_error_model(tile_diam: int, sub_rounds: int):
    circuit = generate_honeycomb_circuit(
        tile_diam=tile_diam,
        sub_rounds=sub_rounds,
        noise=0.001,
    )
    _ = circuit.detector_error_model(decompose_errors=True)


def test_circuit_details():
    actual = generate_honeycomb_circuit(
        tile_diam=1,
        sub_rounds=1003,
        noise=0.001,
    )
    cleaned = stim.Circuit(str(actual))
    assert cleaned == stim.Circuit("""
        QUBIT_COORDS(0, 1) 0
        QUBIT_COORDS(0, 3) 1
        QUBIT_COORDS(0, 5) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 0.5) 4
        QUBIT_COORDS(1, 1) 5
        QUBIT_COORDS(1, 1.5) 6
        QUBIT_COORDS(1, 2) 7
        QUBIT_COORDS(1, 2.5) 8
        QUBIT_COORDS(1, 3) 9
        QUBIT_COORDS(1, 3.5) 10
        QUBIT_COORDS(1, 4) 11
        QUBIT_COORDS(1, 4.5) 12
        QUBIT_COORDS(1, 5) 13
        QUBIT_COORDS(1, 5.5) 14
        QUBIT_COORDS(2, 0) 15
        QUBIT_COORDS(2, 2) 16
        QUBIT_COORDS(2, 4) 17
        QUBIT_COORDS(3, 0) 18
        QUBIT_COORDS(3, 0.5) 19
        QUBIT_COORDS(3, 1) 20
        QUBIT_COORDS(3, 1.5) 21
        QUBIT_COORDS(3, 2) 22
        QUBIT_COORDS(3, 2.5) 23
        QUBIT_COORDS(3, 3) 24
        QUBIT_COORDS(3, 3.5) 25
        QUBIT_COORDS(3, 4) 26
        QUBIT_COORDS(3, 4.5) 27
        QUBIT_COORDS(3, 5) 28
        QUBIT_COORDS(3, 5.5) 29

        R 3 5 7 9 11 13 18 20 22 24 26 28
        X_ERROR(0.001) 3 5 7 9 11 13 18 20 22 24 26 28
        DEPOLARIZE1(0.001) 0 1 2 4 6 8 10 12 14 15 16 17 19 21 23 25 27 29
        TICK

        R 1 6 12 15 21 27
        C_XYZ 3 7 11 20 24 28
        X_ERROR(0.001) 1 6 12 15 21 27
        DEPOLARIZE1(0.001) 3 7 11 20 24 28 0 2 4 5 8 9 10 13 14 16 17 18 19 22 23 25 26 29
        TICK

        CX 24 1 7 6 11 12 3 15 20 21 28 27
        DEPOLARIZE2(0.001) 24 1 7 6 11 12 3 15 20 21 28 27
        DEPOLARIZE1(0.001) 0 2 4 5 8 9 10 13 14 16 17 18 19 22 23 25 26 29
        TICK

        R 0 8 14 17 23 29
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        X_ERROR(0.001) 0 8 14 17 23 29
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 1 2 4 6 10 12 15 16 19 21 25 27
        TICK

        CX 20 0 7 8 3 14 11 17 24 23 28 29
        CX 9 1 5 6 13 12 18 15 22 21 26 27
        DEPOLARIZE2(0.001) 20 0 7 8 3 14 11 17 24 23 28 29 9 1 5 6 13 12 18 15 22 21 26 27
        DEPOLARIZE1(0.001) 2 4 10 16 19 25
        TICK

        X_ERROR(0.001) 1 6 12 15 21 27
        R 2 4 10 16 19 25
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        M 1 6 12 15 21 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 2 4 10 16 19 25
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 0 8 14 17 23 29
        TICK

        CX 28 2 3 4 11 10 7 16 20 19 24 25
        CX 5 0 9 8 13 14 26 17 22 23 18 29
        DEPOLARIZE2(0.001) 28 2 3 4 11 10 7 16 20 19 24 25 5 0 9 8 13 14 26 17 22 23 18 29
        DEPOLARIZE1(0.001) 1 6 12 15 21 27
        TICK

        X_ERROR(0.001) 0 8 14 17 23 29
        R 1 6 12 15 21 27
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        M 0 8 14 17 23 29
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 1 6 12 15 21 27
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 2 4 10 16 19 25
        TICK

        CX 24 1 7 6 11 12 3 15 20 21 28 27
        CX 13 2 5 4 9 10 22 16 18 19 26 25
        DEPOLARIZE2(0.001) 24 1 7 6 11 12 3 15 20 21 28 27 13 2 5 4 9 10 22 16 18 19 26 25
        DEPOLARIZE1(0.001) 0 8 14 17 23 29
        TICK

        X_ERROR(0.001) 2 4 10 16 19 25
        R 0 8 14 17 23 29
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        M 2 4 10 16 19 25
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 0 8 14 17 23 29
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 1 6 12 15 21 27
        TICK

        CX 20 0 7 8 3 14 11 17 24 23 28 29
        CX 9 1 5 6 13 12 18 15 22 21 26 27
        DEPOLARIZE2(0.001) 20 0 7 8 3 14 11 17 24 23 28 29 9 1 5 6 13 12 18 15 22 21 26 27
        DEPOLARIZE1(0.001) 2 4 10 16 19 25
        TICK

        X_ERROR(0.001) 1 6 12 15 21 27
        R 2 4 10 16 19 25
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        M 1 6 12 15 21 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 2 4 10 16 19 25
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 0 8 14 17 23 29
        TICK

        REPEAT 332 {
            CX 28 2 3 4 11 10 7 16 20 19 24 25
            CX 5 0 9 8 13 14 26 17 22 23 18 29
            DEPOLARIZE2(0.001) 28 2 3 4 11 10 7 16 20 19 24 25 5 0 9 8 13 14 26 17 22 23 18 29
            DEPOLARIZE1(0.001) 1 6 12 15 21 27
            TICK

            X_ERROR(0.001) 0 8 14 17 23 29
            R 1 6 12 15 21 27
            C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
            M 0 8 14 17 23 29
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            X_ERROR(0.001) 1 6 12 15 21 27
            DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 2 4 10 16 19 25
            TICK

            CX 24 1 7 6 11 12 3 15 20 21 28 27
            CX 13 2 5 4 9 10 22 16 18 19 26 25
            DEPOLARIZE2(0.001) 24 1 7 6 11 12 3 15 20 21 28 27 13 2 5 4 9 10 22 16 18 19 26 25
            DEPOLARIZE1(0.001) 0 8 14 17 23 29
            TICK

            X_ERROR(0.001) 2 4 10 16 19 25
            R 0 8 14 17 23 29
            C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
            M 2 4 10 16 19 25
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
            DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            X_ERROR(0.001) 0 8 14 17 23 29
            DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 1 6 12 15 21 27
            TICK

            CX 20 0 7 8 3 14 11 17 24 23 28 29
            CX 9 1 5 6 13 12 18 15 22 21 26 27
            DEPOLARIZE2(0.001) 20 0 7 8 3 14 11 17 24 23 28 29 9 1 5 6 13 12 18 15 22 21 26 27
            DEPOLARIZE1(0.001) 2 4 10 16 19 25
            TICK

            X_ERROR(0.001) 1 6 12 15 21 27
            R 2 4 10 16 19 25
            C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
            M 1 6 12 15 21 27
            OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
            DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
            DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
            SHIFT_COORDS(0, 0, 1)
            X_ERROR(0.001) 2 4 10 16 19 25
            DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 0 8 14 17 23 29
            TICK
        }

        CX 28 2 3 4 11 10 7 16 20 19 24 25
        CX 5 0 9 8 13 14 26 17 22 23 18 29
        DEPOLARIZE2(0.001) 28 2 3 4 11 10 7 16 20 19 24 25 5 0 9 8 13 14 26 17 22 23 18 29
        DEPOLARIZE1(0.001) 1 6 12 15 21 27
        TICK

        X_ERROR(0.001) 0 8 14 17 23 29
        R 1 6 12 15 21 27
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        M 0 8 14 17 23 29
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 2, 0) rec[-30] rec[-29] rec[-26] rec[-24] rec[-23] rec[-20] rec[-12] rec[-11] rec[-8] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 5, 0) rec[-28] rec[-27] rec[-25] rec[-22] rec[-21] rec[-19] rec[-10] rec[-9] rec[-7] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        X_ERROR(0.001) 1 6 12 15 21 27
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 2 4 10 16 19 25
        TICK

        CX 24 1 7 6 11 12 3 15 20 21 28 27
        CX 13 2 5 4 9 10 22 16 18 19 26 25
        DEPOLARIZE2(0.001) 24 1 7 6 11 12 3 15 20 21 28 27 13 2 5 4 9 10 22 16 18 19 26 25
        DEPOLARIZE1(0.001) 0 8 14 17 23 29
        TICK

        X_ERROR(0.001) 2 4 10 16 19 25
        M 2 4 10 16 19 25
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 0, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-23] rec[-20] rec[-12] rec[-10] rec[-7] rec[-6] rec[-5] rec[-2]
        DETECTOR(2, 3, 0) rec[-29] rec[-27] rec[-26] rec[-22] rec[-21] rec[-19] rec[-11] rec[-9] rec[-8] rec[-4] rec[-3] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 1 3 5 6 7 8 9 11 12 13 14 15 17 18 20 21 22 23 24 26 27 28 29
        TICK

        C_XYZ 5 9 13 18 22 26
        DEPOLARIZE1(0.001) 5 9 13 18 22 26 0 1 2 3 4 6 7 8 10 11 12 14 15 16 17 19 20 21 23 24 25 27 28 29
        TICK

        CX 9 1 5 6 13 12 18 15 22 21 26 27
        DEPOLARIZE2(0.001) 9 1 5 6 13 12 18 15 22 21 26 27
        DEPOLARIZE1(0.001) 0 2 3 4 7 8 10 11 14 16 17 19 20 23 24 25 28 29
        TICK

        X_ERROR(0.001) 1 6 12 15 21 27
        M 1 6 12 15 21 27
        OBSERVABLE_INCLUDE(0) rec[-5] rec[-4]
        DETECTOR(0, 4, 0) rec[-30] rec[-28] rec[-25] rec[-24] rec[-22] rec[-19] rec[-12] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1]
        DETECTOR(2, 1, 0) rec[-29] rec[-27] rec[-26] rec[-23] rec[-21] rec[-20] rec[-11] rec[-9] rec[-8] rec[-5] rec[-3] rec[-2]
        SHIFT_COORDS(0, 0, 1)
        C_XYZ 3 5 7 9 11 13 18 20 22 24 26 28
        DEPOLARIZE1(0.001) 3 5 7 9 11 13 18 20 22 24 26 28 0 2 4 8 10 14 16 17 19 23 25 29
        TICK

        X_ERROR(0.001) 3 5 7 9 11 13 18 20 22 24 26 28
        M 3 5 7 9 11 13 18 20 22 24 26 28
        DETECTOR(0, 2, 0) rec[-36] rec[-35] rec[-32] rec[-30] rec[-29] rec[-26] rec[-18] rec[-17] rec[-14] rec[-11] rec[-10] rec[-9] rec[-5] rec[-4] rec[-3]
        DETECTOR(2, 5, 0) rec[-34] rec[-33] rec[-31] rec[-28] rec[-27] rec[-25] rec[-16] rec[-15] rec[-13] rec[-12] rec[-8] rec[-7] rec[-6] rec[-2] rec[-1]
        DETECTOR(0, 4, 0) rec[-24] rec[-22] rec[-19] rec[-18] rec[-16] rec[-13] rec[-9] rec[-8] rec[-7] rec[-3] rec[-2] rec[-1]
        DETECTOR(2, 1, 0) rec[-23] rec[-21] rec[-20] rec[-17] rec[-15] rec[-14] rec[-12] rec[-11] rec[-10] rec[-6] rec[-5] rec[-4]
        OBSERVABLE_INCLUDE(0) rec[-11] rec[-10] rec[-8] rec[-7]
        DEPOLARIZE1(0.001) 0 1 2 4 6 8 10 12 14 15 16 17 19 21 23 25 27 29
    """)
