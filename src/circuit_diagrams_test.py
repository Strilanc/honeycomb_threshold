from circuit_diagrams import plot_circuit
from honeycomb_circuit import generate_honeycomb_circuit


def assert_same_diagram(actual, expected):
    s1 = str(actual)
    s2 = str(expected)
    if s1.strip() != s2.strip():
        import cirq
        print()
        print("!!!!!!")
        print(s1)
        print("!!!!!!")
        assert False, "Actual:\n.\n" + s1 + "\n.\nExpected:\n.\n" + s2 + "\n.\nDIFF:\n\n" + cirq.testing.highlight_text_differences(s1, s2)


def test_plot_six_cycle():
    assert_same_diagram(plot_circuit(generate_honeycomb_circuit(
        noise=0.001,
        tile_diam=1,
        sub_rounds=13,
        style="6",
    ), only_repeat_block=True), r"""
                                                                                                            ~
                                                                                                            |
                                                                          ~                                 |
                                                                          |                                 |
                                    X   ~                               \ |                                 X
                                    |M  |                                R|                                  \
                                    | \ |                                 X                                   X
                                    |  R|                                  \                                  |M
                                    |   X                                   X                                 | \
                                    |    \                                  |M                                |  R
                                    |                                       |                                 |
                                    |                                       |                                 |
                  @--------X        @        \        @        X--------@   |    X--------@        \        @-+------X
                   C        \        C        R       |C        M        C  |     \        C        R        C|       M
                    @        X--------@        X------+-@        \        @-+------X        @--------X        @        \
                    |C        M        C        \     |  C        R        C|       M        C        \        C        R
                  ~-+-@        \        @--------X    |   @--------X        @        \        @        X--------@        X----~
                    |  C        R        C        M   |    C        \        C        R       |C        M        C        \
                    |                                 |                                       |
                    |                                 |                                       |
                  \ |                                 X                                   X   |
                   R|                                  \                                  |M  |
                    X                                   X                                 | \ |
                     \                                  |M                                |  R|
                      X                                 | \                               |   X
                      |M                                |  R                              |    \
                      |                                 |                                 |
                      |                                 |                                 |
              ~---@   |    X--------@        \        @-+------X        @--------X        @        \        @        X----~
                   C  |     \        C        R        C|       M        C        \        C        R       |C        M
                    @-+------X        @--------X        @        \        @        X--------@        X------+-@        \
                     C|       M        C        \        C        R       |C        M        C        \     |  C        R
                      @        \        @        X--------@        X------+-@        \        @--------X    ~   @--------X
                       C        R       |C        M        C        \     |  C        R        C        M        C        \
                                        |                                 ~
                                        |
                                        ~
    """)


def test_plot_three_cycle():
    assert_same_diagram(plot_circuit(generate_honeycomb_circuit(
        noise=0.001,
        tile_diam=1,
        sub_rounds=20,
        style="3",
    ), only_repeat_block=True), r"""
                        ~
                        |                                                ~
                        X                       X ~                     D|
                         X                      |D|                      X
                         |D                     | X                       X
                         |                      |                         |
                         |                      |                         |
         ~--Y     D     X+----X     Y-----X     X     D     Y     X-----X |   X--~
             @-----X     Y     D     @     X-----Y     X----+@     D     Y+----X
              X     X-----@     X----+X     D     @-----X   | X-----X     @     D
              |                      |                      |
              |                      |                      |
            X |                     D|                      X
            |D|                      X                       X
            | X                       X                      |D
            |                         |                      |
            |                         |                      |
            X     D     Y     X-----X |   X-----Y     D     X+----X     Y-----X
          ~--Y     X----+@     D     Y+----X     @-----X     Y     D     @     X--~
              @-----X   | X-----X     @     D     X     X-----@     X----+X     D
                        ~                         |                      |
                                                  |                      ~
                                                  ~
    """)
