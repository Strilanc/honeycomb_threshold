from circuit_diagrams import plot_circuit
from honeycomb_circuit import generate_honeycomb_circuit
from honeycomb_layout import HoneycombLayout


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


def test_plot_SD6():
    assert_same_diagram(plot_circuit(generate_honeycomb_circuit(HoneycombLayout(
        noise=0.001,
        data_width=2,
        data_height=6,
        sub_rounds=13,
        style="SD6",
        obs="V",
    )), only_repeat_block=True), r"""
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


def test_plot_PC3():
    assert_same_diagram(plot_circuit(generate_honeycomb_circuit(HoneycombLayout(
        noise=0.001,
        data_width=2,
        data_height=6,
        sub_rounds=20,
        style="PC3",
        obs="V",
    )), only_repeat_block=True), r"""
                                                ~
                         ~                      |
                        M|                      X                       X ~
                         X                       X                      |M|
                          X                      |M                     | X
                          |                      |                      |
                          |                      |                      |
            X     X-----@ |   X-----X     M     @+----X     X-----X     @     M
          ~-+Y     M     X+----X     Y-----X     X     M     Y     X-----X     X--~
            | @-----X     Y     M     @     X-----Y     X----+@     M     Y-----X
            |                         |                      |
            |                         |                      |
            X                       X |                     M|
             X                      |M|                      X
             |M                     | X                       X
             |                      |                         |
             |                      |                         |
            @+----X     X-----X     @     M     X     X-----@ |   X-----X     M
             X     M     Y     X-----X     X----+Y     M     X+----X     Y-----X
           ~--Y     X----+@     M     Y-----X   | @-----X     Y     M     @     X--~
                         |                      ~                         |
                         ~                                                |
                                                                          ~
    """)


def test_plot_EM3():
    assert_same_diagram(plot_circuit(generate_honeycomb_circuit(HoneycombLayout(
        noise=0.001,
        data_width=2 * 2,
        data_height=6 * 2,
        sub_rounds=20,
        style="EM3",
        obs="V",
    )), only_repeat_block=True), r"""
                        ~                                                                       ~
                        |                                                ~                      |                                                ~
         ~--y           y           y-----------y ~         y           y+----------y           y           y-----------y ~         y           y+-~
             z-----------z           z           z+---------+z           z           z-----------z           z           z+---------+z           z
              x           x----------+x           x         | x-----------x           x           x----------+x           x         | x-----------x
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
         ~--y |         y           y+----------y           y           y-----------y |         y           y+----------y           y           y--~
             z+---------+z           z           z-----------z           z           z+---------+z           z           z-----------z           z
              x         | x-----------x           x           x----------+x           x         | x-----------x           x           x----------+x
                        |                         |                      |                      |                         |                      |
                        |                         |                      |                      |                         |                      |
                        |                         |                      |                      |                         |                      |
                        |                         |                      |                      |                         |                      |
                        |                         |                      |                      |                         |                      |
                        |                         |                      |                      |                         |                      |
                        |                         |                      |                      |                         |                      |
         ~--y           y           y-----------y |         y           y+----------y           y           y-----------y |         y           y+-~
             z-----------z           z           z+---------+z           z           z-----------z           z           z+---------+z           z
              x           x----------+x           x         | x-----------x           x           x----------+x           x         | x-----------x
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
              |                      |                      |                         |                      |                      |
         ~--y |         y           y+----------y           y           y-----------y |         y           y+----------y           y           y--~
             z+---------+z           z           z-----------z           z           z+---------+z           z           z-----------z           z
              x         | x-----------x           x           x----------+x           x         | x-----------x           x           x----------+x
                        ~                         |                      |                      ~                         |                      |
                                                  |                      ~                                                |                      ~
                                                  ~                                                                       ~
    """)
