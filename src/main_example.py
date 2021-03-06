from collect_data import collect_simulated_experiment_data, read_recorded_data
from honeycomb_layout import HoneycombLayout
from plotting import plot_data


def main():
    collect_simulated_experiment_data(
        [
            HoneycombLayout(
                noise=p,
                data_width=d * 4,
                data_height=d * 6,
                sub_rounds=30,
                style="EM3_v2",
                obs="H",
            ).as_decoder_problem("pymatching")
            for d in [1, 2]
            for p in [0.001, 0.002, 0.003]
        ],
        out_path="test.csv",
        discard_previous_data=True,
        min_shots=10**3,
        max_shots=10**6,
        min_seen_logical_errors=10**2,
    )

    plot_data(
        read_recorded_data("test.csv"),
        title="LogLog per-sub-round error rates in periodic Honeycomb code under circuit noise",
        show=True)


if __name__ == '__main__':
    main()
