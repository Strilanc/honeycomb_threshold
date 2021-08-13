import pathlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

from collect_data import read_recorded_data, ProblemShotData, CSV_HEADER, CSV_HEADER_VERSION

def main():
    if len(sys.argv) == 1:
        raise ValueError("Specify csv files to include as command line arguments.")

    csvs = []
    for path in sys.argv[1:]:
        p = pathlib.Path(path)
        if p.is_dir():
            csvs.extend(p.glob("*.csv"))
        else:
            csvs.append(p)

    all_data = read_recorded_data(*csvs)
    all_data.data = {k: v for k, v in sorted(all_data.data.items())}

    print(CSV_HEADER)
    for desc, shot_data in all_data.data.items():
        print(",".join(str(e) for e in [
            desc.data_width,
            desc.data_height,
            desc.rounds,
            desc.noise,
            desc.circuit_style,
            desc.preserved_observable,
            desc.code_distance,
            desc.num_qubits,
            shot_data.num_shots,
            shot_data.num_correct,
            shot_data.total_processing_seconds,
            desc.decoder,
            CSV_HEADER_VERSION,
        ]))


if __name__ == '__main__':
    main()
