import argparse
import pathlib
import sys
from pathlib import Path

from paper.main_collect_all import honeycomb_problems, surface_code_problems

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Non-package import directory hack.

def main():
    parser = argparse.ArgumentParser(description="Find uncompleted jobs.")
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--surface_code_problems_directory', type=str, required=False, help="A directory of surface code problems to also do.")
    args = vars(parser.parse_args())
    surface_dir = args.get('surface_code_problems_directory') or f"{pathlib.Path(__file__).parent}/surface_code_circuits"
    csv_dir = args.get('csv_dir')

    problems = honeycomb_problems() + surface_code_problems(surface_dir)

    csvs = []
    p = pathlib.Path(csv_dir)
    if p.is_dir():
        csvs.extend(p.glob("*.csv"))
    else:
        csvs.append(p)

    seen = {int(str(e).split('/')[-1].split('.')[0]) for e in csvs}
    missed = [k for k in range(len(problems)) if k not in seen]
    for k in missed:
        if k not in seen:
            print(k)


if __name__ == '__main__':
    main()
