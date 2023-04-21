"""
Validate data according to a schema.

Usage:
python3 validate_data.py --csv_fp path/to/your_data.csv
"""

import argparse
from transfer.validation import validate_data
import pandas as pd


def main(input_fp: str):
    print(f"[INFO] validating data at {input_fp}")
    if input_fp.endswith(".pkl"):
        df = pd.read_pickle(input_fp)
    elif input_fp.endswith(".feather"):
        df = pd.read_feather(input_fp)
    else:
        raise ValueError(f"Unsupported file extension for {input_fp}")
    validate_data(df)
    print("[INFO] validation complete.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", help="path to the data file.",
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
