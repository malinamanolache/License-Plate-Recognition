"""Generates rodosol .json file used for testing as ground truth."""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rodosol_path",
        type=str,
        required=True,
        help="Path to rodosol dataset",
    )
