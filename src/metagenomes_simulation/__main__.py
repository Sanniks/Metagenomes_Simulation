#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
import json
import argparse
import platform
from time import time as now
from metagenomes_simulation.network import (
    create_correlation_matrix,
    create_network_from_correlation_matrix,
    create_correlation_matrix_parallel,
)
from metagenomes_simulation.data_processing import (
    absolute_to_relative_abundance,
    group_otu_by_subject,
)

from metagenomes_simulation import __version__

__author__ = ["Carlo Mattia Lovecchio"]
__email__ = ["carlo.mattia02@gmail.com"]

RESET_COLOR_CODE = "\033[0m"
GREEN_COLOR_CODE = "\033[38;5;40m"
ORANGE_COLOR_CODE = "\033[38;5;208m"
VIOLET_COLOR_CODE = "\033[38;5;141m"
RED_COLOR_CODE = "\033[38;5;196m"
CRLF = "\r\x1b[K" if platform.system() != "Windows" else "\r\x1b[2K"


DEFAULT_RAW_OTU = Path("data/raw/otu_HMP2_16S.csv")
DEFAULT_META = Path("data/raw/meta_HMP2.csv")
DEFAULT_RELATIVE_OTU = Path("data/processed/otus_relative_abundance.csv")
DEFAULT_ABSOLUTE_SUBJECTS_DIR = Path("data/processed/otu_separated_by_subjects/absolute")
DEFAULT_RELATIVE_SUBJECTS_DIR = Path("data/processed/otu_separated_by_subjects/relative")
DEFAULT_MIN_SAMPLES = 5


def run_prepare_processed(args):
    print("This command will prepare processed data from raw OTU and metadata files.")
    print()
    print(f"Raw OTU file: {args.raw_otu}")
    print(f"Metadata file: {args.meta}")
    print(f"Relative OTU output: {args.relative_otu_output}")
    print(f"Absolute subject files directory: {args.absolute_subjects_dir}")
    print(f"Relative subject files directory: {args.relative_subjects_dir}")
    print(f"Minimum samples per subject: {args.min_samples}")
    print()

    if not args.yes:
        answer = input("Continue? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return
    print("Processing files...")

    otu_df = pd.read_csv(args.raw_otu, index_col=0)
    meta_df = pd.read_csv(args.meta, index_col=0)

    print("Creating relative abundance OTU table...")
    relative_otu_df = absolute_to_relative_abundance(otu_df)

    args.relative_otu_output.parent.mkdir(parents=True, exist_ok=True)
    args.absolute_subjects_dir.mkdir(parents=True, exist_ok=True)
    args.relative_subjects_dir.mkdir(parents=True, exist_ok=True)

    relative_otu_df.to_csv(args.relative_otu_output)
    print(f"Saved relative abundance table: {args.relative_otu_output}")

    print("Splitting absolute OTU table by subject...")
    absolute_by_subject = group_otu_by_subject(otu_df, meta_df)

    n_subjects = 0
    for subject_id, subject_df in absolute_by_subject.items():
        if len(subject_df) >= args.min_samples:
            subject_df.to_csv(args.absolute_subjects_dir / f"{subject_id}.csv")
            n_subjects += 1

    print(f"Saved {n_subjects} absolute subject files: {args.absolute_subjects_dir}")

    print("Splitting relative OTU table by subject...")
    relative_by_subject = group_otu_by_subject(relative_otu_df, meta_df)

    n_subjects = 0
    for subject_id, subject_df in relative_by_subject.items():
        if len(subject_df) >= args.min_samples:
            subject_df.to_csv(args.relative_subjects_dir / f"{subject_id}.csv")
            n_subjects += 1

    print(f"Saved {n_subjects} relative subject files: {args.relative_subjects_dir}")
    print("Done.")


def parse_args():
    """
    Parse command line arguments for the metagenomes_simulation package.
    This function sets up the argument parser, defines the expected arguments,
    and returns the parsed arguments.

    Returns
    -------
    argparse.Namespace
      The parsed command line arguments as a Namespace object.

    Raises
    -------
    argparse.ArgumentError
      If there is an error in the argument parsing, such as a missing required
      argument or an invalid value.
    """
    # global sofware information
    parser = argparse.ArgumentParser(
        prog="metagenomes_simulation",
        argument_default=None,
        add_help=True,
        prefix_chars="-",
        allow_abbrev=True,
        exit_on_error=True,
        description="A Python package for simulating metagenomic data and analyzing microbial communities.",
    )

    # prepare-data subcommand
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Prepare the data for analysis by processing raw OTU and metadata files.",
        description=(
            "Prepare raw metagenomic data for analysis. "
            "This command reads the raw OTU table and metadata table, creates a "
            "relative abundance OTU table, and splits both absolute and relative "
            "OTU tables into one CSV file per subject."
        ),
    )

    prepare_parser.add_argument(
        "--raw-otu",
        type=Path,
        default=DEFAULT_RAW_OTU,
        help=f"Raw OTU input CSV file. Default: {DEFAULT_RAW_OTU}",
    )
    prepare_parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help=f"Metadata input CSV file. Default: {DEFAULT_META}",
    )
    prepare_parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Minimum number of samples per subject to include. Default: {DEFAULT_MIN_SAMPLES}",
    )
    prepare_parser.add_argument(
        "--relative-otu-output",
        type=Path,
        default=DEFAULT_RELATIVE_OTU,
        help=f"Relative abundance OTU output CSV. Default: {DEFAULT_RELATIVE_OTU}",
    )
    prepare_parser.add_argument(
        "--absolute-subjects-dir",
        type=Path,
        default=DEFAULT_ABSOLUTE_SUBJECTS_DIR,
        help=(
            "Output directory for absolute OTU tables split by subject. "
            f"Default: {DEFAULT_ABSOLUTE_SUBJECTS_DIR}"
        ),
    )
    prepare_parser.add_argument(
        "--relative-subjects-dir",
        type=Path,
        default=DEFAULT_RELATIVE_SUBJECTS_DIR,
        help=(
            "Output directory for relative OTU tables split by subject. "
            f"Default: {DEFAULT_RELATIVE_SUBJECTS_DIR}"
        ),
    )
    prepare_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically confirm the operation without prompting.",
    )

    prepare_parser.set_defaults(func=run_prepare_processed)

    """
    # metagenomes_simulation --raw_otu <otu_file>
    # This option allows the user to specify the input files from which to read the data.
    # It should be a list of files separated by spaces.
    parser.add_argument(
        "--raw_otu",
        type=str,
        nargs=1,
        required=False,
        default=None,
        help=(
            "The input OTU file for which statistics will be computed."
            "It should be a .csv file."
            "Example: --raw_otu otu.csv"
        ),
    )

    # metagenomes_simulation --meta <metadata_file>
    # This option allows the user to specify the input files from which to read the data.
    # It should be a list of files separated by spaces.
    parser.add_argument(
        "--meta",
        type=str,
        nargs=1,
        required=False,
        default=None,
        help=(
            "The input metadata file for which statistics will be computed."
            "It should be a .csv file."
            "Example: --meta metadata.csv"
        ),
    )

    # metagenomes_simulation --num-workers <int>
    # This option allows the user to specify the number of worker threads
    # to use for parallel computation.
    parser.add_argument(
        "--num-workers",
        "-n",
        dest="num_workers",
        type=int,
        required=False,
        default=4,
        help="The number of worker threads to use for parallel computation. Default is 4.",
    )

    # metagenomes_simulation --output-dir <dir>
    # This option allows the user to specify the output directory.
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        required=False,
        default="./results",
        help="The directory where the output files will be saved.",
    )

    # metagenomes_simulation --split_raw
    # This option allows the user to split the raw data into multiple files.
    parser.add_argument(
        "--split_raw",
        "-S",
        dest="split",
        action="store_true",
        default=False,
        help=(
            "Split the raw data into multiple files based on the metadata."
            " Each file will contain data for a specific subject."
        ),
    )

    # metagenomes_simulation --relative
    # This option allows the user to transform the data to relative values.
    parser.add_argument(
        "--relative",
        "-R",
        dest="relative",
        action="store_true",
        default=False,
        help="Transform the data to relative values.",
    )

    # metagenomes_simulation --clr
    # This option allows the user to transform the data to central log ratio values.
    parser.add_argument(
        "--clr",
        "-C",
        dest="clr",
        action="store_true",
        default=False,
        help="Transform the data to central log ratio values.",
    )

    # metagenomes_simulation --noise
    # This option allows the user to add noise to the data.
    # Specify type of noise, mean and std; default is gaussian, mean = 0, std = 1.
    parser.add_argument(
        "--noise",
        "-N",
        dest="noise",
        nargs=3,
        default=["gaussian", "0", "1"],
        metavar=("TYPE", "MEAN", "STD"),
        help=(
            "Add noise to the data."
            " Specify type of noise, mean and standard deviation."
            " Default: type = gaussian, mean = 0, std = 1."
            " Example: --noise gaussian 0 1"
        ),
    )

    # metagenomes_simulation --sparsity
    # This option allows the user to compute the sparsity of the data.
    parser.add_argument(
        "--sparsity",
        dest="sparsity",
        action="store_true",
        default=False,
        help="Compute the sparsity of the data.",
    )
"""
    # metagenomes_simulation --version
    parser.add_argument(
        "--version",
        "-v",
        dest="version",
        required=False,
        action="store_true",
        default=False,
        help="Get the current version installed",
    )

    return parser


def main():
    # extract the arguments of the cmd
    parser = parse_args()
    args = parser.parse_args()

    # source: https://patorjk.com/software/taag
    print(
        rf"""{VIOLET_COLOR_CODE}
                                                                                              
 _____     _                                        _____ _           _     _   _             
|     |___| |_ ___ ___ ___ ___ ___ _____ ___ ___   |   __|_|_____ _ _| |___| |_|_|___ ___ ___ 
| | | | -_|  _| .'| . | -_|   | . |     | -_|_ -|  |__   | |     | | | | .'|  _| | . |   |_ -|
|_|_|_|___|_| |__,|_  |___|_|_|___|_|_|_|___|___|  |_____|_|_|_|_|___|_|__,|_| |_|___|_|_|___|
                  |___|                                                                       

    {RESET_COLOR_CODE}""",
        file=sys.stdout,
        flush=True,
    )

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)

    if args.version:
        print(__version__, file=sys.stdout, flush=True)
        exit(0)


if __name__ == "__main__":
    main()
