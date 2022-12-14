import sys
sys.path.append("./lib/")

import argparse
import os
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

from raman_lib.misc import load_data
from raman_lib.preprocessing import BaselineCorrector, RangeLimiter, SavGolFilter

# Prepare logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logdir  = Path("./log/03_preprocess_data/")

if not os.path.exists(logdir):
    os.makedirs(logdir)

dt = datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = logdir / f"{dt}.log"

handler_c = logging.StreamHandler()
handler_f = logging.FileHandler(logfile)

handler_c.setLevel(logging.INFO)
handler_f.setLevel(logging.DEBUG)

format_c = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
format_f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_c.setFormatter(format_c)
handler_f.setFormatter(format_f)

logger.addHandler(handler_c)
logger.addHandler(handler_f)


def preprocess(data, limits=(None, None), sg_window=15):
    X = data.drop(columns=["label", "file"], errors="ignore")
    wns = np.asarray(X.columns.astype(float))
    X = np.asarray(X)
    y = np.asarray(data.label)
    files = np.asarray(data.file)

    # Subtract baseline
    X = BaselineCorrector().fit_transform(X)

    # Reduce spectral range
    rl = RangeLimiter(lim=limits, reference=wns)
    X = rl.fit_transform(X)
    wns_reduced = wns[rl.lim_[0]:rl.lim_[1]]

    # Smooth spetra
    X = SavGolFilter(window=sg_window).fit_transform(X)

    # Normalize intensity
    X = Normalizer().fit_transform(X)

    # Combine data back to Dataframe
    data_prep = pd.DataFrame(X, columns=wns_reduced)
    data_prep.insert(0, "label", y)
    data_prep.insert(1, "file", files)

    return data_prep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-f", "--file", metavar="PATH", type=str, action="store",
                        help=".csv file containing the spectral data.", required=True)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, action="store",
                        help="Path for the output file.", required=True)
    parser.add_argument("-l", "--limits", metavar=("LOW", "HIGH"), type=float, nargs=2, action="store",
                        help="Limits for reducing the spectral range.", required=False, default=(None, None))
    parser.add_argument("-w", "--window", metavar="INT", type=int, action="store",
                        help="Window size for Savitzky-Golay smoothing", required=False, default=15)

    # TODO: Add arguments for other preprocessing steps

    logger.info("Parsing arguments")
    args = parser.parse_args()

    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    return args


if __name__ == "__main__":
    logger.info("Starting data preprocessing")
    args = parse_args()

    path_in = Path(args.file)
    path_out = Path(args.out)

    filename = path_in.stem.removesuffix("_qc")

    if not os.path.exists(path_out):
        logger.info("Creating output directory")
        os.makedirs(path_out)

    path_out = path_out / (filename + "_preprocessed.csv")

    logger.info(f"Loading data from {path_in}")
    data = load_data(path_in)
    logger.info("Finished loading data")

    data_prep = preprocess(data, limits=args.limits, sg_window=args.window)

    logger.info("Preprocessing complete")
    logger.info(f"Saving preprocessed data to {path_out}")
    data_prep.to_csv(path_out, index=False)
