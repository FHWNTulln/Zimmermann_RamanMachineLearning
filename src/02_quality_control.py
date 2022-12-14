import sys

sys.path.append("./lib/")

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path

import pandas as pd
from raman_lib.preprocessing import RangeLimiter
from raman_lib.spectra_scoring import score_names, score_sort_spectra
from raman_lib.visualization import plot_spectra_peaks

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logdir  = Path("./log/02_quality_control/")

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sort spectra by quality score using Savitzky Golay based peak detection")

    parser.add_argument_group()

    parser.add_argument("-f", "--file", type=str,
                        help="csv-file containing the dataset.", required=True)
    parser.add_argument("-o", "--out", type=str,
                        help="Path for the output directory.", required=True)
    parser.add_argument("-l", "--limits", metavar=("LOW", "HIGH"), type=float, nargs=2, default=[None, None],
                        help="Set limits to reduce the range of x-values. Default: None")
    parser.add_argument("-b", "--baseline", type=str, choices={
                        "asls", "arpls", "mormol", "rollingball", "beads"}, default="asls")
    parser.add_argument("-w", "--window", type=int, default=35,
                        help="Window half-width used for the Savitzky-Golay-Filter before detecting peaks. Default: 17")
    parser.add_argument("-t", "--threshold", type=float, default=1,
                        help="Threshold for the 1st derivative for a peak to be accepted. Default: 1")
    parser.add_argument("-m", "--min-height", type=float, default=0,
                        help="Minimum height for a peak to be accepted. Default: 5")
    parser.add_argument("-s", "--score", type=int, choices={0, 1, 2, 3, 4}, default=1,
                        help="Measure to use for scoring spectra; 0: None, use 1 as the base score; 1: Median peak height; 2: Mean peak height; 3: Mean peak area; 4: Total peak area. Default: 1")
    parser.add_argument("-p", "--peaks", type=int, choices={0, 1, 2}, default=1,
                        help="How the number of peaks influences the score; 0: No influence; 1: Multiplicative, 2: Exponential. Default: 1")
    parser.add_argument("-n", "--numspectra", type=int, default=None,
                        help="Number of spectra to keep for each class. If not provided, all spectra are used.")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Use if scoring results should be plotted", required=False)

    logger.info("Parsing arguments")
    args = parser.parse_args()

    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    logger.info("Checking arguments")
    check_args_validity(args)

    return args


def check_args_validity(args):
    assert args.score != 0 and args.peaks != 0, "Peak intensity and peak number cannot both be ignored."
    assert args.window % 2 != 0 and args.window > 0, "Savitzky Golay window size must be a positive odd number."
    assert args.threshold >= 0, "Threshold must be positive or zero."


if __name__ == "__main__":
    logger.info("Starting quality control")
    args = parse_arguments()

    path_in = Path(args.file)
    path_out = Path(args.out)

    if not os.path.exists(path_out):
        logger.info("Creating output directory")
        os.makedirs(path_out)

    path_out_data = path_out / (path_in.stem + "_qc.csv")
    path_out_scores = path_out / (path_in.stem + "_qc_scores.csv")

    logger.info(f"Loading data from file {path_in}")
    data = pd.read_csv(path_in)
    logger.info("Data loaded sucessfully")

    logger.info("Running quality control")
    data_out, _, score_dict = score_sort_spectra(data,
                                              n=args.numspectra,
                                              limits=args.limits,
                                              bl_method=args.baseline,
                                              sg_window=args.window,
                                              threshold=args.threshold,
                                              min_height=args.min_height,
                                              score_measure=args.score,
                                              n_peaks_influence=args.peaks,
                                              detailed=True)

    if args.visualize:

        data_vis = data.drop(columns=["label", "file"]).values.astype(float)
        wns_vis = data.drop(columns=["label", "file"]).columns.astype(float)

        rl = RangeLimiter(lim=args.limits,
                          reference=wns_vis)

        data_rl = rl.fit_transform(data_vis)
        wns_rl = wns_vis[rl.lim_[0]:rl.lim_[1]]

        logger.info("Plotting spectra")
        plot_spectra_peaks(wns_rl,
                           data_rl,
                           score_dict["peak_pos"],
                           labels=score_dict["total_scores"])

    logger.info(f"Saving data to file {path_out_data}")
    data_out.to_csv(path_out_data, index=False)

    logger.info(f"Saving quality scores to file {path_out_scores}")
    pd.DataFrame({score_names[args.score]: score_dict["intensity_scores"], 
                  "N Peaks": score_dict["peak_scores"]}).to_csv(
        path_out_scores, index=False
    )
