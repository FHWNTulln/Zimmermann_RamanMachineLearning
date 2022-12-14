import sys

sys.path.append("./lib/")

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from raman_lib.crossvalidation import CrossValidator
from raman_lib.misc import TqdmStreamHandler, load_data
from raman_lib.preprocessing import PeakPicker

# Prepare logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


logdir  = Path("./log/04_lda_dim_reduction/")

if not os.path.exists(logdir):
    os.makedirs(logdir)

dt = datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = logdir / f"{dt}.log"

handler_c = TqdmStreamHandler()
handler_f = logging.FileHandler(logfile)

handler_c.setLevel(logging.INFO)
handler_f.setLevel(logging.DEBUG)

format_c = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
format_f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_c.setFormatter(format_c)
handler_f.setFormatter(format_f)

logger.addHandler(handler_c)
logger.addHandler(handler_f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-f", "--file", metavar="PATH", type=str, action="store",
                        help=".csv file containing the spectral data.", required=True)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, action="store",
                        help="Path for the output directory.", required=True)
    parser.add_argument("-s", "--scoring", metavar="SCORE", type=str, nargs="*", action="store", 
                        help="Scoring metrics to use. The first one will be used for hyperparameter selection.", default="accuracy")
    parser.add_argument("-t", "--trials", metavar="INT", type=int, action="store", 
                        help="Number of trials for the randomized nested crossvalidation.", default=20)
    parser.add_argument("-k", "--folds", metavar="INT", type=int, action="store", 
                        help="Number of folds for crossvalidation.", default=5)
    parser.add_argument("-j", "--jobs", metavar="INT", type=int, action="store", 
                        help="Number of parallel jobs. Set as -1 to use all available processors", default=1)
    parser.add_argument("-p", "--pca-components", metavar=("min", "max", "step"), type=int, nargs=3, action="store", 
                        help="Used to set the range of principal components for crossvalidation.", default=[1, 20, 1])
    parser.add_argument("-n", "--nmf-components", metavar=("min", "max", "step"), type=int, nargs=3, action="store", 
                        help="Used to set the range of NMF components for crossvalidation.", default=[1, 20, 1])
    parser.add_argument("-c", "--fa-clusters", metavar=("min", "max", "step"), type=int, nargs=3, action="store", 
                        help="Used to set the range of feature agglomeration clusters for crossvalidation.", default=[1, 20, 1])
    parser.add_argument("-d", "--peak-distance", metavar=("min", "max", "step"), type=int, nargs=3, action="store", 
                        help="Used to set the range of minimal peak distance for crossvalidation.", default=[1, 20, 1])

    logger.info("Parsing arguments")
    args = parser.parse_args()

    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    return args


if __name__ == "__main__":
    logger.info("Classification with LDA and dimensionality reduction")
    args = parse_args()

    path_in = Path(args.file)
    path_out = Path(args.out)

    filename = path_in.stem

    logger.info(f"Loading data from {path_in}")
    data = load_data(path_in)
    
    logger.info("Parsing data")
    X = data.drop(columns=["label", "file"], errors="ignore")
    wns = np.asarray(X.columns.astype(float))
    X = np.asarray(X)
    y = np.asarray(data.label)
    y, y_key = pd.factorize(y)
    logger.info("Data import complete")

    if isinstance(args.scoring, str):
        refit = True
    else:
        refit = args.scoring[0]

    # LDA only

    logger.info("Classifier 1: LDA only")
    lda_path_out = path_out / "lda"

    if not os.path.exists(lda_path_out):
        logger.debug("Creating output directory")
        os.makedirs(lda_path_out)

    clf = LinearDiscriminantAnalysis()

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        scoring=args.scoring,
                        coef_func=lambda x: x.scalings_,
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(lda_path_out)
    logger.info("Cross validation complete")

    # PCA-LDA

    logger.info("Classifier 2: PCA-LDA")
    pca_path_out = path_out / "pca_lda"

    if not os.path.exists(pca_path_out):
        logger.debug("Creating output directory")
        os.makedirs(pca_path_out)

    clf = Pipeline([("pca", PCA()),
                    ("lda", LinearDiscriminantAnalysis())])

    param_grid = {"pca__n_components": range(
        args.pca_components[0],
        args.pca_components[1],
        args.pca_components[2]
    )}

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: np.matmul(x[0].components_.T,
                                                      x[1].scalings_),
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(pca_path_out)
    logger.info("Cross validation complete")

    # NMF-LDA

    logger.info("Classifier 3: NMF-LDA")
    nmf_path_out = path_out / "nmf_lda"

    if not os.path.exists(nmf_path_out):
        os.makedirs(nmf_path_out)

    clf = Pipeline([("nmf", NMF(init="nndsvda", tol=1e-2, max_iter=5000)),
                    ("lda", LinearDiscriminantAnalysis())])

    param_grid = {"nmf__n_components": range(
        args.nmf_components[0],
        args.nmf_components[1],
        args.nmf_components[2]
    )}

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: np.matmul(x[0].components_.T,
                                                    x[1].scalings_),
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)
    
    cv.to_csv(nmf_path_out)
    logger.info("Cross validation complete")

    # FA-LDA

    logger.info("Classifier 4: FA-LDA")
    fa_path_out = path_out / "fa_lda"

    if not os.path.exists(fa_path_out):
        os.makedirs(fa_path_out)

    clf = Pipeline([("agglo", FeatureAgglomeration(connectivity=np.diag(np.ones(len(wns))) +
                                                                np.diag(np.ones(len(wns)-1), 1) +
                                                                np.diag(np.ones(len(wns)-1), -1))),
                    ("lda", LinearDiscriminantAnalysis())])

    param_grid = {"agglo__n_clusters": range(
        args.fa_clusters[0],
        args.fa_clusters[1],
        args.fa_clusters[2]
    )}

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: x[1].scalings_[x[0].labels_],
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(fa_path_out)
    logger.info("Cross validation complete")

    # Peak-LDA

    logger.info("Classifier 5: Peak-LDA")
    peak_path_out = path_out / "peak_lda"

    if not os.path.exists(peak_path_out):
        os.makedirs(peak_path_out)

    clf = Pipeline([("peaks", PeakPicker()),
                    ("lda", LinearDiscriminantAnalysis())])

    param_grid = {"peaks__min_dist": range(
        args.peak_distance[0],
        args.peak_distance[1],
        args.peak_distance[2]
    )}

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: np.matmul(x[0].peaks_.T,
                                                      x[1].scalings_),
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(peak_path_out)
    logger.info("Cross validation complete")
