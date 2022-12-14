import sys

sys.path.append("./lib/")

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from raman_lib.crossvalidation import CrossValidator
from raman_lib.misc import TqdmStreamHandler, load_data, int_float

# Prepare logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


logdir = Path("./log/05_regularized_models/")

if not os.path.exists(logdir):
    os.makedirs(logdir)

dt = datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = logdir / f"{dt}.log"

handler_c = TqdmStreamHandler()
handler_f = logging.FileHandler(logfile)

handler_c.setLevel(logging.INFO)
handler_f.setLevel(logging.DEBUG)

format_c = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
format_f = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    parser.add_argument("--logreg-l1-c", metavar=("min", "max", "n steps"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range of C values for crossvalidation of LogReg with l1 term using numpy.logspace.", 
                        default=[-2, 2, 5])
    parser.add_argument("--logreg-l2-c", metavar=("min", "max", "n steps"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range of C values for crossvalidation of LogReg with l2 term using numpy.logspace.", 
                        default=[-2, 2, 5])
    parser.add_argument("--svm-l1-c", metavar=("min", "max", "n steps"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range of C values for crossvalidation of Linear SVM with l1 term using numpy.logspace.", 
                        default=[-2, 2, 5])
    parser.add_argument("--svm-l2-c", metavar=("min", "max", "n steps"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range of C values for crossvalidation of Linear SVM with l2 term using numpy.logspace.", 
                        default=[-2, 2, 5])


    logger.info("Parsing arguments")
    args = parser.parse_args()

    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    return args


if __name__ == "__main__":
    logger.info("Classification with regularized logistic regression and SVM")
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

    # Logistic Regression with l1 term

    logger.info("Classifier 1: Logistic Regression with l1 Regularization")
    lg_l1_path_out = path_out / "logreg_l1"

    if not os.path.exists(lg_l1_path_out):
        logger.debug("Creating output directory")
        os.makedirs(lg_l1_path_out)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear",
                                      penalty="l1",
                                      max_iter=1000,
                                      random_state=41))
    ])

    param_grid = {
        "logreg__C": np.logspace(args.logreg_l1_c[0],
                                 args.logreg_l1_c[1],
                                 args.logreg_l1_c[2])
    }

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: x[1].coef_,
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(lg_l1_path_out)
    logger.info("Cross validation complete")

    # Logistic Regression with l2 term

    logger.info("Classifier 2: Logistic Regression with l2 Regularization")
    lg_l2_path_out = path_out / "logreg_l2"

    if not os.path.exists(lg_l2_path_out):
        logger.debug("Creating output directory")
        os.makedirs(lg_l2_path_out)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear",
                                      penalty="l2",
                                      random_state=51,
                                      max_iter=1000))
    ])

    param_grid = {
        "logreg__C": np.logspace(args.logreg_l2_c[0],
                                 args.logreg_l2_c[1],
                                 args.logreg_l2_c[2])
    }

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: x[1].coef_,
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(lg_l2_path_out)
    logger.info("Cross validation complete")


    # Linear SVM with l1 term

    logger.info("Classifier 3: Linear SVM with l1 Regularization")
    svm_l1_path_out = path_out / "svm_l1"

    if not os.path.exists(svm_l1_path_out):
        logger.debug("Creating output directory")
        os.makedirs(svm_l1_path_out)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(penalty="l1", dual=False, max_iter=10000))
    ])

    param_grid = {
        "svm__C": np.logspace(args.svm_l1_c[0],
                              args.svm_l1_c[1],     
                              args.svm_l1_c[2])
    }

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: x[1].coef_,
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(svm_l1_path_out)
    logger.info("Cross validation complete")

    # Linear SVM with l2 term

    logger.info("Classifier 4: Linear SVM with l2 Regularization")
    svm_l2_path_out = path_out / "svm_l2"

    if not os.path.exists(svm_l2_path_out):
        logger.debug("Creating output directory")
        os.makedirs(svm_l2_path_out)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(penalty="l2", max_iter=5000))
    ])

    param_grid = {
        "svm__C": np.logspace(args.svm_l2_c[0],
                              args.svm_l2_c[1],     
                              args.svm_l2_c[2])
    }

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        coef_func=lambda x: x[1].coef_,
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(svm_l2_path_out)
    logger.info("Cross validation complete")

