import sys

sys.path.append("./lib/")

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from lightgbm import LGBMClassifier

from raman_lib.crossvalidation import CrossValidator
from raman_lib.misc import load_data, int_float, TqdmStreamHandler

# Prepare logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


logdir = Path("./log/06_tree_based_models/")

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
    parser.add_argument("--tree-alpha", metavar=("min", "max", "n steps"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range of alpha values for pruning of decision trees using numpy.logspace.", 
                        default=[-2, 2, 5])
    parser.add_argument("--rf-feature-sample", metavar=("min", "max", "step"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range for subsampling of features in random forests using numpy.linspace.", 
                        default=[0.1, 1, 10])
    parser.add_argument("--gbdt-learning-rate", metavar=("min", "max", "n steps"), type=int_float, nargs=3, action="store", 
                        help="Used to set the range of learning rate values for gradient-boosted trees using np.linspace.", 
                        default=[0.05, 0.5, 10])

    logger.info("Parsing arguments")
    args = parser.parse_args()

    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    return args


if __name__ == "__main__":
    logger.info("Classification with tree-based models")
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

    # Basic Decision Tree

    logger.info("Classifier 1: Basic decision tree")
    tree_path_out = path_out / "decision_tree"

    if not os.path.exists(tree_path_out):
        logger.debug("Creating output directory")
        os.makedirs(tree_path_out)

    clf = DecisionTreeClassifier(random_state=653)

    param_grid = {
        "ccp_alpha": np.logspace(args.tree_alpha[0],
                                 args.tree_alpha[1],
                                 args.tree_alpha[2])
    }

    logger.info("Starting cross validation")
    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        feature_names=wns,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs
                        ).fit(X, y)

    cv.to_csv(tree_path_out)
    export_graphviz(cv.estimator_, 
                    out_file=str(tree_path_out / "tree.dot"), 
                    feature_names=wns,  
                    class_names=y_key,  
                    filled=True, rounded=True,  
                    special_characters=True,
                    leaves_parallel=True)
    
    logger.info("Cross validation complete")

    # Random Forest

    logger.info("Classifier 2: Random Forest")
    rf_path_out = path_out / "random_forest"

    if not os.path.exists(rf_path_out):
        logger.debug("Creating output directory")
        os.makedirs(rf_path_out)

    clf = LGBMClassifier(boosting_type="rf",
			 subsample=0.8,
                         subsample_freq=1,
                         max_bin=10,
                         max_depth=8,
                         random_state=2434)

    param_grid = {
        "colsample_bytree": np.linspace(args.rf_feature_sample[0],
                                        args.rf_feature_sample[1],
                                        args.rf_feature_sample[2]),
        }

    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs,
                        explainer=True,
                        feature_names=wns
                        ).fit(X, y)

    cv.to_csv(rf_path_out)
    
    logger.info("Cross validation complete")

    # Gradient-boosted Decision Trees

    logger.info("Classifier 3: Gradient-boosted Decision Trees")
    gbdt_path_out = path_out / "gbdt"

    if not os.path.exists(gbdt_path_out):
        logger.debug("Creating output directory")
        os.makedirs(gbdt_path_out)

    clf = LGBMClassifier(colsample_bytree=0.2,
                         max_bin=10,
                         max_depth=5,
                         random_state=6233)

    param_grid = {
        "learning_rate": np.linspace(args.gbdt_learning_rate[0],
                                     args.gbdt_learning_rate[1],
                                     args.gbdt_learning_rate[2])
        }

    cv = CrossValidator(clf,
                        param_grid,
                        scoring=args.scoring,
                        refit=refit,
                        n_folds=args.folds,
                        n_trials=args.trials,
                        n_jobs=args.jobs,
                        explainer=True,
                        feature_names=wns
                        ).fit(X, y)

    cv.to_csv(gbdt_path_out)
    
    logger.info("Cross validation complete")
