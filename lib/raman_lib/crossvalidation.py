from collections import defaultdict
import os
import logging

import numpy as np
import pandas as pd
import shap
from mlxtend.evaluate import mcnemar, mcnemar_table
from sklearn.base import (BaseEstimator, MetaEstimatorMixin, clone,
                          is_classifier)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (GridSearchCV, KFold, ParameterGrid,
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from tqdm.autonotebook import trange

from .misc import mode

cv_logger = logging.getLogger(__name__)

class CrossValidator(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        estimator,
        param_grid=None,
        scoring="accuracy",
        refit=True,
        coef_func=None,
        explainer=False,
        n_folds=5,
        n_trials=20,
        n_jobs=1,
        verbose=0,
        feature_names=None
    ):
        cv_logger.debug("Creating instance of CrossValidator")
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.coef_func = coef_func
        self.explainer = explainer
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.feature_names = feature_names

    def fit(self, X, y=None):

        self.do_gs = bool(self.param_grid)
        self.multi_score = not isinstance(self.scoring, str)

        cv_logger.debug("Setting up result arrays")
        self._prep_results(X, y)

        cv_logger.debug("Starting CV repetitions")
        for i in trange(self.n_trials):
            cv_logger.debug(f"Repetition {i}")

            cv_logger.debug("Getting CV splits")
            outer_cv, inner_cv = self._get_cv(random_state=i)

            dummy_results = self._get_dummy_results(X, y, outer_cv)

            if self.do_gs:
                cv_logger.debug("Starting grid search")
                estimator = self._do_gridsearch(X, y, inner_cv)
                cv_logger.debug("Grid search complete. Storing results...")
                self._store_cv_results(estimator, i)
                ct_jobs = 1

            else:
                estimator = self.estimator
                ct_jobs = self.n_jobs

            cv_logger.debug("Starting cross testing")
            ct_results_tmp = cross_validate(estimator,
                                            X, y,
                                            scoring=self.scoring,
                                            cv=outer_cv,
                                            return_estimator=True,
                                            return_train_score=True,
                                            n_jobs=ct_jobs,
                                            verbose=self.verbose)
            cv_logger.debug("Cross testing complete. Storing results...")
            self._store_ct_results(X, y, i,
                                   outer_cv,
                                   dummy_results,
                                   ct_results_tmp)

        if self.explainer:
            cv_logger.debug("Storing SHAP results")
            self.shap_results_ = self.shap_results_.mean()

        cv_logger.debug("Fitting final estimator")
        self._fit_final_estimator(X, y)

        return self

    def predict(self, X):
        """Pass-through from underlying estimator"""
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Pass-through from underlying estimator"""
        return self.estimator_.predict_proba(X)
    
    def decision_function(self, X):
        """Pass-through from underlying estimator"""
        return self.estimator_.decision_function(X)
    
    def transform(self, X):
        """Pass-through from underlying estimator"""
        return self.estimator_.transform(X)

    def score(self, X, y):
        """Pass-through from underlying estimator"""
        return self.estimator_.score(X, y)

    def _prep_results(self, X, y):
        if self.do_gs:
            cv_logger.debug("Creating arrays for grid search results")
            self.param_results_ = defaultdict(list)
            self.cv_results_ = defaultdict(list)
            for param in ParameterGrid(self.param_grid):
                for name, val in param.items():
                    self.cv_results_[f"param_{name}"].append(val)
        
        cv_logger.debug("Creating arrays for cross testing results")
        self.ct_results_ = defaultdict(list)
        self.predictions_ = defaultdict(
            lambda: np.zeros((self.n_trials, len(y))))

        if self.coef_func:
            cv_logger.debug("Creating array for model coefficients")
            self.coefs_ = np.zeros((self.n_trials, X.shape[1]))

        if self.explainer:
            cv_logger.debug("Creating array for SHAP explanations")
            self.shap_results_ = np.empty(self.n_trials, dtype=object)

    def _get_cv(self, random_state=None):

        if is_classifier(self.estimator):
            # StratifiedKFold to preserve class percentages
            outer_cv = StratifiedKFold(
                self.n_folds, shuffle=True, random_state=random_state)
            inner_cv = StratifiedKFold(
                self.n_folds, shuffle=True, random_state=random_state)
        else:
            outer_cv = KFold(self.n_folds, shuffle=True,
                             random_state=random_state)
            inner_cv = KFold(self.n_folds, shuffle=True,
                             random_state=random_state)

        return outer_cv, inner_cv

    def _get_dummy_results(self, X, y, outer_cv):
        if is_classifier(self.estimator):
            cv_logger.debug("Getting predictions from dummy classifier")
            dummy_results = cross_val_predict(
                DummyClassifier(), X, y, cv=outer_cv)
        else:
            cv_logger.debug("Getting predictions from dummy regressor")
            dummy_results = cross_val_predict(
                DummyRegressor(), X, y, cv=outer_cv)

        return dummy_results


    def _do_gridsearch(self, X, y, cv):

        gridsearch = GridSearchCV(self.estimator,
                                  param_grid=self.param_grid,
                                  scoring=self.scoring,
                                  refit=self.refit,
                                  return_train_score=True,
                                  cv=cv,
                                  verbose=self.verbose,
                                  n_jobs=self.n_jobs)

        gridsearch.fit(X, y)

        return gridsearch

    def _store_cv_results(self, gridsearch, i):
        if self.multi_score:
            for score in self.scoring:
                self.cv_results_[f"train_{score}_{i}"] = gridsearch.cv_results_[
                    f"mean_train_{score}"]
                self.cv_results_[f"test_{score}_{i}"] = gridsearch.cv_results_[
                    f"mean_test_{score}"]
        else:
            self.cv_results_[f"train_score_{i}"] = gridsearch.cv_results_[
                f"mean_train_score"]
            self.cv_results_[f"test_score_{i}"] = gridsearch.cv_results_[
                f"mean_test_score"]
        for name, val in gridsearch.best_params_.items():
            self.param_results_[name].append(val)


    def _store_ct_results(self, X, y, i, outer_cv, dummy_results, ct_results_tmp):
        if self.multi_score:
            for score in self.scoring:
                self.ct_results_[f"train_{score}"].append(ct_results_tmp[f"train_{score}"].mean())
                self.ct_results_[f"test_{score}"].append(ct_results_tmp[f"test_{score}"].mean())
        else:
            self.ct_results_[f"train_score"].append(ct_results_tmp["train_score"].mean())
            self.ct_results_[f"test_score"].append(ct_results_tmp["test_score"].mean())

        self.ct_results_[f"fit_time"].append(ct_results_tmp["fit_time"].mean())
        self.ct_results_[f"predict_time"].append(ct_results_tmp["score_time"].mean())

        if self.coef_func:
            coef_tmp = np.zeros((self.n_folds, X.shape[1]))

        if self.explainer:
            shap_vals = np.zeros((len(y), X.shape[1]))
            shap_base_vals = np.zeros(len(y))
            shap_data = np.zeros((len(y), X.shape[1]))

        for j, (train, test) in enumerate(outer_cv.split(X, y)):
            cv_logger.debug(f"Storing CT results of fold {j}")
            X_train, X_test = X[train], X[test]

            current_estimator = ct_results_tmp["estimator"][j]

            if self.do_gs:
                current_estimator = current_estimator.best_estimator_

            if self.coef_func:
                cv_logger.debug("Storing coefficients")
                coef_tmp[j,:] = self.coef_func(current_estimator).squeeze()

            cv_logger.debug("Storing predictions")
            y_pred = current_estimator.predict(X_test)
            self.predictions_["y_pred"][i, test] = y_pred

            if hasattr(current_estimator, "decision_function"):
                conf_scores = current_estimator.decision_function(X_test)
                self.predictions_["conf_scores"][i, test] = conf_scores

            if hasattr(current_estimator, "predict_proba"):
                proba = current_estimator.predict_proba(X_test)[:,1]
                self.predictions_["probability"][i, test] = proba

            if self.explainer:
                cv_logger.debug("Creating SHAP explanation")
                if isinstance(self.estimator, Pipeline):
                    current_prep = current_estimator[:-1]
                    current_estimator = current_estimator[-1]

                    X_train = current_prep.transform(X_train)
                    X_test = current_prep.transform(X_test)
                explainer = shap.Explainer(current_estimator, X_train)

                shap_tmp = explainer(X_test, check_additivity=False)
                shap_vals[test, :] = shap_tmp.values
                shap_base_vals[test] = shap_tmp.base_values
                shap_data[test, :] = shap_tmp.data

        cv_logger.debug("Performing McNemar test")
        mcn_table = mcnemar_table(y.ravel(),
                                  dummy_results.ravel(),
                                  self.predictions_["y_pred"][i].ravel())
        _, p_val = mcnemar(mcn_table)
        self.ct_results_["p_value"].append(p_val)

        cv_logger.debug("Averaging results")
        if self.coef_func:
            self.coefs_[i,:] = coef_tmp.mean(axis=0)

        if self.explainer:
            self.shap_results_[i] = shap.Explanation(shap_vals,
                                                     base_values=shap_base_vals,
                                                     data=shap_data,
                                                     feature_names=self.feature_names)
                                            
    
    def _fit_final_estimator(self, X, y):
        self.estimator_ = clone(self.estimator)

        if self.do_gs:
            cv_logger.debug("Getting parameters from grid search results")
            params_final_model = {}

            for param, vals in self.param_results_.items():
                if np.asarray(vals).dtype == float:
                    val_final = np.mean(vals)
                else:
                    val_final = mode(vals)

                params_final_model[param] = val_final

            cv_logger.debug("Setting parameters for final model")
            self.estimator_.set_params(**params_final_model)

        cv_logger.debug("Fitting final model")
        self.estimator_.fit(X, y)

    def to_csv(self, path):

        cv_logger.info(f"Saving results to: {path}")
        pd.DataFrame(self.ct_results_).to_csv(
            path / "ct_results.csv",
            index=False
        )
        pd.DataFrame(self.predictions_["y_pred"], dtype=int).to_csv(
            path / "predictions.csv",
            header=False, index=False
        )

        if "conf_scores" in self.predictions_:
            pd.DataFrame(self.predictions_["conf_scores"]).to_csv(
                path / "confidence_scores.csv",
                header=False, index=False
            )

        if "probability" in self.predictions_:
            pd.DataFrame(self.predictions_["probability"]).to_csv(
                path / "probabilities.csv",
                header=False, index=False
            )

        if self.do_gs:
            pd.DataFrame(self.cv_results_).to_csv(
                path / "cv_results.csv",
                index=False
            )

            pd.DataFrame(self.param_results_).to_csv(
                path / "param_results.csv",
                index=False
            )

        if self.coef_func:
            pd.DataFrame(self.coefs_, columns=self.feature_names).to_csv(
                path / "coefficients.csv",
                index=False
            )

        if self.explainer:
            os.makedirs(path / "shap", exist_ok=True)
            
            pd.DataFrame(self.shap_results_.values).to_csv(
                path / "shap" / "values.csv",
                index=False, header=False
            )

            pd.DataFrame(self.shap_results_.base_values).to_csv(
                path / "shap" / "base_values.csv",
                index=False, header=False
            )

            pd.DataFrame(self.shap_results_.data).to_csv(
                path / "shap" / "data.csv",
                index=False, header=False
            )

            pd.DataFrame(self.shap_results_.feature_names).to_csv(
                path / "shap" / "feature_names.csv",
                index=False, header=False
            )
