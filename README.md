# RamanMachineLearning

This repository contains the machine-learning-based data analysis workflow for Raman spectra developed during the master's thesis "Classification of Cell Systems using Raman Spectroscopy and Machine Learning" by Daniel Zimmermann.

## Setup

After downloading this tool, first adjust the contents of `raman_ml.conf` according to your needs. The following tables list the parameters that can be adjusted.

### General Parameters
| Parameter | Description |
| --------- | ----------- |
| `FILE_PREFIX` | Filename prefix that will be used for all output files |
| `N_TRIALS` | Number of randomized repetitions for cross-validation |
| `N_FOLDS` | Number of folds for (nested) k-fold cross-validation |
| `N_CORES` | Number of CPU cores that will be used (-1 for all available) |
| `SCORING` | Performance metrics to be calculated during cross-validation |
| `CONDA_DIR` | Directory of your *conda* installation |
| `ENV_NAME` | Name that will be used for the *conda* environment |
| `DIR1`/`DIR2` | Directories where the individual spectra are stored |
| `LAB1`/`LAB2` | Class labels corresponding to `DIR1`/`DIR2` |

### QC/Preprocessing Parameters
| Parameter | Description |
| --------- | ----------- |
| `QC_LIM_LOW`/`HIGH` | Wavenumber range that should be considered for quality control |
| `QC_WINDOW` | Window size for the Savitzky-Golay filter during quality control |
| `QC_THRESHOLD` | Minimum value of the derivate spectrum for a peak to be detected |
| `QC_MIN_HEIGHT` | Minimum height for a peak to be detected |
| `QC_SCORE` | How the intensity of the spectrum influences the quality score. 0 - None, 1 - Median peak height, 2 - Mean peak height, 3 - Mean area, 4 - Total area |
| `QC_PEAKS` | How the number of peaks influences the quality score. 0 - None, 1 - Linear, 2 - Quadratic |
| `QC_NUM` | How many spectra to keep from each class | 
| `PREP_LIM_LOW`/`HIGH` | Wavenumber range that will be retained during preprocessing |
| `PREP_WINDOW` | Window size for the Savitzky-Golay filter during preprocessing |

### Classification Model Parameters
Here, value ranges must be entered which will be optimized during cross-validation. Some value ranges are log-scaled. For these, the entered value represents the power of 10 of the actual parameter value (see also `numpy.logspace`). For a more detailed description of each parameter refer to the `scikit-learn` documentation.

| Parameter | Description |
| --------- | ----------- |
| `PCA_COMP` | Range of pca-components to test. Format: (min max+1 stepsize) |
| `NMF_COMP` | Range of nmf-components to test. Format: (min max+1 stepsize) |
| `FA_CLUST` | Range of the number of clusters to test in Feature Agglomeration-LDA. Format: (min max+1 stepsize) |
| `PEAK_DIST` | Range of the minimum peak distance to test in Peak-LDA. Format: (min max+1 stepsize) |
| `LR1_C` | Range of values for the regularization parameter C to test in logistic regression (l1). Logarithmic. Format: (min max n_steps) |
| `LR2_C` | Range of values for C to test in logistic regression (l2). Logarithmic. Format: (min max n_steps)` |
| `SVM1_C` | Range of values for C to test in SVM (l1). Logarithmic. Format: (min max n_steps) |
| `SVM2_C` | Range of values for C to test in SVM (l2). Logarithmic. Format: (min max n_steps) |
| `DT_ALPHA` | Range of values for the parameter α of cost-complexity-pruning to test in the decision tree model. Logarithmic. Format: (min max n_steps) |
| `RF_FEATURE_SAMPLE` | Range of the feature subsample parameter in the random forest model. Format: (min max n_steps) |
| `GBDT_LEARNING_RATE` | Range of the learning rate parameter in the gradient-boosting model. Format: (min max n_steps) |

To run the workflow, execute the file `run.sh` from the terminal and follow the instructions on screen.
## Examining Results
To examine the result of each ML method,  *IPython* notebooks are provided in the `notebooks` directory.
`inspect_data.ipynb` can be used to take a look at the raw spectra. 
`inspect_results_full.ipynb` shows the following results for each model:
- Table of scoring metrics
- Validation curve (if applicable)
- Distribution of optimal parameter values (if applicable)
- Confidence scores or probabilities per class
- Model coefficients (if applicable)
- Confusion matrix
- ROC curve
- Shapley values for interpretation (only random forest & gradient-boosting)