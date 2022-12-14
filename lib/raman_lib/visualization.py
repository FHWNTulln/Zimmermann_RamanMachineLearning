import re
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from ipywidgets import Button, HBox
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.signal import find_peaks
from sklearn.metrics import (auc, confusion_matrix, roc_curve)


def plot_spectra_peaks(wns, signal, deriv, peaks, scores, labels=None):

    signal = np.asarray(signal)
    deriv = np.asarray(deriv)
    fig, (ax1, ax2) = plt.subplots(2,1)
    #plt.subplots_adjust(bottom=0.2)

    if labels is not None:
        fig.suptitle(labels[0])

    line1, = ax1.plot(wns, signal[0, :])
    peakmarks = ax1.scatter(wns[peaks[0]], signal[0, :][peaks[0]],
                           c="red", marker="x", s=50, zorder=3)
    score_text = ax1.text(0.01, 0.01, f"Score: {int(scores[0])}", transform=ax1.transAxes)

    ax1.set_xlim(wns[0], wns[-1])
    ax1.grid()

    ax1.set_xlabel("Raman Shift ($\mathregular{cm^{-1}}$)",
                  fontdict={"weight": "bold", "size": 12})
    ax2.set_ylabel("Intensity (-)")

    line2, = ax2.plot(wns, deriv[0,:])

    ax2.set_xlim(wns[0], wns[-1])
    ax2.grid()

    ax2.set_xlabel("Raman Shift ($\mathregular{cm^{-1}}$)",
                  fontdict={"weight": "bold", "size": 12})
    ax2.set_ylabel("1st derivative")

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(signal)
            ydata = signal[i, :]
            line1.set_ydata(ydata)

            marks = np.array([[wns[peak], signal[i][peak]]
                             for peak in peaks[i]])
            if len(marks) == 0:
                peakmarks.set_visible(False)
            else:
                peakmarks.set_visible(True)
                peakmarks.set_offsets(marks)
            
            score_text.set_text(f"Score: {int(scores[i])}")
            if labels is not None:
                fig.suptitle(labels[i])

            ax1.relim()
            ax1.autoscale_view()

            line2.set_ydata(deriv[i,:])
            ax2.relim()
            ax2.autoscale_view()

            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(signal)
            ydata = signal[i, :]
            line1.set_ydata(ydata)

            marks = np.array([[wns[peak], signal[i][peak]]
                             for peak in peaks[i]])
            if len(marks) == 0:
                peakmarks.set_visible(False)
            else:
                peakmarks.set_visible(True)
                peakmarks.set_offsets(marks)

            score_text.set_text(f"Score: {int(scores[i])}")
            if labels is not None:
                fig.suptitle(labels[i])

            ax1.relim()
            ax1.autoscale_view()

            line2.set_ydata(deriv[i,:])
            ax2.relim()
            ax2.autoscale_view()

            plt.draw()

    callback = Index()

    bnext = Button(description='Next')
    bprev = Button(description='Previous')

    buttons = HBox(children=[bprev, bnext])
    display(buttons)

    bnext.on_click(callback.next)
    bprev.on_click(callback.prev)

    plt.show()



def split_by_sign(x, y):
    x1, x2, y1, y2 = np.stack(
        [x[:-1],  x[1:], y[:-1], y[1:]])[:, np.diff(y < 0)]
    xf = x1 + -y1 * (x2 - x1) / (y2 - y1)

    i = np.searchsorted(x, xf)
    x0 = np.insert(x, i, xf)
    y0 = np.insert(y, i, 0)

    y_neg = np.ma.masked_array(y0, mask=y0 > 0)
    y_zero = np.ma.masked_array(y0, mask=y0 != 0)
    y_pos = np.ma.masked_array(y0, mask=y0 < 0)

    return x0, y_neg, y_zero, y_pos


def plot_validation_curve(cv_results, score="accuracy", ax=None):

    param = np.asarray(cv_results.filter(regex="param_")).squeeze()
    train_scores = np.asarray(cv_results.filter(regex=f"train_{score}_"))
    test_scores = np.asarray(cv_results.filter(regex=f"test_{score}_"))
    
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)

    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    if ax is None:
        ax = plt.gca()

    ax.plot(param, train_scores_mean, color="k", linestyle="dashed", label="Training")
    ax.plot(param, test_scores_mean, color="k", label="Validation")

    ax.fill_between(param, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, color="k", alpha=0.3)
    ax.fill_between(param, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, color="k", alpha=0.3)

  


# def plot_val_curves(cv_results, score="accuracy", x_labels=None, y_label="Score", log_scale=False, figsize=(8,6)):
#     if not isinstance(cv_results, pd.DataFrame):
#         raise TypeError("Pandas DataFrame required.")
#     if isinstance(x_labels, str):
#         x_labels = [x_labels]

#     params = cv_results.filter(regex="param_")
#     train_scores = np.asarray(cv_results.filter(regex=f"train_{score}_"))
#     test_scores = np.asarray(cv_results.filter(regex=f"test_{score}_"))

#     if len(params.columns) == 1:
        
#         if x_labels is None:
#             x_labels = [params.columns[0]]

#         x = np.asarray(params).ravel()
#         plot_validation_curve(x, train_scores, test_scores, x_label=x_labels[0], y_label=y_label, log_scale=log_scale, figsize=figsize)
    
#     else:
#         if x_labels is None:
#             x_labels = params.columns
#         if isinstance(log_scale, bool):
#             log_scale = np.full(len(params.columns), log_scale, dtype=bool)

#         max_i = np.unravel_index(np.argmax(test_scores, axis=None), test_scores.shape)[0]
#         max_params = params.iloc[max_i,:]

#         for i, (param, vals) in enumerate(params.iteritems()):
#             max_params_tmp = max_params.drop(param)
#             indices = np.ones(len(params), dtype=bool)
#             for p, val in max_params_tmp.iteritems():
#                 indices = np.bitwise_and(indices, params[p] == val)
#             x = np.asarray(vals[indices])
#             plot_validation_curve(x, train_scores[indices], test_scores[indices], x_label=x_labels[i], y_label=y_label, log_scale=log_scale[i], figsize=figsize)


def annotate_peaks(x, y, ax, min_height=0, min_dist=None, offset=6, fontsize=None):

    if min_dist is None:
        min_dist = len(x) // 100

    peaks_pos = find_peaks(y, height=min_height, distance=min_dist)[0]
    peaks_neg = find_peaks(np.negative(
        y), height=min_height, distance=min_dist)[0]

    for peak in peaks_pos:
        ax.annotate(str(int(x[peak])),
                    (x[peak], y[peak]),
                    fontsize=fontsize,
                    xytext=(0, offset),
                    textcoords="offset points",
                    rotation=90,
                    ha="center")

    for peak in peaks_neg:
        ax.annotate(str(int(x[peak])),
                    (x[peak], y[peak]),
                    fontsize=fontsize,
                    xytext=(0, -offset),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="top")


def plot_coefs(coefs, features=None, show_range=False, ax=None,
               col=True, annotate=False, annot_kw=None):

    if isinstance(coefs, pd.DataFrame):
        features = np.asarray(coefs.columns.astype(float))
    elif isinstance(coefs, pd.Series):
        features = np.asarray(coefs.index.astype(float))
    elif features is None:
        features = range(len(coefs[0]))

    coefs = np.asarray(coefs)

    if len(coefs.shape) == 1:
        coefs_plot = coefs
    elif len(coefs.shape) == 2:
        coefs_plot = np.mean(coefs, axis=0)
    else:
        raise ValueError("Only 1 or 2-dimensional arrays are supported.")

    if ax is None:
        ax=plt.gca()

    ax.axhline(c="black", alpha=0.5, linewidth=1)

    if col:
        features_0, coefs_neg, coefs_0, coefs_pos = split_by_sign(features, 
                                                                  coefs_plot)

        ax.plot(features_0, coefs_neg, color="C0")
        ax.plot(features_0, coefs_pos, color="C1")
        ax.plot(features_0, coefs_0, color="k")
    else:
        ax.plot(features, coefs_plot, color="k")

    if show_range:
        coefs_std = np.std(coefs, axis=0)
        coefs_lower = coefs_plot - coefs_std
        coefs_upper = coefs_plot + coefs_std

        ax.fill_between(features, coefs_lower, coefs_upper,
                        color="grey", alpha=0.7, edgecolor=None)

    if annotate:
        annotate_peaks(features, coefs_plot, ax, **annot_kw)


def plot_confidence_scores(scores, groups, order=None, markersize=5, ax=None):

    scores_plot = scores.mean(axis=0)

    if ax is None:
        ax=plt.gca()

    sns.boxplot(
        x=groups,
        y=scores_plot,
        order=order,
        ax=ax,
        showfliers=False,
        boxprops={"facecolor": "white"}
    )

    sns.stripplot(
        x=groups,
        y=scores_plot,
        order=order,
        size=markersize,
        ax=ax
    )


def plot_confusion_matrix(y_pred, y_true, labels= (0, 1), ax=None, colorbar=False, **kwargs):

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    conf_matrices = np.asarray(
        [confusion_matrix(y_true, y_pred[i, :]) for i in range(len(y_pred))]
    )

    conf_matrix_plot = conf_matrices.mean(axis=0)
    vmax = conf_matrix_plot.sum(axis=1).max()
    n_classes = conf_matrix_plot.shape[0]

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(conf_matrix_plot, vmin=0, vmax=vmax, interpolation="none", **kwargs)
    col_min, col_max = im.cmap(0), im.cmap(1.0)

    for i in range(n_classes):
        for j in range(n_classes):
            text = f"{conf_matrix_plot[i,j]:.1f}"
            textcol = col_max if conf_matrix_plot[i,j] < (vmax / 2) else col_min

            ax.text(j, i, text, ha="center", va="center", color=textcol)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    if colorbar:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="3%")
        ax.figure.colorbar(im, cax=cax)


    # ConfusionMatrixDisplay(conf_matrix_plot).plot(values_format=".1f", 
    #                                               ax=ax, 
    #                                               im_kw={
    #                                                 "vmin":0,
    #                                                 "vmax":vmax
    #                                               }, 
    #                                               **kwargs)

  

def plot_roc_curve(conf_scores, y, label, ax=None):

    if not isinstance(conf_scores, np.ndarray):
        conf_scores = np.asarray(conf_scores)
    mean_fpr = np.linspace(0, 1, 200)
    aucs = []
    tprs = []

    if ax is None:
        ax = plt.gca()

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for row in conf_scores:
        fpr, tpr, _ = roc_curve(y, row)
        ax.plot(fpr, tpr, color="k", alpha=0.2, linewidth=1)
        aucs.append(auc(fpr, tpr))
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1

    aucs_mean = np.mean(aucs)
    aucs_std = np.std(aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="k", linewidth=2,
        label=f"{label} (AUC = {aucs_mean:.4f} $\pm$ {aucs_std:.4f})"
    )

    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    

    return np.array((mean_fpr, mean_tpr)), np.array((aucs_mean, aucs_std))


def plot_qc_summary(qc_results, 
                    binrange_peaks=None, 
                    binwidth_peaks=None, 
                    binrange_score=None, 
                    binwidth_score=None,
                    ymax_peaks=None,
                    ymax_score=None, 
                    fig=None):

    if fig is None:
        fig = plt.gcf()


    (ax_box1, ax_box2), (ax_hist1, ax_hist2) = fig.subplots(
        2, 2, sharex="col", gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(x=qc_results.iloc[:,1], ax=ax_box1)
    sns.boxplot(x=qc_results.iloc[:,0], ax=ax_box2)
    sns.histplot(qc_results.iloc[:,1], ax=ax_hist1, binrange=binrange_peaks, binwidth=binwidth_peaks)
    sns.histplot(qc_results.iloc[:,0], ax=ax_hist2, binrange=binrange_score, binwidth=binwidth_score)

    ax_box1.set(yticks=[])
    ax_box2.set(yticks=[])
    sns.despine(ax=ax_hist1)
    sns.despine(ax=ax_hist2)
    sns.despine(ax=ax_box1, left=True)
    sns.despine(ax=ax_box2, left=True)

    ax_hist1.set_xlabel("Number of Peaks")
    ax_hist2.set_xlabel(qc_results.columns[0])

    ax_hist1.set_ylim([None, ymax_peaks])
    ax_hist2.set_ylim([None, ymax_score])

    ax_box1.tick_params(axis="x", labelbottom=True)
    ax_box2.tick_params(axis="x", labelbottom=True)      


def plot_roc_comparison(rocs, aucs, regex=None, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for name, curve in rocs.items():
        if regex:
            if re.search(regex, name):
                ax.plot(curve[0], curve[1], 
                    label=f"{name} (AUC = {aucs[name][0]:.4f})")
        else:
            ax.plot(curve[0], curve[1], 
                label=f"{name} (AUC = {aucs[name][0]:.4f})")
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))



def boxplot_comparison(data, regex=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if regex:
        data = data.filter(regex=regex)

    sns.boxplot(data=data, 
                ax=ax,
                boxprops={"facecolor": "w"})
