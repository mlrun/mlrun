from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from ..artifacts import PlotArtifact
import pandas as pd
import numpy as np
from scipy import interp
from itertools import cycle


def gcf_clear(plt):
    """Utility to clear matplotlib figure
    Run this inside every plot method before calling any matplotlib
    methods
    :param plot:    matloblib figure object
    """
    plt.cla()
    plt.clf()
    plt.close()


def feature_importances(model, header):
    """Display estimated feature importances
    Only works for models with attribute 'feature_importances_`
    :param model:       fitted model
    :param header:      feature labels
    """
    if not hasattr(model, "feature_importances_"):
        raise Exception(
            "feature importances are only available for some models, if you got"
            "here then please make sure to check your estimated model for a "
            "`feature_importances_` attribute before calling this method"
        )

    # create a feature importance table with desired labels
    zipped = zip(model.feature_importances_, header)
    feature_imp = pd.DataFrame(sorted(zipped), columns=["freq", "feature"]).sort_values(
        by="freq", ascending=False
    )

    plt.clf()  # gcf_clear(plt)
    plt.figure()
    sns.barplot(x="freq", y="feature", data=feature_imp)
    plt.title("features")
    plt.tight_layout()

    return (
        PlotArtifact(
            "feature-importances", body=plt.gcf(), title="Feature Importances"
        ),
        feature_imp,
    )


def plot_importance(
    context, model, key: str = "feature-importances", plots_dest: str = "plots"
):
    """Display estimated feature importances
    Only works for models with attribute 'feature_importances_`

    **legacy version please deprecate in functions and demos**

    :param context:     function context
    :param model:       fitted model
    :param key:         key of feature importances plot and table in artifact
                        store
    :param plots_dest:  subfolder  in artifact store
    """
    if not hasattr(model, "feature_importances_"):
        raise Exception("feature importaces are only available for some models")

    # create a feature importance table with desired labels
    zipped = zip(model.feature_importances_, context.header)
    feature_imp = pd.DataFrame(sorted(zipped), columns=["freq", "feature"]).sort_values(
        by="freq", ascending=False
    )

    gcf_clear(plt)
    plt.figure(figsize=(20, 10))
    sns.barplot(x="freq", y="feature", data=feature_imp)
    plt.title("features")
    plt.tight_layout()

    fname = f"{plots_dest}/{key}.html"
    context.log_artifact(PlotArtifact(key, body=plt.gcf()), local_path=fname)

    # feature importances are also saved as a csv table (generally small):
    fname = key + "-tbl.csv"
    return context.log_dataset(key + "-tbl", df=feature_imp, local_path=fname)


def learning_curves(model):
    """model class dependent

    WIP

    get training history plots for xgboost, lightgbm

    returns list of PlotArtifacts, can be empty if no history
    is found
    """
    plots = []

    # do this here and not in the call to learning_curve plots,
    # this is default approach for xgboost and lightgbm
    if hasattr(model, "evals_result"):
        results = model.evals_result()
        train_set = list(results.items())[0]
        valid_set = list(results.items())[1]

        learning_curves = pd.DataFrame(
            {
                "train_error": train_set[1]["error"],
                "train_auc": train_set[1]["auc"],
                "valid_error": valid_set[1]["error"],
                "valid_auc": valid_set[1]["auc"],
            }
        )

        plt.clf()  # gcf_clear(plt)
        fig, ax = plt.subplots()
        plt.xlabel("# training examples")
        plt.ylabel("auc")
        plt.title("learning curve - auc")
        ax.plot(learning_curves.train_auc, label="train")
        ax.plot(learning_curves.valid_auc, label="valid")
        ax.legend(loc="lower left")
        plots.append(PlotArtifact("learning curve - auc", body=plt.gcf()))

        plt.clf()  # gcf_clear(plt)
        fig, ax = plt.subplots()
        plt.xlabel("# training examples")
        plt.ylabel("error rate")
        plt.title("learning curve - error")
        ax.plot(learning_curves.train_error, label="train")
        ax.plot(learning_curves.valid_error, label="valid")
        ax.legend(loc="lower left")
        plots.append(PlotArtifact("learning curve - taoot", body=plt.gcf()))

    # elif some other model history api...

    return plots


def confusion_matrix(model, xtest, ytest, cmap="Blues"):
    cmd = metrics.plot_confusion_matrix(
        model,
        xtest,
        ytest,
        normalize="all",
        values_format=".2g",
        cmap=plt.get_cmap(cmap),
    )
    # for now only 1, add different views to this array for display in UI
    cmd.plot()
    return PlotArtifact(
        "confusion-matrix-normalized",
        body=cmd.figure_,
        title="Confusion Matrix - Normalized Plot",
    )


def precision_recall_multi(ytest_b, yprob, labels, scoring="micro"):
    """"""
    n_classes = len(labels)

    precision = dict()
    recall = dict()
    avg_prec = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(
            ytest_b[:, i], yprob[:, i]
        )
        avg_prec[i] = metrics.average_precision_score(ytest_b[:, i], yprob[:, i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(
        ytest_b.ravel(), yprob.ravel()
    )
    avg_prec["micro"] = metrics.average_precision_score(ytest_b, yprob, average="micro")
    ap_micro = avg_prec["micro"]
    # model_metrics.update({'precision-micro-avg-classes': ap_micro})

    # gcf_clear(plt)
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    plt.figure()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall["micro"], precision["micro"], color="gold", lw=10)
    lines.append(l)
    labels.append(f"micro-average precision-recall (area = {ap_micro:0.2f})")

    for i, color in zip(range(n_classes), colors):
        (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(f"precision-recall for class {i} (area = {avg_prec[i]:0.2f})")

    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("precision recall - multiclass")
    plt.legend(lines, labels, loc=(0, -0.41), prop=dict(size=10))

    return PlotArtifact(
        "precision-recall-multiclass",
        body=plt.gcf(),
        title="Multiclass Precision Recall",
    )


def roc_multi(ytest_b, yprob, labels):
    """"""
    n_classes = len(labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(ytest_b[:, i], yprob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ytest_b.ravel(), yprob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    gcf_clear(plt)
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("receiver operating characteristic - multiclass")
    plt.legend(loc=(0, -0.68), prop=dict(size=10))

    return PlotArtifact("roc-multiclass", body=plt.gcf(), title="Multiclass ROC Curve")


def roc_bin(ytest, yprob, clear: bool = False):
    """"""
    # ROC plot
    if clear:
        gcf_clear(plt)
    fpr, tpr, _ = metrics.roc_curve(ytest, yprob)
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label="a label")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="best")

    return PlotArtifact("roc-binary", body=plt.gcf(), title="Binary ROC Curve")


def precision_recall_bin(model, xtest, ytest, yprob, clear=False):
    """"""
    if clear:
        gcf_clear(plt)
    disp = metrics.plot_precision_recall_curve(model, xtest, ytest)
    disp.ax_.set_title(
        f"precision recall: AP={metrics.average_precision_score(ytest, yprob):0.2f}"
    )

    return PlotArtifact(
        "precision-recall-binary", body=disp.figure_, title="Binary Precision Recall"
    )


def plot_roc(
    context,
    y_labels,
    y_probs,
    key="roc",
    plots_dir: str = "plots",
    fmt="png",
    fpr_label: str = "false positive rate",
    tpr_label: str = "true positive rate",
    title: str = "roc curve",
    legend_loc: str = "best",
    clear: bool = True,
):
    """plot roc curves

    **legacy version please deprecate in functions and demos**

    :param context:      the function context
    :param y_labels:     ground truth labels, hot encoded for multiclass
    :param y_probs:      model prediction probabilities
    :param key:          ("roc") key of plot in artifact store
    :param plots_dir:    ("plots") destination folder relative path to artifact path
    :param fmt:          ("png") plot format
    :param fpr_label:    ("false positive rate") x-axis labels
    :param tpr_label:    ("true positive rate") y-axis labels
    :param title:        ("roc curve") title of plot
    :param legend_loc:   ("best") location of plot legend
    :param clear:        (True) clear the matplotlib figure before drawing
    """
    # clear matplotlib current figure
    if clear:
        gcf_clear(plt)

    # draw 45 degree line
    plt.plot([0, 1], [0, 1], "k--")

    # labelling
    plt.xlabel(fpr_label)
    plt.ylabel(tpr_label)
    plt.title(title)
    plt.legend(loc=legend_loc)

    # single ROC or multiple
    if y_labels.shape[1] > 1:

        # data accumulators by class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_labels[:, :-1].shape[1]):
            fpr[i], tpr[i], _ = metrics.roc_curve(
                y_labels[:, i], y_probs[:, i], pos_label=1
            )
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"class {i}")
    else:
        fpr, tpr, _ = metrics.roc_curve(y_labels, y_probs[:, 1], pos_label=1)
        plt.plot(fpr, tpr, label="positive class")

    fname = f"{plots_dir}/{key}.html"
    return context.log_artifact(PlotArtifact(key, body=plt.gcf()), local_path=fname)
