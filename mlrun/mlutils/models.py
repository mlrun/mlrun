import json
from inspect import signature, _empty
from importlib import import_module
from typing import List
from itertools import cycle

from ..artifacts import PlotArtifact
from .plots import (
    gcf_clear,
    feature_importances,
    precision_recall_bin,
    precision_recall_multi,
    roc_multi,
    roc_bin,
)

import numpy as np
import pandas as pd
from scipy import interp
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from scikitplot.metrics import plot_calibration_curve


def get_class_fit(module_pkg_class: str):
    """generate a model config
    :param module_pkg_class:  str description of model, e.g.
        `sklearn.ensemble.RandomForestClassifier`
    """
    splits = module_pkg_class.split(".")
    model_ = getattr(import_module(".".join(splits[:-1])), splits[-1])
    f = list(signature(model_().fit).parameters.items())
    d = {}
    for i in range(len(f)):
        d.update({f[i][0]: None if f[i][1].default is _empty else f[i][1].default})

    return {
        "CLASS": model_().get_params(),
        "FIT": d,
        "META": {
            "pkg_version": import_module(splits[0]).__version__,
            "class": module_pkg_class,
        },
    }


def gen_sklearn_model(model_pkg, skparams):
    """generate an sklearn model configuration

    input can be either a "package.module.class" or
    a json file
    """
    if model_pkg.endswith("json"):
        model_config = json.load(open(model_pkg, "r"))
    else:
        model_config = get_class_fit(model_pkg)

    for k, v in skparams:
        if k.startswith("CLASS_"):
            model_config["CLASS"][k[6:]] = v
        if k.startswith("FIT_"):
            model_config["FIT"][k[4:]] = v

    return model_config


def eval_class_model(
    xtest, ytest, model, labels: str = "labels", pred_params: dict = {}
):
    """generate predictions and validation stats

    pred_params are non-default, scikit-learn api prediction-function parameters.
    For example, a tree-type of model may have a tree depth limit for its prediction
    function.

    :param xtest:        features array type Union(DataItem, DataFrame, np. Array)
    :param ytest:        ground-truth labels Union(DataItem, DataFrame, Series, np. Array, List)
    :param model:        estimated model
    :param labels:       ('labels') labels in ytest is a pd.DataFrame or Series
    :param pred_params:  (None) dict of predict function parameters
    """
    if isinstance(ytest, (pd.DataFrame, pd.Series)):
        unique_labels = ytest[labels].unique()
        ytest = ytest.values
    elif isinstance(ytest, np.ndarray):
        unique_labels = np.unique(ytest)
    elif isinstance(ytest, list):
        unique_labels = set(ytest)

    n_classes = len(unique_labels)
    is_multiclass = True if n_classes > 2 else False

    # PROBS
    ypred = model.predict(xtest, **pred_params)
    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(xtest, **pred_params)
    else:
        # todo if decision fn...
        raise Exception("not implemented for this classifier")

    # todo - calibrate
    # outputs are some stats and some plots and...
    # should be option, some classifiers don't need, some do it already, many don't

    model_metrics = {
        "plots": [],  # placeholder for plots
        "accuracy": float(metrics.accuracy_score(ytest, ypred)),
        "test-error-rate": np.sum(ytest != ypred) / ytest.shape[0],
    }

    # CONFUSION MATRIX
    gcf_clear(plt)
    cmd = metrics.plot_confusion_matrix(
        model, xtest, ytest, normalize="all", cmap=plt.cm.Blues
    )
    model_metrics["plots"].append(PlotArtifact("confusion-matrix", body=cmd.figure_))

    if is_multiclass:
        # PRECISION-RECALL CURVES MICRO AVGED
        # binarize/hot-encode here since we look at each class
        lb = LabelBinarizer()
        ytest_b = lb.fit_transform(ytest)

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
        avg_prec["micro"] = metrics.average_precision_score(
            ytest_b, yprob, average="micro"
        )
        ap_micro = avg_prec["micro"]
        model_metrics.update({"precision-micro-avg-classes": ap_micro})

        gcf_clear(plt)
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
        plt.figure(figsize=(7, 8))
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

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision recall - multiclass")
        plt.legend(lines, labels, loc=(0, -0.38), prop=dict(size=10))
        model_metrics["plots"].append(
            PlotArtifact("precision-recall-multiclass", body=plt.gcf())
        )

        # ROC CURVES
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(ytest_b[:, i], yprob[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
            ytest_b.ravel(), yprob.ravel()
        )
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
            label="micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["macro"]),
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
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("receiver operating characteristic - multiclass")
        plt.legend(loc="lower right")
        model_metrics["plots"].append(PlotArtifact("roc-multiclass", body=plt.gcf()))
        # AUC multiclass
        model_metrics.update(
            {
                "auc-macro": metrics.roc_auc_score(
                    ytest_b, yprob, multi_class="ovo", average="macro"
                ),
                "auc-weighted": metrics.roc_auc_score(
                    ytest_b, yprob, multi_class="ovo", average="weighted"
                ),
            }
        )

        # others (todo - macro, micro...)
        model_metrics.update(
            {
                "f1-score": metrics.f1_score(ytest, ypred, average="macro"),
                "recall_score": metrics.recall_score(ytest, ypred, average="macro"),
            }
        )
    else:
        # binary
        yprob_pos = yprob[:, 1]

        model_metrics.update(
            {
                "rocauc": metrics.roc_auc_score(ytest, yprob_pos),
                "brier_score": metrics.brier_score_loss(
                    ytest, yprob_pos, pos_label=ytest.max()
                ),
            }
        )

        # precision-recall

        # ROC plot

    return model_metrics


def eval_model_v2(
    context,
    xtest,
    ytest,
    model,
    pcurve_bins: int = 10,
    pcurve_names: List[str] = ["my classifier"],
    plots_artifact_path: str = "",
    pred_params: dict = {},
    cmap="Blues",
):
    """generate predictions and validation stats

    pred_params are non-default, scikit-learn api prediction-function
    parameters. For example, a tree-type of model may have a tree depth
    limit for its prediction function.

    :param xtest:        features array type Union(DataItem, DataFrame,
                         numpy array)
    :param ytest:        ground-truth labels Union(DataItem, DataFrame,
                         Series, numpy array, List)
    :param model:        estimated model
    :param pcurve_bins:  (10) subdivide [0,1] interval into n bins, x-axis
    :param pcurve_names: label for each calibration curve
    :param pred_params:  (None) dict of predict function parameters
    :param cmap:         ('Blues') matplotlib color map
    """

    import numpy as np

    def df_blob(df):
        return bytes(df.to_csv(index=False), encoding="utf-8")

    if isinstance(ytest, np.ndarray):
        unique_labels = np.unique(ytest)
    elif isinstance(ytest, list):
        unique_labels = set(ytest)
    else:
        try:
            ytest = ytest.values
            unique_labels = np.unique(ytest)
        except Exception as e:
            raise Exception(f"unrecognized data type for ytest {e}")

    n_classes = len(unique_labels)
    is_multiclass = True if n_classes > 2 else False

    # INIT DICT...OR SOME OTHER COLLECTOR THAT CAN BE ACCESSED
    plots_path = plots_artifact_path or context.artifact_subpath("plots")
    extra_data = {}

    ypred = model.predict(xtest, **pred_params)
    context.log_results(
        {
            "accuracy": float(metrics.accuracy_score(ytest, ypred)),
            "test-error": np.sum(ytest != ypred) / ytest.shape[0],
        }
    )

    # PROBABILITIES
    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(xtest, **pred_params)
        if not is_multiclass:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                ytest, yprob[:, -1], n_bins=pcurve_bins, strategy="uniform"
            )
            cmd = plot_calibration_curve(ytest, [yprob], pcurve_names)
            calibration = context.log_artifact(
                PlotArtifact(
                    "probability-calibration",
                    body=cmd.get_figure(),
                    title="probability calibration plot",
                ),
                artifact_path=plots_path,
                db_key=False,
            )
            extra_data["probability calibration"] = calibration

    # CONFUSION MATRIX
    cm = confusion_matrix(ytest, ypred, normalize="all")
    df = pd.DataFrame(data=cm)
    extra_data["confusion matrix table.csv"] = df_blob(df)

    cmd = metrics.plot_confusion_matrix(
        model,
        xtest,
        ytest,
        normalize="all",
        values_format=".2g",
        cmap=plt.get_cmap(cmap),
    )
    confusion = context.log_artifact(
        PlotArtifact(
            "confusion-matrix",
            body=cmd.figure_,
            title="Confusion Matrix - Normalized Plot",
        ),
        artifact_path=plots_path,
        db_key=False,
    )
    extra_data["confusion matrix"] = confusion

    # LEARNING CURVES
    if hasattr(model, "evals_result"):
        results = model.evals_result()
        train_set = list(results.items())[0]
        valid_set = list(results.items())[1]

        learning_curves_df = None
        if is_multiclass:
            if hasattr(train_set[1], "merror"):
                learning_curves_df = pd.DataFrame(
                    {
                        "train_error": train_set[1]["merror"],
                        "valid_error": valid_set[1]["merror"],
                    }
                )
        else:
            if hasattr(train_set[1], "error"):
                learning_curves_df = pd.DataFrame(
                    {
                        "train_error": train_set[1]["error"],
                        "valid_error": valid_set[1]["error"],
                    }
                )

        if learning_curves_df:
            extra_data["learning curve table.csv"] = df_blob(learning_curves_df)

            _, ax = plt.subplots()
            plt.xlabel("# training examples")
            plt.ylabel("error rate")
            plt.title("learning curve - error")
            ax.plot(learning_curves_df["train_error"], label="train")
            ax.plot(learning_curves_df["valid_error"], label="valid")
            learning = context.log_artifact(
                PlotArtifact(
                    "learning-curve", body=plt.gcf(), title="Learning Curve - erreur"
                ),
                artifact_path=plots_path,
                db_key=False,
            )
            extra_data["learning curve"] = learning

    # FEATURE IMPORTANCES
    if hasattr(model, "feature_importances_"):
        (fi_plot, fi_tbl) = feature_importances(model, xtest.columns)
        extra_data["feature importances"] = context.log_artifact(
            fi_plot, db_key=False, artifact_path=plots_path
        )
        extra_data["feature importances table.csv"] = df_blob(fi_tbl)

    # AUC - ROC - PR CURVES
    if is_multiclass:
        lb = LabelBinarizer()
        ytest_b = lb.fit_transform(ytest)

        extra_data["precision_recall_multi"] = context.log_artifact(
            precision_recall_multi(ytest_b, yprob, unique_labels),
            artifact_path=plots_path,
            db_key=False,
        )
        extra_data["roc_multi"] = context.log_artifact(
            roc_multi(ytest_b, yprob, unique_labels),
            artifact_path=plots_path,
            db_key=False,
        )

        # AUC multiclass
        aucmicro = metrics.roc_auc_score(
            ytest_b, yprob, multi_class="ovo", average="micro"
        )
        aucweighted = metrics.roc_auc_score(
            ytest_b, yprob, multi_class="ovo", average="weighted"
        )

        context.log_results({"auc-micro": aucmicro, "auc-weighted": aucweighted})

        # others (todo - macro, micro...)
        f1 = metrics.f1_score(ytest, ypred, average="macro")
        ps = metrics.precision_score(ytest, ypred, average="macro")
        rs = metrics.recall_score(ytest, ypred, average="macro")
        context.log_results({"f1-score": f1, "precision_score": ps, "recall_score": rs})

    else:
        yprob_pos = yprob[:, 1]
        extra_data["precision_recall_bin"] = context.log_artifact(
            precision_recall_bin(model, xtest, ytest, yprob_pos),
            artifact_path=plots_path,
            db_key=False,
        )
        extra_data["roc_bin"] = context.log_artifact(
            roc_bin(ytest, yprob_pos, clear=True),
            artifact_path=plots_path,
            db_key=False,
        )

        rocauc = metrics.roc_auc_score(ytest, yprob_pos)
        brier_score = metrics.brier_score_loss(ytest, yprob_pos, pos_label=ytest.max())
        f1 = metrics.f1_score(ytest, ypred)
        ps = metrics.precision_score(ytest, ypred)
        rs = metrics.recall_score(ytest, ypred)
        context.log_results(
            {
                "rocauc": rocauc,
                "brier_score": brier_score,
                "f1-score": f1,
                "precision_score": ps,
                "recall_score": rs,
            }
        )

    # return all model metrics and plots
    return extra_data
