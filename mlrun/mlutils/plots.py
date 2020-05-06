from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from ..artifacts import PlotArtifact, TableArtifact
import pandas as pd


def gcf_clear(plt):
    """Utility to clear matplotlib figure
    Run this inside every plot method before calling any matplotlib
    methods
    :param plot:    matloblib figure object
    """
    plt.cla()
    plt.clf()
    plt.close()


def plot_importance(
    context,
    model,
    key: str = "feature-importances",
    plots_dest: str = "plots"
):
    """Display estimated feature importances
    Only works for models with attribute 'feature_importances_`
    :param context:     function context
    :param model:       fitted model
    :param key:         key of feature importances plot and table in artifact
                        store
    :param plots_dest:  subfolder  in artifact store
    """
    if not hasattr(model, "feature_importances_"):
        raise Exception(
            "feature importaces are only available for some models")

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
    return context.log_artifact(TableArtifact(key + "-tbl", df=feature_imp), local_path=fname)


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
):
    """plot roc curves

    TODO:  add averaging method (as string) that was used to create probs,
    display in legend

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
    """
    # clear matplotlib current figure
    gcf_clear(plt)

    # draw 45 degree line
    plt.plot([0, 1], [0, 1], "k--")

    # labelling
    plt.xlabel(fpr_label)
    plt.ylabel(tpr_label)
    plt.title(title)
    plt.legend(loc=legend_loc)

    # single ROC or mutliple
    if y_labels.shape[1] > 1:
        # data accummulators by class
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
        plt.plot(fpr, tpr, label=f"positive class")

    fname = f"{plots_dir}/{key}.html"
    return context.log_artifact(PlotArtifact(key, body=plt.gcf()), local_path=fname)
