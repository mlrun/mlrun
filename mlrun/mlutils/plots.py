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
