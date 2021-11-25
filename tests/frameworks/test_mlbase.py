import subprocess
import sys

import pandas as pd
import pytest
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlrun
from mlrun import new_function


def _is_installed(lib) -> bool:
    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]
    return lib not in installed_packages


def get_dataset(classification=True):
    if classification:
        iris = load_iris()
        X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        y = pd.DataFrame(data=iris.target, columns=["species"])
    else:
        boston = load_boston()
        X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
        y = pd.DataFrame(data=boston.target, columns=["target"])
    return train_test_split(X, y, test_size=0.2)


def run_mlbase_sklearn_classification(context):
    from mlrun.frameworks.sklearn import apply_mlrun

    model = LogisticRegression()
    X_train, X_test, y_train, y_test = get_dataset()
    apply_mlrun(
        model, context, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    model.fit(X_train, y_train)


def run_mlbase_xgboost_regression(context: mlrun.MLClientCtx):
    import json

    import xgboost as xgb

    from mlrun.frameworks.xgboost import apply_mlrun

    model = xgb.XGBRegressor()
    X_train, X_test, y_train, y_test = get_dataset(classification=False)
    model_handler = apply_mlrun(
        model, context, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )
    model.fit(X_train, y_train)

    json_artifact = "test.json"
    with open(json_artifact, "w") as json_file:
        json.dump({"test": 0}, json_file, indent=4)

    model_handler.register_artifacts(
        context.log_artifact(
            json_artifact,
            local_path=json_artifact,
            artifact_path=context.artifact_path,
            db_key=False,
        )
    )
    model_handler.update()


def run_mlbase_lgbm_classification(context):
    import lightgbm as lgb

    from mlrun.frameworks.lgbm import apply_mlrun

    model = lgb.LGBMClassifier()
    X_train, X_test, y_train, y_test = get_dataset()
    apply_mlrun(
        model, context, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )
    model.fit(X_train, y_train)


def test_run_mlbase_sklearn_classification():
    sklearn_run = new_function().run(
        artifact_path="./temp", handler=run_mlbase_sklearn_classification
    )
    assert (sklearn_run.artifact("model").meta.to_dict()["metrics"]["accuracy"]) > 0
    assert (sklearn_run.artifact("model").meta.to_dict()["model_file"]) == "model.pkl"


@pytest.mark.skipif(_is_installed("xgboost"), reason="xgboost package missing")
def test_run_mlbase_xgboost_regression():
    xgb_run = new_function().run(
        artifact_path="./temp", handler=run_mlbase_xgboost_regression
    )
    assert (xgb_run.artifact("model").meta.to_dict()["metrics"]["accuracy"]) > 0
    assert "confusion matrix" not in (
        xgb_run.artifact("model").meta.to_dict()["extra_data"]
    )
    assert (xgb_run.artifact("model").meta.to_dict()["model_file"]) == "model.pkl"


@pytest.mark.skipif(_is_installed("lightgbm"), reason="missing packages")
def test_run_mlbase_lgbm_classification():
    lgbm_run = new_function().run(
        artifact_path="./temp", handler=run_mlbase_lgbm_classification
    )
    assert (lgbm_run.artifact("model").meta.to_dict()["metrics"]["accuracy"]) > 0
    assert (lgbm_run.artifact("model").meta.to_dict()["model_file"]) == "model.pkl"
