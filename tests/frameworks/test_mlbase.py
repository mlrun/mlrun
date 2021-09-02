import pandas as pd
import pytest
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlrun import new_function

skip_tests = True


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
    from mlrun.frameworks.sklearn import apply_mlrun as apply_mlrun_sklearn

    model = LogisticRegression(solver="liblinear")
    X_train, X_test, y_train, y_test = get_dataset()
    model = apply_mlrun_sklearn(
        model, context, model_name="my_model_name", X_test=X_test, y_test=y_test
    )
    model.fit(X_train, y_train)


def run_mlbase_xgboost_regression(context):
    import xgboost as xgb

    from mlrun.frameworks.xgboost import apply_mlrun as apply_mlrun_xgb

    model = xgb.XGBRegressor()
    X_train, X_test, y_train, y_test = get_dataset(classification=False)
    model = apply_mlrun_xgb(model, context, X_test=X_test, y_test=y_test)
    model.fit(X_train, y_train)


@pytest.mark.skipif(skip_tests, reason="missing packages")
def test_run_mlbase_sklearn_classification():
    sklearn_run = new_function().run(handler=run_mlbase_sklearn_classification)
    model = sklearn_run.artifact("my_model_name").meta
    assert model.metrics["accuracy"] > 0, "wrong accuracy"
    assert model.model_file == "LogisticRegression.pkl"


@pytest.mark.skipif(skip_tests, reason="missing packages")
def test_run_mlbase_xgboost_regression():
    xgb_run = new_function().run(handler=run_mlbase_xgboost_regression)
    assert xgb_run.artifact("test_set").meta, "test set not generated"
