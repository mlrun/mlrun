# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib
import tempfile
from random import randint, random

import lightgbm as lgb
import matplotlib as mpl
import mlflow
import mlflow.xgboost
import pytest
import xgboost as xgb
from mlflow import log_artifacts, log_metric, log_param
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from mlrun.utils import logger
import mlrun
from mlrun.track.trackers.mlflow_tracker import MLFlowTracker

mpl.use("Agg")


# simple general mlflow example of hand logging
def simple_run(context):
    mlflow.set_tracking_uri(context.artifact_path)
    # Log some random params and metrics
    log_param("param1", randint(0, 100))
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)
    # Create an artifact and log it
    test_directory = tempfile.TemporaryDirectory()
    with open(f"{test_directory.name}/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts(test_directory.name)
    test_directory.cleanup()


def lgb_run(context):
    mlflow.set_tracking_uri(context.artifact_path)
    # prepare example dataset
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # enable auto logging
    # this includes lightgbm.sklearn estimators
    mlflow.lightgbm.autolog()

    regressor = lgb.LGBMClassifier(n_estimators=20, reg_lambda=1.0)
    regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = regressor.predict(X_test)
    f1_score(y_test, y_pred, average="micro")
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))


def xgb_run(context):
    mlflow.set_tracking_uri(context.artifact_path)
    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    with mlflow.start_run():
        # train model
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": 0.3,
            "eval_metric": "mlogloss",
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "seed": 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, "train")])
        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})


def test_is_enabled(rundb_mock):
    # enable tracking in config for inspection
    mlrun.mlconf.tracking.enabled = True
    # see if mlflow is in scope
    try:
        importlib.import_module("mlflow")
        relevant = True
    except Exception:
        relevant = False
    mlflow_tracker = MLFlowTracker()
    # check all the stuff we check in is_enabled
    enabled = (
        mlflow_tracker._tracked_platform is not None
        and getattr(mlrun.mlconf.tracking, "mlflow", {}).mode == "enabled"
        and relevant
    )

    assert mlflow_tracker.is_enabled() == enabled


@pytest.mark.parametrize("handler", ["xgb_run", "lgb_run", "simple_run"])
def test_run(rundb_mock, handler):
    mlrun.mlconf.tracking.enabled = True
    test_directory = tempfile.TemporaryDirectory()
    mlflow.set_tracking_uri(test_directory.name)

    # in order to tell mlflow where to look for logged run for comparison
    client = mlflow.MlflowClient()

    # Create a project for this tester:
    project = mlrun.get_or_create_project(name="default", context=test_directory.name)

    # Create a MLRun function using the tester source file (all the functions must be located in it):
    func = project.set_function(
        func=__file__,
        name=f"{handler}-test",
        kind="job",
        image="mlrun/mlrun",
        requirements=["mlflow"],
    )
    # mlflow creates a dir to log the run, this makes it in the tmpdir we create
    try:
        trainer_run = func.run(
            local=True, artifact_path=test_directory.name, handler=handler
        )
        _validate_run(trainer_run, test_directory, client)
    except Exception as e:
        test_directory.cleanup()
        raise e
    test_directory.cleanup()


def _validate_run(run, test_directory, client):
    runs = []
    # returns a list of mlflow.entities.Experiment
    experiments = client.search_experiments()
    for experiment in experiments:
        runs.append(client.search_runs(experiment.experiment_id))
    # find the right run
    run_to_comp = None
    for run_list in runs:
        for mlflow_run in run_list:
            if mlflow_run.info.run_id == run.metadata.labels["mlflow-runid"]:
                run_to_comp = mlflow_run
    if not run_to_comp:
        logger.warning("Run not found, test failed")
        assert False
    # check that values correspond
    for param in run_to_comp.data.params:
        assert run_to_comp.data.params[param] == run.spec.parameters[param]
    for metric in run_to_comp.data.metrics:
        assert run_to_comp.data.metrics[metric] == run.outputs[metric]
    assert len(run_to_comp.data.params) == len(run.spec.parameters)
    # check the number of artifacts corresponds
    num_artifacts = len(client.list_artifacts(run_to_comp.info.run_id))
    assert num_artifacts == len(run.status.artifacts)
