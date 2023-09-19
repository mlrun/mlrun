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
import tempfile
from random import randint, random

import lightgbm as lgb
import matplotlib as mpl
import mlflow
import mlflow.environment_variables
import mlflow.xgboost
import pytest
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

import mlrun
from mlrun.track.trackers.mlflow_tracker import MLFlowTracker

mpl.use("Agg")


# simple general mlflow example of hand logging
def simple_run(context):
    mlflow.set_tracking_uri(context.artifact_path)
    with mlflow.start_run():
        # Log some random params and metrics
        mlflow.log_param("param1", randint(0, 100))
        mlflow.log_metric("foo", random())
        mlflow.log_metric("foo", random() + 1)
        mlflow.log_metric("foo", random() + 2)
        # Create an artifact and log it
        with tempfile.TemporaryDirectory() as test_dir:
            with open(f"{test_dir}/test.txt", "w") as f:
                f.write("hello world!")
                mlflow.log_artifacts(test_dir)


def lgb_run(context):
    mlflow.set_tracking_uri(context.artifact_path)
    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # enable auto logging
    mlflow.lightgbm.autolog()

    train_set = lgb.Dataset(X_train, label=y_train)

    with mlflow.start_run():
        # train model
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.1,
            "metric": "multi_logloss",
            "colsample_bytree": 1.0,
            "subsample": 1.0,
            "seed": 42,
        }
        model = lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=[train_set],
            valid_names=["train"],
        )

        # evaluate model
        y_proba = model.predict(X_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})


def xgb_run(context):
    mlflow.set_tracking_uri(context.artifact_path)
    # prepare train and test data
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

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


@pytest.mark.parametrize("enable_tracking", [True, False])
def test_is_enabled(rundb_mock, enable_tracking):
    # enable tracking in config for inspection
    mlrun.mlconf.external_platform_tracking.enabled = enable_tracking
    # see if mlflow is in scope
    mlflow_tracker = MLFlowTracker()
    # check all the stuff we check in is_enabled
    enabled = (
        getattr(mlrun.mlconf.external_platform_tracking, "mlflow", {}).enabled is True
    )
    assert mlflow_tracker.is_enabled() == enabled


@pytest.mark.parametrize("handler", ["xgb_run", "lgb_run", "simple_run"])
def test_track_run(rundb_mock, handler):
    mlrun.mlconf.external_platform_tracking.enabled = True
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(handler)
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)

        # in order to tell mlflow where to look for logged run for comparison
        client = mlflow.MlflowClient()

        # Create a project for this tester:
        project = mlrun.get_or_create_project(name="default", context=test_directory)

        # Create a MLRun function using the tester source file (all the functions must be located in it):
        func = project.set_function(
            func=__file__,
            name=f"{handler}-test",
            kind="job",
            image="mlrun/mlrun",
            requirements=["mlflow"],
        )
        # mlflow creates a dir to log the run, this makes it in the tmpdir we create
        trainer_run = func.run(
            local=True, artifact_path=test_directory, handler=handler
        )
        _validate_run(trainer_run, client)


def _validate_run(run: mlrun.run, client: mlflow.MlflowClient):
    runs = []
    # returns a list of mlflow.entities.Experiment
    experiments = client.search_experiments()
    for experiment in experiments:
        runs.append(client.search_runs(experiment.experiment_id))
    # find the right run
    run_to_comp = None
    for run_list in runs:
        for mlflow_run in run_list:
            if mlflow_run.info.run_id == run.metadata.labels["mlflow-run-id"]:
                run_to_comp = mlflow_run
    if not run_to_comp:
        assert False, "Run not found, test failed"
    # check that values correspond
    for param in run_to_comp.data.params:
        assert run_to_comp.data.params[param] == run.spec.parameters[param]
    for metric in run_to_comp.data.metrics:
        assert run_to_comp.data.metrics[metric] == run.status.results[metric]
    assert len(run_to_comp.data.params) == len(run.spec.parameters)
    # check the number of artifacts corresponds
    num_artifacts = len(client.list_artifacts(run_to_comp.info.run_id))
    assert num_artifacts == len(run.status.artifacts)
