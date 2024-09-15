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
import pathlib
import tempfile
from random import randint, random

import lightgbm as lgb
import mlflow
import mlflow.environment_variables
import mlflow.xgboost
import pytest
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

import mlrun
import mlrun.launcher.local
from mlrun.track.trackers.mlflow_tracker import MLFlowTracker

#  Important:
#  unlike mlconf which resets back to default after each test run, the mlflow configurations
#  and env vars don't, so at the end of each test we need to redo anything we set in that test.
#  what we cover in these tests: logging "regular" runs with, experiment name, run id and context
#  name (last two using mlconf), failing run mid-way, and a run with no handler.
#  we also test here importing of runs, artifacts and models from a previous run.


# simple general mlflow example of hand logging
def simple_run():
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


# simple general mlflow example with interruption in the middle
def interrupted_run():
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
        raise Exception("Testing logging an interrupted run")
        # more code ...


# simple mlflow example of lgb logging
def lgb_run():
    # prepare train and test data
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # enable auto logging
    mlflow.lightgbm.autolog()

    train_set = lgb.Dataset(x_train, label=y_train)

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
        # model and training data are being logged automatically
        model = lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=[train_set],
            valid_names=["train"],
        )

        # evaluate model
        y_proba = model.predict(x_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})


# simple mlflow example of xgb logging
def xgb_run():
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
        # model and training data are being logged automatically
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
    """
    This is a small test to validate tracking enablement mechanisms
    """
    # enable tracking in config for inspection
    mlrun.mlconf.external_platform_tracking.enabled = enable_tracking
    # see if mlflow is in scope
    mlflow_tracker = MLFlowTracker()
    # check all the stuff we check in is_enabled
    enabled = mlrun.mlconf.external_platform_tracking.mlflow.enabled is True

    assert mlflow_tracker.is_enabled() == enabled


@pytest.mark.parametrize("handler", ["xgb_run", "lgb_run", "simple_run"])
def test_track_run_with_experiment_name(rundb_mock, handler):
    """
    This test is for tracking a run logged by mlflow into mlrun while it's running using the experiment name.
    first activate the tracking option in mlconf, then we name the mlflow experiment,
    then we run some code that is being logged by mlflow using mlrun,
    and finally compare the mlrun we tracked with the original mlflow run using the validate func
    """
    # Enable general tracking
    mlrun.mlconf.external_platform_tracking.enabled = True
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(f"{handler}_test_track")
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )

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
            local=True,
            handler=handler,
            artifact_path=test_directory,
        )

        _validate_run(
            run=trainer_run, run_id=trainer_run.metadata.labels.get("mlflow-run-id")
        )
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


@pytest.mark.parametrize("handler", ["xgb_run", "lgb_run", "simple_run"])
def test_track_run_with_control_run(rundb_mock, handler):
    """
    This test is for tracking a run logged by mlflow into mlrun while it's running using the run id.
    first activate the tracking option in mlconf, then we name the mlflow experiment,
    then we run some code that is being logged by mlflow using mlrun,
    and finally compare the mlrun we tracked with the original mlflow run using the validate func
    """
    # Enable general tracking
    mlrun.mlconf.external_platform_tracking.enabled = True
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(
        f"{handler}_with_control_run"
    )
    # Tell mlrun to create a mlflow run in advance, and by so knowing it's run id
    mlrun.mlconf.external_platform_tracking.mlflow.control_run = True
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )

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
            local=True,
            handler=handler,
            artifact_path=test_directory,
        )

        _validate_run(
            run=trainer_run, run_id=trainer_run.metadata.labels.get("mlflow-run-id")
        )
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


@pytest.mark.parametrize("handler", ["xgb_run", "lgb_run", "simple_run"])
def test_track_run_with_match_experiment_to_runtime(rundb_mock, handler):
    """
    This test is for tracking a run logged by mlflow into mlrun while it's running by setting
    'mlconf.external_platform_tracking.mlflow.match_experiment_to_runtime` to True.
    first activate the tracking option in mlconf, then we name the mlflow experiment,
    then we run some code that is being logged by mlflow using mlrun,
    and finally compare the mlrun we tracked with the original mlflow run using the validate func
    """
    # Enable general tracking
    mlrun.mlconf.external_platform_tracking.enabled = True
    # Tell mlrun to set experiment name to context.name
    mlrun.mlconf.external_platform_tracking.mlflow.match_experiment_to_runtime = True
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )

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
            local=True,
            handler=handler,
            artifact_path=test_directory,
        )

        _validate_run(
            run=trainer_run, run_id=trainer_run.metadata.labels.get("mlflow-run-id")
        )
        # unset mlflow experiment name to default
        mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


@pytest.mark.parametrize("run_name", ["simple_run"])
def test_track_run_no_handler(rundb_mock, run_name):
    """
    This test is for tracking a run logged by mlflow into mlrun while it's running without a handler.
    first activate the tracking option in mlconf, then we name the mlflow experiment,
    then we run some code that is being logged by mlflow using mlrun(from a different file in the assets dir),
    and finally compare the mlrun we tracked with the original mlflow run using the validate func
    """
    mlrun.mlconf.external_platform_tracking.enabled = True
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(f"{run_name}_no_handler")
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )
        # Get the script path from assets:
        script_path = str(
            pathlib.Path(__file__).parent.parent
            / "assets"
            / f"{run_name}_no_handler.py"
        )
        # Create a MLRun function using the tester source file (all the functions must be located in it):
        func = project.set_function(
            script_path,
            name=run_name,
            kind="job",
            image="mlrun/mlrun",
        )
        # Mlflow creates a dir to log the run, this makes it in the tmpdir we create
        trainer_run = func.run(
            name=f"{run_name}_no_handler",
            project=project.name,
            artifact_path=test_directory,
            params={"tracking_uri": test_directory},
            local=True,
        )
        # Was added for the tracking in different processes, now remove it before validation
        trainer_run.spec.parameters.pop("tracking_uri")
        # Need run id in order to find correct run when loggen in different process
        run_id = trainer_run.status.results.pop("run_id")
        _validate_run(run=trainer_run, run_id=run_id)
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


def _mock_wrap_run_result(monkeypatch):
    """Mock function for `_wrap_run_result`, in order to examine tracked run and avoid run crash"""

    def _wrap_run_result(*args, **kwargs):
        return mlrun.run.RunObject.from_dict(args[2])

    monkeypatch.setattr(
        mlrun.launcher.local.ClientLocalLauncher, "_wrap_run_result", _wrap_run_result
    )


@pytest.mark.parametrize("handler", ["interrupted_run"])
def test_track_interrupted_run(monkeypatch, rundb_mock, handler):
    """
    This test is for tracking a run logged by mlflow into mlrun while it's running and then crashing.
    first we mock `_wrap_run_result` in order to catch the run object, then we activate the tracking option in mlconf,
     then we name the mlflow experiment, then we run some code that is being logged by mlflow using mlrun,
    and finally compare the mlrun we tracked with the original mlflow run using the validate func
    """
    # Need to wrap the code in order to get back the run from mlrun, so we can compare
    # it to what we wanted to log from mlflow
    _mock_wrap_run_result(monkeypatch)
    mlrun.mlconf.external_platform_tracking.enabled = True
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(handler)

    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )

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
            local=True,
            handler=handler,
            artifact_path=test_directory,
            watch=False,
        )

        _validate_run(
            run=trainer_run, run_id=trainer_run.metadata.labels.get("mlflow-run-id")
        )
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


@pytest.mark.parametrize("handler", [xgb_run, lgb_run, simple_run])
def test_import_run(rundb_mock, handler):
    """
    This test is for importing a run logged by mlflow into mlrun.
    first we run some code that's logged by mlflow, then we import it
    to mlrun, and then we use the validate function to compare between original run and imported
    """
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(
        f"{handler.__name__}_import_run"
    )
    with tempfile.TemporaryDirectory() as test_directory:
        # Tell mlflow where to save logged data
        mlflow.set_tracking_uri(test_directory)

        # Run mlflow wrapped code
        handler()

        # Set mlconf path to artifacts
        mlrun.mlconf.artifact_path = f"{test_directory}/artifact"

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )

        # Create a MLRun function using the tester source file (all the functions must be located in it):
        project.set_function(
            func=__file__,
            name=f"{handler.__name__}-test",
            kind="job",
            image="mlrun/mlrun",
            requirements=["mlflow"],
        )

        mlflow_run = mlflow.last_active_run()
        imported_run = MLFlowTracker().import_run(
            project=project,
            reference_id=mlflow_run.info.run_id,
            function_name=f"{handler.__name__}-test",
        )

        _validate_run(run=imported_run, run_id=mlflow_run.info.run_id)
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


@pytest.mark.parametrize("handler", [xgb_run, lgb_run])
def test_import_model(rundb_mock, handler):
    """
    This test is for importing a model logged by mlflow into mlrun.
    first we run some code that's logged by mlflow, then we import the logged model
    to mlrun, and then we validate
    """
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(
        f"{handler.__name__}_import_model"
    )
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Run mlflow code
        handler()

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default", context=test_directory, allow_cross_project=True
        )

        # Access model's uri through mlflow's last run
        mlflow_run = mlflow.last_active_run()
        model_uri = f"{mlflow_run.info.artifact_uri}/model"

        key = "test_model"
        MLFlowTracker().import_model(
            project=project,
            reference_id=model_uri,
            key=key,
            metrics=mlflow_run.data.metrics,
        )

        # Validate model was logged into project
        assert project.get_artifact(key)
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


@pytest.mark.parametrize("handler", [xgb_run, lgb_run, simple_run])
def test_import_artifact(rundb_mock, handler):
    """
    This test is for importing an artifact logged by mlflow into mlrun.
    first we run some code that's logged by mlflow, then we import logged artifacts
    to mlrun, and then we use validate by comparing to original artifacts
    """
    # Set the mlflow experiment name
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(
        f"{handler.__name__}_import_artifact"
    )
    with tempfile.TemporaryDirectory() as test_directory:
        mlflow.set_tracking_uri(test_directory)  # Tell mlflow where to save logged data

        # Run mlflow code
        handler()

        # Create a project for this tester:
        project = mlrun.get_or_create_project(
            name="default1", context=test_directory, allow_cross_project=True
        )

        # Get a list of all artifacts logged by mlflow during last run
        mlflow_run = mlflow.last_active_run()
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(mlflow_run.info.run_id)

        # Try importing all artifacts
        for artifact in artifacts:
            # We don't want to log models here
            if not artifact.is_dir:
                artifact_uri = f"{mlflow_run.info.artifact_uri}/{artifact.path}"

                key = f"test_artifact_{artifact.path}"
                MLFlowTracker().import_artifact(
                    project=project,
                    reference_id=artifact_uri,
                    key=key,
                )

                # Validate artifact in project
                assert project.get_artifact(key)
    # unset mlflow experiment name to default
    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.unset()


def _validate_run(run: mlrun.run, run_id: str = None):
    # in order to tell mlflow where to look for logged run for comparison
    client = mlflow.MlflowClient()
    if run_id:
        run_to_comp = mlflow.get_run(run_id)
    else:
        run_to_comp = mlflow.last_active_run()
    if not run_to_comp:
        assert False, "Run not found, test failed"

    # check that values correspond
    print(f"run params: {run.spec.parameters}")
    print(f"run to comp params: {run_to_comp.data.params}")
    for param in run_to_comp.data.params:
        assert run_to_comp.data.params[param] == run.spec.parameters[param]
    for metric in run_to_comp.data.metrics:
        assert run_to_comp.data.metrics[metric] == run.status.results[metric]
    assert len(run_to_comp.data.params) == len(run.spec.parameters)
    # check the number of artifacts corresponds
    num_artifacts = len(client.list_artifacts(run_to_comp.info.run_id))
    assert num_artifacts == len(run.status.artifacts), "Wrong number of artifacts"
