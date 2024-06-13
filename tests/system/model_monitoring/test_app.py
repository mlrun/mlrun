# Copyright 2023 Iguazio
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

import concurrent.futures
import json
import os
import pickle
import time
import typing
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.types
import mlrun.db.httpdb
import mlrun.feature_store
import mlrun.feature_store as fstore
import mlrun.model_monitoring.api
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationBaseV2,
)
from mlrun.model_monitoring.applications.histogram_data_drift import (
    HistogramDataDriftApplication,
)
from mlrun.model_monitoring.evidently_application import SUPPORTED_EVIDENTLY_VERSION
from mlrun.utils.logger import Logger
from tests.system.base import TestMLRunSystem

from .assets.application import (
    EXPECTED_EVENTS_COUNT,
    DemoMonitoringApp,
    DemoMonitoringAppV2,
    NoCheckDemoMonitoringApp,
)
from .assets.custom_evidently_app import (
    CustomEvidentlyMonitoringApp,
    CustomEvidentlyMonitoringAppV2,
)


@dataclass
class _AppData:
    class_: type[
        typing.Union[ModelMonitoringApplicationBase, ModelMonitoringApplicationBaseV2]
    ]
    rel_path: str
    requirements: list[str] = field(default_factory=list)
    kwargs: dict[str, typing.Any] = field(default_factory=dict)
    abs_path: str = field(init=False)
    results: typing.Optional[set[str]] = None  # only for testing
    metrics: typing.Optional[set[str]] = None  # only for testing (future use)
    deploy: bool = True  # Set `False` for the default app

    def __post_init__(self) -> None:
        assert hasattr(self.class_, "NAME")

        path = Path(__file__).parent / self.rel_path
        assert path.exists()
        self.abs_path = str(path.absolute())


_DefaultDataDriftAppData = _AppData(
    class_=HistogramDataDriftApplication,
    rel_path="",
    deploy=False,
    results={"general_drift"},
    metrics={"hellinger_mean", "kld_mean", "tvd_mean"},
)


class _V3IORecordsChecker:
    project_name: str
    _logger: Logger
    apps_data: list[_AppData]
    app_interval: int

    @classmethod
    def custom_setup(cls, project_name: str) -> None:
        # By default, the TSDB connection is based on V3IO TSDB
        # To use TDEngine, set the `TSDB_CONNECTION` environment variable to the TDEngine connection string,
        # e.g. "taosws://user:password@host:port"

        if os.getenv(mm_constants.ProjectSecretKeys.TSDB_CONNECTION):
            project = mlrun.get_or_create_project(
                project_name, "./", allow_cross_project=True
            )
            project.set_model_monitoring_credentials(
                tsdb_connection=os.getenv(
                    mm_constants.ProjectSecretKeys.TSDB_CONNECTION
                )
            )

            cls._tsdb_storage = mlrun.model_monitoring.get_tsdb_connector(
                project=project_name,
                TSDB_CONNECTION=os.getenv(
                    mm_constants.ProjectSecretKeys.TSDB_CONNECTION
                ),
            )
        else:
            cls._tsdb_storage = mlrun.model_monitoring.get_tsdb_connector(
                project=project_name
            )
        cls._kv_storage = mlrun.model_monitoring.get_store_object(project=project_name)
        cls._v3io_container = f"users/pipelines/{project_name}/monitoring-apps/"

    @classmethod
    def _test_results_kv_record(cls, ep_id: str) -> None:
        for app_data in cls.apps_data:
            app_name = app_data.class_.NAME
            cls._logger.debug(
                "Checking the results KV record of app", app_name=app_name
            )

            resp = cls._kv_storage.client.kv.get(
                container=cls._v3io_container, table_path=ep_id, key=app_name
            )
            assert (
                data := resp.output.item
            ), f"V3IO KV app data is empty for app {app_name}"
            if app_data.results:
                assert (
                    data.keys() == app_data.results
                ), "The KV saved metrics are different than expected"

    @classmethod
    def _test_metrics_kv_record(cls, ep_id: str) -> None:
        for app_data in cls.apps_data:
            if not app_data.metrics:
                return

            app_name = app_data.class_.NAME
            table_path = f"{ep_id}_metrics"

            for metric in app_data.metrics:
                cls._logger.debug(
                    "Checking a metric KV record of app",
                    app_name=app_name,
                    metric=metric,
                )
                resp = cls._kv_storage.client.kv.get(
                    container=cls._v3io_container,
                    table_path=table_path,
                    key=f"{app_name}.{metric}",
                )
                assert (
                    resp.output.item
                ), f"V3IO KV app data is empty for app {app_name} and metric {metric}"

    @classmethod
    def _test_tsdb_record(cls, ep_id: str) -> None:
        if cls._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
            # V3IO TSDB
            df: pd.DataFrame = cls._tsdb_storage._get_records(
                table=mm_constants.V3IOTSDBTables.APP_RESULTS,
                start=f"now-{10 * cls.app_interval}m",
                end="now",
            )
        else:
            # TDEngine
            df: pd.DataFrame = cls._tsdb_storage._get_records(
                table=mm_constants.TDEngineSuperTables.APP_RESULTS,
                start=datetime.now().astimezone()
                - timedelta(minutes=10 * cls.app_interval),
                end=datetime.now().astimezone(),
                timestamp_column=mm_constants.WriterEvent.END_INFER_TIME,
            )

        assert not df.empty, "No TSDB data"
        assert (
            df.endpoint_id == ep_id
        ).all(), "The endpoint IDs are different than expected"

        assert set(df.application_name) == {
            app_data.class_.NAME for app_data in cls.apps_data
        }, "The application names are different than expected"

        tsdb_metrics = df.groupby("application_name").result_name.unique()
        for app_data in cls.apps_data:
            if app_metrics := app_data.results:
                app_name = app_data.class_.NAME
                cls._logger.debug("Checking the TSDB record of app", app_name=app_name)
                assert (
                    set(tsdb_metrics[app_name]) == app_metrics
                ), "The TSDB saved metrics are different than expected"

    @classmethod
    def _test_predictions_table(cls, ep_id: str, should_be_empty: bool = False) -> None:
        if cls._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
            predictions_df: pd.DataFrame = cls._tsdb_storage._get_records(
                table=mm_constants.FileTargetKind.PREDICTIONS, start="0", end="now"
            )
        else:
            # TDEngine
            predictions_df: pd.DataFrame = cls._tsdb_storage._get_records(
                table=mm_constants.TDEngineSuperTables.PREDICTIONS,
                start=datetime.min,
                end=datetime.now().astimezone(),
            )
        if should_be_empty:
            assert predictions_df.empty, "Predictions should be empty"
        else:
            assert not predictions_df.empty, "No TSDB predictions data"
            assert (
                predictions_df.endpoint_id == ep_id
            ).all(), "The endpoint IDs are different than expected"

    @classmethod
    def _test_apps_parquet(
        cls, ep_id: str, inputs: set[str], outputs: set[str]
    ) -> None:  # TODO : delete in 1.9.0  (V1 app deprecation)
        parquet_apps_directory = (
            mlrun.model_monitoring.helpers.get_monitoring_parquet_path(
                mlrun.get_or_create_project(cls.project_name, allow_cross_project=True),
                kind=mm_constants.FileTargetKind.APPS_PARQUET,
            )
        )
        df = ParquetTarget(
            path=f"{parquet_apps_directory}/key={ep_id}",
        ).as_df()

        is_inputs_saved = inputs.issubset(df.columns)
        assert is_inputs_saved, "Dataframe does not contain the input columns"
        is_output_saved = outputs.issubset(df.columns)
        assert is_output_saved, "Dataframe does not contain the output columns"
        is_metadata_saved = set(mm_constants.FeatureSetFeatures.list()).issubset(
            df.columns
        )
        assert is_metadata_saved, "Dataframe does not contain the metadata columns"

    @classmethod
    def _test_v3io_records(
        cls, ep_id: str, inputs: set[str], outputs: set[str]
    ) -> None:
        cls._test_apps_parquet(ep_id, inputs, outputs)
        cls._test_results_kv_record(ep_id)
        cls._test_metrics_kv_record(ep_id)
        cls._test_tsdb_record(ep_id)

    @classmethod
    def _test_api_get_metrics(
        cls,
        ep_id: str,
        app_data: _AppData,
        run_db: mlrun.db.httpdb.HTTPRunDB,
        type: typing.Literal["metrics", "results"] = "results",
    ) -> list[str]:
        cls._logger.debug("Checking GET /metrics API", type=type)
        response = run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=f"projects/{cls.project_name}/model-endpoints/{ep_id}/metrics?type={type}",
        )
        get_app_results: set[str] = set()
        app_results_full_names: list[str] = []

        parsed_response = json.loads(response.content.decode())

        if type == "metrics":
            assert (
                mlrun.model_monitoring.helpers.get_invocations_metric(
                    cls.project_name
                ).dict()
                in parsed_response
            ), "The invocations metric is missing"

        for result in parsed_response:
            if result["app"] in [app_data.class_.NAME, "mlrun-infra"]:
                get_app_results.add(result["name"])
                app_results_full_names.append(result["full_name"])

        expected_results = getattr(app_data, type)
        if type == "metrics":
            expected_results.add(mm_constants.PredictionsQueryConstants.INVOCATIONS)

        assert get_app_results == expected_results
        assert app_results_full_names, f"No {type}"
        return app_results_full_names

    @classmethod
    def _test_api_get_values(
        cls,
        ep_id: str,
        results_full_names: list[str],
        run_db: mlrun.db.httpdb.HTTPRunDB,
    ) -> None:
        cls._logger.debug("Checking GET /metrics-values API")
        names_query = f"?name={'&name='.join(results_full_names)}"
        response = run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=f"projects/{cls.project_name}/model-endpoints/{ep_id}/metrics-values{names_query}",
        )
        for result_values in json.loads(response.content.decode()):
            assert result_values[
                "data"
            ], f"No data for result {result_values['full_name']}"
            assert result_values[
                "values"
            ], f"The values list is empty for result {result_values['full_name']}"

    @classmethod
    def _test_api(cls, ep_id: str, app_data: _AppData) -> None:
        cls._logger.debug("Checking model endpoint monitoring APIs")
        run_db = mlrun.db.httpdb.HTTPRunDB(mlrun.mlconf.dbpath)
        metrics_full_names = cls._test_api_get_metrics(
            ep_id=ep_id, app_data=app_data, run_db=run_db, type="metrics"
        )
        results_full_names = cls._test_api_get_metrics(
            ep_id=ep_id, app_data=app_data, run_db=run_db, type="results"
        )

        cls._test_api_get_values(
            ep_id=ep_id,
            results_full_names=metrics_full_names + results_full_names,
            run_db=run_db,
        )


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestMonitoringAppFlow(TestMLRunSystem, _V3IORecordsChecker):
    project_name = "test-app-flow-v2"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        assert (
            typing.cast(
                int, mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events
            )
            == EXPECTED_EVENTS_COUNT
        )

        cls.model_name = "classification"
        cls.num_features = 4

        cls.app_interval: int = 1  # every 1 minute
        cls.app_interval_seconds = timedelta(minutes=cls.app_interval).total_seconds()

        cls.evidently_workspace_path = (
            f"/v3io/projects/{cls.project_name}/artifacts/evidently-workspace"
        )
        cls.evidently_project_id = str(uuid.uuid4())

        cls.apps_data: list[_AppData] = [
            _DefaultDataDriftAppData,
            _AppData(
                class_=DemoMonitoringAppV2,
                rel_path="assets/application.py",
                results={"data_drift_test", "model_perf"},
            ),
            _AppData(
                class_=CustomEvidentlyMonitoringAppV2,
                rel_path="assets/custom_evidently_app.py",
                requirements=[f"evidently=={SUPPORTED_EVIDENTLY_VERSION}"],
                kwargs={
                    "evidently_workspace_path": cls.evidently_workspace_path,
                    "evidently_project_id": cls.evidently_project_id,
                },
                results={"data_drift_test"},
            ),
        ]

        cls.run_db = mlrun.get_run_db()

    @classmethod
    def custom_setup(cls) -> None:
        _V3IORecordsChecker.custom_setup(project_name=cls.project_name)

    def _submit_controller_and_deploy_writer(
        self, deploy_histogram_data_drift_app
    ) -> None:
        self.project.enable_model_monitoring(
            base_period=self.app_interval,
            **({} if self.image is None else {"image": self.image}),
            deploy_histogram_data_drift_app=deploy_histogram_data_drift_app,
        )

    def _set_and_deploy_monitoring_apps(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for app_data in self.apps_data:
                if app_data.deploy:
                    fn = self.project.set_model_monitoring_function(
                        func=app_data.abs_path,
                        application_class=app_data.class_.__name__,
                        name=app_data.class_.NAME,
                        image="mlrun/mlrun" if self.image is None else self.image,
                        requirements=app_data.requirements,
                        **app_data.kwargs,
                    )
                    executor.submit(fn.deploy)

    def _log_model(self, with_training_set: bool) -> tuple[set[str], set[str]]:
        train_set = None
        dataset = load_iris()
        if with_training_set:
            train_set = pd.DataFrame(
                dataset.data,
                columns=dataset.feature_names,
            )
            inputs = {
                mlrun.feature_store.api.norm_column_name(feature)
                for feature in dataset.feature_names
            }
        else:
            inputs = {f"f{i}" for i in range(len(dataset.feature_names))}

        self.project.log_model(
            f"{self.model_name}_{with_training_set}",
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
            training_set=train_set,
        )
        outputs = {"p0"}

        return inputs, outputs

    @classmethod
    def _deploy_model_serving(
        cls, with_training_set: bool
    ) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        serving_fn = typing.cast(
            mlrun.runtimes.nuclio.serving.ServingRuntime,
            mlrun.import_function(
                "hub://v2_model_server",
                project=cls.project_name,
                new_name="model-serving",
            ),
        )
        serving_fn.add_model(
            f"{cls.model_name}_{with_training_set}",
            model_path=f"store://models/{cls.project_name}/{cls.model_name}_{with_training_set}:latest",
        )
        serving_fn.set_tracking()
        if cls.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = cls.image

        serving_fn.deploy()
        return serving_fn

    @classmethod
    def _infer(
        cls,
        serving_fn: mlrun.runtimes.nuclio.serving.ServingRuntime,
        *,
        num_events: int = 10_000,
        with_training_set: bool = True,
    ) -> None:
        result = serving_fn.invoke(
            f"v2/models/{cls.model_name}_{with_training_set}/infer",
            json.dumps({"inputs": [[0.0] * cls.num_features] * num_events}),
        )
        assert isinstance(result, dict), "Unexpected result type"
        assert "outputs" in result, "Result should have 'outputs' key"
        assert (
            len(result["outputs"]) == num_events
        ), "Outputs length does not match inputs"

    @classmethod
    def _get_model_endpoint_id(cls) -> str:
        endpoints = cls.run_db.list_model_endpoints(project=cls.project_name)
        assert endpoints and len(endpoints) == 1
        return endpoints[0].metadata.uid

    @classmethod
    def _test_model_endpoint_stats(cls, ep_id: str) -> None:
        cls._logger.debug("Checking model endpoint", ep_id=ep_id)
        ep = cls.run_db.get_model_endpoint(project=cls.project_name, endpoint_id=ep_id)
        assert (
            ep.status.current_stats.keys()
            == ep.status.feature_stats.keys()
            == set(ep.spec.feature_names)
        ), "The endpoint current stats keys are not the same as feature stats and feature names"
        assert ep.status.drift_status, "The general drift status is empty"
        assert ep.status.drift_measures, "The drift measures are empty"

        drift_table = pd.DataFrame.from_dict(ep.status.drift_measures, orient="index")
        assert set(drift_table.columns) == {
            "hellinger",
            "kld",
            "tvd",
        }, "The drift metrics are not as expected"
        assert set(drift_table.index) == set(
            ep.spec.feature_names
        ), "The feature names are not as expected"

    @pytest.mark.parametrize("with_training_set", [True, False])
    def test_app_flow(self, with_training_set: bool) -> None:
        self.project = typing.cast(mlrun.projects.MlrunProject, self.project)
        inputs, outputs = self._log_model(with_training_set)

        for i in range(len(self.apps_data)):
            if "with_training_set" in self.apps_data[i].kwargs:
                self.apps_data[i].kwargs["with_training_set"] = with_training_set

        # workaround for ML-5997
        if not with_training_set and _DefaultDataDriftAppData in self.apps_data:
            self.apps_data.remove(_DefaultDataDriftAppData)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(
                self._submit_controller_and_deploy_writer,
                deploy_histogram_data_drift_app=_DefaultDataDriftAppData
                in self.apps_data,
                # workaround for ML-5997
            )
            executor.submit(self._set_and_deploy_monitoring_apps)
            future = executor.submit(self._deploy_model_serving, with_training_set)

        serving_fn = future.result()

        time.sleep(5)
        self._infer(serving_fn, with_training_set=with_training_set)
        # mark the first window as "done" with another request
        time.sleep(
            self.app_interval_seconds
            + mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
            + 2
        )
        self._infer(serving_fn, num_events=1, with_training_set=with_training_set)
        # wait for the completed window to be processed
        time.sleep(1.2 * self.app_interval_seconds)

        ep_id = self._get_model_endpoint_id()
        self._test_v3io_records(ep_id=ep_id, inputs=inputs, outputs=outputs)
        self._test_predictions_table(ep_id)

        if with_training_set:
            self._test_api(ep_id=ep_id, app_data=_DefaultDataDriftAppData)
            if _DefaultDataDriftAppData in self.apps_data:
                self._test_model_endpoint_stats(ep_id=ep_id)


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestMonitoringAppFlowV1(TestMonitoringAppFlow):
    # TODO : delete in 1.9.0 (V1 app deprecation)
    project_name = "test-app-flow-v1"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        super().custom_setup_class()
        cls.apps_data: list[_AppData] = [
            _AppData(
                class_=DemoMonitoringApp,
                rel_path="assets/application.py",
                results={"data_drift_test", "model_perf"},
            ),
            _AppData(
                class_=CustomEvidentlyMonitoringApp,
                rel_path="assets/custom_evidently_app.py",
                requirements=[f"evidently=={SUPPORTED_EVIDENTLY_VERSION}"],
                kwargs={
                    "evidently_workspace_path": cls.evidently_workspace_path,
                    "evidently_project_id": cls.evidently_project_id,
                    "with_training_set": True,
                },
                results={"data_drift_test"},
            ),
        ]

    @pytest.mark.parametrize("with_training_set", [True, False])
    def test_app_flow(self, with_training_set) -> None:
        super().test_app_flow(with_training_set)


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestRecordResults(TestMLRunSystem, _V3IORecordsChecker):
    project_name = "test-mm-record-results"
    name_prefix = "infer-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        # model
        cls.classif = SVC()
        cls.model_name = "svc"
        # data
        cls.columns = ["a1", "a2", "b"]
        cls.y_name = "t"
        cls.num_rows = 15
        cls.num_cols = len(cls.columns)
        cls.num_classes = 2
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = cls._generate_data()
        cls.training_set = cls.x_train.join(cls.y_train)
        cls.test_set = cls.x_test.join(cls.y_test)
        cls.infer_results_df = cls.test_set
        # endpoint
        cls.endpoint_id = "58d42fdd76ad999c377fad1adcafd2790b5a89b9"
        cls.function_name = f"{cls.name_prefix}-function"
        # training
        cls._train()

        # model monitoring app
        cls.app_data = _AppData(
            class_=NoCheckDemoMonitoringApp, rel_path="assets/application.py"
        )

        # model monitoring infra
        cls.app_interval: int = 1  # every 1 minute
        cls.app_interval_seconds = timedelta(minutes=cls.app_interval).total_seconds()
        cls.apps_data = [_DefaultDataDriftAppData, cls.app_data]
        _V3IORecordsChecker.custom_setup(project_name=cls.project_name)

    @classmethod
    def _generate_data(cls) -> list[typing.Union[pd.DataFrame, pd.Series]]:
        rng = np.random.default_rng(seed=1)
        x = pd.DataFrame(rng.random((cls.num_rows, cls.num_cols)), columns=cls.columns)
        y = pd.Series(np.arange(cls.num_rows) % cls.num_classes, name=cls.y_name)
        assert cls.num_rows > cls.num_classes
        return train_test_split(x, y, train_size=0.75, random_state=1)

    @classmethod
    def _train(cls) -> None:
        cls.classif.fit(
            cls.x_train,
            cls.y_train,  # pyright: ignore[reportGeneralTypeIssues]
        )

    def _log_model(self) -> None:
        self.project.log_model(  # pyright: ignore[reportOptionalMemberAccess]
            self.model_name,
            body=pickle.dumps(self.classif),
            model_file="classif.pkl",
            framework="sklearn",
            training_set=self.training_set,
            label_column=self.y_name,
        )

    def _deploy_monitoring_app(self) -> None:
        self.project = typing.cast(mlrun.projects.MlrunProject, self.project)
        fn = self.project.set_model_monitoring_function(
            func=self.app_data.abs_path,
            application_class=self.app_data.class_.__name__,
            name=self.app_data.class_.NAME,
            requirements=self.app_data.requirements,
            image="mlrun/mlrun" if self.image is None else self.image,
            **self.app_data.kwargs,
        )
        self.project.deploy_function(fn)

    def _record_results(self) -> None:
        mlrun.model_monitoring.api.record_results(
            project=self.project_name,
            model_path=self.project.get_artifact_uri(  # pyright: ignore[reportOptionalMemberAccess]
                key=self.model_name, category="model", tag="latest"
            ),
            model_endpoint_name=f"{self.name_prefix}-test",
            function_name=self.function_name,
            endpoint_id=self.endpoint_id,
            context=mlrun.get_or_create_ctx(name=f"{self.name_prefix}-context"),  # pyright: ignore[reportGeneralTypeIssues]
            infer_results_df=self.infer_results_df,
        )

    def _deploy_monitoring_infra(self) -> None:
        self.project.enable_model_monitoring(  # pyright: ignore[reportOptionalMemberAccess]
            base_period=self.app_interval,
            **({} if self.image is None else {"image": self.image}),
        )

    def test_inference_feature_set(self) -> None:
        self._log_model()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self._deploy_monitoring_app)
            executor.submit(self._deploy_monitoring_infra)

        self._record_results()

        time.sleep(2.4 * self.app_interval_seconds)

        self._test_v3io_records(
            self.endpoint_id, inputs=set(self.columns), outputs=set(self.y_name)
        )
        self._test_predictions_table(self.endpoint_id, should_be_empty=True)


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelMonitoringInitialize(TestMLRunSystem):
    project_name = "test-mm-initialize"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    def test_model_monitoring_crud(self) -> None:
        import v3io.dataplane

        all_functions = mm_constants.MonitoringFunctionNames.list() + [
            mm_constants.HistogramDataDriftApplicationConstants.NAME
        ]
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.update_model_monitoring_controller(
                image=self.image or "mlrun/mlrun"
            )

        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun", wait_for_deployment=True
        )

        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            ignore_cache=True,
        )
        assert (
            controller.spec.config["spec.triggers.cron_interval"]["attributes"][
                "interval"
            ]
            == "10m"
        )
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            wait_for_deployment=False,
            rebuild_images=False,
        )
        # check that all the function are still deployed
        for name in all_functions:
            func = self.project.get_function(
                key=name,
                ignore_cache=True,
            )
            func._get_db().get_nuclio_deploy_status(func, verbose=False)
            assert func.status.state == "ready"

        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            wait_for_deployment=False,
            rebuild_images=True,
        )

        # check that all the function are in building state
        for name in all_functions:
            func = self.project.get_function(
                key=name,
                ignore_cache=True,
            )
            func._get_db().get_nuclio_deploy_status(func, verbose=False)
            assert func.status.state == "building"

        self.project._wait_for_functions_deployment(all_functions)

        self.project.update_model_monitoring_controller(
            image=self.image or "mlrun/mlrun", base_period=1, wait_for_deployment=True
        )
        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            ignore_cache=True,
        )
        assert (
            controller.spec.config["spec.triggers.cron_interval"]["attributes"][
                "interval"
            ]
            == "1m"
        )

        self.project.disable_model_monitoring(delete_histogram_data_drift_app=False)
        v3io_client = v3io.dataplane.Client(endpoint=mlrun.mlconf.v3io_api)

        # controller and writer(with hus stream) should be deleted
        for name in mm_constants.MonitoringFunctionNames.list():
            stream_path = mlrun.model_monitoring.helpers.get_stream_path(
                project=self.project.name, function_name=name
            )
            _, container, stream_path = (
                mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                    stream_path
                )
            )
            if name != mm_constants.MonitoringFunctionNames.STREAM:
                with pytest.raises(mlrun.errors.MLRunNotFoundError):
                    self.project.get_function(
                        key=name,
                        ignore_cache=True,
                    )
                with pytest.raises(v3io.dataplane.response.HttpResponseError):
                    v3io_client.stream.describe(container, stream_path)
            else:
                self.project.get_function(
                    key=name,
                    ignore_cache=True,
                )
                v3io_client.stream.describe(container, stream_path)

        self.project.disable_model_monitoring(
            delete_histogram_data_drift_app=False, delete_stream_function=True
        )

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.get_function(
                key=mm_constants.MonitoringFunctionNames.STREAM,
                ignore_cache=True,
            )

        # check that the stream of the stream pod is not deleted
        stream_path = mlrun.model_monitoring.helpers.get_stream_path(
            project=self.project.name,
            function_name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
        )
        _, container, stream_path = (
            mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                stream_path
            )
        )
        v3io_client.stream.describe(container, stream_path)

        self.project.delete_model_monitoring_function(
            mm_constants.HistogramDataDriftApplicationConstants.NAME
        )
        # check that the histogram data drift app and it's stream is deleted
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.get_function(
                key=mm_constants.HistogramDataDriftApplicationConstants.NAME,
                ignore_cache=True,
            )

        with pytest.raises(v3io.dataplane.response.HttpResponseError):
            stream_path = mlrun.model_monitoring.helpers.get_stream_path(
                project=self.project.name,
                function_name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
            )
            _, container, stream_path = (
                mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                    stream_path
                )
            )
            v3io_client.stream.describe(container, stream_path)


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestAllKindOfServing(TestMLRunSystem):
    project_name = "test-mm-serving"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        random_rgb_image_list = (
            np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
            .reshape(-1, 3)
            .tolist()
        )
        cls.models = {
            "int_one_to_one": {
                "name": "serving_1",
                "model_name": "int_one_to_one",
                "class_name": "OneToOne",
                "data_point": [1, 2, 3],
                "schema": ["f0", "f1", "f2", "p0"],
            },
            "int_one_to_many": {
                "name": "serving_2",
                "model_name": "int_one_to_many",
                "class_name": "OneToMany",
                "data_point": [1, 2, 3],
                "schema": ["f0", "f1", "f2", "p0", "p1", "p2", "p3", "p4"],
            },
            "str_one_to_one": {
                "name": "serving_3",
                "model_name": "str_one_to_one",
                "class_name": "OneToOne",
                "data_point": "input_str",
                "schema": ["f0", "p0"],
            },
            "str_one_to_one_with_train": {
                "name": "serving_4",
                "model_name": "str_one_to_one_with_train",
                "class_name": "OneToOne",
                "data_point": "input_str",
                "schema": ["str_in", "str_out"],
                "training_set": pd.DataFrame(
                    data={"str_in": ["str_1", "str_2"], "str_out": ["str_3", "str_4"]}
                ),
                "label_column": "str_out",
            },
            "str_one_to_many": {
                "name": "serving_5",
                "model_name": "str_one_to_many",
                "class_name": "OneToMany",
                "data_point": "input_str",
                "schema": ["f0", "p0", "p1", "p2", "p3", "p4"],
            },
            "img_one_to_one": {
                "name": "serving_6",
                "model_name": "img_one_to_one",
                "class_name": "OneToOne",
                "data_point": random_rgb_image_list,
                "schema": [f"f{i}" for i in range(600)] + ["p0"],
            },
            "int_and_str_one_to_one": {
                "name": "serving_7",
                "model_name": "int_and_str_one_to_one",
                "class_name": "OneToOne",
                "data_point": [1, "a", 3],
                "schema": ["f0", "f1", "f2", "p0"],
            },
        }

    def _log_model(
        self,
        model_name: str,
        training_set: pd.DataFrame = None,
        label_column: typing.Union[str, list[str]] = None,
    ) -> None:
        self.project.log_model(
            model_name,
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
            training_set=training_set,
            label_column=label_column,
        )

    @classmethod
    def _deploy_model_serving(
        cls,
        name: str,
        model_name: str,
        class_name: str,
        enable_tracking: bool = True,
        **kwargs,
    ) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        serving_fn = mlrun.code_to_function(
            project=cls.project_name,
            name=name,
            filename=f"{str((Path(__file__).parent / 'assets').absolute())}/models.py",
            kind="serving",
        )
        serving_fn.add_model(
            model_name,
            model_path=f"store://models/{cls.project_name}/{model_name}:latest",
            class_name=class_name,
        )
        serving_fn.set_tracking(enable_tracking=enable_tracking)
        if cls.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = cls.image

        serving_fn.deploy()
        return typing.cast(mlrun.runtimes.nuclio.serving.ServingRuntime, serving_fn)

    def _test_endpoint(self, model_name, feature_set_uri) -> dict[str, typing.Any]:
        model_dict = self.models[model_name]
        serving_fn = self.project.get_function(model_dict.get("name"))
        data_point = model_dict.get("data_point")

        serving_fn.invoke(
            f"v2/models/{model_name}/infer",
            json.dumps(
                {"inputs": [data_point]},
            ),
        )
        serving_fn.invoke(
            f"v2/models/{model_name}/infer",
            json.dumps({"inputs": [data_point, data_point]}),
        )
        time.sleep(
            mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs + 10
        )

        offline_response_df = ParquetTarget(
            name="temp",
            path=fstore.get_feature_set(feature_set_uri).spec.targets[0].path,
        ).as_df()

        is_schema_saved = set(model_dict.get("schema")).issubset(
            offline_response_df.columns
        )
        has_all_the_events = offline_response_df.shape[0] == 3

        return {
            "model_name": model_name,
            "is_schema_saved": is_schema_saved,
            "has_all_the_events": has_all_the_events,
        }

    def test_all(self) -> None:
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            base_period=1,
            deploy_histogram_data_drift_app=False,
        )
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for model_name, model_dict in self.models.items():
                self._log_model(
                    model_name,
                    training_set=model_dict.get("training_set"),
                    label_column=model_dict.get("label_column"),
                )
                future = executor.submit(self._deploy_model_serving, **model_dict)
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()

        futures_2 = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.db = mlrun.model_monitoring.get_store_object(project=self.project_name)
            endpoints = self.db.list_model_endpoints()
            for endpoint in endpoints:
                future = executor.submit(
                    self._test_endpoint,
                    model_name=endpoint[mm_constants.EventFieldType.MODEL].split(":")[
                        0
                    ],
                    feature_set_uri=endpoint[
                        mm_constants.EventFieldType.FEATURE_SET_URI
                    ],
                )
                futures_2.append(future)

        for future in concurrent.futures.as_completed(futures_2):
            res_dict = future.result()
            assert res_dict[
                "is_schema_saved"
            ], f"For {res_dict['model_name']} the schema of parquet is missing columns"

            assert res_dict[
                "has_all_the_events"
            ], f"For {res_dict['model_name']} Not all the events were saved"


class TestTracking(TestAllKindOfServing):
    project_name = "test-tracking"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        cls.models = {
            "int_one_to_one": {
                "name": "serving_1",
                "model_name": "int_one_to_one",
                "class_name": "OneToOne",
                "data_point": [1, 2, 3],
                "schema": ["f0", "f1", "f2", "p0"],
            },
        }

    def test_tracking(self) -> None:
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            base_period=1,
            deploy_histogram_data_drift_app=False,
        )

        for model_name, model_dict in self.models.items():
            self._log_model(
                model_name,
                training_set=model_dict.get("training_set"),
                label_column=model_dict.get("label_column"),
            )
            self._deploy_model_serving(**model_dict, enable_tracking=False)

        self.db = mlrun.model_monitoring.get_store_object(project=self.project_name)
        endpoints = self.db.list_model_endpoints()
        assert len(endpoints) == 0

        for model_name, model_dict in self.models.items():
            self._deploy_model_serving(**model_dict, enable_tracking=True)

        self.db = mlrun.model_monitoring.get_store_object(project=self.project_name)
        endpoints = self.db.list_model_endpoints()
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        assert (
            endpoint["monitoring_mode"]
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled
        )

        res_dict = self._test_endpoint(
            model_name=endpoint[mm_constants.EventFieldType.MODEL].split(":")[0],
            feature_set_uri=endpoint[mm_constants.EventFieldType.FEATURE_SET_URI],
        )
        assert res_dict[
            "is_schema_saved"
        ], f"For {res_dict['model_name']} the schema of parquet is missing columns"

        assert res_dict[
            "has_all_the_events"
        ], f"For {res_dict['model_name']} Not all the events were saved"

        for model_name, model_dict in self.models.items():
            self._deploy_model_serving(**model_dict, enable_tracking=False)

        self.db = mlrun.model_monitoring.get_store_object(project=self.project_name)
        endpoints = self.db.list_model_endpoints()
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        assert (
            endpoint["monitoring_mode"]
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.disabled
        )

        res_dict = self._test_endpoint(
            model_name=endpoint[mm_constants.EventFieldType.MODEL].split(":")[0],
            feature_set_uri=endpoint[mm_constants.EventFieldType.FEATURE_SET_URI],
        )

        assert res_dict[
            "has_all_the_events"
        ], f"For {res_dict['model_name']}, Despite tracking being disabled, there is new data in the parquet."
