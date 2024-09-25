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
import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.types
import mlrun.db.httpdb
import mlrun.feature_store
import mlrun.feature_store as fstore
import mlrun.model_monitoring
import mlrun.model_monitoring.api
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.applications import (
    SUPPORTED_EVIDENTLY_VERSION,
    ModelMonitoringApplicationBase,
)
from mlrun.model_monitoring.applications.histogram_data_drift import (
    HistogramDataDriftApplication,
)
from mlrun.utils.logger import Logger
from tests.system.base import TestMLRunSystem

from .assets.application import (
    EXPECTED_EVENTS_COUNT,
    DemoMonitoringApp,
    ErrApp,
    NoCheckDemoMonitoringApp,
)
from .assets.custom_evidently_app import (
    CustomEvidentlyMonitoringApp,
)


@dataclass
class _AppData:
    class_: type[ModelMonitoringApplicationBase]
    rel_path: str
    requirements: list[str] = field(default_factory=list)
    kwargs: dict[str, typing.Any] = field(default_factory=dict)
    abs_path: str = field(init=False)
    results: set[str] = field(default_factory=set)  # only for testing
    metrics: set[str] = field(default_factory=set)  # only for testing
    artifacts: set[str] = field(default_factory=set)  # only for testing
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
    artifacts={"features_drift_results", "drift_table_plot"},
)


class _V3IORecordsChecker:
    project_name: str
    _logger: Logger
    apps_data: list[_AppData]
    app_interval: int

    @classmethod
    def custom_setup(cls, project_name: str) -> None:
        # By default, the TSDB connection is based on V3IO TSDB
        # To use TDEngine, set the `MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION`
        # environment variable to the TDEngine connection string,
        # e.g. "taosws://user:password@host:port"

        project = mlrun.get_or_create_project(
            project_name, "./", allow_cross_project=True
        )
        project.set_model_monitoring_credentials(
            endpoint_store_connection=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
            stream_path=mlrun.mlconf.model_endpoint_monitoring.stream_connection,
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

        cls._tsdb_storage = mlrun.model_monitoring.get_tsdb_connector(
            project=project_name,
            tsdb_connection_string=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        cls._kv_storage = mlrun.model_monitoring.get_store_object(
            project=project_name,
            store_connection_string=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
        )
        cls._v3io_container = f"users/pipelines/{project_name}/monitoring-apps/"

    @classmethod
    def _test_results_kv_record(cls, ep_id: str) -> None:
        for app_data in cls.apps_data:
            if not app_data.results:
                continue
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
    def _test_tsdb_record(
        cls, ep_id: str, last_request: datetime, error_count: float
    ) -> None:
        if cls._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
            # V3IO TSDB
            df: pd.DataFrame = cls._tsdb_storage.get_results_metadata(endpoint_id=ep_id)
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
            app_data.class_.NAME for app_data in cls.apps_data if app_data.results
        }, "The application names are different than expected"

        tsdb_metrics = df.groupby("application_name").result_name.unique()
        for app_data in cls.apps_data:
            if app_metrics := app_data.results:
                app_name = app_data.class_.NAME
                cls._logger.debug("Checking the TSDB record of app", app_name=app_name)
                assert (
                    set(tsdb_metrics[app_name]) == app_metrics
                ), "The TSDB saved metrics are different than expected"

        if cls._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
            cls._logger.debug("Checking the MEP status")
            rs_tsdb = cls._tsdb_storage.get_drift_status(endpoint_ids=ep_id)
            cls._check_valid_tsdb_result(rs_tsdb, ep_id, "result_status", 2.0)

            if last_request:
                cls._logger.debug("Checking the MEP last_request")
                lr_tsdb = cls._tsdb_storage.get_last_request(endpoint_ids=ep_id)
                cls._check_valid_tsdb_result(
                    lr_tsdb, ep_id, "last_request", last_request
                )

            if error_count:
                cls._logger.debug("Checking the MEP error_count")
                ec_tsdb = cls._tsdb_storage.get_error_count(endpoint_ids=ep_id)
                cls._check_valid_tsdb_result(ec_tsdb, ep_id, "error_count", error_count)

    @classmethod
    def _check_valid_tsdb_result(
        cls, df: pd.DataFrame, ep_id: str, result_name: str, result_value: typing.Any
    ):
        assert not df.empty, "No TSDB data"
        assert (
            df.endpoint_id == ep_id
        ).all(), "The endpoint IDs are different than expected"
        assert (
            df[df["endpoint_id"] == ep_id][result_name].item() == result_value
        ), f"The {result_name} is different than expected for {ep_id}"

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
    def _test_parquet(
        cls, ep_id: str, inputs: set[str], outputs: set[str]
    ) -> None:  # TODO : delete in 1.9.0  (V1 app deprecation)
        parquet_apps_directory = (
            mlrun.model_monitoring.helpers.get_monitoring_parquet_path(
                mlrun.get_or_create_project(cls.project_name, allow_cross_project=True),
                kind=mm_constants.FileTargetKind.PARQUET,
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
        cls,
        ep_id: str,
        inputs: set[str],
        outputs: set[str],
        last_request: datetime = None,
        error_count: float = None,
    ) -> None:
        cls._test_parquet(ep_id, inputs, outputs)
        cls._test_results_kv_record(ep_id)
        cls._test_metrics_kv_record(ep_id)
        cls._test_tsdb_record(ep_id, last_request=last_request, error_count=error_count)

    @classmethod
    def _test_api_get_metrics(
        cls,
        ep_id: str,
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
            get_app_results.add(result["name"])
            app_results_full_names.append(result["full_name"])

        expected_results = set().union(
            *[getattr(app_data, type) for app_data in cls.apps_data]
        )

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

        # ML-6940
        end = int(time.time() * 1000)
        start = end - 1000 * 60 * 60 * 24 * 30  # 30 days in the past
        base_query = f"?name={'&name='.join(results_full_names)}"
        query_with_start_and_end_times = f"{base_query}&start={start}&end={end}"

        for query in (base_query, query_with_start_and_end_times):
            response = run_db.api_call(
                method=mlrun.common.types.HTTPMethod.GET,
                path=f"projects/{cls.project_name}/model-endpoints/{ep_id}/metrics-values{query}",
            )
            for result_values in json.loads(response.content.decode()):
                assert result_values[
                    "data"
                ], f"No data for result {result_values['full_name']}"
                assert result_values[
                    "values"
                ], f"The values list is empty for result {result_values['full_name']}"

    @classmethod
    def _test_api(cls, ep_id: str) -> None:
        cls._logger.debug("Checking model endpoint monitoring APIs")
        run_db = mlrun.db.httpdb.HTTPRunDB(mlrun.mlconf.dbpath)
        metrics_full_names = cls._test_api_get_metrics(
            ep_id=ep_id, run_db=run_db, type="metrics"
        )
        results_full_names = cls._test_api_get_metrics(
            ep_id=ep_id, run_db=run_db, type="results"
        )

        cls._test_api_get_values(
            ep_id=ep_id,
            results_full_names=metrics_full_names + results_full_names,
            run_db=run_db,
        )


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
class TestMonitoringAppFlow(TestMLRunSystem, _V3IORecordsChecker):
    project_name = "test-app-flow"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None
    error_count = 10

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

        # The main inference task event count
        cls.num_events = 10_000

        cls.app_interval: int = 1  # every 1 minute
        cls.app_interval_seconds = timedelta(minutes=cls.app_interval).total_seconds()

        cls.evidently_workspace_path = (
            f"/v3io/projects/{cls.project_name}/artifacts/evidently-workspace"
        )
        cls.evidently_project_id = str(uuid.uuid4())

        cls.apps_data: list[_AppData] = [
            _DefaultDataDriftAppData,
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
                },
                results={"data_drift_test"},
                artifacts={"evidently_report", "evidently_suite", "dashboard"},
            ),
            _AppData(
                class_=ErrApp,
                rel_path="assets/application.py",
            ),
        ]

        cls.run_db = mlrun.get_run_db()

    def custom_setup(self) -> None:
        # skips TestMLRunSystem, calls custom_setup() of _V3IORecordsChecker
        super(TestMLRunSystem, self).custom_setup(project_name=self.project_name)

    def custom_teardown(self) -> None:
        # validate that stream resources were deleted as expected
        stream_path = self._test_env[
            "MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"
        ]

        func_to_validate = [
            "model-monitoring-writer",
            "histogram-data-drift",
            "evidently-app-test-v2",
        ]

        if stream_path.startswith("v3io:///"):
            from v3io.dataplane.response import HttpResponseError

            from mlrun.utils.v3io_clients import get_v3io_client

            client = get_v3io_client(endpoint=mlrun.mlconf.v3io_api)

            for func in func_to_validate:
                with pytest.raises(HttpResponseError):
                    client.object.get(
                        container="projects",
                        path=f"{self.project_name}/model-endpoints/stream-{func}/serving-state.json",
                    )

            # validate that the monitoring stream was deleted
            with pytest.raises(HttpResponseError):
                client.object.get(
                    container="projects",
                    path=f"{self.project_name}/model-endpoints/stream/serving-state.json",
                )

        elif stream_path.startswith("kafka://"):
            import kafka

            _, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
            consumer = kafka.KafkaConsumer(
                bootstrap_servers=brokers,
            )
            topics = consumer.topics()

            project_topics_list = [f"monitoring_stream_{self.project_name}"]
            for func in func_to_validate:
                project_topics_list.append(
                    f"monitoring_stream_{self.project_name}_{func}"
                )

            for topic in project_topics_list:
                assert topic not in topics

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

    def _add_error_alert(self) -> None:
        self._logger.debug("Create an error alert")
        entity_kind = alert_objects.EventEntityKind.MODEL_MONITORING_APPLICATION

        dummy_notification = mlrun.common.schemas.Notification(
            kind="webhook",
            name=mlrun.common.schemas.alert.EventKind.MM_APP_FAILED,
            condition="",
            params={"url": "some-url"},
            severity="debug",
            message="mm app failed!",
        )

        alert_config = mlrun.alerts.alert.AlertConfig(
            project=self.project_name,
            name=mlrun.common.schemas.alert.EventKind.MM_APP_FAILED,
            summary="An invalid event has been detected in the model monitoring application",
            severity=alert_objects.AlertSeverity.HIGH,
            entities=alert_objects.EventEntities(
                kind=entity_kind,
                project=self.project_name,
                ids=[f"{self.project_name}_err-app"],
            ),
            trigger=alert_objects.AlertTrigger(
                events=[mlrun.common.schemas.alert.EventKind.MM_APP_FAILED]
            ),
            criteria=alert_objects.AlertCriteria(count=1, period="10m"),
            notifications=[
                alert_objects.AlertNotification(notification=dummy_notification)
            ],
            reset_policy=mlrun.common.schemas.alert.ResetPolicy.AUTO,
        )

        self.project.store_alert_config(alert_config)

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
        num_events: int,
        with_training_set: bool = True,
    ) -> datetime:
        result = serving_fn.invoke(
            f"v2/models/{cls.model_name}_{with_training_set}/infer",
            json.dumps({"inputs": [[0.0] * cls.num_features] * num_events}),
        )
        assert isinstance(result, dict), "Unexpected result type"
        assert "outputs" in result, "Result should have 'outputs' key"
        assert (
            len(result["outputs"]) == num_events
        ), "Outputs length does not match inputs"
        return datetime.fromisoformat(result["timestamp"])

    @classmethod
    def _infer_with_error(
        cls,
        serving_fn: mlrun.runtimes.nuclio.serving.ServingRuntime,
        *,
        with_training_set: bool = True,
    ):
        for i in range(cls.error_count):
            try:
                serving_fn.invoke(
                    f"v2/models/{cls.model_name}_{with_training_set}/infer",
                    json.dumps({"inputs": [[0.0] * (cls.num_features + 1)]}),
                )
            except Exception:
                pass

    @classmethod
    def _get_model_endpoint_id(cls) -> str:
        endpoints = cls.run_db.list_model_endpoints(project=cls.project_name)
        assert endpoints and len(endpoints) == 1, "Expects a single model endpoint"
        endpoint = typing.cast(mlrun.model_monitoring.ModelEndpoint, endpoints[0])
        assert endpoint.spec.stream_path == mlrun.model_monitoring.get_stream_path(
            project=cls.project_name,
            stream_uri=mlrun.mlconf.model_endpoint_monitoring.stream_connection,
        ), "The model endpoint stream path is different than expected"
        return endpoint.metadata.uid

    def _test_artifacts(self, ep_id: str) -> None:
        for app_data in self.apps_data:
            if app_data.artifacts:
                app_name = app_data.class_.NAME
                self._logger.debug("Checking app artifacts", app_name=app_name)
                for key in app_data.artifacts:
                    self._logger.debug("Checking artifact existence", key=key)
                    artifact = self.project.get_artifact(key)
                    self._logger.debug("Checking artifact labels", key=key)
                    assert {
                        "mlrun/producer-type": "model-monitoring-app",
                        "mlrun/app-name": app_name,
                        "mlrun/endpoint-id": ep_id,
                    }.items() <= artifact.labels.items()
                    self._logger.debug(
                        "Test the artifact can be fetched from the store", key=key
                    )
                    artifact.to_dataitem().get()

    @classmethod
    def _test_model_endpoint_stats(cls, ep_id: str) -> None:
        cls._logger.debug("Checking model endpoint", ep_id=ep_id)
        ep = cls.run_db.get_model_endpoint(project=cls.project_name, endpoint_id=ep_id)
        assert ep.status.feature_stats.keys() == set(
            ep.spec.feature_names
        ), "The endpoint's feature stats keys are not the same as the feature names"
        assert set(ep.status.current_stats.keys()) == set(
            ep.status.feature_stats.keys()
        ), "The endpoint's current stats is different than expected"
        assert ep.status.drift_status, "The general drift status is empty"
        assert ep.status.drift_measures, "The drift measures are empty"

        for measure in ["hellinger_mean", "kld_mean", "tvd_mean"]:
            assert isinstance(
                ep.status.drift_measures.pop(measure, None), float
            ), f"Expected '{measure}' in drift measures"

        drift_table = pd.DataFrame.from_dict(ep.status.drift_measures, orient="index")
        assert set(drift_table.columns) == {
            "hellinger",
            "kld",
            "tvd",
        }, "The drift metrics are not as expected"
        assert set(drift_table.index) == set(
            ep.spec.feature_names
        ), "The feature names are not as expected"

        assert (
            ep.status.current_stats["sepal_length_cm"]["count"] == cls.num_events
        ), "Different number of events than expected"

    @classmethod
    def _test_error_alert(cls) -> None:
        cls._logger.debug("Checking the error alert")
        alerts = cls.run_db.list_alerts_configs(cls.project_name)
        assert len(alerts) == 1, "Expects a single alert"

        # Validate alert configuration
        alert = alerts[0]
        assert alert.name == mlrun.common.schemas.alert.EventKind.MM_APP_FAILED
        assert alert.trigger["events"] == [
            mlrun.common.schemas.alert.EventKind.MM_APP_FAILED
        ]
        assert (
            alert.entities["kind"]
            == alert_objects.EventEntityKind.MODEL_MONITORING_APPLICATION
        )
        assert alert.entities["ids"] == [f"{cls.project_name}_err-app"]

        # Validate alert notification
        assert alert.count == 1

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

        self._submit_controller_and_deploy_writer(
            deploy_histogram_data_drift_app=_DefaultDataDriftAppData in self.apps_data,
            # workaround for ML-5997
        )
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self._set_and_deploy_monitoring_apps)
            future = executor.submit(self._deploy_model_serving, with_training_set)

        serving_fn = future.result()
        self._add_error_alert()

        time.sleep(5)
        self._infer(
            serving_fn, num_events=self.num_events, with_training_set=with_training_set
        )

        self._infer_with_error(serving_fn, with_training_set=with_training_set)
        # mark the first window as "done" with another request
        time.sleep(
            self.app_interval_seconds
            + mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
            + 2
        )
        for i in range(10):
            last_request = self._infer(
                serving_fn, num_events=1, with_training_set=with_training_set
            )
        # wait for the completed window to be processed
        time.sleep(1.2 * self.app_interval_seconds)

        ep_id = self._get_model_endpoint_id()

        self._test_v3io_records(
            ep_id=ep_id,
            inputs=inputs,
            outputs=outputs,
            last_request=last_request,
            error_count=self.error_count,
        )
        self._test_predictions_table(ep_id)

        self._test_artifacts(ep_id=ep_id)
        self._test_api(ep_id=ep_id)
        if _DefaultDataDriftAppData in self.apps_data:
            self._test_model_endpoint_stats(ep_id=ep_id)
        self._test_error_alert()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
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
            class_=NoCheckDemoMonitoringApp,
            rel_path="assets/application.py",
            results={"data_drift_test", "model_perf"},
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
@pytest.mark.model_monitoring
class TestModelMonitoringInitialize(TestMLRunSystem):
    """Test model monitoring infrastructure initialization and cleanup, including the usage of
    disable the model monitoring and delete a specific model monitoring application."""

    project_name = "test-mm-initialize"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    def test_model_monitoring_crud(self) -> None:
        # Main validations:
        # 1 - Deploy model monitoring infrastructure and validate controller cron trigger
        # 2 - Validate that all the model monitoring functions are deployed
        # 3 - Update the controller cron trigger and validate the change
        # 4 - Disable model monitoring and validate the related resources are deleted
        # 5 - Disable the monitoring stream pod and validate the stream resource is not deleted
        # 6 - Delete the histogram data drift application and validate the related resources are deleted

        all_functions = mm_constants.MonitoringFunctionNames.list() + [
            mm_constants.HistogramDataDriftApplicationConstants.NAME
        ]
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.update_model_monitoring_controller(
                image=self.image or "mlrun/mlrun"
            )
        self.project.set_model_monitoring_credentials(
            endpoint_store_connection=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
            stream_path=mlrun.mlconf.model_endpoint_monitoring.stream_connection,
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            wait_for_deployment=True,
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

        stream_path = self._test_env[
            "MLRUN_MODEL_ENDPOINT_MONITORING__STREAM_CONNECTION"
        ]
        if stream_path.startswith("v3io:///"):
            import v3io.dataplane

            v3io_client = v3io.dataplane.Client(endpoint=mlrun.mlconf.v3io_api)

            # controller and writer(with has stream) should be deleted
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

            self._disable_stream_function()

            # check that the stream of the stream resource is not deleted
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

            # check that the stream of the histogram data drift app is deleted
            self._delete_histogram_app()

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

        elif stream_path.startswith("kafka://"):
            import kafka

            _, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
            consumer = kafka.KafkaConsumer(
                bootstrap_servers=brokers,
            )
            topics = consumer.topics()

            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                self.project.get_function(
                    key=mm_constants.MonitoringFunctionNames.WRITER,
                    ignore_cache=True,
                )
            assert (
                f"monitoring_stream_{self.project_name}_{mm_constants.MonitoringFunctionNames.WRITER}"
                not in topics
            )

            self.project.get_function(
                key=mm_constants.MonitoringFunctionNames.STREAM,
                ignore_cache=True,
            )

            assert f"monitoring_stream_{self.project_name}" in topics

            self._disable_stream_function()

            # check that the topic of the stream resource is not deleted
            consumer = kafka.KafkaConsumer(
                bootstrap_servers=brokers,
            )
            topics = consumer.topics()
            assert f"monitoring_stream_{self.project_name}" in topics

            self._delete_histogram_app()

            # check that the topic of the histogram data drift app is deleted
            consumer = kafka.KafkaConsumer(
                bootstrap_servers=brokers,
            )
            topics = consumer.topics()
            assert (
                f"monitoring_stream_{self.project_name}_{mm_constants.HistogramDataDriftApplicationConstants.NAME}"
                not in topics
            )

    def _disable_stream_function(self):
        self.project.disable_model_monitoring(
            delete_histogram_data_drift_app=False, delete_stream_function=True
        )

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.get_function(
                key=mm_constants.MonitoringFunctionNames.STREAM,
                ignore_cache=True,
            )

    def _delete_histogram_app(self):
        self.project.delete_model_monitoring_function(
            mm_constants.HistogramDataDriftApplicationConstants.NAME
        )
        # check that the histogram data drift app and it's stream is deleted
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.get_function(
                key=mm_constants.HistogramDataDriftApplicationConstants.NAME,
                ignore_cache=True,
            )


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.model_monitoring
class TestMonitoredServings(TestMLRunSystem):
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
        cls.router_models = {
            "int_one_to_one": {
                "model_name": "int_one_to_one",
                "class_name": "OneToOne",
                "data_point": [1, 2, 3],
                "schema": ["f0", "f1", "f2", "p0"],
            },
            "int_one_to_many": {
                "model_name": "int_one_to_many",
                "class_name": "OneToMany",
                "data_point": [1, 2, 3],
                "schema": ["f0", "f1", "f2", "p0", "p1", "p2", "p3", "p4"],
            },
            "str_one_to_one": {
                "model_name": "str_one_to_one",
                "class_name": "OneToOne",
                "data_point": ["input_str"],
                "schema": ["f0", "p0"],
            },
            "str_one_to_one_with_train": {
                "model_name": "str_one_to_one_with_train",
                "class_name": "OneToOne",
                "data_point": ["input_str"],
                "schema": ["str_in", "str_out"],
                "training_set": pd.DataFrame(
                    data={"str_in": ["str_1", "str_2"], "str_out": ["str_3", "str_4"]}
                ),
                "label_column": "str_out",
            },
            "str_one_to_many": {
                "model_name": "str_one_to_many",
                "class_name": "OneToMany",
                "data_point": ["input_str"],
                "schema": ["f0", "p0", "p1", "p2", "p3", "p4"],
            },
            "img_one_to_one": {
                "model_name": "img_one_to_one",
                "class_name": "OneToOne",
                "data_point": random_rgb_image_list,
                "schema": [f"f{i}" for i in range(600)] + ["p0"],
            },
            "int_and_str_one_to_one": {
                "model_name": "int_and_str_one_to_one",
                "class_name": "OneToOne",
                "data_point": [1, "a", 3],
                "schema": ["f0", "f1", "f2", "p0"],
            },
        }

        cls.test_models_tracking = {
            "int_one_to_one": {
                "model_name": "int_one_to_one",
                "class_name": "OneToOne",
                "data_point": [1, 2, 3],
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

    def _deploy_model_router(
        self,
        name: str,
        enable_tracking: bool = True,
    ) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        serving_fn = mlrun.code_to_function(
            project=self.project_name,
            name=name,
            filename=f"{str((Path(__file__).parent / 'assets').absolute())}/models.py",
            kind="serving",
        )
        serving_fn.set_topology("router")
        for model_name, model_dict in self.router_models.items():
            self._log_model(
                model_name=model_name,
                training_set=model_dict.get("training_set"),
                label_column=model_dict.get("label_column"),
            )
            serving_fn.add_model(
                model_name,
                model_path=f"store://models/{self.project_name}/{model_name}:latest",
                class_name=model_dict.get("class_name"),
            )
        serving_fn.set_tracking(enable_tracking=enable_tracking)
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image

        serving_fn.deploy()
        return typing.cast(mlrun.runtimes.nuclio.serving.ServingRuntime, serving_fn)

    def _deploy_model_serving(
        self,
        model_name: str,
        class_name: str,
        enable_tracking: bool = True,
        **kwargs,
    ) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        serving_fn = mlrun.code_to_function(
            project=self.project_name,
            name=self.function_name,
            filename=f"{str((Path(__file__).parent / 'assets').absolute())}/models.py",
            kind="serving",
        )
        serving_fn.add_model(
            model_name,
            model_path=f"store://models/{self.project_name}/{model_name}:latest",
            class_name=class_name,
        )
        serving_fn.set_tracking(enable_tracking=enable_tracking)
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image

        serving_fn.deploy()
        return typing.cast(mlrun.runtimes.nuclio.serving.ServingRuntime, serving_fn)

    def _test_endpoint(
        self, model_name, feature_set_uri, model_dict
    ) -> dict[str, typing.Any]:
        serving_fn = self.project.get_function(self.function_name)
        data_point = model_dict.get("data_point")
        if model_name == "img_one_to_one":
            data_point = [data_point]
        serving_fn.invoke(
            f"v2/models/{model_name}/infer",
            json.dumps(
                {"inputs": data_point},
            ),
        )
        if model_name == "img_one_to_one":
            data_point = data_point[0]
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
            "df": offline_response_df,
        }

    def test_different_kind_of_serving(self) -> None:
        self.function_name = "serving-router"
        self.project.set_model_monitoring_credentials(
            endpoint_store_connection=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
            stream_path=mlrun.mlconf.model_endpoint_monitoring.stream_connection,
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            base_period=1,
            deploy_histogram_data_drift_app=False,
        )
        self._deploy_model_router(self.function_name)

        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.db = mlrun.model_monitoring.get_store_object(
                project=self.project_name,
                store_connection_string=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
            )
            endpoints = self.db.list_model_endpoints()
            assert len(endpoints) == 7
            for endpoint in endpoints:
                future = executor.submit(
                    self._test_endpoint,
                    model_name=endpoint[mm_constants.EventFieldType.MODEL].split(":")[
                        0
                    ],
                    feature_set_uri=endpoint[
                        mm_constants.EventFieldType.FEATURE_SET_URI
                    ],
                    model_dict=self.router_models[
                        endpoint[mm_constants.EventFieldType.MODEL].split(":")[0]
                    ],
                )
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            res_dict = future.result()
            assert res_dict[
                "is_schema_saved"
            ], f"For {res_dict['model_name']} the schema of parquet is missing columns"

            assert res_dict[
                "has_all_the_events"
            ], f"For {res_dict['model_name']} Not all the events were saved"

    def test_tracking(self) -> None:
        self.function_name = "serving-1"
        self.project.set_model_monitoring_credentials(
            endpoint_store_connection=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
            stream_path=mlrun.mlconf.model_endpoint_monitoring.stream_connection,
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )
        self.db = mlrun.model_monitoring.get_store_object(
            project=self.project_name,
            store_connection_string=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
        )
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            base_period=1,
            deploy_histogram_data_drift_app=False,
        )

        for model_name, model_dict in self.test_models_tracking.items():
            self._log_model(
                model_name,
                training_set=model_dict.get("training_set"),
                label_column=model_dict.get("label_column"),
            )
            self._deploy_model_serving(**model_dict, enable_tracking=False)

        endpoints = self.db.list_model_endpoints()
        assert len(endpoints) == 0

        for model_name, model_dict in self.test_models_tracking.items():
            self._deploy_model_serving(**model_dict, enable_tracking=True)

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
            model_dict=self.test_models_tracking[
                endpoint[mm_constants.EventFieldType.MODEL].split(":")[0]
            ],
        )
        assert res_dict[
            "is_schema_saved"
        ], f"For {res_dict['model_name']} the schema of parquet is missing columns"

        assert res_dict[
            "has_all_the_events"
        ], f"For {res_dict['model_name']} Not all the events were saved"

        for model_name, model_dict in self.test_models_tracking.items():
            self._deploy_model_serving(**model_dict, enable_tracking=False)

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
            model_dict=self.test_models_tracking[
                endpoint[mm_constants.EventFieldType.MODEL].split(":")[0]
            ],
        )

        assert res_dict[
            "has_all_the_events"
        ], f"For {res_dict['model_name']}, Despite tracking being disabled, there is new data in the parquet."

    def test_enable_model_monitoring_after_failure(self) -> None:
        self.function_name = "test-function"
        self.project.set_model_monitoring_credentials(
            endpoint_store_connection=mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection,
            stream_path=mlrun.mlconf.model_endpoint_monitoring.stream_connection,
            tsdb_connection=mlrun.mlconf.model_endpoint_monitoring.tsdb_connection,
        )

        with pytest.raises(
            mlrun.runtimes.utils.RunError,
            match="Function .* deployment failed",
        ):
            self.project.enable_model_monitoring(
                image="nonexistent-image:1.0.0",
                wait_for_deployment=True,
            )
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
            wait_for_deployment=True,
        )
        self.project.enable_model_monitoring(
            image=self.image or "mlrun/mlrun",
        )
        # check that all the function are still deployed
        all_functions = mm_constants.MonitoringFunctionNames.list() + [
            mm_constants.HistogramDataDriftApplicationConstants.NAME
        ]
        for name in all_functions:
            func = self.project.get_function(
                key=name,
                ignore_cache=True,
            )
            func._get_db().get_nuclio_deploy_status(func, verbose=False)
            assert func.status.state == "ready"
