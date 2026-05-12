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
import ipaddress
import json
import pickle
import tempfile
import time
import typing
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import kafka
import numpy as np
import pandas as pd
import pytest
import v3io.dataplane
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from v3io.dataplane.response import HttpResponseError

import mlrun
import mlrun.alerts.alert
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.types
import mlrun.db.httpdb
import mlrun.feature_store
import mlrun.feature_store as fstore
import mlrun.model_monitoring
import mlrun.model_monitoring.api
import mlrun.serving
from mlrun.common.schemas.model_monitoring import ResultKindApp
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpointDriftValues,
    ModelEndpointMonitoringMetric,
)
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaStream,
    DatastoreProfileV3io,
)
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.applications import (
    ExistingDataHandling,
    ModelMonitoringApplicationBase,
    histogram_data_drift,
)
from mlrun.model_monitoring.applications.evidently import SUPPORTED_EVIDENTLY_VERSION
from mlrun.model_monitoring.db._schedules import (
    delete_model_monitoring_schedules_user_folder,
)
from mlrun.utils.logger import Logger
from mlrun.utils.v3io_clients import get_v3io_client
from tests.system.base import TestMLRunSystem

from . import TestMLRunSystemModelMonitoring
from .assets.application import (
    EXPECTED_EVENTS_COUNT,
    CountApp,
    DemoMonitoringApp,
    ErrApp,
    NoCheckDemoMonitoringApp,
)
from .assets.custom_evidently_app import DemoEvidentlyMonitoringApp


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
    class_=histogram_data_drift.HistogramDataDriftApplication,
    rel_path="",
    deploy=False,
    results={"general_drift"},
    metrics={"hellinger_mean", "kld_mean", "tvd_mean"},
)


class _V3IORecordsChecker:
    project_name: str
    _logger: Logger
    app_interval: int
    mm_tsdb_profile: DatastoreProfile

    @classmethod
    def custom_setup(cls, project_name: str) -> None:
        cls._tsdb_storage = mlrun.model_monitoring.get_tsdb_connector(
            project=project_name, profile=cls.mm_tsdb_profile
        )
        cls._v3io_container = f"users/pipelines/{project_name}/monitoring-apps/"

    @classmethod
    def _test_tsdb_record(
        cls,
        ep_id: str,
        last_request: datetime,
        error_count: float,
        apps_data: list[_AppData],
    ) -> None:
        df: pd.DataFrame = cls._tsdb_storage.get_results_metadata(endpoint_id=ep_id)

        assert not df.empty, "No TSDB data"
        assert (df.endpoint_id == ep_id).all(), (
            "The endpoint IDs are different than expected"
        )

        assert set(df.application_name) == {
            app_data.class_.NAME for app_data in apps_data if app_data.results
        }, "The application names are different than expected"

        tsdb_metrics = df.groupby("application_name").result_name.unique()
        for app_data in apps_data:
            if app_metrics := app_data.results:
                app_name = app_data.class_.NAME
                cls._logger.debug("Checking the TSDB record of app", app_name=app_name)
                assert set(tsdb_metrics[app_name]) == app_metrics, (
                    "The TSDB saved metrics are different than expected"
                )

        cls._logger.debug("Checking the MEP status")
        rs_tsdb = cls._tsdb_storage.get_drift_status(endpoint_ids=ep_id)
        cls._check_valid_tsdb_result(rs_tsdb, ep_id, "result_status", 2.0)

        if last_request:
            cls._logger.debug("Checking the MEP last_request")
            lr_tsdb = cls._tsdb_storage.get_last_request(endpoint_ids=ep_id)
            if isinstance(lr_tsdb, pd.DataFrame):
                cls._check_valid_tsdb_result(
                    lr_tsdb, ep_id, "last_request", pd.Timestamp(last_request)
                )
            else:
                cls._check_last_request_dict(
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
        assert (df.endpoint_id == ep_id).all(), (
            "The endpoint IDs are different than expected"
        )
        assert df[df["endpoint_id"] == ep_id][result_name].item() == result_value, (
            f"The {result_name} is different than expected for {ep_id}"
        )

    @classmethod
    def _check_last_request_dict(
        cls,
        data: dict[str, float],
        ep_id: str,
        result_name: str,
        result_value: datetime,
    ):
        assert data, "No last request data"
        assert list(data.keys())[0] == ep_id, (
            "The endpoint IDs are different than expected"
        )
        assert data[ep_id] == result_value.timestamp(), (
            f"The {result_name} is different than expected for {ep_id}"
        )

    @classmethod
    def _test_predictions_table(cls, ep_id: str, should_be_empty: bool = False) -> None:
        if cls._tsdb_storage.type == mm_constants.TSDBTarget.TimescaleDB:
            table = cls._tsdb_storage._metrics_queries.tables[
                mm_constants.TimescaleDBTables.PREDICTIONS
            ]
            full_query = table._get_records_query(
                start=datetime.min, end=datetime.now().astimezone()
            )
            query_result = cls._tsdb_storage._connection.run(
                query=full_query,
            )
            df_columns = query_result.fields
            predictions_df = pd.DataFrame(query_result.data, columns=df_columns)
        elif cls._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
            predictions_df: pd.DataFrame = cls._tsdb_storage._get_records(
                table=mm_constants.V3IOTSDBTables.PREDICTIONS, start="0", end="now"
            )
        else:
            raise ValueError(f"Unsupported TSDB type: {cls._tsdb_storage.type}")
        if should_be_empty:
            assert predictions_df.empty, "Predictions should be empty"
        else:
            assert not predictions_df.empty, "No TSDB predictions data"
            assert (predictions_df.endpoint_id == ep_id).all(), (
                "The endpoint IDs are different than expected"
            )

    @classmethod
    def _test_v3io_records(
        cls,
        ep_id: str,
        apps_data: list[_AppData],
        last_request: datetime | None = None,
        error_count: float | None = None,
    ) -> None:
        cls._test_tsdb_record(
            ep_id,
            last_request=last_request,
            error_count=error_count,
            apps_data=apps_data,
        )

    @classmethod
    def _test_api_get_metrics(
        cls,
        ep_id: str,
        run_db: mlrun.db.httpdb.HTTPRunDB,
        apps_data: list[_AppData],
        type: typing.Literal["metrics", "results"] = "results",
    ) -> list[str]:
        cls._logger.debug("Checking the metrics", type=type)
        monitoring_metrics = run_db.get_model_endpoint_monitoring_metrics(
            project=cls.project_name, endpoint_id=ep_id, type=type
        )
        get_app_results: set[str] = set()
        app_results_full_names: list[str] = []
        if type == "metrics":
            assert (
                mlrun.model_monitoring.helpers.get_invocations_metric(
                    cls.project_name
                ).dict()
                in monitoring_metrics
            ), "The invocations metric is missing"

        for result in monitoring_metrics:
            get_app_results.add(result.name)
            app_results_full_names.append(result.full_name)

        expected_results = set().union(
            *[getattr(app_data, type) for app_data in apps_data]
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
                assert result_values["data"], (
                    f"No data for result {result_values['full_name']}"
                )
                assert result_values["values"], (
                    f"The values list is empty for result {result_values['full_name']}"
                )

    @classmethod
    def _test_api(cls, ep_id: str, apps_data: list[_AppData]) -> None:
        cls._logger.debug("Checking model endpoint monitoring APIs")
        run_db = mlrun.db.httpdb.HTTPRunDB(mlrun.mlconf.dbpath)
        metrics_full_names = cls._test_api_get_metrics(
            ep_id=ep_id, run_db=run_db, apps_data=apps_data, type="metrics"
        )
        results_full_names = cls._test_api_get_metrics(
            ep_id=ep_id, run_db=run_db, apps_data=apps_data, type="results"
        )

        cls._test_api_get_values(
            ep_id=ep_id,
            results_full_names=metrics_full_names + results_full_names,
            run_db=run_db,
        )


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
class TestMonitoringAppFlow(TestMLRunSystemModelMonitoring, _V3IORecordsChecker):
    project_name = "test-app-flow"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: str | None = None
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

        cls.run_db = mlrun.get_run_db()

    def custom_setup(self) -> None:
        self.set_mm_credentials()
        self._external_stream_delay = 0
        if isinstance(
            self.mm_stream_profile, DatastoreProfileKafkaStream
        ) and self.mm_stream_profile.attributes()["brokers"][0].endswith(
            ".confluent.cloud:9092"
        ):
            # external Confluent Cloud degrades the streams latency
            self._external_stream_delay = 90  # seconds
        super(TestMLRunSystem, self).custom_setup(project_name=self.project_name)

    def custom_teardown(self) -> None:
        # validate that stream resources were deleted as expected
        stream_profile = self.mm_stream_profile

        func_to_validate = [mm_constants.MonitoringFunctionNames.WRITER] + [
            app_data.class_.NAME for app_data in self.apps_data
        ]

        if isinstance(stream_profile, DatastoreProfileV3io):
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

            # validate that the controller stream was deleted
            with pytest.raises(HttpResponseError):
                client.object.get(
                    container="users",
                    path=f"pipelines/{self.project_name}/model-endpoints/{mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER}/serving-state.json",
                )

        elif isinstance(stream_profile, DatastoreProfileKafkaStream):
            kafka_profile_attributes = stream_profile.attributes()
            kafka_consumer_kwargs = mlrun.datastore.utils.KafkaParameters(
                kafka_profile_attributes
            ).consumer()
            consumer = kafka.KafkaConsumer(
                bootstrap_servers=stream_profile.brokers, **kafka_consumer_kwargs
            )
            topics = consumer.topics()

            project_topics_list = [
                f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}"
            ]
            for func in func_to_validate + [
                mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER
            ]:
                project_topics_list.append(
                    f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}_{func}"
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
            wait_for_deployment=True,
        )

    def _set_and_deploy_monitoring_apps(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for app_data in self.apps_data:
                if app_data.deploy:
                    fn = self.project.set_model_monitoring_function(
                        func=app_data.abs_path,
                        application_class=app_data.class_.__name__,
                        name=app_data.class_.NAME,
                        image=mlrun.mlconf.function_defaults.image_by_kind.job
                        if self.image is None
                        else self.image,
                        requirements=app_data.requirements,
                        **app_data.kwargs,
                    )

                    def deploy_function():
                        nonlocal fn
                        fn.deploy()
                        fn._wait_for_function_deployment(db=mlrun.get_run_db())

                    executor.submit(deploy_function)

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
        cls,
        with_training_set: bool,
        with_model_runner: bool = False,
    ) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        if with_model_runner:
            code_path = (
                f"{str((Path(__file__).parent / 'assets').absolute())}/models.py"
            )
            serving_fn = mlrun.code_to_function(
                name="model-serving",
                kind="serving",
                project=cls.project_name,
                filename=code_path,
            )
            model_runner_step = mlrun.serving.ModelRunnerStep(
                name="ModelRunner",
                full_event=True,
            )
            model_runner_step.add_model(
                endpoint_name=f"{cls.model_name}_{with_training_set}",
                model_class="MyModel",
                execution_mechanism="naive",
                model_artifact=f"store://models/{cls.project_name}/{cls.model_name}_{with_training_set}:latest",
                input_path="inputs",
                result_path="outputs",
            )
            graph = serving_fn.set_topology("flow", engine="async")
            graph.to(model_runner_step).respond()
        else:
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
        cls._workaround_ml_12522_force_http_for_nodeport(serving_fn)
        return serving_fn

    @staticmethod
    def _workaround_ml_12522_force_http_for_nodeport(
        fn: mlrun.runtimes.nuclio.serving.ServingRuntime,
    ) -> None:
        """Workaround for ML-12522.

        Nuclio reports ``status.external_invocation_urls`` as scheme-less.
        Since mlrun #9578, ``_resolve_invocation_url`` prepends ``https://``
        to scheme-less external URLs, which breaks plain-HTTP NodePort
        setups (open-source CE / k3s) with ``SSL: WRONG_VERSION_NUMBER``.
        Iguazio setups expose functions via TLS-terminated ingress
        (hostname/path form) and must keep the ``https://`` default.

        Only rewrite the NodePort form (IPv4 ``host:port`` with no path)
        to ``http://`` here. Remove once ML-12522 routes API-gateway
        invocations through ``APIGateway.invoke_url`` and
        ``_resolve_invocation_url`` defaults back to ``http://``.
        """
        for i, url in enumerate(fn.status.external_invocation_urls or []):
            if "://" in url or "/" in url:
                continue
            host = url.split(":", 1)[0]
            try:
                ipaddress.ip_address(host)
            except ValueError:
                continue
            fn.status.external_invocation_urls[i] = f"http://{url}"

    @classmethod
    def _infer(
        cls,
        serving_fn: mlrun.runtimes.nuclio.serving.ServingRuntime,
        *,
        num_events: int,
        with_training_set: bool = True,
        with_model_runner: bool = False,
    ) -> datetime:
        result = serving_fn.invoke(
            path="/"
            if with_model_runner
            else f"v2/models/{cls.model_name}_{with_training_set}/infer",
            body=json.dumps({"inputs": [[0.0] * cls.num_features] * num_events}),
        )
        assert isinstance(result, dict), "Unexpected result type"
        assert "outputs" in result, "Result should have 'outputs' key"
        assert len(result["outputs"]) == num_events, (
            "Outputs length does not match inputs"
        )
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

    def _test_artifacts(self, ep_id: str) -> None:
        for app_data in self.apps_data:
            if app_data.artifacts:
                app_name = app_data.class_.NAME
                self._logger.debug("Checking app artifacts", app_name=app_name)
                for key in app_data.artifacts:
                    self._logger.debug("Checking artifact existence", key=key)
                    artifact = self.project.get_artifact(f"{key}-{ep_id}")
                    self._logger.debug("Checking artifact labels", key=f"{key}-{ep_id}")
                    assert {
                        "mlrun/producer-type": "model-monitoring-app",
                        "mlrun/app-name": app_name,
                        "mlrun/endpoint-id": ep_id,
                    }.items() <= artifact.labels.items()
                    self._logger.debug(
                        "Test the artifact can be fetched from the store",
                        key=f"{key}-{ep_id}",
                    )
                    artifact.to_dataitem().get()

    @classmethod
    def _test_model_endpoint_stats(
        cls, mep: mlrun.common.schemas.ModelEndpoint
    ) -> None:
        cls._logger.debug("Checking model endpoint", ep_id=mep.metadata.uid)
        assert mep.spec.feature_stats.keys() == set(mep.spec.feature_names), (
            "The endpoint's feature stats keys are not the same as the feature names"
        )
        ep_current_stats = mep.status.current_stats

        ep_drift_measures = mep.status.drift_measures

        assert set(ep_current_stats.keys()) == set(mep.spec.feature_stats.keys()), (
            "The endpoint's current stats is different than expected"
        )

        assert ep_drift_measures, "The general drift status is empty"
        assert ep_drift_measures, "The drift measures are empty"

        for measure in ["hellinger_mean", "kld_mean", "tvd_mean"]:
            assert isinstance(ep_drift_measures.pop(measure, None), float), (
                f"Expected '{measure}' in drift measures"
            )

        drift_table = pd.DataFrame.from_dict(ep_drift_measures, orient="index")
        assert set(drift_table.columns) == {
            "hellinger",
            "kld",
            "tvd",
        }, "The drift metrics are not as expected"
        assert set(drift_table.index) == set(mep.spec.feature_names), (
            "The feature names are not as expected"
        )

        assert ep_current_stats["sepal_length_cm"]["count"] == cls.num_events, (
            "Different number of events than expected"
        )

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

    def _get_apps_data(self, with_training_set: bool) -> list[_AppData]:
        apps_data = [
            _AppData(
                class_=DemoMonitoringApp,
                rel_path="assets/application.py",
                results={"data_drift_test", "model_perf"},
            ),
            _AppData(
                class_=ErrApp,
                rel_path="assets/application.py",
            ),
        ]
        if with_training_set:
            # Applications that need training set
            apps_data.extend(
                [
                    _DefaultDataDriftAppData,
                    _AppData(
                        class_=DemoEvidentlyMonitoringApp,
                        rel_path="assets/custom_evidently_app.py",
                        requirements=[f"evidently=={SUPPORTED_EVIDENTLY_VERSION}"],
                        kwargs={
                            "evidently_workspace_path": (
                                f"/v3io/projects/{self.project_name}/artifacts/evidently-workspace"
                            ),
                            "evidently_project_id": str(uuid.uuid4()),
                        },
                        results={"data_drift_test"},
                        artifacts={"evidently_report"},
                    ),
                ]
            )
        return apps_data

    def _test_function_summaries(self) -> None:
        self._logger.debug("Checking function summaries")
        function_summaries = self.project.get_monitoring_function_summaries()
        assert len(function_summaries) == 3 + len(self.apps_data)
        function_summaries = self.project.get_monitoring_function_summaries(
            include_infra=False, start=datetime(2020, 1, 1, tzinfo=UTC)
        )
        assert len(function_summaries) == len(self.apps_data)

        try:
            # Check that Evidently app is in `self.apps_data`
            self.project.get_function(
                key=DemoEvidentlyMonitoringApp.NAME, ignore_cache=True
            )
            self._logger.debug("Checking Evidently function summary")
            evidently_func_summary_list = (
                self.project.get_monitoring_function_summaries(
                    include_infra=False, names=[DemoEvidentlyMonitoringApp.NAME]
                )
            )

            assert len(evidently_func_summary_list) == 1
            evidently_func_summary = evidently_func_summary_list[0]

            assert evidently_func_summary.name == DemoEvidentlyMonitoringApp.NAME
            assert (
                evidently_func_summary.status
                == mlrun.common.schemas.FunctionState.ready
            )
            assert evidently_func_summary.base_period == self.app_interval
            assert not evidently_func_summary.stats

            # now get function summary with stats
            evidently_func_summary_list = (
                self.project.get_monitoring_function_summaries(
                    include_infra=False,
                    names=[DemoEvidentlyMonitoringApp.NAME],
                    include_stats=True,
                )
            )
            evidently_func_summary = evidently_func_summary_list[0]

            evidently_stats = evidently_func_summary.stats

            assert evidently_stats["potential_detection"] == 1
            assert evidently_stats["detected"] == 0

            assert evidently_stats["stream_stats"]
            assert evidently_stats["stream_stats"]["committed"] == 1
            assert evidently_stats["stream_stats"]["lag"] == 0

        except mlrun.errors.MLRunNotFoundError:
            # Evidently app was not deployed
            pass

        if _DefaultDataDriftAppData in self.apps_data:
            # test a specific function summary
            hist_function_summary = self.project.get_monitoring_function_summary(
                name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
                include_latest_metrics=True,
            )
            assert hist_function_summary.stats
            assert len(hist_function_summary.stats["metrics"]) == 4

            first_metric = hist_function_summary.stats["metrics"][0]
            assert first_metric["type"] == "result"
            # verify the expected keys of a result
            assert first_metric.keys() == {
                "kind",
                "result_name",
                "status",
                "time",
                "type",
                "value",
            }, "The result keys are not as expected"

            assert first_metric["result_name"] == "general_drift"
            assert first_metric["value"] == 1

            second_metric = hist_function_summary.stats["metrics"][1]
            assert second_metric["type"] == "metric"
            # verify the expected keys of a metric
            assert second_metric.keys() == {
                "metric_name",
                "time",
                "type",
                "value",
            }, "The metric keys are not as expected"

            assert hist_function_summary.stats["stream_stats"]

            # verify the stream stats
            shards = hist_function_summary.stats["stream_stats"].keys()
            expected_committed = 1
            actual_committed = 0
            for shard in shards:
                actual_committed += hist_function_summary.stats["stream_stats"][shard][
                    "committed"
                ]
                # Verify that the lag is 0
                assert hist_function_summary.stats["stream_stats"][shard]["lag"] == 0
            assert actual_committed == expected_committed, (
                f"Expected {expected_committed} committed events, but got {actual_committed}"
            )

    def _test_drift_over_time(self) -> None:
        self._logger.debug("Checking drift over time")
        end = datetime.now().astimezone() + timedelta(
            hours=1
        )  # add 1 hour because end is rounded to the start of the hour
        drift_over_time: ModelEndpointDriftValues = self.project.get_drift_over_time(
            end=end
        )
        assert drift_over_time is not None
        assert len(drift_over_time.values) == 1, "Drift over time should have one value"
        assert drift_over_time.values[0].count_detected == 1, (
            "Drift over time should have one detected drift"
        )
        assert drift_over_time.values[0].count_suspected == 0, (
            "Drift over time should not have potential drift"
        )
        end = datetime.now().astimezone() - timedelta(hours=1)
        drift_over_time: ModelEndpointDriftValues = self.project.get_drift_over_time(
            end=end
        )
        assert drift_over_time is not None
        assert len(drift_over_time.values) == 0, (
            "No drift over time should be detected in the past"
        )

    @pytest.mark.parametrize(
        "with_training_set, with_model_runner",
        [
            pytest.param(True, True, marks=pytest.mark.smoke),
            pytest.param(True, False),
            pytest.param(False, True),
            pytest.param(False, False),
        ],
    )
    def test_app_flow(self, with_training_set: bool, with_model_runner: bool) -> None:
        self.apps_data = self._get_apps_data(with_training_set)
        self.project = typing.cast(mlrun.projects.MlrunProject, self.project)

        self._log_model(with_training_set)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            infra_future = executor.submit(
                self._submit_controller_and_deploy_writer,
                _DefaultDataDriftAppData in self.apps_data,
            )
            monitoring_apps_future = executor.submit(
                self._set_and_deploy_monitoring_apps
            )
            serving_future = executor.submit(
                self._deploy_model_serving, with_training_set, with_model_runner
            )

        infra_future.result()
        monitoring_apps_future.result()
        serving_fn = serving_future.result()
        self._add_error_alert()

        time.sleep(5)
        last_request = self._infer(
            serving_fn,
            num_events=self.num_events,
            with_training_set=with_training_set,
            with_model_runner=with_model_runner,
        )

        self._infer_with_error(serving_fn, with_training_set=with_training_set)
        # wait for the NO-OP event to close the window
        initial_wait = (
            2 * self.app_interval_seconds
            + mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
            + mlrun.mlconf.model_endpoint_monitoring.writer_graph.flush_after_seconds
            + self._external_stream_delay
        )

        mep_result = {}

        def check_model_endpoint_data() -> None:
            mep = mlrun.db.get_run_db().get_model_endpoint(
                name=f"{self.model_name}_{with_training_set}",
                project=self.project.name,
                function_name="model-serving",
                function_tag="latest",
                feature_analysis=True,
                tsdb_metrics=True,
            )
            # Verify endpoint has required data
            assert mep is not None, "Model endpoint is None"
            assert mep.status.last_request is not None, "last_request is None"

            # Verify TSDB actually has data (not just endpoint metadata)
            df = self._tsdb_storage.get_results_metadata(endpoint_id=mep.metadata.uid)
            assert not df.empty, "TSDB data not yet available"

            # Store for later use (avoids duplicate fetch)
            mep_result["mep"] = mep

        self.wait_for_condition(
            condition_check=check_model_endpoint_data,
            initial_wait=initial_wait,
            condition_description="model endpoint to have monitoring data and TSDB to be populated",
        )

        # Use the endpoint captured during the successful check
        mep = mep_result["mep"]

        # Model predict timestamp is slightly differ than storey timestamp
        assert (mep.status.last_request - last_request) < timedelta(milliseconds=1), (
            "The saved `last_request` in the model endpoint is different than the last result timestamp"
        )

        self._test_v3io_records(
            ep_id=mep.metadata.uid,
            last_request=mep.status.last_request,
            apps_data=self.apps_data,
            error_count=self.error_count,
        )

        self._test_predictions_table(mep.metadata.uid)
        self._test_artifacts(ep_id=mep.metadata.uid)
        self._test_api(ep_id=mep.metadata.uid, apps_data=self.apps_data)
        if _DefaultDataDriftAppData in self.apps_data:
            self._test_model_endpoint_stats(mep=mep)
        self._test_error_alert()
        self._test_function_summaries()
        self._test_drift_over_time()


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestRecordResults(TestMLRunSystemModelMonitoring, _V3IORecordsChecker):
    project_name = "test-mm-record"
    name_prefix = "infer-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: str | None = None

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

    def custom_setup(self) -> None:
        self.set_mm_credentials()
        super(TestMLRunSystem, self).custom_setup(project_name=self.project_name)

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
            image=mlrun.mlconf.function_defaults.image_by_kind.job
            if self.image is None
            else self.image,
            **self.app_data.kwargs,
        )
        self.project.deploy_function(fn)

    def _record_results(self) -> str:
        model_endpoint = mlrun.model_monitoring.api.record_results(
            project=self.project_name,
            model_path=self.project.get_artifact_uri(  # pyright: ignore[reportOptionalMemberAccess]
                key=self.model_name, category="model", tag="latest"
            ),
            model_endpoint_name=f"{self.name_prefix}-test",
            function_name=self.function_name,
            context=mlrun.get_or_create_ctx(name=f"{self.name_prefix}-context"),  # pyright: ignore[reportGeneralTypeIssues]
            infer_results_df=self.infer_results_df,
        )

        return model_endpoint.metadata.uid

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

        endpoint_id = self._record_results()

        # Wait for TSDB data to be processed with retry pattern
        initial_wait = (
            2 * self.app_interval_seconds
            + mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
        )

        def check_tsdb_data() -> None:
            mep = mlrun.db.get_run_db().get_model_endpoint(
                name=f"{self.name_prefix}-test",
                project=self.project.name,
                endpoint_id=endpoint_id,
                feature_analysis=True,
                tsdb_metrics=True,
            )
            self._test_v3io_records(
                mep.metadata.uid,
                apps_data=self.apps_data,
            )
            self._test_predictions_table(mep.metadata.uid, should_be_empty=True)

        self.wait_for_condition(
            condition_check=check_tsdb_data,
            initial_wait=initial_wait,
            condition_description="TSDB data to be available for batch endpoint",
        )


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestServingJobEndpoint(TestMLRunSystemModelMonitoring, _V3IORecordsChecker):
    """
    Demonstrates running a serving job with model monitoring enabled.  In this test, we deploy a simple serving model
    and then validate the newly created batch model endpoint along with its application metrics.
    Also tests the deployment of a monitoring-application from the MLRun hub (count-events)
    """

    project_name = "test-mm-serving-job"
    name_prefix = "infer-monitoring"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: str | None = None

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
        cls.function_name = f"{cls.name_prefix}-function"
        # training
        cls._train()

        # model monitoring app
        cls.app_data = {
            "url": "hub://count_events",
            "class_name": "CountApp",
            "app_name": "count",
            "metric_name": "count",
        }

        # model monitoring infra
        cls.app_interval: int = 1  # every 1 minute
        cls.app_interval_seconds = timedelta(minutes=cls.app_interval).total_seconds()
        cls.apps_data = [cls.app_data]

    def custom_setup(self) -> None:
        self.set_mm_credentials()
        super(TestMLRunSystem, self).custom_setup(project_name=self.project_name)

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

    @staticmethod
    def _generate_input_df() -> pd.DataFrame:
        d = {
            "column_0": {0: 0.1, 1: 1.3, 2: 0.7},
            "column_1": {0: 0.3, 1: -2.2, 2: -2.0},
            "column_2": {0: 0.01, 1: 2.3, 2: 1.59},
            "column_3": {0: -0.38, 1: 1.83, 2: 1.86},
            "column_4": {0: -0.7, 1: 0.8, 2: 1.27},
        }

        df = pd.DataFrame(d)
        input_df = df.set_index(
            pd.date_range("2025-07-24 05:00:10", freq="120s", periods=len(df))
        )
        input_df.reset_index(inplace=True)
        return input_df

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
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = str(Path(temp_dir))
            fn = self.project.set_model_monitoring_function(
                func=self.app_data["url"],
                application_class=self.app_data["class_name"],
                name=self.app_data["app_name"],
                image=mlrun.mlconf.function_defaults.image_by_kind.job
                if self.image is None
                else self.image,
                local_path=temp_path,
            )
            self.project.deploy_function(fn)
            return fn

    def _deploy_monitoring_infra(self) -> None:
        self.project.enable_model_monitoring(  # pyright: ignore[reportOptionalMemberAccess]
            base_period=self.app_interval,
            **({} if self.image is None else {"image": self.image}),
            deploy_histogram_data_drift_app=False,
            wait_for_deployment=True,
        )

    def _run_serving_job(self, input_df: pd.DataFrame) -> mlrun.runtimes.BaseRuntime:
        function = self.project.set_function(
            func=str(self.assets_path / "function_with_model.py"),
            name="test",
            kind="serving",
            image=self.image,
        )
        graph = function.set_topology("flow", engine="async")

        model_runner_step = mlrun.serving.ModelRunnerStep(name="my_model_runner")

        model_runner_step.add_model(
            endpoint_name="my_model",
            model_class="DummyModel",
            execution_mechanism="naive",
            model_endpoint_creation_strategy=mm_constants.ModelEndpointCreationStrategy.OVERWRITE,
        )

        graph.to(model_runner_step)
        function.set_tracking()
        job = function.to_job()

        input_dataset = self.project.log_dataset("input_dataset", input_df)
        inputs = {"data": input_dataset.uri}
        params = {"timestamp_column": "index"}
        self.project.run_function(job, inputs=inputs, params=params)
        return job

    def _test_batch_ep_metrics(
        self, function_name: str, input_df: pd.DataFrame
    ) -> None:
        model_endpoint = mlrun.get_run_db().get_model_endpoint(
            name="my_model",
            project=self.project_name,
            function_name=function_name,
            function_tag="latest",
        )

        assert model_endpoint is not None, "Model endpoint was not created"

        self._test_predictions_table(model_endpoint.metadata.uid, should_be_empty=False)

        assert model_endpoint.status.first_request == input_df[
            "index"
        ].min().to_pydatetime().replace(tzinfo=UTC)
        assert model_endpoint.status.last_request == input_df[
            "index"
        ].max().to_pydatetime().replace(tzinfo=UTC)

        run_db = mlrun.get_run_db()

        monitoring_metrics = run_db.get_model_endpoint_monitoring_metrics(
            project=self.project_name,
            endpoint_id=model_endpoint.metadata.uid,
            type="metrics",
        )

        assert len(monitoring_metrics) == 2
        metric_name = self.app_data["metric_name"]
        metric_names = [
            metric.full_name
            for metric in monitoring_metrics
            if metric_name in metric.full_name
        ]

        self._test_metric_values(
            ep_id=model_endpoint.metadata.uid,
            metrics_full_names=metric_names,
            run_db=run_db,
            start=0,
            end=datetime.now().timestamp() * 1000,
        )

    def _test_metric_values(
        self,
        ep_id: str,
        metrics_full_names: list[str],
        run_db: mlrun.db.httpdb.HTTPRunDB,
        start: float | None,
        end: float | None,
    ) -> None:
        base_query = f"?name={'&name='.join(metrics_full_names)}"
        query = f"{base_query}&start={start}&end={end}"

        response = run_db.api_call(
            method=mlrun.common.types.HTTPMethod.GET,
            path=f"projects/{self.project_name}/model-endpoints/{ep_id}/metrics-values{query}",
        )
        response_content = json.loads(response.content.decode())
        for metric_values in response_content:
            assert metric_values["data"], (
                f"No data for metric {metric_values['full_name']}"
            )
            assert metric_values["values"], (
                f"The values list is empty for metric {metric_values['full_name']}"
            )
            assert len(metric_values["values"]) == 3

        first_metric = response_content[0]
        assert first_metric["full_name"] in metrics_full_names

    def test_serving_as_a_job(self) -> None:
        self._log_model()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            monitoring_app = executor.submit(self._deploy_monitoring_app)
            executor.submit(self._deploy_monitoring_infra)

        fn = monitoring_app.result()
        self._assert_replicas(fn)

        input_df = self._generate_input_df()
        function = self._run_serving_job(input_df=input_df)
        initial_wait = (
            mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs + 20
        )

        def check_batch_metrics() -> None:
            self._test_batch_ep_metrics(
                function_name=function.metadata.name, input_df=input_df
            )

        self.wait_for_condition(
            condition_check=check_batch_metrics,
            initial_wait=initial_wait,
            condition_description="batch job metrics to be available (invocations + count)",
        )

    @staticmethod
    def _assert_replicas(fn):
        """
        Validate that the 'min_replicas' and 'max_replicas' values in the function's spec are correct after deployment.
        This check ensures that the replica settings, which are modified on the server side during deployment, are
        properly reflected on the client side.
        """
        assert fn.spec.min_replicas == 1
        assert fn.spec.max_replicas == 1


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelMonitoringInitialize(TestMLRunSystemModelMonitoring):
    """Test model monitoring infrastructure initialization and cleanup, including the usage of
    disable the model monitoring and delete a specific model monitoring application."""

    project_name = "test-mm-initialize"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: str | None = None

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
                image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job
            )
        self.set_mm_credentials()
        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            wait_for_deployment=True,
        )

        with pytest.raises(mlrun.errors.MLRunConflictError):
            self.project.enable_model_monitoring(
                image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            )

        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            ignore_cache=True,
        )
        assert (
            controller.spec.config["spec.triggers.cron_interval"]["attributes"][
                "interval"
            ]
            == "3m"
        )
        # check that all the function are still deployed
        for name in all_functions:
            func = self.project.get_function(
                key=name,
                ignore_cache=True,
            )
            func._get_db().get_nuclio_deploy_status(func, verbose=False)
            assert func.status.state == "ready"

        self.project.update_model_monitoring_controller(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            base_period=1,
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
            == "1m"
        )

        self.project.disable_model_monitoring(delete_histogram_data_drift_app=False)

        stream_profile = self.mm_stream_profile
        if isinstance(stream_profile, DatastoreProfileV3io):
            v3io_client = v3io.dataplane.Client(endpoint=mlrun.mlconf.v3io_api)

            # controller and writer(with has stream) should be deleted
            for name in mm_constants.MonitoringFunctionNames.list():
                container, stream_path = self.get_stream_path(name)
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
            container, stream_path = self.get_stream_path(
                mm_constants.HistogramDataDriftApplicationConstants.NAME
            )
            v3io_client.stream.describe(container, stream_path)
            self._deploy_demo_app()

            # check that the stream of the histogram data drift app is deleted
            self._delete_histogram_app()

            # check that only the demo app is remaining
            monitoring_functions = self.project.list_model_monitoring_functions(
                tag="latest"
            )
            assert len(monitoring_functions) == 1, (
                "expected a single monitoring function after deletion of histogram app"
            )
            assert [fn.metadata.name for fn in monitoring_functions] == [
                DemoMonitoringApp.NAME
            ], "the remaining function should be the demo app"

            with pytest.raises(v3io.dataplane.response.HttpResponseError):
                v3io_client.stream.describe(container, stream_path)

        elif isinstance(stream_profile, DatastoreProfileKafkaStream):
            consumer = kafka.KafkaConsumer(bootstrap_servers=stream_profile.brokers)
            topics = consumer.topics()

            # Verify that controller resources were deleted
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                self.project.get_function(
                    key=mm_constants.MonitoringFunctionNames.WRITER,
                    ignore_cache=True,
                )
            assert (
                f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}_{mm_constants.MonitoringFunctionNames.WRITER}"
                not in topics
            )

            # Verify that controller resources were deleted
            with pytest.raises(mlrun.errors.MLRunNotFoundError):
                self.project.get_function(
                    key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
                    ignore_cache=True,
                )
            assert (
                f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}_{mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER}_v1"
                not in topics
            )

            # Verify that monitoring stream resources were not deleted
            self.project.get_function(
                key=mm_constants.MonitoringFunctionNames.STREAM,
                ignore_cache=True,
            )

            assert (
                f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}_v1"
                in topics
            )

            self._disable_stream_function()

            # check that the topic of the stream resource is not deleted
            consumer = kafka.KafkaConsumer(bootstrap_servers=stream_profile.brokers)
            topics = consumer.topics()
            assert (
                f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}_v1"
                in topics
            )

            self._delete_histogram_app()

            # check that the topic of the histogram data drift app is deleted
            consumer = kafka.KafkaConsumer(bootstrap_servers=stream_profile.brokers)
            topics = consumer.topics()
            assert (
                f"monitoring_stream_{mlrun.mlconf.system_id}_{self.project_name}_{mm_constants.HistogramDataDriftApplicationConstants.NAME}_v1"
                not in topics
            )

    def _deploy_demo_app(self):
        demo_app = _AppData(
            class_=DemoMonitoringApp,
            rel_path="assets/application.py",
            results={"data_drift_test", "model_perf"},
        )
        fn = self.project.set_model_monitoring_function(
            func=demo_app.abs_path,
            application_class=demo_app.class_.__name__,
            name=demo_app.class_.NAME,
            image=mlrun.mlconf.function_defaults.image_by_kind.job
            if self.image is None
            else self.image,
            requirements=demo_app.requirements,
            **demo_app.kwargs,
        )
        fn.deploy()

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


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestUpdateControllerPreservesAuthToken(TestMLRunSystemModelMonitoring):
    """ML-12021: Verify that update_model_monitoring_controller preserves
    the auth token that was set during enable_model_monitoring."""

    project_name = "test-mm-auth-token"
    image: str | None = None

    @pytest.mark.timeout(600)
    def test_auth_token_preserved_after_controller_update(self) -> None:
        self.set_mm_credentials()

        # Clean up any leftover monitoring from a previous run
        try:
            self.project.disable_model_monitoring()
        except Exception:
            pass

        token_name = "test-auth-token"
        with mlrun.RuntimeConfigurationContext(auth_token_name=token_name):
            self.project.enable_model_monitoring(
                image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
                deploy_histogram_data_drift_app=False,
                wait_for_deployment=True,
            )

        # Verify the controller has the auth token after initial deploy
        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            ignore_cache=True,
        )
        assert controller.spec.auth.get("token_name") == token_name

        # Update controller (no RuntimeConfigurationContext active)
        self.project.update_model_monitoring_controller(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            base_period=1,
            wait_for_deployment=True,
        )

        # Verify the auth token is still preserved after update
        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            ignore_cache=True,
        )
        assert controller.spec.auth.get("token_name") == token_name


@TestMLRunSystemModelMonitoring.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestMonitoredServings(TestMLRunSystemModelMonitoring):
    project_name = "test-mm-serving"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: str | None = None

    @classmethod
    def custom_setup_class(cls) -> None:
        random_rgb_image_list = (
            np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
            .reshape(-1, 3)
            .tolist()
        )
        cls.model_by_endpoint_name = {
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
                "schema": ["feature0", "feature1", "feature2", "override_label"],
                "training_set": pd.DataFrame(
                    data={
                        "feature0": [1, 2],
                        "feature1": [1, 2],
                        "feature2": [1, 2],
                        "label": [1, 1],
                    }
                ),
                "label_column": "label",
            },
        }

    def custom_setup(self) -> None:
        self.set_mm_credentials()

    def _log_model(
        self,
        model_name: str,
        training_set: pd.DataFrame = None,
        label_column: typing.Union[str, list[str]] | None = None,
    ) -> None:
        self.project.log_model(
            model_name,
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
            training_set=training_set,
            label_column=label_column,
        )

    def _log_iris_model(self) -> tuple[set[str], set[str]]:
        dataset = load_iris()
        train_set = pd.DataFrame(
            dataset.data,
            columns=dataset.feature_names,
        )
        inputs = {
            mlrun.feature_store.api.norm_column_name(feature)
            for feature in dataset.feature_names
        }

        self.project.log_model(
            "classification",
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
            training_set=train_set,
        )
        outputs = {"p0"}

        return inputs, outputs

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
        for endpoint_name, model_dict in self.model_by_endpoint_name.items():
            model_name = model_dict["model_name"]
            self._log_model(
                model_name=model_name,
                training_set=model_dict.get("training_set"),
                label_column=model_dict.get("label_column"),
            )
            serving_fn.add_model(
                endpoint_name,
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
            outputs=kwargs.get("outputs"),
        )
        serving_fn.set_tracking(enable_tracking=enable_tracking)
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image

        serving_fn.deploy()
        return typing.cast(mlrun.runtimes.nuclio.serving.ServingRuntime, serving_fn)

    def _test_endpoint(
        self, endpoint_name, feature_set_uri, model_dict
    ) -> dict[str, typing.Any]:
        serving_fn = self.project.get_function(self.function_name)
        self._infer_by_endpoint(endpoint_name, model_dict, serving_fn)

        initial_wait = (
            mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs + 20
        )

        result = {}

        def check_parquet_data() -> None:
            nonlocal result
            result = self._test_parquet(feature_set_uri, model_dict)

        self.wait_for_condition(
            condition_check=check_parquet_data,
            initial_wait=initial_wait,
            condition_description=f"parquet data for endpoint {endpoint_name}",
        )

        return result

    @staticmethod
    def _infer_by_endpoint(endpoint_name, model_dict, serving_fn):
        data_point = model_dict.get("data_point")
        if endpoint_name == "img_one_to_one":
            data_point = [data_point]
        serving_fn.invoke(
            f"v2/models/{endpoint_name}/infer",
            json.dumps(
                {"inputs": data_point},
            ),
        )
        if endpoint_name == "img_one_to_one":
            data_point = data_point[0]
        serving_fn.invoke(
            f"v2/models/{endpoint_name}/infer",
            json.dumps({"inputs": [data_point, data_point]}),
        )

    @staticmethod
    def _test_parquet(feature_set_uri, model_dict):
        offline_response_df = ParquetTarget(
            name="temp",
            path=fstore.get_feature_set(feature_set_uri).spec.targets[0].path,
        ).as_df()
        is_schema_saved = set(model_dict.get("schema")).issubset(
            offline_response_df.columns
        )
        has_all_the_events = offline_response_df.shape[0] == 3
        return {
            "is_schema_saved": is_schema_saved,
            "has_all_the_events": has_all_the_events,
            "df": offline_response_df,
        }

    def test_different_kind_of_serving(self) -> None:
        self.function_name = "serving-router"
        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            base_period=1,
            deploy_histogram_data_drift_app=False,
        )
        self._deploy_model_router(self.function_name)

        endpoints_list = mlrun.db.get_run_db().list_model_endpoints(
            project=self.project_name, tsdb_metrics=True
        )
        endpoints = endpoints_list.endpoints
        assert len(endpoints) == 7
        serving_fn = self.project.get_function(self.function_name)
        for endpoint in endpoints:
            self._infer_by_endpoint(
                endpoint.metadata.name,
                self.model_by_endpoint_name[endpoint.metadata.name],
                serving_fn,
            )

        initial_wait = (
            mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs + 20
        )

        def check_all_endpoints_parquet() -> None:
            for endpoint in endpoints:
                res_dict = self._test_parquet(
                    endpoint.spec.monitoring_feature_set_uri,
                    self.model_by_endpoint_name[endpoint.metadata.name],
                )
                assert res_dict["is_schema_saved"], (
                    f"For {endpoint.metadata.name} the schema of parquet is missing columns"
                )

        self.wait_for_condition(
            condition_check=check_all_endpoints_parquet,
            initial_wait=initial_wait,
            condition_description="parquet data for all 7 endpoints",
        )

        for endpoint in endpoints:
            res_dict = self._test_parquet(
                endpoint.spec.monitoring_feature_set_uri,
                self.model_by_endpoint_name[endpoint.metadata.name],
            )
            assert res_dict["is_schema_saved"], (
                f"For {endpoint.metadata.name} the schema of parquet is missing columns"
            )

            assert res_dict["has_all_the_events"], (
                f"For {endpoint.metadata.name} Not all the events were saved"
            )

    def test_tracking(self) -> None:
        self.function_name = "serving-1"
        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            base_period=1,
            deploy_histogram_data_drift_app=False,
        )
        kwargs = {"outputs": ["override_label"]}
        for model_name, model_dict in self.test_models_tracking.items():
            self._log_model(
                model_name,
                training_set=model_dict.get("training_set"),
                label_column=model_dict.get("label_column"),
            )
            self._deploy_model_serving(**model_dict, enable_tracking=False)

        endpoints_list = mlrun.db.get_run_db().list_model_endpoints(
            project=self.project_name, tsdb_metrics=True
        )
        endpoints = endpoints_list.endpoints
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        assert (
            endpoint.status.monitoring_mode
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.disabled
        )

        for model_name, model_dict in self.test_models_tracking.items():
            self._deploy_model_serving(**model_dict, enable_tracking=True, **kwargs)

        endpoints_list = mlrun.db.get_run_db().list_model_endpoints(
            project=self.project_name
        )
        endpoints = endpoints_list.endpoints
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        assert (
            endpoint.status.monitoring_mode
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled
        )

        res_dict = self._test_endpoint(
            endpoint_name=endpoint.metadata.name,
            feature_set_uri=endpoint.spec.monitoring_feature_set_uri,
            model_dict=self.test_models_tracking[endpoint.metadata.name],
        )
        assert res_dict["is_schema_saved"], (
            f"For {endpoint.metadata.name} the schema of parquet is missing columns"
        )

        assert res_dict["has_all_the_events"], (
            f"For {endpoint.metadata.name} Not all the events were saved"
        )

        for model_name, model_dict in self.test_models_tracking.items():
            self._deploy_model_serving(**model_dict, enable_tracking=False)

        endpoints_list = mlrun.db.get_run_db().list_model_endpoints(
            project=self.project_name, tsdb_metrics=True
        )
        endpoints = endpoints_list.endpoints
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        assert (
            endpoint.status.monitoring_mode
            == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.disabled
        )

        res_dict = self._test_endpoint(
            endpoint_name=endpoint.metadata.name,
            feature_set_uri=endpoint.spec.monitoring_feature_set_uri,
            model_dict=self.test_models_tracking[endpoint.metadata.name],
        )

        assert res_dict["has_all_the_events"], (
            f"For {res_dict['model_name']}, Despite tracking being disabled, there is new data in the parquet."
        )

    def test_enable_model_monitoring_after_failure(self) -> None:
        self.function_name = "test-function"

        # non-exstent-image, should fail
        with pytest.raises(
            mlrun.runtimes.utils.RunError,
            match="Function .* deployment failed",
        ):
            self.project.enable_model_monitoring(
                image="nonexistent-image:1.0.0",
                wait_for_deployment=True,
            )

        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            wait_for_deployment=True,
        )

        # double enable should fail
        with pytest.raises(
            mlrun.errors.MLRunConflictError,
            match="The following model-montioring infrastructure functions are already deployed, aborting: ",
        ):
            self.project.enable_model_monitoring(
                image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
                wait_for_deployment=True,
            )

        # disable + enable should succeed
        self.project.disable_model_monitoring()
        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            wait_for_deployment=True,
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

    def test_monitored_model_runner_with_labels(self):
        self.function_name = "model-runner-function"
        self.project.enable_model_monitoring(
            image=self.image or mlrun.mlconf.function_defaults.image_by_kind.job,
            base_period=1,
            deploy_histogram_data_drift_app=True,
        )
        self._log_iris_model()

        function = self.project.set_function(
            func=str(self.assets_path / "models.py"),
            name=self.function_name,
            kind="serving",
            image=self.image,
        )
        graph = function.set_topology("flow", engine="async")

        model_runner_step = mlrun.serving.ModelRunnerStep(name="my_model_runner")

        model_runner_step.add_model(
            endpoint_name="my_model",
            model_class="MyModel",
            execution_mechanism="naive",
            model_artifact=f"store://models/{self.project_name}/classification:latest",
            input_path="inputs",
            result_path="outputs",
        )
        graph.to(model_runner_step)
        function.set_tracking()
        function.deploy()
        serving_fn = self.project.get_function(self.function_name)
        serving_fn.invoke("/", body=json.dumps({"inputs": [[0, 0, 0, 0]]}))
        time.sleep(1)
        serving_fn.invoke(
            "/", body=json.dumps({"inputs": [[1, 1, 1, 1]], "labels": {"user": "test"}})
        )

        initial_wait = (
            mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs + 60
        )

        def check_parquet_with_labels() -> None:
            endpoints_list = (
                mlrun.db.get_run_db()
                .list_model_endpoints(project=self.project_name, tsdb_metrics=True)
                .endpoints
            )
            feature_set_uri = endpoints_list[0].spec.monitoring_feature_set_uri
            offline_response_df = ParquetTarget(
                name="temp",
                path=fstore.get_feature_set(feature_set_uri).spec.targets[0].path,
            ).as_df()
            assert len(offline_response_df) == 2, "Not all the events were saved"
            assert offline_response_df["labels"].iloc[1] == {"user": "test"}, (
                "Labels were not saved correctly"
            )

        self.wait_for_condition(
            condition_check=check_parquet_with_labels,
            initial_wait=initial_wait,
            condition_description="parquet data with labels to be saved",
        )


class TestAppJob(TestMLRunSystem):
    """
    Test the histogram data drift application as a job.
    This is performed via the `evaluate` method of the application.
    Note: the local test can probably be moved to the integration tests.
    """

    project_name = "mm-app-as-job"
    image: str | None = None

    @pytest.mark.parametrize("run_local", [False, True])
    def test_histogram_app(self, run_local: bool) -> None:
        # Prepare the data
        sample_data = pd.DataFrame({"a": [9, 10, -2, 1], "b": [0.11, 2.03, 0.55, 0]})
        reference_data = pd.DataFrame({"a": [12, 13], "b": [3.12, 4.12]})
        reference_data_uri = self.project.log_dataset(
            "reference_data", reference_data
        ).uri

        # Call `.evaluate(...)`
        run_result = histogram_data_drift.HistogramDataDriftApplication.evaluate(
            func_path=histogram_data_drift.__file__,
            sample_data=sample_data,
            reference_data=reference_data_uri,
            run_local=run_local,
            image=self.image,  # Relevant for remote runs only
            class_arguments={
                # Produce artifacts for testing
                "produce_json_artifact": True,
                "produce_plotly_artifact": True,
            },
        )

        # Test the state
        assert run_result.state() == "completed", (
            "The job did not complete successfully"
        )
        # Test the inputs
        assert run_result.spec.inputs.keys() == {
            "sample_data",
            "reference_data",
        }, "The run inputs are different than the passed ones"
        # Test the results
        returned_results = run_result.output("return")
        assert returned_results, "No returned results"
        assert [
            {"metric_name": "hellinger_mean", "metric_value": 1.0},
            # Ignore KLD due to varying numerical accuracy on different systems
            # {"metric_name": "kld_mean", "metric_value": 8.517193191416238},
            {"metric_name": "tvd_mean", "metric_value": 0.5},
            {
                "result_name": "general_drift",
                "result_value": 0.75,
                "result_kind": 0,
                "result_status": 2,
                "result_extra_data": "{}",
            },
        ] == [returned_results[0]] + returned_results[2:4], (
            "The returned metrics are different than the expected ones"
        )
        # Test the artifacts
        for artifact_name in {"features_drift_results", "drift_table_plot"}:
            assert run_result.output(artifact_name), (
                f"The artifact '{artifact_name}' is not listed in the run's output"
            )
            # The artifact is logged with the run's name
            artifact_key = f"{run_result.metadata.name}_{artifact_name}"
            artifact = self.project.get_artifact(artifact_key)
            artifact.to_dataitem().get()


class TestAppJobModelEndpointData(TestMLRunSystemModelMonitoring):
    """
    Test getting the model endpoint data in a simple count application.
    This is performed via the ``evaluate`` method of the application, with ``base_period``.
    """

    project_name = "mm-job-mep-data"
    image: str | None = None
    _serving_function_name = "model-server"
    _model_name = "classifier-0"

    def _set_infra(self) -> None:
        self.project.enable_model_monitoring(
            **({} if self.image is None else {"image": self.image}),
            wait_for_deployment=True,
            deploy_histogram_data_drift_app=False,
        )

    def _log_model(self) -> str:
        return self.project.log_model(
            "classifier",
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
        ).uri

    def _deploy_model_serving(self) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        model_uri = self._log_model()
        serving_fn = typing.cast(
            mlrun.runtimes.nuclio.serving.ServingRuntime,
            self.project.set_function(
                "hub://v2_model_server", name=self._serving_function_name
            ),
        )
        serving_fn.add_model(self._model_name, model_path=model_uri)
        serving_fn.set_tracking()
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image

        serving_fn.deploy()
        return serving_fn

    def _setup_resources(self) -> None:
        self.set_mm_credentials()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self._set_infra)
            executor.submit(self._deploy_model_serving)

    def custom_teardown(self) -> None:
        delete_model_monitoring_schedules_user_folder(self.project_name)
        return super().custom_teardown()

    @pytest.mark.parametrize("run_local", [False, True])
    @pytest.mark.parametrize("write_output", [True])
    def test_count_app(self, run_local: bool, write_output: bool) -> None:
        # Set up the serving function with a model endpoint, and the necessary infrastructure
        self._setup_resources()

        # Invoke the serving function with some data
        serving_fn = typing.cast(
            mlrun.runtimes.nuclio.serving.ServingRuntime,
            self.project.get_function(self._serving_function_name),
        )
        serving_fn.invoke(
            f"v2/models/{self._model_name}/infer",
            body=json.dumps({"inputs": [[0, 0, 0, 0]] * 14}),
        )

        time.sleep(65)

        # second window
        serving_fn.invoke(
            f"v2/models/{self._model_name}/infer",
            body=json.dumps({"inputs": [[0, 1, 0, 0]] * 3}),
        )
        serving_fn.invoke(
            f"v2/models/{self._model_name}/infer",
            body=json.dumps({"inputs": [[0, 1, 0, 4.4]]}),
        )

        initial_wait = 80
        endpoint_result = {}

        def check_model_endpoint_ready() -> None:
            endpoint = mlrun.get_run_db().get_model_endpoint(
                name=self._model_name,
                project=self.project_name,
                function_name=self._serving_function_name,
                function_tag="latest",
            )
            # Verify endpoint has request timestamps from both windows
            assert endpoint.status.first_request is not None, "first_request is None"
            assert endpoint.status.last_request is not None, "last_request is None"

            # Store for later use (avoids duplicate fetch)
            endpoint_result["endpoint"] = endpoint

        self.wait_for_condition(
            condition_check=check_model_endpoint_ready,
            initial_wait=initial_wait,
            condition_description="model endpoint to have request data from both windows",
        )

        # Use the endpoint captured during the successful check
        model_endpoint = endpoint_result["endpoint"]

        # Call `.evaluate(...)` with a base period of 1 minute

        # To include the first request, make a small offset
        start = model_endpoint.status.first_request - timedelta(microseconds=1)

        end = model_endpoint.status.last_request
        # Make sure `end - start` is a multiple of the `base_period` in `evaluate`
        end = start + timedelta(minutes=(end - start).total_seconds() // 60 + 1)

        endpoints_params = [
            [(model_endpoint.metadata.name, model_endpoint.metadata.uid)],
            [model_endpoint.metadata.name],
            "all",
        ]

        for i, endpoints in enumerate(endpoints_params):
            # Do not write except the first and last time
            last = i == len(endpoints_params) - 1
            write_output_this_time = write_output if (i == 0 or last) else False

            run_result = CountApp.evaluate(
                func_path=str(Path(__file__).parent / "assets/application.py"),
                func_name="count-app-batch",
                endpoints=endpoints,
                start=start,
                end=end,
                run_local=run_local,
                image=self.image,
                base_period=1,
                write_output=write_output_this_time,
                stream_profile=(
                    self.mm_stream_profile
                    if run_local and write_output_this_time
                    else None
                ),
                existing_data_handling=ExistingDataHandling.delete_all
                if last
                else ExistingDataHandling.fail_on_overlap,
            )

            # Test the state
            assert run_result.state() == "completed", (
                "The job did not complete successfully"
            )

            # Test the passed base period
            assert run_result.spec.parameters["base_period"] == 1, (
                "The base period is different than the passed one"
            )

            # Test the results
            outputs = run_result.outputs
            assert outputs, "No returned results"
            assert len(outputs) == 2, (
                "The number of outputs is different than the number of windows"
            )
            assert list(outputs.values()) == [
                {
                    "result_name": "count",
                    "result_value": 14.0,
                    "result_kind": 2,
                    "result_status": 0,
                    "result_extra_data": "{}",
                },
                {
                    "result_name": "count",
                    "result_value": 4.0,
                    "result_kind": 2,
                    "result_status": 0,
                    "result_extra_data": "{}",
                },
            ], "The outputs are different than expected"

            if write_output:
                # Test that the outputs were written in the database
                db = typing.cast(mlrun.db.httpdb.HTTPRunDB, mlrun.get_run_db())

                expected_metrics = [
                    ModelEndpointMonitoringMetric(
                        project=self.project_name,
                        app="count-app-batch",
                        type="result",
                        name="count",
                        full_name=f"{self.project_name}.count-app-batch.result.count",
                        kind=ResultKindApp.model_performance,
                    ),
                    ModelEndpointMonitoringMetric(
                        project=self.project_name,
                        app="mlrun-infra",
                        type="metric",
                        name="invocations",
                        full_name=f"{self.project_name}.mlrun-infra.metric.invocations",
                    ),
                ]

                def check_metrics_written() -> None:
                    metrics = db.get_model_endpoint_monitoring_metrics(
                        project=self.project_name,
                        endpoint_id=model_endpoint.metadata.uid,
                    )
                    assert metrics == expected_metrics, (
                        "The metrics from the database are different than expected"
                    )

                self.wait_for_condition(
                    condition_check=check_metrics_written,
                    initial_wait=5,
                    retry_interval=2.0,  # Faster retry for quick database writes
                    condition_description="metrics to be written to database by writer",
                )


class TestBatchServingWithSampling(TestMLRunSystemModelMonitoring):
    """
    Test that the model monitoring infrastructure can handle batch serving with sampling percentage.
    In this test, two serving functions are deployed, one with a pre-defined sampling percentage and one without.
    After invoking the serving functions, the predictions table is checked for both the effective sample count and the
    estimated prediction count.
    """

    project_name = "mm-sampling"
    image: str | None = None
    _serving_function_name_with_sample = "model-server-v1"
    _serving_function_name_without_sample = "model-server-v2"
    _model_name = "classifier-0"

    def _set_infra(self) -> None:
        self.project.enable_model_monitoring(
            **({} if self.image is None else {"image": self.image}),
            wait_for_deployment=True,
            deploy_histogram_data_drift_app=False,
        )

    def _log_model(self) -> str:
        return self.project.log_model(
            "classifier",
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
        ).uri

    def _deploy_model_serving(
        self,
        model_uri: str,
        sampling_percentage: float | None = None,
        with_model_runner: bool | None = False,
    ) -> mlrun.runtimes.nuclio.serving.ServingRuntime:
        if with_model_runner:
            code_path = (
                f"{str((Path(__file__).parent / 'assets').absolute())}/models.py"
            )
            serving_fn = mlrun.code_to_function(
                name=self._serving_function_name_with_sample
                if sampling_percentage
                else self._serving_function_name_without_sample,
                kind="serving",
                project=self.project_name,
                filename=code_path,
            )
            model_runner_step = mlrun.serving.ModelRunnerStep(
                name="ModelRunner",
                full_event=True,
            )
            model_runner_step.add_model(
                endpoint_name=self._model_name,
                model_class="MyModel",
                execution_mechanism="naive",
                model_artifact=model_uri,
                input_path="inputs",
                result_path="outputs",
            )
            graph = serving_fn.set_topology("flow", engine="async")
            graph.to(model_runner_step).respond()
        else:
            serving_fn = typing.cast(
                mlrun.runtimes.nuclio.serving.ServingRuntime,
                self.project.set_function(
                    "hub://v2_model_server",
                    name=self._serving_function_name_with_sample
                    if sampling_percentage
                    else self._serving_function_name_without_sample,
                ),
            )
            serving_fn.add_model(self._model_name, model_path=model_uri)
        if sampling_percentage:
            serving_fn.set_tracking(sampling_percentage=sampling_percentage)
        else:
            serving_fn.set_tracking()
        if self.image is not None:
            serving_fn.spec.image = serving_fn.spec.build.image = self.image

        serving_fn.deploy()
        return serving_fn

    def _setup_resources(self, with_model_runner: bool | None = False) -> None:
        self.set_mm_credentials()
        model_uri = self._log_model()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(
                self._deploy_model_serving, model_uri, 15.5, with_model_runner
            )  # with sampling
            executor.submit(
                self._deploy_model_serving, model_uri, None, with_model_runner
            )  # without sampling
            executor.submit(self._set_infra)
        self._tsdb_storage = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project_name, profile=self.mm_tsdb_profile
        )

    @pytest.mark.parametrize("with_model_runner", [False, True])
    def test_serving(self, with_model_runner: bool) -> None:
        # Set up the serving function with a model endpoint, and the necessary infrastructure
        self._setup_resources(with_model_runner)

        # Send 10 requests to the serving functions, with each request containing 100 data points
        serving_fn_v1 = typing.cast(
            mlrun.runtimes.nuclio.serving.ServingRuntime,
            self.project.get_function(self._serving_function_name_with_sample),
        )

        serving_fn_v2 = typing.cast(
            mlrun.runtimes.nuclio.serving.ServingRuntime,
            self.project.get_function(self._serving_function_name_without_sample),
        )

        for i in range(10):
            serving_fn_v1.invoke(
                path="/"
                if with_model_runner
                else f"v2/models/{self._model_name}/infer",
                body=json.dumps({"inputs": [[0, 0, 0, 0]] * 100}),
            )
            serving_fn_v2.invoke(
                path="/"
                if with_model_runner
                else f"v2/models/{self._model_name}/infer",
                body=json.dumps({"inputs": [[0, 0, 0, 0]] * 100}),
            )

        # Wait for model endpoints to have sampling data
        endpoints = {}

        def check_endpoints_with_sampling() -> None:
            ep_with = mlrun.get_run_db().get_model_endpoint(
                name=self._model_name,
                project=self.project_name,
                function_name=self._serving_function_name_with_sample,
                function_tag="latest",
            )
            ep_without = mlrun.get_run_db().get_model_endpoint(
                name=self._model_name,
                project=self.project_name,
                function_name=self._serving_function_name_without_sample,
                function_tag="latest",
            )
            # Check if both endpoints have the expected sampling percentages
            assert ep_with.status.sampling_percentage == 15.5
            assert ep_without.status.sampling_percentage == 100

            # Verify TSDB actually has predictions data (not just endpoint metadata)
            if self._tsdb_storage.type == mm_constants.TSDBTarget.TimescaleDB:
                table = self._tsdb_storage._metrics_queries.tables[
                    mm_constants.TimescaleDBTables.PREDICTIONS
                ]
                full_query = table._get_records_query(
                    start=datetime.min, end=datetime.now().astimezone()
                )
                query_result = self._tsdb_storage._connection.run(query=full_query)
                df_columns = query_result.fields
                predictions_df = pd.DataFrame(query_result.data, columns=df_columns)
            elif self._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
                predictions_df = self._tsdb_storage._get_records(
                    table=mm_constants.V3IOTSDBTables.PREDICTIONS,
                    start="0",
                    end="now",
                )
            else:
                raise ValueError(f"Unsupported TSDB type: {self._tsdb_storage.type}")
            assert predictions_df.shape[0] == 20, (
                "TSDB predictions data not yet available"
            )

            # Store for later use (avoids duplicate fetch)
            endpoints["with_sample"] = ep_with
            endpoints["without_sample"] = ep_without

        self.wait_for_condition(
            condition_check=check_endpoints_with_sampling,
            initial_wait=30,
            condition_description="model endpoints to have sampling data and TSDB predictions",
        )

        # Use the endpoints captured during the successful check
        model_endpoint_with_sample = endpoints["with_sample"]
        model_endpoint_without_sample = endpoints["without_sample"]

        self._test_predictions_table(
            ep_id_with_sample=model_endpoint_with_sample.metadata.uid,
            ep_id_without_sample=model_endpoint_without_sample.metadata.uid,
        )

    def _test_predictions_table(
        self, ep_id_with_sample: str, ep_id_without_sample: str
    ) -> None:
        if self._tsdb_storage.type == mm_constants.TSDBTarget.TimescaleDB:
            table = self._tsdb_storage._metrics_queries.tables[
                mm_constants.TimescaleDBTables.PREDICTIONS
            ]
            full_query = table._get_records_query(
                start=datetime.min, end=datetime.now().astimezone()
            )
            query_result = self._tsdb_storage._connection.run(
                query=full_query,
            )
            df_columns = query_result.fields
            predictions_df = pd.DataFrame(query_result.data, columns=df_columns)
        elif self._tsdb_storage.type == mm_constants.TSDBTarget.V3IO_TSDB:
            predictions_df: pd.DataFrame = self._tsdb_storage._get_records(
                table=mm_constants.V3IOTSDBTables.PREDICTIONS, start="0", end="now"
            )
        else:
            raise ValueError(f"Unsupported TSDB type: {self._tsdb_storage.type}")

        assert "effective_sample_count" in predictions_df.columns
        assert "estimated_prediction_count" in predictions_df.columns
        assert predictions_df.shape[0] == 20

        predictions_df_with_sample = predictions_df[
            predictions_df["endpoint_id"] == ep_id_with_sample
        ]
        predictions_df_without_sample = predictions_df[
            predictions_df["endpoint_id"] == ep_id_without_sample
        ]

        # Validate that the model endpoint without sampling includes all the data points
        assert predictions_df_without_sample["effective_sample_count"].sum() == 1000
        assert predictions_df_without_sample["estimated_prediction_count"].sum() == 1000
        # As for the model endpoint with sampling, the effective sample count should be around 155
        # corresponding to the 15.5% sampling. We will validate that it is not equal to 1000.
        assert predictions_df_with_sample["effective_sample_count"].sum() != 1000
