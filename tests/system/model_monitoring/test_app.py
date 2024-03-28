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

import concurrent
import concurrent.futures
import json
import pickle
import time
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
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
import mlrun.feature_store
import mlrun.feature_store as fstore
import mlrun.model_monitoring.api
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring import TrackingPolicy
from mlrun.model_monitoring.application import ModelMonitoringApplicationBase
from mlrun.model_monitoring.evidently_application import SUPPORTED_EVIDENTLY_VERSION
from mlrun.model_monitoring.writer import _TSDB_BE, _TSDB_TABLE, ModelMonitoringWriter
from mlrun.utils.logger import Logger
from tests.system.base import TestMLRunSystem

from .assets.application import (
    EXPECTED_EVENTS_COUNT,
    DemoMonitoringApp,
    NoCheckDemoMonitoringApp,
)
from .assets.custom_evidently_app import CustomEvidentlyMonitoringApp


@dataclass
class _AppData:
    class_: type[ModelMonitoringApplicationBase]
    rel_path: str
    requirements: list[str] = field(default_factory=list)
    kwargs: dict[str, typing.Any] = field(default_factory=dict)
    abs_path: str = field(init=False)
    metrics: typing.Optional[set[str]] = None  # only for testing

    def __post_init__(self) -> None:
        path = Path(__file__).parent / self.rel_path
        assert path.exists()
        self.abs_path = str(path.absolute())


class _V3IORecordsChecker:
    _logger: Logger
    apps_data: list[_AppData]
    app_interval: int

    @classmethod
    def custom_setup_class(cls, project_name: str) -> None:
        cls._v3io_container = ModelMonitoringWriter.get_v3io_container(project_name)
        cls._kv_storage = ModelMonitoringWriter._get_v3io_client().kv
        cls._tsdb_storage = ModelMonitoringWriter._get_v3io_frames_client(
            cls._v3io_container
        )

    @classmethod
    def _test_kv_record(cls, ep_id: str) -> None:
        for app_data in cls.apps_data:
            app_name = app_data.class_.name
            cls._logger.debug("Checking the KV record of app", app_name=app_name)
            resp = ModelMonitoringWriter._get_v3io_client().kv.get(
                container=cls._v3io_container, table_path=ep_id, key=app_name
            )
            assert (
                data := resp.output.item
            ), f"V3IO KV app data is empty for app {app_name}"
            if app_data.metrics:
                assert (
                    data.keys() == app_data.metrics
                ), "The KV saved metrics are different than expected"

    @classmethod
    def _test_tsdb_record(cls, ep_id: str) -> None:
        df: pd.DataFrame = cls._tsdb_storage.read(
            backend=_TSDB_BE,
            table=_TSDB_TABLE,
            start=f"now-{5 * cls.app_interval}m",
            end="now",
        )
        assert not df.empty, "No TSDB data"
        assert (
            df.endpoint_id == ep_id
        ).all(), "The endpoint IDs are different than expected"

        assert set(df.application_name) == {
            app_data.class_.name for app_data in cls.apps_data
        }, "The application names are different than expected"

        tsdb_metrics = df.groupby("application_name").result_name.unique()
        for app_data in cls.apps_data:
            if app_metrics := app_data.metrics:
                app_name = app_data.class_.name
                cls._logger.debug("Checking the TSDB record of app", app_name=app_name)
                assert (
                    set(tsdb_metrics[app_name]) == app_metrics
                ), "The TSDB saved metrics are different than expected"

    @classmethod
    def _test_apps_parquet(
        cls, ep_id: str, inputs: set[str], outputs: set[str]
    ) -> None:
        parquet_apps_directory = (
            mlrun.model_monitoring.helpers.get_monitoring_parquet_path(
                mlrun.get_or_create_project(cls.project_name),
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
        cls._test_kv_record(ep_id)
        cls._test_tsdb_record(ep_id)


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestMonitoringAppFlow(TestMLRunSystem, _V3IORecordsChecker):
    project_name = "test-app-flow"
    project_name += datetime.now().strftime(  # remove when ML-5588 is fixed
        "%y%m%d%H%M"
    )
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    @classmethod
    def custom_setup_class(cls) -> None:
        cls.max_events = typing.cast(
            int, mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events
        )
        assert cls.max_events == EXPECTED_EVENTS_COUNT

        cls.model_name = "classification"
        cls.num_features = 4

        cls.app_interval: int = 1  # every 1 minute
        cls.app_interval_seconds = timedelta(minutes=cls.app_interval).total_seconds()

        cls.evidently_workspace_path = (
            f"/v3io/projects/{cls.project_name}/artifacts/evidently-workspace"
        )
        cls.evidently_project_id = str(uuid.uuid4())

        cls.apps_data: list[_AppData] = [
            _AppData(
                class_=DemoMonitoringApp,
                rel_path="assets/application.py",
                metrics={"data_drift_test", "model_perf"},
            ),
            _AppData(
                class_=CustomEvidentlyMonitoringApp,
                rel_path="assets/custom_evidently_app.py",
                requirements=[f"evidently=={SUPPORTED_EVIDENTLY_VERSION}"],
                kwargs={
                    "evidently_workspace_path": cls.evidently_workspace_path,
                    "evidently_project_id": cls.evidently_project_id,
                },
                metrics={"data_drift_test"},
            ),
        ]
        cls.infer_path = f"v2/models/{cls.model_name}/infer"
        cls.infer_input = cls._generate_infer_input()
        cls.next_window_input = cls._generate_infer_input(num_events=1)

        _V3IORecordsChecker.custom_setup_class(project_name=cls.project_name)

    def _submit_controller_and_deploy_writer(
        self, deploy_histogram_data_drift_app
    ) -> None:
        self.project.enable_model_monitoring(
            base_period=self.app_interval,
            **({} if self.image is None else {"default_controller_image": self.image}),
        )

    def _set_and_deploy_monitoring_apps(self) -> None:
        with ThreadPoolExecutor() as executor:
            for app_data in self.apps_data:
                fn = self.project.set_model_monitoring_function(
                    func=app_data.abs_path,
                    application_class=app_data.class_.__name__,
                    name=app_data.class_.name,
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
        serving_fn = mlrun.import_function(
            "hub://v2_model_server", project=cls.project_name, new_name="model-serving"
        )
        serving_fn.add_model(
            f"{cls.model_name}_{with_training_set}",
            model_path=f"store://models/{cls.project_name}/{cls.model_name}_{with_training_set}:latest",
        )
        serving_fn.set_tracking(tracking_policy=TrackingPolicy())
        if cls.image is not None:
            for attr in (
                "stream_image",
                "default_batch_image",
                "default_controller_image",
            ):
                setattr(serving_fn.spec.tracking_policy, attr, cls.image)
            serving_fn.spec.image = serving_fn.spec.build.image = cls.image

        serving_fn.deploy()
        return typing.cast(mlrun.runtimes.serving.ServingRuntime, serving_fn)

    @classmethod
    def _get_model_endpoint_id(cls) -> str:
        endpoints = mlrun.get_run_db().list_model_endpoints(project=cls.project_name)
        assert endpoints and len(endpoints) == 1
        return endpoints[0].metadata.uid

    @classmethod
    def _generate_infer_input(cls, num_events: typing.Optional[int] = None) -> str:
        if num_events is None:
            num_events = cls.max_events
        return json.dumps({"inputs": [[0] * cls.num_features] * num_events})

    def test_app_flow(self) -> None:
    @pytest.mark.parametrize("with_training_set", [True, False])
    def test_app_flow(self, with_training_set) -> None:
        self.project = typing.cast(mlrun.projects.MlrunProject, self.project)
        inputs, outputs = self._log_model(with_training_set)

        for i in range(len(self.apps_data)):
            if "with_training_set" in self.apps_data[i].kwargs:
                self.apps_data[i].kwargs["with_training_set"] = with_training_set

        # workaround for ML-5997
        if not with_training_set:
            self.apps_data.pop(0)

        with ThreadPoolExecutor() as executor:
            executor.submit(
                self._submit_controller_and_deploy_writer,
                deploy_histogram_data_drift_app=with_training_set,  # workaround for ML-5997
            )
            executor.submit(self._set_and_deploy_monitoring_apps)
            future = executor.submit(self._deploy_model_serving, with_training_set)

        self.serving_fn = future.result()

        time.sleep(5)
        self.serving_fn.invoke(self.infer_path, self.infer_input)
        # mark the first window as "done" with another request
        time.sleep(
            self.app_interval_seconds
            + mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
            + 2
        )
        self.serving_fn.invoke(self.infer_path, self.next_window_input)
        # wait for the completed window to be processed
        time.sleep(1.2 * self.app_interval_seconds)

        self._test_v3io_records(
            ep_id=self._get_model_endpoint_id(), inputs=inputs, outputs=outputs
        )


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
        cls.apps_data = [cls.app_data]
        _V3IORecordsChecker.custom_setup_class(project_name=cls.project_name)

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
            name=self.app_data.class_.name,
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
            **({} if self.image is None else {"default_controller_image": self.image}),
        )

    def test_inference_feature_set(self) -> None:
        self._log_model()

        with ThreadPoolExecutor() as executor:
            executor.submit(self._deploy_monitoring_app)
            executor.submit(self._deploy_monitoring_infra)

        self._record_results()

        time.sleep(2.4 * self.app_interval_seconds)

        self._test_v3io_records(
            self.endpoint_id, inputs=set(self.columns), outputs=set(self.y_name)
        )


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestModelMonitoringInitialize(TestMLRunSystem):
    project_name = "test-mm-initialize"
    # Set image to "<repo>/mlrun:<tag>" for local testing
    image: typing.Optional[str] = None

    def test_enable_model_monitoring(self) -> None:
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            self.project.update_model_monitoring_controller(
                image=self.image or "mlrun/mlrun"
            )

        self.project.enable_model_monitoring(image=self.image or "mlrun/mlrun")

        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER
        )
        writer = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.WRITER
        )
        stream = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.STREAM
        )

        controller._wait_for_function_deployment(db=controller._get_db())
        writer._wait_for_function_deployment(db=writer._get_db())
        stream._wait_for_function_deployment(db=stream._get_db())
        assert (
            controller.spec.config["spec.triggers.cron_interval"]["attributes"][
                "interval"
            ]
            == "10m"
        )

        self.project.update_model_monitoring_controller(
            image=self.image or "mlrun/mlrun", base_period=1
        )
        controller = self.project.get_function(
            key=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            ignore_cache=True,
        )
        controller._wait_for_function_deployment(db=controller._get_db())
        assert (
            controller.spec.config["spec.triggers.cron_interval"]["attributes"][
                "interval"
            ]
            == "1m"
        )


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
        cls, name: str, model_name: str, class_name: str, **kwargs
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
        serving_fn.set_tracking()
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
        with ThreadPoolExecutor() as executor:
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
        with ThreadPoolExecutor() as executor:
            self.db = mlrun.model_monitoring.get_model_endpoint_store(
                project=self.project_name
            )
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
