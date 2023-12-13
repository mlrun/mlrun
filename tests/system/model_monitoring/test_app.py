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

import json
import time
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_iris

import mlrun
from mlrun.model_monitoring import TrackingPolicy
from mlrun.model_monitoring.application import ModelMonitoringApplication
from mlrun.model_monitoring.writer import _TSDB_BE, _TSDB_TABLE, ModelMonitoringWriter
from tests.system.base import TestMLRunSystem

from .assets.application import EXPECTED_EVENTS_COUNT, DemoMonitoringApp
from .assets.custom_evidently_app import CustomEvidentlyMonitoringApp


@dataclass
class _AppData:
    class_: typing.Type[ModelMonitoringApplication]
    rel_path: str
    requirements: list[str] = field(default_factory=list)
    kwargs: dict[str, typing.Any] = field(default_factory=dict)
    abs_path: str = field(init=False)

    def __post_init__(self) -> None:
        path = Path(__file__).parent / self.rel_path
        assert path.exists()
        self.abs_path = str(path.absolute())


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestMonitoringAppFlow(TestMLRunSystem):
    project_name = "test-monitoring-app-flow"
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
            _AppData(class_=DemoMonitoringApp, rel_path="assets/application.py"),
            _AppData(
                class_=CustomEvidentlyMonitoringApp,
                rel_path="assets/custom_evidently_app.py",
                requirements=["evidently~=0.4.7"],
                kwargs={
                    "evidently_workspace_path": cls.evidently_workspace_path,
                    "evidently_project_id": cls.evidently_project_id,
                },
            ),
        ]
        cls.infer_path = f"v2/models/{cls.model_name}/infer"
        cls.infer_input = cls._generate_infer_input()
        cls.next_window_input = cls._generate_infer_input(num_events=1)

        cls._v3io_container = ModelMonitoringWriter.get_v3io_container(cls.project_name)
        cls._kv_storage = ModelMonitoringWriter._get_v3io_client().kv
        cls._tsdb_storage = ModelMonitoringWriter._get_v3io_frames_client(
            cls._v3io_container
        )

    def _submit_controller_and_deploy_writer(self) -> None:
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
                executor.submit(self.project.deploy_function, fn)

    def _log_model(self) -> None:
        dataset = load_iris()
        train_set = pd.DataFrame(
            dataset.data,
            columns=dataset.feature_names,
        )
        self.project.log_model(
            self.model_name,
            model_dir=str((Path(__file__).parent / "assets").absolute()),
            model_file="model.pkl",
            training_set=train_set,
        )

    @classmethod
    def _deploy_model_serving(cls) -> mlrun.runtimes.serving.ServingRuntime:
        serving_fn = mlrun.import_function(
            "hub://v2_model_server", project=cls.project_name, new_name="model-serving"
        )
        serving_fn.add_model(
            cls.model_name,
            model_path=f"store://models/{cls.project_name}/{cls.model_name}:latest",
        )
        serving_fn.set_tracking(
            tracking_policy=TrackingPolicy(
                default_batch_intervals=f"*/{cls.app_interval} * * * *",
            ),
        )
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
    def _get_model_enpoint_id(cls) -> str:
        endpoints = mlrun.get_run_db().list_model_endpoints(project=cls.project_name)
        assert endpoints and len(endpoints) == 1
        return endpoints[0].metadata.uid

    @classmethod
    def _generate_infer_input(cls, num_events: typing.Optional[int] = None) -> str:
        if num_events is None:
            num_events = cls.max_events
        return json.dumps({"inputs": [[0] * cls.num_features] * num_events})

    @classmethod
    def _test_kv_record(cls, ep_id: str) -> None:
        for app_data in cls.apps_data:
            app_name = app_data.class_.name
            cls._logger.debug("Checking the KV record of app", app_name=app_name)
            resp = ModelMonitoringWriter._get_v3io_client().kv.get(
                container=cls._v3io_container, table_path=ep_id, key=app_name
            )
            assert resp.output.item, f"V3IO KV app data is empty for app {app_name}"

    @classmethod
    def _test_tsdb_record(cls, ep_id: str) -> None:
        df: pd.DataFrame = cls._tsdb_storage.read(
            backend=_TSDB_BE,
            table=_TSDB_TABLE,
            start=f"now-{5 * cls.app_interval}m",
        )
        assert not df.empty, "No TSDB data"
        assert (
            df.endpoint_id == ep_id
        ).all(), "The endpoint IDs are different than expected"
        assert set(df.application_name) == {
            app_data.class_.name for app_data in cls.apps_data
        }, "The application names are different than expected"

    @classmethod
    def _test_v3io_records(cls, ep_id: str) -> None:
        cls._test_kv_record(ep_id)
        cls._test_tsdb_record(ep_id)

    def test_app_flow(self) -> None:
        self.project = typing.cast(mlrun.projects.MlrunProject, self.project)
        self._log_model()

        with ThreadPoolExecutor() as executor:
            executor.submit(self._submit_controller_and_deploy_writer)
            executor.submit(self._set_and_deploy_monitoring_apps)
            future = executor.submit(self._deploy_model_serving)

        self.serving_fn = future.result()

        time.sleep(5)
        self.serving_fn.invoke(self.infer_path, self.infer_input)
        # mark the first window as "done" with another request
        time.sleep(self.app_interval_seconds + 2)
        self.serving_fn.invoke(self.infer_path, self.next_window_input)
        # wait for the completed window to be processed
        time.sleep(1.2 * self.app_interval_seconds)

        self._test_v3io_records(ep_id=self._get_model_enpoint_id())
