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
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_iris

import mlrun
from mlrun.model_monitoring import TrackingPolicy
from mlrun.model_monitoring.writer import _TSDB_BE, _TSDB_TABLE, ModelMonitoringWriter
from tests.system.base import TestMLRunSystem

from .assets.application import DemoMonitoringApp


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestMonitoringAppFlow(TestMLRunSystem):
    project_name = "test-monitoring-app-flow"

    @classmethod
    def custom_setup_class(cls) -> None:
        cls.max_events = 5

        cls.model_name = "classification"
        cls.num_features = 4

        cls._orig_parquet_batching_max_events = (
            mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events
        )
        mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events = (
            cls.max_events
        )

        cls.app_interval: int = 1  # every 1 minute

        cls.app_name = DemoMonitoringApp.name
        cls.infer_path = f"v2/models/{cls.model_name}/infer"
        cls.infer_input = cls._generate_infer_input()

        cls._v3io_container = ModelMonitoringWriter.get_v3io_container(cls.project_name)
        cls._kv_storage = ModelMonitoringWriter._get_v3io_client().kv
        cls._tsdb_storage = ModelMonitoringWriter._get_v3io_frames_client(
            cls._v3io_container
        )

    @classmethod
    def custom_teardown_class(cls) -> None:
        mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events = (
            cls._orig_parquet_batching_max_events
        )

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
                application_batch=True,
            ),
        )
        serving_fn.deploy()
        return typing.cast(mlrun.runtimes.serving.ServingRuntime, serving_fn)

    @classmethod
    def _get_model_enpoint_id(cls) -> str:
        endpoints = mlrun.get_run_db().list_model_endpoints(project=cls.project_name)
        assert endpoints and len(endpoints) == 1
        return endpoints[0].metadata.uid

    @classmethod
    def _generate_infer_input(cls) -> str:
        return json.dumps({"inputs": [[0] * cls.num_features] * cls.max_events})

    @classmethod
    def _test_kv_record(cls, ep_id: str) -> None:
        resp = ModelMonitoringWriter._get_v3io_client().kv.get(
            container=cls._v3io_container, table_path=ep_id, key=cls.app_name
        )
        assert resp.output.item, "V3IO KV app data is empty"

    @classmethod
    def _test_tsdb_record(cls, ep_id: str) -> None:
        df: pd.DataFrame = cls._tsdb_storage.read(
            backend=_TSDB_BE,
            table=_TSDB_TABLE,
            start=f"now-{2 * cls.app_interval}m",
        )
        assert not df.empty, "No TSDB data"
        assert (
            df.iloc[0].endpoint_id == ep_id
        ), "The endpoint ID is different than expected"

    @classmethod
    def _test_v3io_records(cls, ep_id: str) -> None:
        cls._test_kv_record(ep_id)
        cls._test_tsdb_record(ep_id)

    def test_app_flow(self) -> None:
        self.project = typing.cast(mlrun.projects.MlrunProject, self.project)
        self.project.set_model_monitoring_application(
            func=str((Path(__file__).parent / "assets/application.py").absolute()),
            application_class="DemoMonitoringApp",
            name=self.app_name,
            image="mlrun/mlrun",
        )
        self._log_model()
        self.serving_fn = self._deploy_model_serving()

        time.sleep(5)
        self.serving_fn.invoke(self.infer_path, self.infer_input)
        time.sleep(1.2 * timedelta(minutes=self.app_interval).total_seconds())

        ep_id = self._get_model_enpoint_id()
        self._test_v3io_records(ep_id)
