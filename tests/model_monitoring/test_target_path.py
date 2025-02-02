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

import os
from collections.abc import Iterator
from unittest import mock

import pytest

import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.config
import mlrun.model_monitoring
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    DatastoreProfileV3io,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)

TEST_PROJECT = "test-model-endpoints"


@mock.patch.dict(os.environ, {"MLRUN_ARTIFACT_PATH": "s3://some-bucket/"}, clear=True)
def test_get_file_target_path():
    # offline target with relative path
    offline_parquet_relative = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=TEST_PROJECT,
        kind="parquet",
        target="offline",
        artifact_path=os.environ["MLRUN_ARTIFACT_PATH"],
    )
    assert (
        offline_parquet_relative
        == os.environ["MLRUN_ARTIFACT_PATH"] + "model-endpoints/parquet"
    )

    # online target
    online_target = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=TEST_PROJECT, kind="some_kind", target="online"
    )
    assert (
        online_target
        == f"v3io:///users/pipelines/{TEST_PROJECT}/model-endpoints/some_kind"
    )

    # offline target with absolute path
    mlrun.mlconf.model_endpoint_monitoring.offline_storage_path = (
        "schema://projects/test-path"
    )
    offline_parquet_abs = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=TEST_PROJECT, kind="parquet", target="offline"
    )
    assert (
        offline_parquet_abs + f"/{TEST_PROJECT}/parquet"
        == f"schema://projects/test-path/{TEST_PROJECT}/parquet"
    )

    tsdb_monitoring_application_full_path = (
        mlrun.mlconf.get_model_monitoring_file_target_path(
            project=TEST_PROJECT,
            kind=mm_constants.FileTargetKind.MONITORING_APPLICATION,
        )
    )
    assert (
        tsdb_monitoring_application_full_path
        == f"v3io:///users/pipelines/{TEST_PROJECT}/monitoring-apps/"
    )


def test_get_v3io_stream_path() -> None:
    stream_path = mlrun.model_monitoring.get_stream_path(
        project=TEST_PROJECT, profile=DatastoreProfileV3io(name="tmp")
    )
    assert stream_path == f"v3io:///projects/{TEST_PROJECT}/model-endpoints/stream-v1"


@pytest.fixture
def kafka_profile_name() -> Iterator[str]:
    profile_name = "kafka-prof"
    profile = DatastoreProfileKafkaSource(
        name=profile_name, brokers=["some_kafka_broker:8080"], topics=[]
    )
    register_temporary_client_datastore_profile(profile)
    yield profile_name
    remove_temporary_client_datastore_profile(profile_name)


def test_get_kafka_profile_stream_path(kafka_profile_name: str) -> None:
    # kafka stream path from datastore profile
    stream_path = mlrun.model_monitoring.get_stream_path(
        project=TEST_PROJECT, secret_provider=lambda _: kafka_profile_name
    )
    assert (
        stream_path
        == f"kafka://some_kafka_broker:8080?topic=monitoring_stream_{mlrun.mlconf.system_id}_{TEST_PROJECT}_v1"
    )
