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
import os

import mlrun.config
import mlrun.utils.model_monitoring

TEST_PROJECT = "test-model-endpoints"
os.environ["MLRUN_ARTIFACT_PATH"] = "s3://some-bucket/"


def test_get_file_target_path():

    # offline target with relative path
    offline_parquet_relative = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=TEST_PROJECT, kind="parquet", target="offline"
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


def test_get_stream_path():
    # default stream path
    stream_path = mlrun.utils.model_monitoring.get_stream_path(project=TEST_PROJECT)
    assert (
        stream_path == f"v3io:///users/pipelines/{TEST_PROJECT}/model-endpoints/stream"
    )

    # kafka stream path from env
    os.environ["STREAM_PATH"] = "kafka://some_kafka_bootstrap_servers:8080"
    stream_path = mlrun.utils.model_monitoring.get_stream_path(project=TEST_PROJECT)
    assert (
        stream_path
        == f"kafka://some_kafka_bootstrap_servers:8080?topic=monitoring_stream_{TEST_PROJECT}"
    )
