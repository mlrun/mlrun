# Copyright 2024 Iguazio
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

import pytest

import mlrun
import mlrun.model_monitoring
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaSource,
    DatastoreProfileV3io,
    TDEngineDatastoreProfile,
)
from mlrun.model_monitoring.stream_processing import EventStreamProcessor


@pytest.mark.parametrize(
    "tsdb_profile",
    [
        DatastoreProfileV3io(name="v3io-tsdb-test"),
        TDEngineDatastoreProfile(
            name="tdengine-test", user="root", host="localhost", port=6041
        ),
    ],
)
@pytest.mark.parametrize(
    "stream_profile",
    [
        DatastoreProfileV3io(name="v3io-stream-test"),
        DatastoreProfileKafkaSource(
            name="kafka-test", brokers=["localhost:9092"], topics=[]
        ),
    ],
)
def test_plot_monitoring_serving_graph(
    tsdb_profile: DatastoreProfile, stream_profile: DatastoreProfile
) -> None:
    project_name = "test-stream-processing"
    project = mlrun.get_or_create_project(project_name)

    processor = EventStreamProcessor(project_name, 1000, 10, "mytarget")

    fn = project.set_function(
        kind="serving",
        name="my-fn",
    )

    tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
        project=project_name, profile=tsdb_profile
    )
    stream_path = mlrun.model_monitoring.get_stream_path(
        project=project_name, profile=stream_profile
    )

    processor.apply_monitoring_serving_graph(fn, tsdb_connector, stream_path)

    graph = fn.spec.graph.plot(rankdir="TB")
    print()
    print(
        f"Graphviz graph definition with tsdb_connector={tsdb_connector} and stream_path={stream_path}"
    )
    print("Feed this to graphviz, or to https://dreampuf.github.io/GraphvizOnline")
    print()
    print(graph)
