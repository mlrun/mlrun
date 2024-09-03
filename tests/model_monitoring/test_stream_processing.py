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
from mlrun.model_monitoring.stream_processing import EventStreamProcessor


@pytest.mark.parametrize("tsdb_connector", ["v3io", "taosws"])
@pytest.mark.parametrize("endpoint_store", ["v3io", "mysql"])
def test_plot_monitoring_serving_graph(tsdb_connector, endpoint_store):
    project_name = "test-stream-processing"

    processor = EventStreamProcessor(
        project_name,
        1000,
        10,
        "mytarget",
    )

    fn = mlrun.new_function(
        kind="serving",
        name="my-fn",
    )

    tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
        project=project_name,
        tsdb_connection_string=tsdb_connector,
        initialize=False,
    )
    store_object = mlrun.model_monitoring.get_store_object(
        project=project_name,
        store_connection_string=endpoint_store,
        initialize=False,
    )

    processor.apply_monitoring_serving_graph(fn, tsdb_connector, store_object)

    graph = fn.spec.graph.plot(rankdir="TB")
    print()
    print(
        f"Graphviz graph definition with tsdb_connector={tsdb_connector}, endpoint_store={endpoint_store}"
    )
    print("Feed this to graphviz, or to https://dreampuf.github.io/GraphvizOnline")
    print()
    print(graph)
