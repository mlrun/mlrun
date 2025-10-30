# Copyright 2025 Iguazio
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
    DatastoreProfileTDEngine,
    DatastoreProfileV3io,
)
from mlrun.model_monitoring.writer import WriterGraphFactory


@pytest.mark.parametrize(
    "tsdb_profile",
    [
        DatastoreProfileV3io(name="v3io-tsdb-test"),
        DatastoreProfileTDEngine(
            name="tdengine-test", user="root", host="localhost", port=6041
        ),
    ],
)
def test_plot_writer_graph(
    monkeypatch: pytest.MonkeyPatch, tsdb_profile: DatastoreProfile
) -> None:
    monkeypatch.setattr(mlrun.mlconf, "system_id", "123456")
    # Set system_id for the test to enable TDEngineConnector to construct database name
    mlrun.mlconf.system_id = "123456"
    project_name = "test-writer"
    project = mlrun.get_or_create_project(project_name, allow_cross_project=True)

    fn = project.set_function(
        kind="serving",
        name="my-fn",
    )

    tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
        project=project_name, profile=tsdb_profile
    )

    WriterGraphFactory(parquet_path="").apply_writer_graph(fn, tsdb_connector)

    graph = fn.spec.graph.plot(rankdir="TB")
    print()
    print(f"Graphviz graph definition with tsdb_connector={tsdb_connector}")
    print("Feed this to graphviz, or to https://dreampuf.github.io/GraphvizOnline")
    print()
    print(graph)
