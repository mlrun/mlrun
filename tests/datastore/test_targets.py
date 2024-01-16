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
import os

from mlrun.datastore import StreamTarget
from mlrun.feature_store import FeatureSet


class MockGraph:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def add_step(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


# ML-5484
def test_stream_target_path_is_without_run_id():
    os.environ["V3IO_ACCESS_KEY"] = os.environ.get("V3IO_ACCESS_KEY", "placeholder")

    mock_graph = MockGraph()
    path = "container/dir/subdir/"
    stream_target = StreamTarget(name="my-target", path=f"v3io:///{path}")
    stream_target.run_id = "123"
    fset = FeatureSet(name="my-featureset")
    stream_target.set_resource(fset)
    stream_target.add_writer_step(mock_graph, None, None, key_columns={})
    # make sure that run ID wasn't added to the path
    assert mock_graph.kwargs.get("stream_path") == path
