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
#
import pytest

import mlrun
import mlrun.serving


class Echo:
    """example class"""

    def __init__(self, context, name=None, data={}):
        self.context = context
        self.name = name
        self.data = data

    def do(self, x):
        self.context.logger.info("test text")
        return self.data


def my_hnd(event):
    """example handler"""
    return {"mul": event["x"] * 2}


@pytest.mark.parametrize("executor", mlrun.serving.routers.ParallelRunnerModes.all())
def test_parallel(executor):
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology(
        "router",
        mlrun.serving.routers.ParallelRun(extend_event=True, executor_type=executor),
    )
    graph.add_route("c1", class_name="Echo", data={"a": 1, "b": 2})
    graph.add_route("c2", class_name="Echo", data={"c": 7})
    graph.add_route("c3", handler="my_hnd")

    server = fn.to_mock_server()

    resp = server.test(body={"x": 8})
    assert resp == {"x": 8, "a": 1, "b": 2, "c": 7, "mul": 16}

    resp = server.test("", {"x": 9})
    assert resp == {"x": 9, "a": 1, "b": 2, "c": 7, "mul": 18}
