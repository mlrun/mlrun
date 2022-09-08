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
#
import mlrun
from mlrun.utils import logger
from tests.conftest import results

from .demo_states import *  # noqa


class _DummyStreamRaiser:
    def push(self, data):
        raise ValueError("DummyStreamRaiser raises an error")


def test_async_basic():
    function = mlrun.new_function("tests", kind="serving")
    flow = function.set_topology("flow", engine="async")
    queue = flow.to(name="s1", class_name="ChainWithContext").to(
        "$queue", "q1", path=""
    )

    s2 = queue.to(name="s2", class_name="ChainWithContext")
    s2.to(name="s4", class_name="ChainWithContext")
    s2.to(
        name="s5", class_name="ChainWithContext"
    ).respond()  # this state returns the resp

    queue.to(name="s3", class_name="ChainWithContext")

    # plot the graph for test & debug
    flow.plot(f"{results}/serving/async.png")

    server = function.to_mock_server()
    server.context.visits = {}
    logger.info(f"\nAsync Flow:\n{flow.to_yaml()}")
    resp = server.test(body=[])

    server.wait_for_completion()
    assert resp == ["s1", "s2", "s5"], "flow result is incorrect"
    assert server.context.visits == {
        "s1": 1,
        "s2": 1,
        "s4": 1,
        "s3": 1,
        "s5": 1,
    }, "flow didnt visit expected states"


def test_async_nested():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.add_step(name="s1", class_name="Echo")
    graph.add_step(name="s2", handler="multiply_input", after="s1")
    graph.add_step(name="s3", class_name="Echo", after="s2")

    router_step = graph.add_step("*", name="ensemble", after="s2")
    router_step.add_route("m1", class_name="ModelClass", model_path=".", multiplier=100)
    router_step.add_route("m2", class_name="ModelClass", model_path=".", multiplier=200)
    router_step.add_route(
        "m3:v1", class_name="ModelClass", model_path=".", multiplier=300
    )

    graph.add_step(name="final", class_name="Echo", after="ensemble").respond()

    logger.info(graph.to_yaml())
    server = function.to_mock_server()

    # plot the graph for test & debug
    graph.plot(f"{results}/serving/nested.png")
    resp = server.test("/v2/models/m2/infer", body={"inputs": [5]})
    server.wait_for_completion()
    # resp should be input (5) * multiply_input (2) * m2 multiplier (200)
    assert resp["outputs"] == 5 * 2 * 200, f"wrong health response {resp}"


def test_on_error():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    chain = graph.to("Chain", name="s1")
    chain.to("Raiser").error_handler("catch").to("Chain", name="s3")

    graph.add_step(
        name="catch", class_name="EchoError", after=""
    ).respond().full_event = True
    function.verbose = True
    server = function.to_mock_server()
    logger.info(graph.to_yaml())

    # plot the graph for test & debug
    graph.plot(f"{results}/serving/on_error.png")
    resp = server.test(body=[])
    server.wait_for_completion()
    assert (
        resp["error"] and resp["origin_state"] == "Raiser"
    ), f"error wasnt caught, resp={resp}"


def test_push_error():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    chain = graph.to("Chain", name="s1")
    chain.to("Raiser")

    function.verbose = True
    server = function.to_mock_server()
    server.error_stream = "dummy:///nothing"
    # Force an error inside push_error itself
    server._error_stream_object = _DummyStreamRaiser()
    logger.info(graph.to_yaml())

    server.test(body=[])
    server.wait_for_completion()
