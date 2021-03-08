import pytest

import mlrun
from mlrun.utils import logger

from .demo_states import *  # noqa

engines = [
    "sync",
    "async",
]


def test_basic_flow():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.add_step(name="s1", class_name="Chain")
    graph.add_step(name="s2", class_name="Chain", after="$prev")
    graph.add_step(name="s3", class_name="Chain", after="$prev")

    server = fn.to_mock_server()
    # graph.plot("flow.png")
    print("\nFlow1:\n", graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow1 result is incorrect"

    graph = fn.set_topology("flow", exist_ok=True, engine="sync")
    graph.add_step(name="s2", class_name="Chain")
    graph.add_step(
        name="s1", class_name="Chain", before="s2"
    )  # should place s1 first and s2 after it
    graph.add_step(name="s3", class_name="Chain", after="s2")

    server = fn.to_mock_server()
    logger.info(f"flow: {graph.to_yaml()}")
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow2 result is incorrect"

    graph = fn.set_topology("flow", exist_ok=True, engine="sync")
    graph.add_step(name="s1", class_name="Chain")
    graph.add_step(name="s3", class_name="Chain", after="$prev")
    graph.add_step(name="s2", class_name="Chain", after="s1", before="s3")

    server = fn.to_mock_server()
    logger.info(f"flow: {graph.to_yaml()}")
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow3 result is incorrect"


@pytest.mark.parametrize("engine", engines)
def test_handler(engine):
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine=engine)
    graph.to(name="s1", handler="(event + 1)").to(name="s2", handler="json.dumps")
    if engine == "async":
        graph["s2"].respond()

    server = fn.to_mock_server()
    resp = server.test(body=5)
    if engine == "async":
        server.wait_for_completion()
    # the json.dumps converts the 6 to "6" (string)
    assert resp == "6", f"got unexpected result {resp}"


def test_init_class():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="s1", class_name="Echo").to(name="s2", class_name="RespName")

    server = fn.to_mock_server()
    resp = server.test(body=5)
    assert resp == [5, "s2"], f"got unexpected result {resp}"


def test_on_error():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.add_step(name="s1", class_name="Chain")
    graph.add_step(name="raiser", class_name="Raiser", after="$prev").error_handler(
        "catch"
    )
    graph.add_step(name="s3", class_name="Chain", after="$prev")
    graph.add_step(name="catch", class_name="EchoError").full_event = True

    server = fn.to_mock_server()
    logger.info(f"flow: {graph.to_yaml()}")
    resp = server.test(body=[])
    assert resp["error"] and resp["origin_state"] == "raiser", "error wasnt caught"
