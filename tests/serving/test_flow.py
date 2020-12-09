import mlrun
from .demo_states import *  # noqa


def test_basic_flow():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1")
    graph.add_step("s1", class_name="Chain")
    graph.add_step("s2", class_name="Chain", after="$prev")
    graph.add_step("s3", class_name="Chain", after="$prev")

    server = fn.to_mock_server()
    graph.plot("flow.png")
    print("\nFlow1:\n", graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow1 result is incorrect"

    graph = fn.set_topology("flow", exist_ok=True)
    graph.add_step("s2", class_name="Chain", after="$last")
    graph.add_step(
        "s1", class_name="Chain", after="$start"
    )  # should place s1 first and s2 after it
    graph.add_step("s3", class_name="Chain", after="s2")

    server = fn.to_mock_server()
    print("\nFlow2:\n", graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow2 result is incorrect"

    graph = fn.set_topology("flow", exist_ok=True)
    graph.add_step("s1", class_name="Chain", after="$start")
    graph.add_step("s3", class_name="Chain", after="$last")
    graph.add_step("s2", class_name="Chain", after="s1", before="s3")

    server = fn.to_mock_server()
    print("\nFlow3 (insert):\n", graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow3 result is incorrect"


def test_handler():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1")
    graph.add_step("s1", handler="(event + 1)")
    graph.add_step("s2", handler="json.dumps", after="$prev")

    server = fn.to_mock_server()
    resp = server.test(body=5)
    assert resp == "6", f"got unexpected result {resp}"


def test_on_error():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1")
    graph.add_step("s1", class_name="Chain")
    graph.add_step("raiser", class_name="Raiser", after="$prev").error_handler("catch")
    graph.add_step("s3", class_name="Chain", after="$prev")
    graph.add_step("catch", class_name="EchoError").full_event = True

    server = fn.to_mock_server()
    print(graph.to_yaml())
    resp = server.test(body=[])
    print(resp)
    assert resp["error"] and resp["origin_state"] == "raiser", "error wasnt caught"
