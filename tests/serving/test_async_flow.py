import mlrun
import pytest
from .demo_states import *  # noqa


def has_storey():
    try:
        import storey  # noqa
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not has_storey(), reason="storey not installed")
def test_handler():
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology("flow", start_at="s1", engine="async", result_state="s2")
    fn.add_state("s1", handler="(event + 1)")
    fn.add_state("s2", handler="json.dumps", after="$prev")

    server = fn.to_mock_server()
    resp = server.test(body=5)
    server.wait_for_completion()
    assert resp == "6", f"got unexpected result {resp}"


@pytest.mark.skipif(not has_storey(), reason="storey not installed")
def test_async_basic():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1", engine="async", result_state="s5")
    fn.add_state("s1", class_name="ChainWithContext", after="$start")

    stream_path = ""
    fn.add_state("q", kind="queue", path=stream_path, after="s1")

    fn.add_state("s2", class_name="ChainWithContext", after="q")
    fn.add_state("s3", class_name="ChainWithContext", after="q")
    fn.add_state("s4", class_name="ChainWithContext", after="s2")
    fn.add_state("s5", class_name="ChainWithContext", after="s2")
    fn.plot("async.png")

    server = fn.to_mock_server()
    server.context.visits = {}
    print("\nAsync Flow:\n", graph.to_yaml())
    resp = server.test(body=[])
    print(resp)

    server.wait_for_completion()
    assert resp == ["s1", "s2", "s5"], "flow result is incorrect"
    assert server.context.visits == {
        "s1": 1,
        "s2": 1,
        "s4": 1,
        "s3": 1,
        "s5": 1,
    }, "flow didnt visit expected states"


@pytest.mark.skipif(not has_storey(), reason="storey not installed")
def test_async_nested():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1", engine="async", result_state="final")
    fn.add_state("s1", class_name="Echo")
    fn.add_state("s2", handler="multiply_input", after="s1")
    fn.add_state("s3", class_name="Echo", after="s2")

    fn.add_state("ensemble", kind="router", after="s2")
    fn.add_model(
        "m1", parent="ensemble", class_name="ModelClass", model_path=".", z=100
    )
    fn.add_model(
        "m2", parent="ensemble", class_name="ModelClass", model_path=".", z=200
    )
    fn.add_model(
        "m3:v1", parent="ensemble", class_name="ModelClass", model_path=".", z=300
    )

    fn.add_state("final", class_name="Echo", after="ensemble")
    fn.plot("nested.png")

    print(graph.to_yaml())
    server = fn.to_mock_server()
    resp = server.test("/v2/models/m2/infer", body={"inputs": [5]})
    server.wait_for_completion()
    assert resp["outputs"] == 2000, f"wrong health response {resp}"
