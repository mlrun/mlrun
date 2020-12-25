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
    graph = fn.set_topology("flow", start_at="s1", engine="async")
    graph.add_step(name="s1", handler="(event + 1)")
    graph.add_step(name="s2", handler="json.dumps", after="$prev").respond()

    server = fn.to_mock_server()
    resp = server.test(body=5)
    server.wait_for_completion()
    assert resp == "6", f"got unexpected result {resp}"


@pytest.mark.skipif(not has_storey(), reason="storey not installed")
def test_async_basic():
    fn = mlrun.new_function("tests", kind="serving")
    flow = fn.set_topology("flow", engine="async")
    queue = flow.to(name="s1", class_name="ChainWithContext").to(">", path="")

    s2 = queue.to(name="s2", class_name="ChainWithContext")
    s2.to(name="s4", class_name="ChainWithContext")
    s2.to(name="s5", class_name="ChainWithContext").respond()

    queue.to(name="s3", class_name="ChainWithContext")

    flow.plot("async.png")

    server = fn.to_mock_server()
    server.context.visits = {}
    print("\nAsync Flow:\n", flow.to_yaml())
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
    graph = fn.set_topology("flow", start_at="s1", engine="async")
    graph.add_step(name="s1", class_name="Echo")
    graph.add_step(name="s2", handler="multiply_input", after="s1")
    graph.add_step(name="s3", class_name="Echo", after="s2")

    router = graph.add_step("*", name="ensemble", after="s2")
    router.add_route("m1", class_name="ModelClass", model_path=".", z=100)
    router.add_route("m2", class_name="ModelClass", model_path=".", z=200)
    router.add_route("m3:v1", class_name="ModelClass", model_path=".", z=300)

    graph.add_step(name="final", class_name="Echo", after="ensemble").respond()
    graph.plot("nested.png")

    print(graph.to_yaml())
    server = fn.to_mock_server()
    resp = server.test("/v2/models/m2/infer", body={"inputs": [5]})
    server.wait_for_completion()
    assert resp["outputs"] == 2000, f"wrong health response {resp}"


def test_on_error():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1", engine="async")
    chain = graph.to("Chain", name="s1")
    chain.to("Raiser").error_handler("catch").to("Chain", name="s3")

    graph.add_step(
        name="catch", class_name="EchoError", after=""
    ).respond().full_event = True
    fn.verbose = True
    server = fn.to_mock_server()
    print(graph.to_yaml())
    resp = server.test(body=[])
    server.wait_for_completion()
    print(resp)
    print(dir(resp))
    # assert resp["error"] and resp["origin_state"] == "raiser", "error wasnt caught"
