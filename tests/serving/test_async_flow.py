from copy import copy

import mlrun
import pytest


def has_storey():
    try:
        import storey
    except ImportError:
        return False
    return True


class BaseClass:
    def __init__(self, context, name=None):
        self.context = context
        self.name = name


class Echo(BaseClass):
    def do(self, x):
        print("Echo:", self.name, x)
        return x


class EchoError(BaseClass):
    def do(self, x):
        x.body = {"body": x.body, "origin_state": x.origin_state, "error": x.error}
        return x


class Chain(BaseClass):
    def do(self, x):
        visits = self.context.visits.get(self.name, 0)
        self.context.visits[self.name] = visits + 1
        x = copy(x)
        x.append(self.name)
        return x


@pytest.mark.skipif(not has_storey(), reason="storey not installed")
def test_handler():
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology("flow", start_at="s1", engine="async", result_state="s2")
    fn.add_state("s1", handler="(event + 1)")
    fn.add_state("s2", handler="json.dumps", after='$prev')

    server = fn.to_mock_server()
    resp = server.test(body=5)
    server.wait_for_completion()
    assert resp == "6", f"got unexpected result {resp}"


@pytest.mark.skipif(not has_storey(), reason="storey not installed")
def test_async_basic():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", start_at="s1", engine="async", result_state="s5")
    fn.add_state("s1", class_name="Chain", after="$start")

    stream_path = ""
    fn.add_state("q", kind="queue", path=stream_path, after="s1")

    fn.add_state("s2", class_name="Chain", after="q")
    fn.add_state("s3", class_name="Chain", after="q")
    fn.add_state("s4", class_name="Chain", after="s2")
    fn.add_state("s5", class_name="Chain", after="s2")
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
