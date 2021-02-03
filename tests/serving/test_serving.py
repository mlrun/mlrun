import json
import os
import time
import mlrun

from mlrun.utils import logger
from mlrun.runtimes import nuclio_init_hook
from mlrun.runtimes.serving import serving_subkind
from mlrun.serving import V2ModelServer
from mlrun.serving.server import MockEvent, GraphContext, create_graph_server
from mlrun.serving.states import RouterState, TaskState

router_object = RouterState()
router_object.routes = {
    "m1": TaskState(
        "ModelTestingClass", class_args={"model_path": "", "multiplier": 100}
    ),
    "m2": TaskState(
        "ModelTestingClass", class_args={"model_path": "", "multiplier": 200}
    ),
    "m3:v1": TaskState(
        "ModelTestingClass", class_args={"model_path": "", "multiplier": 300}
    ),
    "m3:v2": TaskState(
        "ModelTestingClass", class_args={"model_path": "", "multiplier": 400}
    ),
}


def generate_spec(graph, mode="sync", params={}):
    return {
        "version": "v2",
        "parameters": params,
        "graph": graph,
        "load_mode": mode,
        "verbose": True,
        "function_uri": "default/func",
    }


asyncspec = generate_spec(
    {
        "kind": "router",
        "routes": {
            "m5": {
                "class_name": "AsyncModelTestingClass",
                "class_args": {"model_path": ""},
            },
        },
    },
    "async",
)


raiser_spec = generate_spec(
    {
        "kind": "router",
        "routes": {
            "m6": {
                "class_name": "RaiserTestingClass",
                "class_args": {"model_path": "."},
            },
        },
    },
    params={"log_stream": "dummy://"},
)


spec = generate_spec(router_object.to_dict())
testdata = '{"inputs": [5]}'


class ModelTestingClass(V2ModelServer):
    def load(self):
        print("loading")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"][0] * self.get_param("multiplier")
        return resp

    def explain(self, request):
        print("predict:", request)
        resp = request["inputs"][0]
        return {"explained": resp}

    def op_myop(self, event):
        return event.body


class RaiserTestingClass(V2ModelServer):
    def load(self):
        print("loading..")

    def predict(self, request):
        print("predict:", request)
        raise ValueError("simulated error..")


class AsyncModelTestingClass(V2ModelServer):
    def load(self):
        print("loading..")
        time.sleep(4)
        print("loaded")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"][0]
        return resp


def init_ctx():
    os.environ["SERVING_SPEC_ENV"] = json.dumps(spec)
    context = GraphContext()
    nuclio_init_hook(context, globals(), serving_subkind)
    return context


def test_v2_get_models():
    context = init_ctx()

    e = MockEvent("", path="/v2/models/", method="GET")
    resp = context.mlrun_handler(context, e)
    data = json.loads(resp.body)

    # expected: {"models": ["m1", "m2", "m3:v1", "m3:v2"]}
    assert len(data["models"]) == 4, f"wrong get models response {resp.body}"


def test_v2_infer():
    def run_model(url, expected):
        event = MockEvent(testdata, path=f"/v2/models/{url}/infer")
        resp = context.mlrun_handler(context, event)
        data = json.loads(resp.body)
        assert data["outputs"] == expected, f"wrong model response {data['outputs']}"

    context = init_ctx()
    run_model("m1", 500)
    run_model("m2", 1000)
    run_model("m3/versions/v1", 1500)
    run_model("m3/versions/v2", 2000)


def test_v2_stream_mode():
    # model and operation are specified inside the message body
    context = init_ctx()
    event = MockEvent('{"model": "m2", "inputs": [5]}')
    resp = context.mlrun_handler(context, event)
    data = json.loads(resp.body)
    assert data["outputs"] == 1000, f"wrong model response {resp.body}"

    event = MockEvent(
        '{"model": "m3:v2", "operation": "explain", "inputs": [5]}', path=""
    )
    resp = context.mlrun_handler(context, event)
    logger.info(f"resp: {resp.body}")
    data = json.loads(resp.body)
    assert data["outputs"]["explained"] == 5, f"wrong model response {data}"


def test_v2_raised_err():
    os.environ["SERVING_SPEC_ENV"] = json.dumps(raiser_spec)
    context = GraphContext()
    nuclio_init_hook(context, globals(), serving_subkind)

    event = MockEvent(testdata, path="/v2/models/m6/infer")
    resp = context.mlrun_handler(context, event)
    context.logger.info(f"model responded: {resp}")
    assert resp.status_code == 400, "expecting Response() with status_code 400"


def test_v2_async_mode():
    # model loading is async
    os.environ["SERVING_SPEC_ENV"] = json.dumps(asyncspec)
    context = GraphContext()
    nuclio_init_hook(context, globals(), serving_subkind)
    context.logger.info("model initialized")

    context.logger.info("test not ready, should return err 408")
    event = MockEvent("", path="/v2/models/m5/ready", method="GET")
    resp = context.mlrun_handler(context, event)
    assert (
        resp.status_code == 408
    ), f"didnt get proper ready resp, expected 408, got {resp.status_code}"

    event = MockEvent(testdata, path="/v2/models/m5/infer")
    resp = context.mlrun_handler(context, event)
    context.logger.info("model responded")
    logger.info(resp)
    assert (
        resp.status_code != 200
    ), f"expected failure, got {resp.status_code} {resp.body}"

    event = MockEvent('{"model": "m5", "inputs": [5]}')
    event.trigger = "stream"
    resp = context.mlrun_handler(context, event)
    context.logger.info("model responded")
    logger.info(resp)
    data = json.loads(resp.body)
    assert data["outputs"] == 5, f"wrong model response {data}"


def test_v2_explain():
    context = init_ctx()
    event = MockEvent(testdata, path="/v2/models/m1/explain")
    resp = context.mlrun_handler(context, event)
    data = json.loads(resp.body)
    assert data["outputs"]["explained"] == 5, f"wrong explain response {resp.body}"


def test_v2_get_modelmeta():
    def get_model(name, version, url):
        event = MockEvent("", path=f"/v2/models/{url}", method="GET")
        resp = context.mlrun_handler(context, event)
        logger.info(f"resp: {resp}")
        data = json.loads(resp.body)

        # expected: {"name": "m3", "version": "v2", "inputs": [], "outputs": []}
        assert (
            data["name"] == name and data["version"] == version
        ), f"wrong get model meta response {resp.body}"

    context = init_ctx()
    get_model("m2", "", "m2")
    get_model("m3", "v2", "m3/versions/v2")


def test_v2_custom_handler():
    context = init_ctx()
    event = MockEvent('{"test": "ok"}', path="/v2/models/m1/myop")
    resp = context.mlrun_handler(context, event)
    data = json.loads(resp.body)
    assert data["test"] == "ok", f"wrong custom op response {resp.body}"


def test_v2_errors():
    context = init_ctx()
    event = MockEvent('{"test": "ok"}', path="/v2/models/m1/xx")
    resp = context.mlrun_handler(context, event)
    # expected: 400, 'illegal model operation xx, method=None'
    assert resp.status_code == 400, f"didnt get proper handler error {resp.body}"

    event = MockEvent('{"test": "ok"}', path="/v2/models/m5/xx")
    resp = context.mlrun_handler(context, event)
    # expected: 400, 'model m5 doesnt exist, available models: m1 | m2 | m3:v1 | m3:v2'
    assert resp.status_code == 400, f"didnt get proper model error {resp.body}"


def test_v2_model_ready():
    context = init_ctx()
    event = MockEvent("", path="/v2/models/m1/ready", method="GET")
    resp = context.mlrun_handler(context, event)
    assert resp.status_code == 200, f"didnt get proper ready resp {resp.body}"


def test_v2_health():
    context = init_ctx()
    event = MockEvent(None, path="/", method="GET")
    resp = context.mlrun_handler(context, event)
    data = json.loads(resp.body)
    # expected: {'name': 'ModelRouter', 'version': 'v2', 'extensions': []}
    assert data["version"] == "v2", f"wrong health response {resp.body}"

    event = MockEvent("", path="/v2/health", method="GET")
    resp = context.mlrun_handler(context, event)
    data = json.loads(resp.body)
    assert data["version"] == "v2", f"wrong health response {resp.body}"


def test_v2_mock():
    host = create_graph_server(graph=RouterState())
    host.graph.add_route(
        "my", class_name=ModelTestingClass, model_path="", multiplier=100
    )
    host.init(None, globals())
    logger.info(host.to_yaml())
    resp = host.test("/v2/models/my/infer", testdata)
    logger.info(f"resp: {resp}")
    # expected: source (5) * multiplier (100)
    assert resp["outputs"] == 5 * 100, f"wrong health response {resp}"


def test_function():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("router")
    fn.add_model("my", class_name="ModelTestingClass", model_path=".", multiplier=100)
    fn.set_tracking("dummy://")  # track using the _DummyStream

    server = fn.to_mock_server()
    logger.info(f"flow: {graph.to_yaml()}")
    resp = server.test("/v2/models/my/infer", testdata)
    # expected: source (5) * multiplier (100)
    assert resp["outputs"] == 5 * 100, f"wrong health response {resp}"

    dummy_stream = server.context.stream.output_stream
    assert len(dummy_stream.event_list) == 1, "expected stream to get one message"
