import json
import os
import time

from mlrun.runtimes import nuclio_init_hook
from mlrun.runtimes.serving import serving_subkind
from mlrun.serving import V2ModelServer
from mlrun.serving.server import MockEvent, MockContext, get_mock_server

spec = {
    "router_class": None,
    "models": {
        "m1": {"model_class": "MClass", "model_path": "", "params": {"z": 100}},
        "m2": {"model_class": "MClass", "model_path": "", "params": {"z": 200}},
        "m3:v1": {"model_class": "MClass", "model_path": "", "params": {"z": 300}},
        "m3:v2": {"model_class": "MClass", "model_path": "", "params": {"z": 400}},
    },
}

testdata = '{"inputs": [5]}'


class MClass(V2ModelServer):
    def load(self):
        print("loading")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"][0] * self.get_param("z")
        return resp

    def explain(self, request):
        print("predict:", request)
        resp = request["inputs"][0]
        return {"explained": resp}

    def op_myop(self, event):
        return event.body


class AsyncMClass(V2ModelServer):
    def load(self):
        print("loading..")
        time.sleep(4)
        print("loaded")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"][0]
        return resp


asyncspec = {
    "router_class": None,
    "models": {
        "m5": {"model_class": "AsyncMClass", "model_path": "", "load_mode": "async"},
    },
}


def init_ctx():
    os.environ["MODELSRV_SPEC_ENV"] = json.dumps(spec)
    context = MockContext()
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
        e = MockEvent(testdata, path=f"/v2/models/{url}/infer")
        resp = context.mlrun_handler(context, e)
        assert resp.body == str(expected), f"wrong model response {resp.body}"

    context = init_ctx()
    run_model("m1", 500)
    run_model("m2", 1000)
    run_model("m3/versions/v1", 1500)
    run_model("m3/versions/v2", 2000)


def test_v2_stream_mode():
    # model and operation are specified inside the message body
    context = init_ctx()
    e = MockEvent('{"model": "m2", "inputs": [5]}')
    resp = context.mlrun_handler(context, e)
    assert resp.body == str(1000), f"wrong model response {resp.body}"

    e = MockEvent('{"model": "m3:v2", "operation": "explain", "inputs": [5]}', path="")
    resp = context.mlrun_handler(context, e)
    data = json.loads(resp.body)
    assert data["explained"] == 5, f"wrong model response {resp.body}"


def test_v2_async_mode():
    # model loading is async
    os.environ["MODELSRV_SPEC_ENV"] = json.dumps(asyncspec)
    context = MockContext()
    nuclio_init_hook(context, globals(), serving_subkind)
    context.logger.info("model initialized")

    e = MockEvent("", path="/v2/models/m5/ready", method="GET")
    resp = context.mlrun_handler(context, e)
    assert (
        resp.status_code == 408
    ), f"didnt get proper ready resp, expected 408, got {resp.status_code}"

    e = MockEvent(testdata, path="/v2/models/m5/infer")
    resp = context.mlrun_handler(context, e)
    context.logger.info("model responded")
    print(resp)
    assert (
        resp.status_code != 200
    ), f"expected failure, got {resp.status_code} {resp.body}"

    e = MockEvent('{"model": "m5", "inputs": [5]}')
    e.trigger = "stream"
    resp = context.mlrun_handler(context, e)
    context.logger.info("model responded")
    print(resp)
    assert resp.body == str(5), f"wrong model response {resp.body}"


def test_v2_explain():
    context = init_ctx()
    e = MockEvent(testdata, path="/v2/models/m1/explain")
    resp = context.mlrun_handler(context, e)
    data = json.loads(resp.body)
    assert data["explained"] == 5, f"wrong explain response {resp.body}"


def test_v2_get_modelmeta():
    def get_model(name, version, url):
        e = MockEvent("", path=f"/v2/models/{url}", method="GET")
        resp = context.mlrun_handler(context, e)
        print(resp)
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
    e = MockEvent('{"test": "ok"}', path="/v2/models/m1/myop")
    resp = context.mlrun_handler(context, e)
    data = json.loads(resp.body)
    assert data["test"] == "ok", f"wrong custom op response {resp.body}"


def test_v2_errors():
    context = init_ctx()
    e = MockEvent('{"test": "ok"}', path="/v2/models/m1/xx")
    resp = context.mlrun_handler(context, e)
    # expected: 400, 'illegal model operation xx, method=None'
    assert resp.status_code == 400, f"didnt get proper handler error {resp.body}"

    e = MockEvent('{"test": "ok"}', path="/v2/models/m5/xx")
    resp = context.mlrun_handler(context, e)
    # expected: 400, 'model m5 doesnt exist, available models: m1 | m2 | m3:v1 | m3:v2'
    assert resp.status_code == 400, f"didnt get proper model error {resp.body}"


def test_v2_model_ready():
    context = init_ctx()
    e = MockEvent("", path="/v2/models/m1/ready", method="GET")
    resp = context.mlrun_handler(context, e)
    assert resp.status_code == 200, f"didnt get proper ready resp {resp.body}"


def test_v2_health():
    context = init_ctx()
    e = MockEvent(None, path="/", method="GET")
    resp = context.mlrun_handler(context, e)
    data = json.loads(resp.body)
    # expected: {'name': 'ModelRouter', 'version': 'v2', 'extensions': []}
    assert data["version"] == "v2", f"wrong health response {resp.body}"

    e = MockEvent("", path="/v2/health", method="GET")
    resp = context.mlrun_handler(context, e)
    data = json.loads(resp.body)
    assert data["version"] == "v2", f"wrong health response {resp.body}"


def test_v2_mock():
    host = get_mock_server()
    host.add_model("my", model_class=MClass, model_path="", params={"z": 100})
    print(host.test("my/infer", testdata))
