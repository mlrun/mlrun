import json
import os
import sys
import uuid

from mlrun.serving.server import v2_serving_init
from mlrun.serving import V2ModelServer
from mlrun.utils import create_logger


class Response(object):
    def __init__(self, headers=None, body=None, content_type=None, status_code=200):
        self.headers = headers or {}
        self.body = body
        self.status_code = status_code
        self.content_type = content_type or "text/plain"

    def __repr__(self):
        cls = self.__class__.__name__
        items = self.__dict__.items()
        args = ("{}={!r}".format(key, value) for key, value in items)
        return "{}({})".format(cls, ", ".join(args))


class MockContext:
    def __init__(self):
        self.state = None
        self.logger = create_logger("debug", "human", "flow", sys.stdout)
        self.worker_id = 0
        self.Response = Response


class Event(object):
    def __init__(
        self, body=None, content_type=None, headers=None, method=None, path=None
    ):
        self.id = uuid.uuid4().hex
        self.key = ""
        self.body = body
        self.time = None

        # optional
        self.headers = headers or {}
        self.method = method
        self.path = path or "/"
        self.content_type = content_type
        self.trigger = None
        self.end = False
        self._trace_log = None

    def __str__(self):
        return f"Event(id={self.id}, body={self.body})"


spec = {  # "router_class": "ModelRouter",
    "models": {"m1": {"model_class": "MClass", "model_path": "", "params": {"z": 100}}}
}


class MClass(V2ModelServer):
    def load(self):
        print("loading")

    def predict(self, request):
        print("predict:", request)
        resp = request["data"][0] * self.get_param("z")
        print("resp:", resp)
        return resp


def test_v2_server():
    os.environ["MODELSRV_SPEC_ENV"] = json.dumps(spec)
    ctx = MockContext()
    v2_serving_init(ctx, globals())

    e = Event('{"data": [5]}', path="/v2/models/m1/infer")
    resp = ctx.mlrun_handler(ctx, e)
    print("resp:", resp)
    assert resp.body == "500", f"wrong model response {resp.body}"
