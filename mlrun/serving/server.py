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
import json
import os
import socket
import sys
import threading
import traceback
import uuid

from ..model import ModelObj
from ..platforms.iguazio import OutputStream
from .routers import ModelRouter
from .v2_serving import HttpTransport
from ..utils import create_class, create_logger


class _ServerContext:
    def __init__(self, parameters):
        self.hostname = socket.gethostname()
        self.output_stream = None
        out_stream = parameters.get("log_stream", "")
        self.stream_sample = int(parameters.get("log_stream_sample", "1"))
        self.stream_batch = int(parameters.get("log_stream_batch", "1"))
        if out_stream:
            self.output_stream = OutputStream(out_stream)


class ModelServerHost(ModelObj):
    _dict_fields = [
        "parameters",
        "models",
        "router_class",
        "router_args",
        "verbose",
        "load_mode",
    ]

    def __init__(self):
        self.router_class = None
        self._router_class = None
        self.router_args = {}
        self.router = None
        self.parameters = {}
        self.models = {}
        self._models_handlers = {}
        self.verbose = False
        self.load_mode = None
        self.context = None

    def add_root_params(self, params={}):
        """for internal use"""
        for key, val in self.parameters.items():
            if key not in params:
                params[key] = val
        return params

    def init(self, context, namespace, init_router=True):
        """for internal use"""
        self.context = context
        self.load_mode = self.load_mode or "sync"
        for name, model in self.models.items():
            self._add_model(name, model, namespace)

        if self.router_class:
            self._router_class = get_class(self.router_class, namespace)
        else:
            self._router_class = ModelRouter
        if init_router:
            self._init_router()

    def add_model(self, name, model_class, model_path, params=None, namespace=None):
        model = {"model_class": model_class, "model_path": model_path, "params": params}
        self._add_model(name, model, namespace)
        self._init_router()

    def _init_router(self):
        router_args = self.router_args or {}
        self.router = self._router_class(
            self.context, self._models_handlers, **router_args
        )
        setattr(self.context, "router", self.router)

    def _add_model(self, name, model, namespace=None):
        model_url = model.get("model_url", None)
        if model_url:
            transport = HttpTransport(model_url)
            self._models_handlers[name] = transport.do
        else:
            class_name = model["model_class"]
            model_path = model["model_path"]
            kwargs = model.get("params", None) or {}
            handler = model.get("handler", "do_event")
            class_object = get_class(class_name, namespace)
            model_object = class_object(self.context, name, model_path, **kwargs)
            if self.load_mode == "sync":
                if not model_object.ready:
                    model_object.load()
                    model_object.ready = True
                self.context.logger.info(f"model {name} was loaded")
            elif self.load_mode == "async":
                t = threading.Thread(target=model_object.async_load)
                t.start()
                self.context.logger.info(f"started async load for model {name}")
            else:
                raise ValueError(
                    f"unsupported model loading mode {self.load_mode} for model {name}"
                )
            self._models_handlers[name] = getattr(model_object, handler)

    def test(
        self, path, body, method="", content_type=None, silent=False, get_body=True
    ):
        if not self.router:
            raise ValueError("no model or router was added, use .add_model()")
        if not path.startswith("/"):
            path = self.router.url_prefix + path
        event = MockEvent(
            body=body, path=path, method=method, content_type=content_type
        )
        resp = v2_serving_handler(self.context, event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code > 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp


def get_class(class_name, namespace):
    if isinstance(class_name, type):
        return class_name
    if class_name in namespace:
        class_object = namespace[class_name]
        return class_object

    try:
        class_object = create_class(class_name)
    except (ImportError, ValueError) as e:
        raise ImportError(f"state init failed, class {class_name} not found, {e}")
    return class_object


def v2_serving_init(context, namespace=None):
    data = os.environ.get("MODELSRV_SPEC_ENV", "")
    if not data:
        raise ValueError("failed to find spec env var")
    spec = json.loads(data)
    server = ModelServerHost.from_dict(spec)

    setattr(context, "server", _ServerContext(server.parameters))
    setattr(context, "add_root_params", server.add_root_params)
    setattr(context, "trace", server.verbose)
    server.init(context, namespace or globals())
    setattr(context, "router", server.router)
    setattr(context, "mlrun_handler", v2_serving_handler)


def v2_serving_handler(context, event, get_body=False):
    try:
        response = context.router.do_event(event)
    except Exception as e:
        if context.trace:
            context.logger.error(traceback.format_exc())
        return context.Response(body=str(e), content_type="text/plain", status_code=400)

    body = response.body
    if isinstance(body, context.Response) or get_body:
        return body

    if body and not isinstance(body, (str, bytes)):
        body = json.dumps(body)
        return context.Response(
            body=body, content_type="application/json", status_code=200
        )
    return body


def get_mock_server(
    context=None,
    router_class=None,
    router_args={},
    parameters={},
    load_mode=None,
    level="debug",
):
    if not context:
        context = MockContext(level)
    host = ModelServerHost()
    host.router_class = router_class
    host.router_args = router_args
    host.parameters = parameters
    host.load_mode = load_mode
    host.verbose = level == "debug"
    host.init(context, {}, init_router=False)

    setattr(host.context, "server", _ServerContext(host.parameters))
    setattr(host.context, "add_root_params", host.add_root_params)
    setattr(host.context, "trace", host.verbose)
    return host


class MockEvent(object):
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

    def __str__(self):
        return f"Event(id={self.id}, body={self.body}, method={self.method}, path={self.path})"


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
    def __init__(self, level="debug"):
        self.state = None
        self.logger = create_logger(level, "human", "flow", sys.stdout)
        self.worker_id = 0
        self.Response = Response
