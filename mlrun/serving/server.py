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
import threading

from ..model import ModelObj
from ..platforms.iguazio import OutputStream
from .routers import ModelRouter
from .v2_serving import HttpTransport
from ..utils import create_class


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
    _dict_fields = ["parameters", "models", "router_class", "router_args", "verbose"]

    def __init__(self):
        self.router_class = None
        self.router_args = {}
        self.router = None
        self.parameters = {}
        self.models = {}
        self._models_handlers = {}
        self.verbose = False

    def add_root_params(self, params={}):
        for key, val in self.parameters.items():
            if key not in params:
                params[key] = val
        return params

    def init(self, context, namespace):
        for name, model in self.models.items():
            model_url = model.get("model_url", None)
            if model_url:
                transport = HttpTransport(model_url)
                self._models_handlers[name] = transport.do
            else:
                class_name = model["model_class"]
                model_path = model["model_path"]
                kwargs = model.get("params", None) or {}
                handler = model.get("handler", "do_event")
                mode = model.get("load_mode", "sync")
                class_object = get_class(class_name, namespace)
                model_object = class_object(context, name, model_path, **kwargs)
                if mode == "sync":
                    if not model_object.ready:
                        model_object.load()
                        model_object.ready = True
                    context.logger.info(f"model {name} was loaded")
                elif mode == "async":
                    t = threading.Thread(target=model_object.async_load)
                    t.start()
                    context.logger.info(f"started async load for model {name}")
                else:
                    raise ValueError(
                        f"unsupported model loading mode {mode} for model {name}"
                    )
                self._models_handlers[name] = getattr(model_object, handler)

        if self.router_class:
            router_class = get_class(self.router_class, namespace)
        else:
            router_class = ModelRouter
        router_args = self.router_args or {}
        self.router = router_class(context, self._models_handlers, **router_args)


def get_class(class_name, namespace):
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


def v2_serving_handler(context, event):
    try:
        response = context.router.do_event(event)
    except Exception as e:
        return context.Response(body=str(e), content_type="text/plain", status_code=400)

    body = response.body
    if isinstance(body, context.Response):
        return body

    if body and not isinstance(body, (str, bytes)):
        print(body)
        body = json.dumps(body)
        return context.Response(
            body=body, content_type="application/json", status_code=200
        )
    return body
