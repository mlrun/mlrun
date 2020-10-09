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
import traceback
import uuid
from copy import deepcopy

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

from ..model import ModelObj
from ..platforms.iguazio import OutputStream
from .routers import ModelRouter
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


_task_state_fields = ["kind", "class_name", "class_args", "handler"]


class MiniTaskState(ModelObj):
    kind = "task"
    _dict_fields = _task_state_fields

    def __init__(self, name=None, class_name=None, class_args=None, handler=None):
        self.name = name
        self.class_name = class_name
        self.class_args = class_args or {}
        self.handler = handler
        self._handler = None
        self._object = None
        self.context = None
        self._class_object = None

    def init_object(self, context, namespace, mode):
        if isinstance(self.class_name, type):
            self._class_object = self.class_name
            self.class_name = self.class_name.__name__

        self.context = context
        if not self._class_object:
            if self.class_name == "$remote":
                self._class_object = RemoteHttpHandler
            else:
                self._class_object = get_class(self.class_name, namespace)

        if not self._object:
            print(self.class_args)
            self._object = self._class_object(context, self.name, **self.class_args)
            self._handler = getattr(self._object, self.handler or "do_event")

        if mode != "skip":
            self._post_init(mode)

    @property
    def object(self):
        return self._object

    def _post_init(self, mode="sync"):
        if self._object and hasattr(self._object, "post_init"):
            self._object.post_init(mode)

    def run(self, event, *args, **kwargs):
        return self._handler(event, *args, **kwargs)


class MiniRouterState(MiniTaskState):
    kind = "router"
    _dict_fields = _task_state_fields + ["routes"]

    def __init__(
        self, name=None, class_name=None, class_args=None, handler=None, routes=None
    ):
        super().__init__(name, class_name, class_args, handler)
        self._routes = {}
        self.routes = routes

    @property
    def routes(self):
        return {name: route.to_dict() for name, route in self._routes.items()}

    @routes.setter
    def routes(self, routes: dict):
        if not routes:
            return
        _routes = {}
        for name, route in routes.items():
            if isinstance(route, dict):
                route = MiniTaskState.from_dict(route)
            elif not hasattr(route, "to_dict"):
                raise ValueError("route must be a dict or state object")
            route.name = name
            _routes[name] = route
        self._routes = _routes

    def add_route(self, route):
        self._routes[route.name] = route

    def init_object(self, context, namespace, mode):
        self.class_name = self.class_name or ModelRouter
        self.class_args = self.class_args or {}
        self.class_args["routes"] = self._routes
        super().init_object(context, namespace, "skip")
        del self.class_args["routes"]

        for route in self._routes.values():
            route.init_object(context, namespace, mode)

        self._post_init(mode)

    def __getitem__(self, name):
        return self._routes[name]


# Model server host currently support a basic topology of single router + multiple
# routes (models/tasks). it will be enhanced later to support more complex topologies
class ModelServerHost(ModelObj):
    def __init__(self, router=None, parameters=None, load_mode=None, verbose=False):
        self.router: MiniRouterState = router
        self.parameters = parameters or {}
        self.verbose = verbose
        self.load_mode = load_mode or "sync"
        self.context = None
        self._namespace = None

    @classmethod
    def from_dict(cls, struct=None, fields=None):
        states = struct["states"]
        router = list(states.values())[0]
        return cls(
            MiniRouterState.from_dict(router),
            parameters=struct.get("parameters", None),
            load_mode=struct.get("load_mode", None),
            verbose=struct.get("verbose", None),
        )

    def to_dict(self, fields=None, exclude=None):
        return {
            "version": "v2",
            "parameters": self.parameters,
            "states": {"router": self.router.to_dict()},
            "load_mode": self.load_mode,
            "verbose": self.verbose,
        }

    def merge_root_params(self, params={}):
        """for internal use, enrich child states with root params"""
        for key, val in self.parameters.items():
            if key not in params:
                params[key] = val
        return params

    def init(self, context, namespace):
        """for internal use, initialize all states (recursively)"""
        self.context = context
        self.router.init_object(context, namespace, self.load_mode)
        setattr(self.context, "root", self.router)
        return v2_serving_handler

    def add_model(
        self, name, class_name, model_path, handler=None, namespace=None, **class_args
    ):
        """add child model/route to the server, will register, init and connect the child class
        the local or global (module.submodule.class) class specified by the class_name
        the context, name, model_path, and **class_args will be used to initialize that class

        every event with "/{router.url_prefix}/{name}/.." or "{name}/.." will be routed to the class.

        keep the handler=None for model server classes, for custom classes you can specify the class handler
        which will be invoked when traffic arrives to that route (class.{handler}(event))

        :param name:        name (and url prefix) used for the route/model
        :param class_name:  class object or name (str) or full path (module.submodule.class)
        :param model_path:  path to mlrun model artifact or model directory file/object path
        :param handler:     for advanced users!, override default class handler name (do_event)
        :param namespace:   class search path when using string_name, for local use py globals()
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        class_args = deepcopy(class_args)
        class_args["model_path"] = model_path
        route = MiniTaskState(name, class_name, class_args, handler)
        route.init_object(self.context, namespace, "sync")
        self.router.add_route(route)

    def test(
        self, path, body, method="", content_type=None, silent=False, get_body=True
    ):
        """invoke a test event into the server to simulate/test server behaviour

        e.g.:
                host = create_mock_server()
                host.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
                print(host.test("my/infer", testdata))


        :param path:     relative ({route-name}/..) or absolute (/{router.url_prefix}/{name}/..) path
        :param body:     message body (dict or json str/bytes)
        :param method:   optional, GET, POST, ..
        :param content_type:  optional, http mime type
        :param silent:   dont raise on error responses (when not 20X)
        :param get_body: return the body (vs serialize response into json)
        """
        if not self.router:
            raise ValueError("no model or router was added, use .add_model()")
        if not path.startswith("/"):
            path = self.router.object.url_prefix + path
        event = MockEvent(
            body=body, path=path, method=method, content_type=content_type
        )
        resp = v2_serving_handler(self.context, event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code > 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp


def get_class(class_name, namespace):
    """return class object from class name string"""
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
    data = os.environ.get("SERVING_SPEC_ENV", "")
    if not data:
        raise ValueError("failed to find spec env var")
    spec = json.loads(data)
    server = ModelServerHost.from_dict(spec)

    # enrich the context with classes and methods which will be used when
    # initializing classes or handling the event
    setattr(context, "server", _ServerContext(server.parameters))
    setattr(context, "merge_root_params", server.merge_root_params)
    setattr(context, "verbose", server.verbose)

    serving_handler = server.init(context, namespace or globals())
    # set the handler hook to point to our handler
    setattr(context, "mlrun_handler", serving_handler)


def v2_serving_handler(context, event, get_body=False):
    try:
        response = context.root.run(event)
    except Exception as e:
        if context.verbose:
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


def create_mock_server(
    context=None,
    router_class=None,
    router_args={},
    parameters={},
    load_mode=None,
    level="debug",
):
    """create serving emulator/tester for locally testing models and servers

        Usage:
                host = create_mock_server()
                host.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
                print(host.test("my/infer", testdata))
    """
    if not context:
        context = MockContext(level)

    router = MiniRouterState(class_name=router_class, class_args=router_args)
    host = ModelServerHost(router, parameters, load_mode, verbose=level == "debug")
    host.init(context, {})

    setattr(host.context, "server", _ServerContext(host.parameters))
    setattr(host.context, "merge_root_params", host.merge_root_params)
    setattr(host.context, "verbose", host.verbose)
    return host


class MockEvent(object):
    """mock basic nuclio event object"""

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
    """mock basic nuclio context object"""

    def __init__(self, level="debug"):
        self.state = None
        self.logger = create_logger(level, "human", "flow", sys.stdout)
        self.worker_id = 0
        self.Response = Response


http_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)


class RemoteHttpHandler:
    """class for calling remote endpoints"""

    def __init__(self, url):
        self.url = url
        self.format = "json"
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._session.mount("https://", http_adapter)

    def do_event(self, event):
        kwargs = {}
        kwargs["headers"] = event.headers or {}
        method = event.method or "POST"
        if method != "GET":
            if isinstance(event.body, (str, bytes)):
                kwargs["data"] = event.body
            else:
                kwargs["json"] = event.body

        url = self.url.strip("/") + event.path
        try:
            resp = self._session.request(method, url, verify=False, **kwargs)
        except OSError as err:
            raise OSError(f"error: cannot run function at url {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.text}")

        data = resp.content
        if self.format == "json" or resp.headers["content-type"] == "application/json":
            data = json.loads(data)
        event.body = data
        return event
