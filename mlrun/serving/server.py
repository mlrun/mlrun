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

from .states import ServingRouterState, ServingTaskState
from ..model import ModelObj
from ..platforms.iguazio import OutputStream
from ..utils import create_logger


class _StreamContext:
    def __init__(self, parameters, function_uri):
        self.hostname = socket.gethostname()
        self.output_stream = None
        self.function_uri = function_uri
        out_stream = parameters.get("log_stream", "")
        self.stream_sample = int(parameters.get("log_stream_sample", "1"))
        self.stream_batch = int(parameters.get("log_stream_batch", "1"))
        if out_stream:
            self.output_stream = OutputStream(out_stream)


# Model server host currently support a basic topology of single router + multiple
# routes (models/tasks). it will be enhanced later to support more complex topologies
class ModelServerHost(ModelObj):
    kind = "server"

    def __init__(
        self,
        graph=None,
        function_uri=None,
        parameters=None,
        load_mode=None,
        verbose=False,
        version=None,
    ):
        self._graph = None
        self.graph: ServingRouterState = graph
        self.function_uri = function_uri
        self.parameters = parameters or {}
        self.verbose = verbose
        self.load_mode = load_mode or "sync"
        self.version = version or "v2"
        self.context = None
        self._namespace = None

    @property
    def graph(self) -> ServingRouterState:
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "spec", ServingRouterState)

    def merge_root_params(self, params={}):
        """for internal use, enrich child states with root params"""
        for key, val in self.parameters.items():
            if key not in params:
                params[key] = val
        return params

    def init(self, context, namespace):
        """for internal use, initialize all states (recursively)"""
        self.context = context
        # enrich the context with classes and methods which will be used when
        # initializing classes or handling the event
        setattr(context, "stream", _StreamContext(self.parameters, self.function_uri))
        setattr(context, "merge_root_params", self.merge_root_params)
        setattr(context, "verbose", self.verbose)

        self.graph.init_object(context, namespace, self.load_mode)
        setattr(self.context, "root", self.graph)
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
        route = ServingTaskState(class_name, class_args, handler)
        self.graph.add_route(name, route).init_object(self.context, namespace)

    def test(
        self, path, body, method="", content_type=None, silent=False, get_body=True
    ):
        """invoke a test event into the server to simulate/test server behaviour

        e.g.:
                server = create_mock_server()
                server.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
                print(server.test("my/infer", testdata))

        :param path:     relative ({route-name}/..) or absolute (/{router.url_prefix}/{name}/..) path
        :param body:     message body (dict or json str/bytes)
        :param method:   optional, GET, POST, ..
        :param content_type:  optional, http mime type
        :param silent:   dont raise on error responses (when not 20X)
        :param get_body: return the body (vs serialize response into json)
        """
        if not self.graph:
            raise ValueError("no model or router was added, use .add_model()")
        if not path.startswith("/"):
            path = self.graph.object.url_prefix + path
        event = MockEvent(
            body=body, path=path, method=method, content_type=content_type
        )
        resp = v2_serving_handler(self.context, event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code > 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp


def v2_serving_init(context, namespace=None):
    data = os.environ.get("SERVING_SPEC_ENV", "")
    if not data:
        raise ValueError("failed to find spec env var")
    spec = json.loads(data)
    server = ModelServerHost.from_dict(spec)
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
    graph=None,
    namespace=None,
    logger=None,
    level="debug",
):
    """create serving emulator/tester for locally testing models and servers

        Usage:
                host = create_mock_server()
                host.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
                print(host.test("my/infer", testdata))
    """
    if not context:
        context = MockContext(level, logger=logger)

    if not graph:
        graph = ServingRouterState(class_name=router_class, class_args=router_args)
    server = ModelServerHost(graph, parameters, load_mode, verbose=level == "debug")
    server.init(context, namespace or {})
    return server


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

    def __init__(self, level="debug", logger=None):
        self.state = None
        self.logger = logger or create_logger(level, "human", "flow", sys.stdout)
        self.worker_id = 0
        self.Response = Response
