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
from typing import Union

from mlrun.config import config

from .states import (
    RouterState,
    TaskState,
    RootFlowState,
    get_function,
    graph_root_setter,
)
from ..model import ModelObj
from ..platforms.iguazio import OutputStream
from ..utils import create_logger, get_caller_globals


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
class GraphServer(ModelObj):
    kind = "server"

    def __init__(
        self,
        graph=None,
        function_uri=None,
        parameters=None,
        load_mode=None,
        verbose=False,
        version=None,
        functions=None,
        graph_initializer=None,
        error_stream=None,
    ):
        self._graph = None
        self.graph: RouterState = graph
        self.function_uri = function_uri
        self.parameters = parameters or {}
        self.verbose = verbose
        self.load_mode = load_mode or "sync"
        self.version = version or "v2"
        self.context = None
        self._namespace = None
        self._current_function = None
        self.functions = functions or []
        self.graph_initializer = graph_initializer
        self.error_stream = error_stream
        self._error_stream = None

    def set_current_function(self, function):
        """set which child function this server is currently running on"""
        self._current_function = function

    @property
    def graph(self) -> Union[RootFlowState, RouterState]:
        return self._graph

    @graph.setter
    def graph(self, graph):
        graph_root_setter(self, graph)

    def set_error_stream(self, error_stream):
        self.error_stream = error_stream
        if error_stream:
            self._error_stream = OutputStream(error_stream)
        else:
            self._error_stream = None

    def init(self, context, namespace):
        """for internal use, initialize all states (recursively)"""
        self.context = context
        # enrich the context with classes and methods which will be used when
        # initializing classes or handling the event

        if self.error_stream:
            self._error_stream = OutputStream(self.error_stream)

        def push_error(event, message, source=None, **kwargs):
            if self._error_stream:
                message = format_error(self, context, source, event, message, kwargs)
                self._error_stream.push(message)

        def get_param(self, key: str, default=None):
            if self.parameters:
                return self.parameters.get(key, default)
            return default

        setattr(context, "stream", _StreamContext(self.parameters, self.function_uri))
        setattr(context, "current_function", self._current_function)
        setattr(context, "get_param", get_param)
        setattr(context, "push_error", push_error)
        setattr(context, "verbose", self.verbose)
        setattr(context, "root", self.graph)

        if self.graph_initializer:
            handler = get_function(self.graph_initializer, namespace)
            handler(self)

        self.graph.init_object(context, namespace, self.load_mode)
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
        route = TaskState(class_name, class_args, handler)
        namespace = namespace or get_caller_globals()
        self.graph.add_route(name, route).init_object(self.context, namespace)

    def test(
        self,
        path="/",
        body=None,
        method="",
        content_type=None,
        silent=False,
        get_body=True,
    ):
        """invoke a test event into the server to simulate/test server behaviour

        e.g.:
                server = create_graph_server()
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
        if path and not path.startswith("/"):
            path = self.graph.object.url_prefix + path
        event = MockEvent(
            body=body, path=path, method=method, content_type=content_type
        )
        resp = v2_serving_handler(self.context, event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code > 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp

    def wait_for_completion(self):
        self.graph.wait_for_completion()


def v2_serving_init(context, namespace=None):
    data = os.environ.get("SERVING_SPEC_ENV", "")
    if not data:
        raise ValueError("failed to find spec env var")
    spec = json.loads(data)
    server = GraphServer.from_dict(spec)
    if config.log_level.lower() == "debug":
        server.verbose = True
    server.set_current_function(os.environ.get("SERVING_CURRENT_FUNCTION", ""))
    serving_handler = server.init(context, namespace or get_caller_globals())
    # set the handler hook to point to our handler
    setattr(context, "mlrun_handler", serving_handler)
    context.logger.info(f"serving was initialized, verbose={server.verbose}")
    if server.verbose:
        context.logger.info(server.to_yaml())


def v2_serving_handler(context, event, get_body=False):
    try:
        response = context.root.run(event)
    except Exception as e:
        message = str(e)
        if context.verbose:
            message += "\n" + str(traceback.format_exc())
        context.logger.error(f"run error, {traceback.format_exc()}")
        context.push_error(event, message, source="_handler")
        return context.Response(
            body=message, content_type="text/plain", status_code=400
        )

    body = response.body
    if isinstance(body, context.Response) or get_body:
        return body

    if body and not isinstance(body, (str, bytes)):
        body = json.dumps(body)
        return context.Response(
            body=body, content_type="application/json", status_code=200
        )
    return body


def create_graph_server(
    context=None,
    router_class=None,
    router_args={},
    parameters={},
    load_mode=None,
    graph=None,
    namespace=None,
    logger=None,
    level="debug",
    current_function=None,
    **kwargs,
) -> GraphServer:
    """create serving emulator/tester for locally testing models and servers

        Usage:
                host = create_graph_server()
                host.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
                print(host.test("my/infer", testdata))
    """
    if not context:
        context = GraphContext(level, logger=logger)

    if not graph:
        graph = RouterState(class_name=router_class, class_args=router_args)
    namespace = namespace or get_caller_globals()
    server = GraphServer(
        graph, parameters, load_mode, verbose=level == "debug", **kwargs
    )
    server.set_current_function(
        current_function or os.environ.get("SERVING_CURRENT_FUNCTION", "")
    )
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
        self.error = None

    def __str__(self):
        error = f", error={self.error}" if self.error else ""
        return f"Event(id={self.id}, body={self.body}, method={self.method}, path={self.path}{error})"


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


class GraphContext:
    """mock basic nuclio context object"""

    def __init__(self, level="debug", logger=None):
        self.state = None
        self.logger = logger or create_logger(level, "human", "flow", sys.stdout)
        self.worker_id = 0
        self.Response = Response
        self.verbose = False


def format_error(server, context, source, event, message, args):
    return {
        "function_uri": server.function_uri,
        "worker": context.worker_id,
        "host": socket.gethostname(),
        "source": source,
        "event": {"id": event.id, "body": event.body},
        "message": message,
        "args": args,
    }
