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

__all__ = ["GraphServer", "create_graph_server", "GraphContext", "MockEvent"]


import json
import os
import socket
import sys
import traceback
import uuid
from typing import Union

import mlrun
from mlrun.secrets import SecretsStore
from mlrun.config import config

from .states import (
    RouterState,
    RootFlowState,
    get_function,
    graph_root_setter,
)
from ..datastore.store_resources import ResourceCache
from ..errors import MLRunInvalidArgumentError
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


class GraphServer(ModelObj):
    kind = "server"

    def __init__(
        self,
        graph=None,
        parameters=None,
        load_mode=None,
        function_uri=None,
        verbose=False,
        version=None,
        functions=None,
        graph_initializer=None,
        error_stream=None,
    ):
        self._graph = None
        self.graph: Union[RouterState, RootFlowState] = graph
        self.function_uri = function_uri
        self.parameters = parameters or {}
        self.verbose = verbose
        self.load_mode = load_mode or "sync"
        self.version = version or "v2"
        self.context = None
        self._current_function = None
        self.functions = functions or {}
        self.graph_initializer = graph_initializer
        self.error_stream = error_stream
        self._error_stream_object = None
        self._secrets = SecretsStore()
        self._db_conn = None
        self.resource_cache = None

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
        """set/initialize the error notification stream"""
        self.error_stream = error_stream
        if error_stream:
            self._error_stream_object = OutputStream(error_stream)
        else:
            self._error_stream_object = None

    def _get_db(self):
        return mlrun.get_run_db(secrets=self._secrets)

    def init(
        self, context, namespace, resource_cache: ResourceCache = None, logger=None
    ):
        """for internal use, initialize all states (recursively)"""

        if self.error_stream:
            self._error_stream_object = OutputStream(self.error_stream)
        self.resource_cache = resource_cache or ResourceCache()
        context = GraphContext(server=self, nuclio_context=context, logger=logger)

        context.stream = _StreamContext(self.parameters, self.function_uri)
        context.current_function = self._current_function
        context.get_store_resource = self.resource_cache.resource_getter(
            self._get_db(), self._secrets
        )
        context.get_table = self.resource_cache.get_table
        context.verbose = self.verbose
        self.context = context

        if self.graph_initializer:
            if callable(self.graph_initializer):
                handler = self.graph_initializer
            else:
                handler = get_function(self.graph_initializer, namespace or [])
            handler(self)

        context.root = self.graph
        self.graph.init_object(context, namespace, self.load_mode, reset=True)
        return v2_serving_handler

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

        example::

            server = create_graph_server()
            server.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
            print(server.test("my/infer", testdata))

        :param path:       api path, e.g. (/{router.url_prefix}/{model-name}/..) path
        :param body:       message body (dict or json str/bytes)
        :param method:     optional, GET, POST, ..
        :param content_type:  optional, http mime type
        :param silent:     dont raise on error responses (when not 20X)
        :param get_body:   return the body as py object (vs serialize response into json)
        """
        if not self.graph:
            raise MLRunInvalidArgumentError(
                "no models or steps were set, use function.set_topology() and add steps"
            )
        event = MockEvent(
            body=body, path=path, method=method, content_type=content_type
        )
        resp = v2_serving_handler(self.context, event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code >= 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp

    def wait_for_completion(self):
        """wait for async operation to complete"""
        self.graph.wait_for_completion()


def v2_serving_init(context, namespace=None):
    """hook for nuclio init_context()"""

    data = os.environ.get("SERVING_SPEC_ENV", "")
    if not data:
        raise MLRunInvalidArgumentError("failed to find spec env var")
    spec = json.loads(data)
    server = GraphServer.from_dict(spec)
    if config.log_level.lower() == "debug":
        server.verbose = True
    server.set_current_function(os.environ.get("SERVING_CURRENT_FUNCTION", ""))
    serving_handler = server.init(context, namespace or get_caller_globals())
    # set the handler hook to point to our handler
    setattr(context, "mlrun_handler", serving_handler)
    setattr(context, "root", server.graph)
    context.logger.info(f"serving was initialized, verbose={server.verbose}")
    if server.verbose:
        context.logger.info(server.to_yaml())


def v2_serving_handler(context, event, get_body=False):
    """hook for nuclio handler()"""

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
    parameters={},
    load_mode=None,
    graph=None,
    verbose=False,
    current_function=None,
    **kwargs,
) -> GraphServer:
    """create graph server host/emulator for local or test runs

    Usage example::

        server = create_graph_server(graph=RouterState(), parameters={})
        server.init(None, globals())
        server.graph.add_route("my", class_name=MyModelClass, model_path="{path}", z=100)
        print(server.test("/v2/models/my/infer", testdata))
    """
    server = GraphServer(graph, parameters, load_mode, verbose=verbose, **kwargs)
    server.set_current_function(
        current_function or os.environ.get("SERVING_CURRENT_FUNCTION", "")
    )
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
    """Graph context object"""

    def __init__(self, level="debug", logger=None, server=None, nuclio_context=None):
        self.state = None
        self.logger = logger
        self.worker_id = 0
        self.Response = Response
        self.verbose = False
        self.stream = None
        self.root = None

        if nuclio_context:
            self.logger = nuclio_context.logger
            self.Response = nuclio_context.Response
            self.worker_id = nuclio_context.worker_id
        elif not logger:
            self.logger = create_logger(level, "human", "flow", sys.stdout)

        self._server = server
        self.current_function = None
        self.get_store_resource = None
        self.get_table = None

    def push_error(self, event, message, source=None, **kwargs):
        if self.verbose:
            self.logger.error(
                f"got error from {source} state:\n{event.body}\n{message}"
            )
        if self._server and self._server._error_stream_object:
            message = format_error(self._server, self, source, event, message, kwargs)
            self._server._error_stream_object.push(message)

    def get_param(self, key: str, default=None):
        if self._server and self._server.parameters:
            return self._server.parameters.get(key, default)
        return default

    def get_secret(self, key: str):
        if self._server and self._server._secrets:
            return self._secrets.get(key)
        return None


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
