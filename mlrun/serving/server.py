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

import asyncio
import json
import os
import socket
import sys
import traceback
import uuid
from typing import Union

import mlrun
from mlrun.config import config
from mlrun.secrets import SecretsStore

from ..datastore import get_stream_pusher
from ..datastore.store_resources import ResourceCache
from ..errors import MLRunInvalidArgumentError
from ..model import ModelObj
from ..utils import create_logger, get_caller_globals, parse_versioned_object_uri
from .states import RootFlowStep, RouterStep, get_function, graph_root_setter


class _StreamContext:
    def __init__(self, enabled, parameters, function_uri):
        self.enabled = False
        self.hostname = socket.gethostname()
        self.function_uri = function_uri
        self.output_stream = None
        self.stream_uri = None

        log_stream = parameters.get("log_stream", "")
        stream_uri = config.model_endpoint_monitoring.store_prefixes.default

        if ((enabled and stream_uri) or log_stream) and function_uri:
            self.enabled = True

            project, _, _, _ = parse_versioned_object_uri(
                function_uri, config.default_project
            )

            stream_uri = stream_uri.format(project=project, kind="stream")

            if log_stream:
                stream_uri = log_stream.format(project=project)

            stream_args = parameters.get("stream_args", {})

            self.stream_uri = stream_uri

            self.output_stream = get_stream_pusher(stream_uri, **stream_args)


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
        track_models=None,
        secret_sources=None,
        default_content_type=None,
    ):
        self._graph = None
        self.graph: Union[RouterStep, RootFlowStep] = graph
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
        self.track_models = track_models
        self._error_stream_object = None
        self.secret_sources = secret_sources
        self._secrets = SecretsStore.from_list(secret_sources)
        self._db_conn = None
        self.resource_cache = None
        self.default_content_type = default_content_type
        self.http_trigger = True

    def set_current_function(self, function):
        """set which child function this server is currently running on"""
        self._current_function = function

    @property
    def graph(self) -> Union[RootFlowStep, RouterStep]:
        return self._graph

    @graph.setter
    def graph(self, graph):
        graph_root_setter(self, graph)

    def set_error_stream(self, error_stream):
        """set/initialize the error notification stream"""
        self.error_stream = error_stream
        if error_stream:
            self._error_stream_object = get_stream_pusher(error_stream)
        else:
            self._error_stream_object = None

    def _get_db(self):
        return mlrun.get_run_db(secrets=self._secrets)

    def init_states(
        self,
        context,
        namespace,
        resource_cache: ResourceCache = None,
        logger=None,
        is_mock=False,
    ):
        """for internal use, initialize all steps (recursively)"""

        if self.secret_sources:
            self._secrets = SecretsStore.from_list(self.secret_sources)

        if self.error_stream:
            self._error_stream_object = get_stream_pusher(self.error_stream)
        self.resource_cache = resource_cache or ResourceCache()

        context = GraphContext(server=self, nuclio_context=context, logger=logger)
        context.is_mock = is_mock
        context.root = self.graph

        context.stream = _StreamContext(
            self.track_models, self.parameters, self.function_uri
        )
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

    def init_object(self, namespace):
        self.graph.init_object(self.context, namespace, self.load_mode, reset=True)
        return (
            v2_serving_async_handler
            if config.datastore.async_source_mode == "enabled"
            else v2_serving_handler
        )

    def test(
        self,
        path="/",
        body=None,
        method="",
        content_type=None,
        silent=False,
        get_body=True,
    ):
        """invoke a test event into the server to simulate/test server behavior

        example::

            server = create_graph_server()
            server.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
            print(server.test("my/infer", testdata))

        :param path:       api path, e.g. (/{router.url_prefix}/{model-name}/..) path
        :param body:       message body (dict or json str/bytes)
        :param method:     optional, GET, POST, ..
        :param content_type:  optional, http mime type
        :param silent:     don't raise on error responses (when not 20X)
        :param get_body:   return the body as py object (vs serialize response into json)
        """
        if not self.graph:
            raise MLRunInvalidArgumentError(
                "no models or steps were set, use function.set_topology() and add steps"
            )
        event = MockEvent(
            body=body, path=path, method=method, content_type=content_type
        )
        resp = self.run(event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code >= 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp

    def run(self, event, context=None, get_body=False, extra_args=None):
        server_context = self.context
        context = context or server_context
        event.content_type = event.content_type or self.default_content_type or ""
        if isinstance(event.body, (str, bytes)) and (
            not event.content_type or event.content_type in ["json", "application/json"]
        ):
            # assume it is json and try to load
            try:
                body = json.loads(event.body)
                event.body = body
            except json.decoder.JSONDecodeError as exc:
                if event.content_type in ["json", "application/json"]:
                    # if its json type and didnt load, raise exception
                    message = f"failed to json decode event, {exc}"
                    context.logger.error(message)
                    server_context.push_error(event, message, source="_handler")
                    return context.Response(
                        body=message, content_type="text/plain", status_code=400
                    )
        try:
            response = self.graph.run(event, **(extra_args or {}))
        except Exception as exc:
            message = str(exc)
            if server_context.verbose:
                message += "\n" + str(traceback.format_exc())
            context.logger.error(f"run error, {traceback.format_exc()}")
            server_context.push_error(event, message, source="_handler")
            return context.Response(
                body=message, content_type="text/plain", status_code=400
            )

        if asyncio.iscoroutine(response):
            return self._process_async_response(context, response, get_body)
        else:
            return self._process_response(context, response, get_body)

    async def _process_async_response(self, context, response, get_body):
        return self._process_response(context, await response, get_body)

    def _process_response(self, context, response, get_body):
        body = response.body
        if isinstance(body, context.Response) or get_body:
            return body

        if body and not isinstance(body, (str, bytes)):
            body = json.dumps(body)
            return context.Response(
                body=body, content_type="application/json", status_code=200
            )
        return body

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
    if hasattr(context, "trigger"):
        server.http_trigger = getattr(context.trigger, "kind", "http") == "http"
    server.set_current_function(os.environ.get("SERVING_CURRENT_FUNCTION", ""))
    server.init_states(context, namespace or get_caller_globals())
    serving_handler = server.init_object(namespace or get_caller_globals())
    # set the handler hook to point to our handler
    setattr(context, "mlrun_handler", serving_handler)
    setattr(context, "server", server)
    context.logger.info(f"serving was initialized, verbose={server.verbose}")
    if server.verbose:
        context.logger.info(server.to_yaml())


def v2_serving_handler(context, event, get_body=False):
    """hook for nuclio handler()"""
    if not context.server.http_trigger:
        event.path = "/"  # fix the issue that non http returns "Unsupported"
    return context.server.run(event, context, get_body)


async def v2_serving_async_handler(context, event, get_body=False):
    """hook for nuclio handler()"""
    return await context.server.run(event, context, get_body)


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

        server = create_graph_server(graph=RouterStep(), parameters={})
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
        args = (f"{key}={repr(value)}" for key, value in items)
        args_str = ", ".join(args)
        return f"{cls}({args_str})"


class GraphContext:
    """Graph context object"""

    def __init__(self, level="info", logger=None, server=None, nuclio_context=None):
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
        self.is_mock = False

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
            return self._server._secrets.get(key)
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
