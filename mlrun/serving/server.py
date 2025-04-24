# Copyright 2023 Iguazio
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
import traceback
import uuid
from typing import Optional, Union

from nuclio import Context as NuclioContext
from nuclio.request import Logger as NuclioLogger

import mlrun
import mlrun.common.constants
import mlrun.common.helpers
import mlrun.model_monitoring
import mlrun.utils
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.secrets import SecretsStore

from ..common.helpers import parse_versioned_object_uri
from ..common.schemas.model_monitoring.constants import FileTargetKind
from ..datastore import get_stream_pusher
from ..datastore.store_resources import ResourceCache
from ..errors import MLRunInvalidArgumentError
from ..model import ModelObj
from ..utils import get_caller_globals
from .states import RootFlowStep, RouterStep, get_function, graph_root_setter
from .utils import event_id_key, event_path_key

DUMMY_STREAM = "dummy://"


class _StreamContext:
    """Handles the stream context for the events stream process. Includes the configuration for the output stream
    that will be used for pushing the events from the nuclio model serving function"""

    def __init__(self, enabled: bool, parameters: dict, function_uri: str):
        """
        Initialize _StreamContext object.
        :param enabled:      A boolean indication for applying the stream context
        :param parameters:   Dictionary of optional parameters, such as `log_stream` and `stream_args`. Note that these
                             parameters might be relevant to the output source such as `kafka_brokers` if
                             the output source is from type Kafka.
        :param function_uri: Full value of the function uri, usually it's <project-name>/<function-name>
        """

        self.enabled = False
        self.hostname = socket.gethostname()
        self.function_uri = function_uri
        self.output_stream = None
        stream_uri = None
        log_stream = parameters.get(FileTargetKind.LOG_STREAM, "")

        if (enabled or log_stream) and function_uri:
            self.enabled = True
            project, _, _, _ = parse_versioned_object_uri(
                function_uri, config.default_project
            )

            stream_args = parameters.get("stream_args", {})

            if log_stream == DUMMY_STREAM:
                # Dummy stream used for testing, see tests/serving/test_serving.py
                stream_uri = DUMMY_STREAM
            elif not stream_args.get("mock"):  # if not a mock: `context.is_mock = True`
                stream_uri = mlrun.model_monitoring.get_stream_path(project=project)

            if log_stream:
                # Update the stream path to the log stream value
                stream_uri = log_stream.format(project=project)
                self.output_stream = get_stream_pusher(stream_uri, **stream_args)
            else:
                # Get the output stream from the profile
                self.output_stream = mlrun.model_monitoring.helpers.get_output_stream(
                    project=project, mock=stream_args.get("mock", False)
                )


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
        tracking_policy=None,
        secret_sources=None,
        default_content_type=None,
        function_name=None,
        function_tag=None,
        project=None,
        model_endpoint_creation_task_name=None,
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
        self.tracking_policy = tracking_policy
        self._error_stream_object = None
        self.secret_sources = secret_sources
        self._secrets = SecretsStore.from_list(secret_sources)
        self._db_conn = None
        self.resource_cache = None
        self.default_content_type = default_content_type
        self.http_trigger = True
        self.function_name = function_name
        self.function_tag = function_tag
        self.project = project
        self.model_endpoint_creation_task_name = model_endpoint_creation_task_name

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
        monitoring_mock=False,
    ):
        """for internal use, initialize all steps (recursively)"""

        if self.secret_sources:
            self._secrets = SecretsStore.from_list(self.secret_sources)

        if self.error_stream:
            self._error_stream_object = get_stream_pusher(self.error_stream)
        self.resource_cache = resource_cache or ResourceCache()

        context = GraphContext(server=self, nuclio_context=context, logger=logger)
        context.is_mock = is_mock
        context.monitoring_mock = monitoring_mock
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

    def test(
        self,
        path: str = "/",
        body: Optional[Union[str, bytes, dict]] = None,
        method: str = "",
        headers: Optional[str] = None,
        content_type: Optional[str] = None,
        silent: bool = False,
        get_body: bool = True,
        event_id: Optional[str] = None,
        trigger: "MockTrigger" = None,
        offset=None,
        time=None,
    ):
        """invoke a test event into the server to simulate/test server behavior

        example::

            server = create_graph_server()
            server.add_model("my", class_name=MyModelClass, model_path="{path}", z=100)
            print(server.test("my/infer", testdata))

        :param path:       api path, e.g. (/{router.url_prefix}/{model-name}/..) path
        :param body:       message body (dict or json str/bytes)
        :param method:     optional, GET, POST, ..
        :param headers:    optional, request headers, ..
        :param content_type:  optional, http mime type
        :param silent:     don't raise on error responses (when not 20X)
        :param get_body:   return the body as py object (vs serialize response into json)
        :param event_id:   specify the unique event ID (by default a random value will be generated)
        :param trigger:    nuclio trigger info or mlrun.serving.server.MockTrigger class (holds kind and name)
        :param offset:     trigger offset (for streams)
        :param time:       event time Datetime or str, default to now()
        """
        if not self.graph:
            raise MLRunInvalidArgumentError(
                "no models or steps were set, use function.set_topology() and add steps"
            )
        if not method:
            method = "POST" if body else "GET"
        event = MockEvent(
            body=body,
            path=path,
            method=method,
            headers=headers,
            content_type=content_type,
            event_id=event_id,
            trigger=trigger,
            offset=offset,
            time=time,
        )
        resp = self.run(event, get_body=get_body)
        if hasattr(resp, "status_code") and resp.status_code >= 300 and not silent:
            raise RuntimeError(f"failed ({resp.status_code}): {resp.body}")
        return resp

    def run(self, event, context=None, get_body=False, extra_args=None):
        server_context = self.context
        context = context or server_context
        event.content_type = event.content_type or self.default_content_type or ""
        if event.headers:
            if event_id_key in event.headers:
                event.id = event.headers.get(event_id_key)
            if event_path_key in event.headers:
                event.path = event.headers.get(event_path_key)

        if isinstance(event.body, (str, bytes)) and (
            not event.content_type or event.content_type in ["json", "application/json"]
        ):
            # assume it is json and try to load
            try:
                body = json.loads(event.body)
                event.body = body
            except (json.decoder.JSONDecodeError, UnicodeDecodeError) as exc:
                if event.content_type in ["json", "application/json"]:
                    # if its json type and didnt load, raise exception
                    message = f"failed to json decode event, {err_to_str(exc)}"
                    context.logger.error(message)
                    server_context.push_error(event, message, source="_handler")
                    return context.Response(
                        body=message, content_type="text/plain", status_code=400
                    )
        try:
            response = self.graph.run(event, **(extra_args or {}))
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {err_to_str(exc)}"
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
        return self.graph.wait_for_completion()


def v2_serving_init(context, namespace=None):
    """hook for nuclio init_context()"""

    context.logger.info("Initializing server from spec")
    spec = mlrun.utils.get_serving_spec()
    server = GraphServer.from_dict(spec)

    if config.log_level.lower() == "debug":
        server.verbose = True
    if hasattr(context, "trigger"):
        server.http_trigger = getattr(context.trigger, "kind", "http") == "http"
    context.logger.info_with(
        "Setting current function",
        current_function=os.getenv("SERVING_CURRENT_FUNCTION", ""),
    )
    server.set_current_function(os.getenv("SERVING_CURRENT_FUNCTION", ""))
    context.logger.info_with(
        "Initializing states", namespace=namespace or get_caller_globals()
    )
    kwargs = {}
    if hasattr(context, "is_mock"):
        kwargs["is_mock"] = context.is_mock
    server.init_states(
        context,
        namespace or get_caller_globals(),
        **kwargs,
    )
    context.logger.info("Initializing graph steps")
    server.init_object(namespace or get_caller_globals())
    # set the handler hook to point to our handler
    setattr(context, "mlrun_handler", v2_serving_handler)
    setattr(context, "_server", server)
    context.logger.info_with("Serving was initialized", verbose=server.verbose)
    if server.verbose:
        context.logger.info(server.to_yaml())

    _set_callbacks(server, context)


def _set_callbacks(server, context):
    if not server.graph.supports_termination() or not hasattr(context, "platform"):
        return

    if hasattr(context.platform, "set_termination_callback"):
        context.logger.info(
            "Setting termination callback to terminate graph on worker shutdown"
        )

        async def termination_callback():
            context.logger.info("Termination callback called")
            maybe_coroutine = server.wait_for_completion()
            if asyncio.iscoroutine(maybe_coroutine):
                await maybe_coroutine
            context.logger.info("Termination of async flow is completed")

        context.platform.set_termination_callback(termination_callback)

    if hasattr(context.platform, "set_drain_callback"):
        context.logger.info(
            "Setting drain callback to terminate and restart the graph on a drain event (such as rebalancing)"
        )

        async def drain_callback():
            context.logger.info("Drain callback called")
            maybe_coroutine = server.wait_for_completion()
            if asyncio.iscoroutine(maybe_coroutine):
                await maybe_coroutine
            context.logger.info(
                "Termination of async flow is completed. Rerunning async flow."
            )
            # Rerun the flow without reconstructing it
            server.graph._run_async_flow()
            context.logger.info("Async flow restarted")

        context.platform.set_drain_callback(drain_callback)


def v2_serving_handler(context, event, get_body=False):
    """hook for nuclio handler()"""
    if context._server.http_trigger:
        # Workaround for a Nuclio bug where it sometimes passes b'' instead of None due to dirty memory
        if event.body == b"":
            event.body = None

    # original path is saved in stream_path so it can be used by explicit ack, but path is reset to / as a
    # workaround for NUC-178
    # nuclio 1.12.12 added the topic attribute, and we must use it as part of the fix for NUC-233
    # TODO: Remove fallback on event.path once support for nuclio<1.12.12 is dropped
    event.stream_path = getattr(event, "topic", event.path)
    if hasattr(event, "trigger") and event.trigger.kind in (
        "kafka",
        "kafka-cluster",
        "v3ioStream",
        "v3io-stream",
        "rabbit-mq",
        "rabbitMq",
    ):
        event.path = "/"

    return context._server.run(event, context, get_body)


def create_graph_server(
    parameters=None,
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
    parameters = parameters or {}
    server = GraphServer(graph, parameters, load_mode, verbose=verbose, **kwargs)
    server.set_current_function(
        current_function or os.getenv("SERVING_CURRENT_FUNCTION", "")
    )
    return server


class MockTrigger:
    """mock nuclio event trigger"""

    def __init__(self, kind="", name=""):
        self.kind = kind
        self.name = name


class MockEvent:
    """mock basic nuclio event object"""

    def __init__(
        self,
        body=None,
        content_type=None,
        headers=None,
        method=None,
        path=None,
        event_id=None,
        trigger: MockTrigger = None,
        offset=None,
        time=None,
    ):
        self.id = event_id or uuid.uuid4().hex
        self.key = ""
        self.body = body

        # optional
        self.headers = headers or {}
        self.method = method
        self.path = path or "/"
        self.content_type = content_type
        self.error = None
        self.trigger = trigger or MockTrigger()
        self.offset = offset or 0

    def __str__(self):
        error = f", error={self.error}" if self.error else ""
        return f"Event(id={self.id}, body={self.body}, method={self.method}, path={self.path}{error})"


class Response:
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

    def __init__(
        self,
        level="info",  # Unused argument
        logger=None,
        server=None,
        nuclio_context: Optional[NuclioContext] = None,
    ) -> None:
        self.state = None
        self.logger = logger
        self.worker_id = 0
        self.Response = Response
        self.verbose = False
        self.stream = None
        self.root = None

        if nuclio_context:
            self.logger: NuclioLogger = nuclio_context.logger
            self.Response = nuclio_context.Response
            if hasattr(nuclio_context, "trigger") and hasattr(
                nuclio_context.trigger, "kind"
            ):
                self.trigger = nuclio_context.trigger.kind
            self.worker_id = nuclio_context.worker_id
            if hasattr(nuclio_context, "platform"):
                self.platform = nuclio_context.platform
        elif not logger:
            self.logger: mlrun.utils.Logger = mlrun.utils.logger

        self._server = server
        self.current_function = None
        self.get_store_resource = None
        self.get_table = None
        self.is_mock = False
        self.monitoring_mock = False
        self._project_obj = None

    @property
    def server(self):
        return self._server

    @property
    def project_obj(self):
        if not self._project_obj:
            self._project_obj = mlrun.get_run_db().get_project(name=self.project)
        return self._project_obj

    @property
    def project(self) -> str:
        """current project name (for the current function)"""
        project, _, _, _ = mlrun.common.helpers.parse_versioned_object_uri(
            self._server.function_uri
        )
        return project

    def push_error(self, event, message, source=None, **kwargs):
        if self.verbose:
            self.logger.error(
                f"got error from {source} state:\n{event.body}\n{message}"
            )
        if self._server and self._server._error_stream_object:
            try:
                message = format_error(
                    self._server, self, source, event, message, kwargs
                )
                self._server._error_stream_object.push(message)
            except Exception as ex:
                message = traceback.format_exc()
                self.logger.error(f"failed to write to error stream: {ex}\n{message}")

    def get_param(self, key: str, default=None):
        if self._server and self._server.parameters:
            return self._server.parameters.get(key, default)
        return default

    def get_secret(self, key: str):
        if self._server and self._server._secrets:
            return self._server._secrets.get(key)
        return None

    def get_remote_endpoint(self, name, external=True):
        """return the remote nuclio/serving function http(s) endpoint given its name

        :param name: the function name/uri in the form [project/]function-name[:tag]
        :param external: return the external url (returns the external url by default)
        """
        if "://" in name:
            return name
        project, uri, tag, _ = mlrun.common.helpers.parse_versioned_object_uri(
            self._server.function_uri
        )
        if name.startswith("."):
            name = f"{uri}-{name[1:]}"
        else:
            project, name, tag, _ = mlrun.common.helpers.parse_versioned_object_uri(
                name, project
            )
        (
            state,
            fullname,
            _,
            _,
            _,
            function_status,
        ) = mlrun.runtimes.nuclio.function.get_nuclio_deploy_status(name, project, tag)

        if state in ["error", "unhealthy"]:
            raise ValueError(
                f"Nuclio function {fullname} is in error state, cannot be accessed"
            )

        key = "externalInvocationUrls" if external else "internalInvocationUrls"
        urls = function_status.get(key)
        if not urls:
            raise ValueError(f"cannot read {key} for nuclio function {fullname}")
        return f"http://{urls[0]}"


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
