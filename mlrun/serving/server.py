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
import base64
import copy
import importlib
import json
import os
import socket
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional, Union

import pandas as pd
import storey
from nuclio import Context as NuclioContext
from nuclio.request import Logger as NuclioLogger

import mlrun
import mlrun.common.helpers
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.datastore.datastore_profile as ds_profile
import mlrun.model_monitoring
import mlrun.utils
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.secrets import SecretsStore

from ..common.helpers import parse_versioned_object_uri
from ..common.schemas.model_monitoring.constants import FileTargetKind
from ..common.schemas.serving import MAX_BATCH_JOB_DURATION
from ..datastore import DataItem, get_stream_pusher
from ..datastore.store_resources import ResourceCache
from ..errors import MLRunInvalidArgumentError
from ..execution import MLClientCtx
from ..model import ModelObj
from ..utils import get_caller_globals, get_relative_module_name_from_path
from .states import (
    FlowStep,
    MonitoredStep,
    RootFlowStep,
    RouterStep,
    get_function,
    graph_root_setter,
)
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
        log_stream = parameters.get(FileTargetKind.LOG_STREAM, "")

        if (enabled or log_stream) and function_uri:
            self.enabled = True
            project, _, _, _ = parse_versioned_object_uri(
                function_uri, config.active_project
            )

            stream_args = parameters.get("stream_args", {})

            if log_stream:
                # Get the output stream from the log stream path
                stream_path = log_stream.format(project=project)
                self.output_stream = get_stream_pusher(stream_path, **stream_args)
            else:
                # Get the output stream from the profile
                self.output_stream = mlrun.model_monitoring.helpers.get_output_stream(
                    project=project,
                    profile=parameters.get("stream_profile"),
                    mock=stream_args.get("mock", False),
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
        resource_cache: Optional[ResourceCache] = None,
        logger=None,
        is_mock=False,
        monitoring_mock=False,
        stream_profile: Optional[ds_profile.DatastoreProfile] = None,
    ) -> None:
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

        if is_mock and monitoring_mock:
            if stream_profile:
                # Add the user-defined stream profile to the parameters
                self.parameters["stream_profile"] = stream_profile
            elif not (
                self.parameters.get(FileTargetKind.LOG_STREAM)
                or mlrun.get_secret_or_env(
                    mm_constants.ProjectSecretKeys.STREAM_PROFILE_NAME
                )
            ):
                # Set a dummy log stream for mocking purposes if there is no direct
                # user-defined stream profile and no information in the environment
                self.parameters[FileTargetKind.LOG_STREAM] = DUMMY_STREAM

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
        if (
            isinstance(context, MLClientCtx)
            or isinstance(body, context.Response)
            or get_body
        ):
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


def add_error_raiser_step(
    graph: RootFlowStep, monitored_steps: dict[str, MonitoredStep]
) -> RootFlowStep:
    monitored_steps_raisers = {}
    user_steps = list(graph.steps.values())
    for monitored_step in monitored_steps.values():
        error_step = graph.add_step(
            class_name="mlrun.serving.states.ModelRunnerErrorRaiser",
            name=f"{monitored_step.name}_error_raise",
            after=monitored_step.name,
            full_event=True,
            raise_exception=monitored_step.raise_exception,
            models_names=list(monitored_step.class_args["models"].keys()),
            model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
            function=monitored_step.function,
        )
        if monitored_step.responder:
            monitored_step.responder = False
            error_step.respond()
        monitored_steps_raisers[monitored_step.name] = error_step.name
        error_step.on_error = monitored_step.on_error
    if monitored_steps_raisers:
        for step in user_steps:
            if step.after:
                if isinstance(step.after, list):
                    for i in range(len(step.after)):
                        if step.after[i] in monitored_steps_raisers:
                            step.after[i] = monitored_steps_raisers[step.after[i]]
                else:
                    if (
                        isinstance(step.after, str)
                        and step.after in monitored_steps_raisers
                    ):
                        step.after = monitored_steps_raisers[step.after]
    return graph


def add_monitoring_general_steps(
    project: str,
    graph: RootFlowStep,
    context,
    serving_spec,
    pause_until_background_task_completion: bool,
) -> tuple[RootFlowStep, FlowStep]:
    """
    Adding the monitoring flow connection steps, this steps allow the graph to reconstruct the serving event enrich it
    and push it to the model monitoring stream
    system_steps structure -
        "background_task_status_step" --> "filter_none" --> "monitoring_pre_processor_step" --> "flatten_events"
        --> "sampling_step" --> "filter_none_sampling" --> "model_monitoring_stream"
    """
    background_task_status_step = None
    if pause_until_background_task_completion:
        background_task_status_step = graph.add_step(
            "mlrun.serving.system_steps.BackgroundTaskStatus",
            "background_task_status_step",
            model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
            full_event=True,
        )
    monitor_flow_step = graph.add_step(
        "storey.Filter",
        "filter_none",
        _fn="(event is not None)",
        after="background_task_status_step" if background_task_status_step else None,
        model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
    )
    if background_task_status_step:
        monitor_flow_step = background_task_status_step
    graph.add_step(
        "mlrun.serving.system_steps.MonitoringPreProcessor",
        "monitoring_pre_processor_step",
        after="filter_none",
        full_event=True,
        model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
    )
    # flatten the events
    graph.add_step(
        "storey.FlatMap",
        "flatten_events",
        _fn="(event)",
        after="monitoring_pre_processor_step",
        model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
    )
    graph.add_step(
        "mlrun.serving.system_steps.SamplingStep",
        "sampling_step",
        after="flatten_events",
        sampling_percentage=float(
            serving_spec.get("parameters", {}).get("sampling_percentage", 100.0)
            if isinstance(serving_spec, dict)
            else getattr(serving_spec, "parameters", {}).get(
                "sampling_percentage", 100.0
            ),
        ),
        model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
    )
    graph.add_step(
        "storey.Filter",
        "filter_none_sampling",
        _fn="(event is not None)",
        after="sampling_step",
        model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
    )

    if getattr(context, "is_mock", False):
        graph.add_step(
            "mlrun.serving.system_steps.MockStreamPusher",
            "model_monitoring_stream",
            after="filter_none_sampling",
            model_endpoint_creation_strategy=mlrun.common.schemas.ModelEndpointCreationStrategy.SKIP,
        )
    else:
        stream_uri = mlrun.model_monitoring.get_stream_path(
            project=project,
            function_name=mlrun.common.schemas.MonitoringFunctionNames.STREAM,
        )
        context.logger.info_with(
            "Creating Model Monitoring stream target using uri:", uri=stream_uri
        )
        graph.add_step(
            ">>",
            "model_monitoring_stream",
            path=stream_uri,
            sharding_func=mlrun.common.schemas.model_monitoring.constants.StreamProcessingEvent.ENDPOINT_ID,
            after="filter_none_sampling",
        )
    return graph, monitor_flow_step


def add_system_steps_to_graph(
    project: str,
    graph: RootFlowStep,
    track_models: bool,
    context,
    serving_spec,
    pause_until_background_task_completion: bool = True,
) -> RootFlowStep:
    if not (isinstance(graph, RootFlowStep) and graph.include_monitored_step()):
        return graph
    monitored_steps = graph.get_monitored_steps()
    graph = add_error_raiser_step(graph, monitored_steps)
    if track_models:
        background_task_status_step = None
        graph, monitor_flow_step = add_monitoring_general_steps(
            project,
            graph,
            context,
            serving_spec,
            pause_until_background_task_completion,
        )
        if background_task_status_step:
            monitor_flow_step = background_task_status_step
        # Connect each model runner to the monitoring step:
        for step_name, step in monitored_steps.items():
            if monitor_flow_step.after:
                if isinstance(monitor_flow_step.after, list):
                    monitor_flow_step.after.append(step_name)
                elif isinstance(monitor_flow_step.after, str):
                    monitor_flow_step.after = [monitor_flow_step.after, step_name]
            else:
                monitor_flow_step.after = [
                    step_name,
                ]
    return graph


def v2_serving_init(context, namespace=None):
    """hook for nuclio init_context()"""

    context.logger.info("Initializing server from spec")
    spec = mlrun.utils.get_serving_spec()
    server = GraphServer.from_dict(spec)
    server.graph = add_system_steps_to_graph(
        server.project,
        copy.deepcopy(server.graph),
        spec.get("track_models"),
        context,
        spec,
    )

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


async def async_execute_graph(
    context: MLClientCtx,
    data: DataItem,
    timestamp_column: Optional[str],
    batching: bool,
    batch_size: Optional[int],
    read_as_lists: bool,
    nest_under_inputs: bool,
) -> None:
    # Validate that data parameter is a DataItem and not passed via params
    if not isinstance(data, DataItem):
        raise MLRunInvalidArgumentError(
            f"Parameter 'data' has type hint 'DataItem' but got {type(data).__name__} instead. "
            f"Data files and artifacts must be passed via the 'inputs' parameter, not 'params'. "
            f"The 'params' parameter is for simple configuration values (strings, numbers, booleans), "
            f"while 'inputs' is for data files that need to be loaded. "
            f"Example: run_function(..., inputs={{'data': 'path/to/data.csv'}}, params={{other_config: value}})"
        )
    run_call_count = 0
    spec = mlrun.utils.get_serving_spec()
    modname = None
    code = os.getenv("MLRUN_EXEC_CODE")
    if code:
        code = base64.b64decode(code).decode("utf-8")
        with open("user_code.py", "w") as fp:
            fp.write(code)
        modname = "user_code"
    else:
        # TODO: find another way to get the local file path, or ensure that MLRUN_EXEC_CODE
        #  gets set in local flow and not just in the remote pod
        source_file_path = spec.get("filename", None)
        if source_file_path:
            source_file_path_object, working_dir_path_object = (
                mlrun.utils.helpers.get_source_and_working_dir_paths(source_file_path)
            )
            if not source_file_path_object.is_relative_to(working_dir_path_object):
                raise mlrun.errors.MLRunRuntimeError(
                    f"Source file path '{source_file_path}' is not under the current working directory "
                    f"(which is required when running with local=True)"
                )
            modname = get_relative_module_name_from_path(
                source_file_path_object, working_dir_path_object
            )

    namespace = {}
    if modname:
        mod = importlib.import_module(modname)
        namespace = mod.__dict__

    server = GraphServer.from_dict(spec)

    if server.model_endpoint_creation_task_name:
        context.logger.info(
            f"Waiting for model endpoint creation task '{server.model_endpoint_creation_task_name}'..."
        )
        background_task = (
            mlrun.get_run_db().wait_for_background_task_to_reach_terminal_state(
                project=server.project,
                name=server.model_endpoint_creation_task_name,
            )
        )
        task_state = background_task.status.state
        if task_state == mlrun.common.schemas.BackgroundTaskState.failed:
            raise mlrun.errors.MLRunRuntimeError(
                "Aborting job due to model endpoint creation background task failure"
            )
        elif task_state != mlrun.common.schemas.BackgroundTaskState.succeeded:
            # this shouldn't happen, but we need to know if it does
            raise mlrun.errors.MLRunRuntimeError(
                "Aborting job because the model endpoint creation background task did not succeed "
                f"(status='{task_state}')"
            )

    df = data.as_df()

    if df.empty:
        context.logger.warn("Job terminated due to empty inputs (0 rows)")
        return

    track_models = spec.get("track_models")

    if track_models and timestamp_column:
        context.logger.info(f"Sorting dataframe by {timestamp_column}")
        df[timestamp_column] = pd.to_datetime(  # in case it's a string
            df[timestamp_column]
        )
        df.sort_values(by=timestamp_column, inplace=True)
        if len(df) > 1:
            start_time = df[timestamp_column].iloc[0]
            end_time = df[timestamp_column].iloc[-1]
            time_range = end_time - start_time
            start_time = start_time.isoformat()
            end_time = end_time.isoformat()
            # TODO: tie this to the controller's base period
            if time_range > pd.Timedelta(MAX_BATCH_JOB_DURATION):
                raise mlrun.errors.MLRunRuntimeError(
                    f"Dataframe time range is too long: {time_range}. "
                    "Please disable tracking or reduce the input dataset's time range below the defined limit "
                    f"of {MAX_BATCH_JOB_DURATION}."
                )
        else:
            start_time = end_time = df["timestamp"].iloc[0].isoformat()
    else:
        # end time will be set from clock time when the batch completes
        start_time = datetime.now(tz=timezone.utc).isoformat()

    server.graph = add_system_steps_to_graph(
        server.project,
        copy.deepcopy(server.graph),
        track_models,
        context,
        spec,
        pause_until_background_task_completion=False,  # we've already awaited it
    )

    if config.log_level.lower() == "debug":
        server.verbose = True
    kwargs = {}
    if hasattr(context, "is_mock"):
        kwargs["is_mock"] = context.is_mock
    server.init_states(
        context=None,  # this context is expected to be a nuclio context, which we don't have in this flow
        namespace=namespace,
        **kwargs,
    )
    context.logger.info("Initializing graph steps")
    server.init_object(namespace)

    context.logger.info_with("Graph was initialized", verbose=server.verbose)

    if server.verbose:
        context.logger.info(server.to_yaml())

    async def run(body):
        nonlocal run_call_count
        event = storey.Event(id=index, body=body)
        if timestamp_column:
            if batching:
                # we use the first row in the batch to determine the timestamp for the whole batch
                body = body[0]
            if not isinstance(body, dict):
                raise mlrun.errors.MLRunRuntimeError(
                    f"When timestamp_column=True, event body must be a dict – got {type(body).__name__} instead"
                )
            if timestamp_column not in body:
                raise mlrun.errors.MLRunRuntimeError(
                    f"Event body '{body}' did not contain timestamp column '{timestamp_column}'"
                )
            event._original_timestamp = body[timestamp_column]
        run_call_count += 1
        return await server.run(event, context)

    if batching and not batch_size:
        batch_size = len(df)

    batch = []
    tasks = []
    for index, row in df.iterrows():
        data = row.to_list() if read_as_lists else row.to_dict()
        if nest_under_inputs:
            data = {"inputs": data}
        if batching:
            batch.append(data)
            if len(batch) == batch_size:
                tasks.append(asyncio.create_task(run(batch)))
                batch = []
        else:
            tasks.append(asyncio.create_task(run(data)))

    if batch:
        tasks.append(asyncio.create_task(run(batch)))

    responses = await asyncio.gather(*tasks)

    termination_result = server.wait_for_completion()
    if asyncio.iscoroutine(termination_result):
        await termination_result

    model_endpoint_uids = spec.get("model_endpoint_uids", [])

    # needed for output_stream to be created
    server = GraphServer.from_dict(spec)
    server.init_states(None, namespace)

    batch_completion_time = datetime.now(tz=timezone.utc).isoformat()

    if not timestamp_column:
        end_time = batch_completion_time

    mm_stream_record = dict(
        kind="batch_complete",
        project=context.project,
        first_timestamp=start_time,
        last_timestamp=end_time,
        batch_completion_time=batch_completion_time,
    )
    output_stream = server.context.stream.output_stream
    for mep_uid in spec.get("model_endpoint_uids", []):
        mm_stream_record["endpoint_id"] = mep_uid
        output_stream.push(mm_stream_record, partition_key=mep_uid)

    context.logger.info(
        f"Job completed processing {len(df)} rows",
        timestamp_column=timestamp_column,
        model_endpoint_uids=model_endpoint_uids,
    )

    has_responder = False
    for step in server.graph.steps.values():
        if getattr(step, "responder", False):
            has_responder = True
            break

    if has_responder:
        # log the results as a dataset artifact
        artifact_path = None
        if (
            "{{run.uid}}" not in context.artifact_path
        ):  # TODO: delete when IG-22841 is resolved
            artifact_path = "+/{{run.uid}}"  # will be concatenated to the context's path in extend_artifact_path
        context.log_dataset(
            "prediction", df=pd.DataFrame(responses), artifact_path=artifact_path
        )

        # if we got responses that appear to be in the right format, try to log per-model datasets too
        if (
            responses
            and responses[0]
            and isinstance(responses[0], dict)
            and isinstance(next(iter(responses[0].values())), (dict, list))
        ):
            try:
                # turn this list of samples into a dict of lists, one per model endpoint
                grouped = defaultdict(list)
                for sample in responses:
                    for model_name, features in sample.items():
                        grouped[model_name].append(features)
                # create a dataframe per model endpoint and log it
                for model_name, features in grouped.items():
                    context.log_dataset(
                        f"prediction_{model_name}",
                        df=pd.DataFrame(features),
                        artifact_path=artifact_path,
                    )
            except Exception as e:
                context.logger.warning(
                    "Failed to log per-model prediction datasets",
                    error=err_to_str(e),
                )

    context.log_result("num_rows", run_call_count)


def _is_inside_asyncio_loop():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


# Workaround for running with local=True in Jupyter (ML-10620)
def _workaround_asyncio_nesting():
    try:
        import nest_asyncio
    except ImportError:
        raise mlrun.errors.MLRunRuntimeError(
            "Cannot execute graph from within an already running asyncio loop. "
            "Attempt to import nest_asyncio as a workaround failed as well."
        )
    nest_asyncio.apply()


def execute_graph(
    context: MLClientCtx,
    data: DataItem,
    timestamp_column: Optional[str] = None,
    batching: bool = False,
    batch_size: Optional[int] = None,
    read_as_lists: bool = False,
    nest_under_inputs: bool = False,
) -> (list[Any], Any):
    """
    Execute graph as a job, from start to finish.

    :param context: The job's execution client context.
    :param data: The input data to the job, to be pushed into the graph row by row, or in batches.
    :param timestamp_column: The name of the column that will be used as the timestamp for model monitoring purposes.
        when timestamp_column is used in conjunction with batching, the first timestamp will be used for the entire
        batch.
    :param batching: Whether to push one or more batches into the graph rather than row by row.
    :param batch_size: The number of rows to push per batch. If not set, and batching=True, the entire dataset will
        be pushed into the graph in one batch.
    :param read_as_lists: Whether to read each row as a list instead of a dictionary.
    :param nest_under_inputs: Whether to wrap each row with {"inputs": ...}.

    :return: A list of responses.
    """
    if _is_inside_asyncio_loop():
        _workaround_asyncio_nesting()

    return asyncio.run(
        async_execute_graph(
            context,
            data,
            timestamp_column,
            batching,
            batch_size,
            read_as_lists,
            nest_under_inputs,
        )
    )


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
        self.executor: Optional[storey.flow.RunnableExecutor] = None

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
