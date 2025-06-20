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

import concurrent
import concurrent.futures
import copy
import json
import traceback
import typing
from datetime import timedelta
from enum import Enum
from io import BytesIO
from typing import Union

import numpy
import numpy as np

import mlrun
import mlrun.common.model_monitoring
import mlrun.common.schemas.model_monitoring
from mlrun.utils import logger, now_date

from .utils import RouterToDict, _extract_input_data, _update_result_body
from .v2_serving import _ModelLogPusher

# Used by `ParallelRun` in process mode, so it can be accessed from different processes.
local_routes = {}


class BaseModelRouter(RouterToDict):
    """base model router class"""

    def __init__(
        self,
        context=None,
        name: typing.Optional[str] = None,
        routes=None,
        protocol: typing.Optional[str] = None,
        url_prefix: typing.Optional[str] = None,
        health_prefix: typing.Optional[str] = None,
        input_path: typing.Optional[str] = None,
        result_path: typing.Optional[str] = None,
        **kwargs,
    ):
        """Model Serving Router, route between child models

        :param context:       for internal use (passed in init)
        :param name:          step name
        :param routes:        for internal use (routes passed in init)
        :param protocol:      serving API protocol (default "v2")
        :param url_prefix:    url prefix for the router (default /v2/models)
        :param health_prefix: health api url prefix (default /v2/health)
        :param input_path:    when specified selects the key/path in the event to use as body
                              this require that the event body will behave like a dict, example:
                              event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means request body will be 7
        :param result_path:   selects the key/path in the event to write the results to
                              this require that the event body will behave like a dict, example:
                              event: {"x": 5} , result_path="resp" means the returned response will be written
                              to event["y"] resulting in {"x": 5, "resp": <result>}
        :param kwargs:        extra arguments
        """
        self.name = name
        self.context = context
        self.routes = routes
        self.protocol = protocol or "v2"
        self.url_prefix = url_prefix or f"/{self.protocol}/models"
        self.health_prefix = health_prefix or f"/{self.protocol}/health"
        self.inputs_key = "instances" if self.protocol == "v1" else "inputs"
        self._input_path = input_path
        self._result_path = result_path
        self._background_task_check_timestamp = None
        self._background_task_terminate = False
        self._background_task_current_state = None
        self.kwargs = kwargs

    def parse_event(self, event):
        parsed_event = {}
        try:
            if not isinstance(event.body, dict):
                body = json.loads(event.body)
            else:
                body = event.body
            if "data_url" in body:
                # Get data from URL
                url = body["data_url"]
                self.context.logger.debug(f"Downloading data url={url}")
                data = mlrun.get_object(url)
                sample = BytesIO(data)
                parsed_event[self.inputs_key] = [sample]
            else:
                parsed_event = body

        except Exception as exc:
            #  if images convert to bytes
            content_type = getattr(event, "content_type", "") or ""
            if content_type.startswith("image/"):
                sample = BytesIO(event.body)
                parsed_event[self.inputs_key] = [sample]
            else:
                raise ValueError("Unrecognized request format") from exc

        return parsed_event

    def post_init(self, mode="sync", **kwargs):
        self.context.logger.info(f"Loaded {list(self.routes.keys())}")

    def get_metadata(self):
        """return the model router/host details"""

        return {"name": self.__class__.__name__, "version": "v2", "extensions": []}

    def _pre_handle_event(self, event):
        method = event.method or "POST"
        if event.body and method != "GET":
            event.body = self.parse_event(event)
        urlpath = getattr(event, "path", "")

        # if health check or "/" return Ok + metadata
        if method == "GET" and (
            urlpath == "/" or urlpath.startswith(self.health_prefix)
        ):
            setattr(event, "terminated", True)
            event.body = self.get_metadata()
            return event

        # check for legal path prefix
        if urlpath and not urlpath.startswith(self.url_prefix) and not urlpath == "/":
            raise ValueError(
                f"illegal path prefix {urlpath}, must start with {self.url_prefix}"
            )
        self._update_background_task_state(event)
        return event

    def do_event(self, event, *args, **kwargs):
        """handle incoming events, event is nuclio event class"""

        original_body = event.body
        event.body = _extract_input_data(self._input_path, event.body)
        event = self.preprocess(event)
        event = self._pre_handle_event(event)
        if not (hasattr(event, "terminated") and event.terminated):
            event = self.postprocess(self._handle_event(event))
        event.body = _update_result_body(self._result_path, original_body, event.body)
        return event

    def _handle_event(self, event):
        return event

    def preprocess(self, event):
        """run tasks before processing the event"""
        return event

    def postprocess(self, event):
        """run tasks after processing the event"""
        return event

    def _get_background_task_status(
        self,
    ) -> mlrun.common.schemas.BackgroundTaskState:
        self._background_task_check_timestamp = now_date()
        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        if not self.context.is_mock:
            if server.model_endpoint_creation_task_name:
                background_task = mlrun.get_run_db().get_project_background_task(
                    server.project, server.model_endpoint_creation_task_name
                )
                logger.debug(
                    "Checking model endpoint creation task status",
                    task_name=server.model_endpoint_creation_task_name,
                )
                if (
                    background_task.status.state
                    in mlrun.common.schemas.BackgroundTaskState.terminal_states()
                ):
                    logger.debug(
                        f"Model endpoint creation task completed with state {background_task.status.state}"
                    )
                    self._background_task_terminate = True
                else:  # in progress
                    logger.debug(
                        f"Model endpoint creation task is still in progress with the current state: "
                        f"{background_task.status.state}. Events will not be monitored for the next 15 seconds",
                        name=self.name,
                        background_task_check_timestamp=self._background_task_check_timestamp.isoformat(),
                    )
                return background_task.status.state
            else:
                logger.debug(
                    "Model endpoint creation task name not provided",
                )
        elif self.context.monitoring_mock:
            self._background_task_terminate = (
                True  # If mock monitoring we return success and terminate task check.
            )
            return mlrun.common.schemas.BackgroundTaskState.succeeded
        self._background_task_terminate = True  # If mock without monitoring we return failed and terminate task check.
        return mlrun.common.schemas.BackgroundTaskState.failed

    def _update_background_task_state(self, event):
        if not self._background_task_terminate and (
            self._background_task_check_timestamp is None
            or now_date() - self._background_task_check_timestamp
            >= timedelta(seconds=15)
        ):
            self._background_task_current_state = self._get_background_task_status()
        if event.body:
            event.body["background_task_state"] = (
                self._background_task_current_state
                or mlrun.common.schemas.BackgroundTaskState.running
            )


class ModelRouter(BaseModelRouter):
    def _resolve_route(self, body, urlpath):
        subpath = None
        model = ""
        if urlpath and not urlpath == "/":
            # process the url <prefix>/<model>[/versions/<ver>]/operation
            subpath = ""
            urlpath = urlpath[len(self.url_prefix) :].strip("/")
            if not urlpath:
                return "", None, ""
            segments = urlpath.split("/")
            model = segments[0]
            if len(segments) > 2 and segments[1] == "versions":
                model = model + ":" + segments[2]
                segments = segments[2:]
            if len(segments) > 1:
                subpath = "/".join(segments[1:])

        if isinstance(body, dict):
            # accepting route information from body as well
            # to support streaming protocols (e.g. Kafka).
            model = model or body.get("model", list(self.routes.keys())[0])
            subpath = body.get("operation", subpath)
        if subpath is None:
            subpath = "infer"

        if model not in self.routes:
            models = " | ".join(self.routes.keys())
            raise ValueError(f"model {model} doesnt exist, available models: {models}")

        return model, self.routes[model], subpath

    def _handle_event(self, event):
        name, route, subpath = self._resolve_route(event.body, event.path)
        if not route:
            # if model wasn't specified return model list
            setattr(event, "terminated", True)
            event.body = {"models": list(self.routes.keys())}
            return event

        self.context.logger.debug(f"router run model {name}, op={subpath}")
        event.path = subpath
        response = route.run(event)
        event.body = response.body if response else None
        return event


class ParallelRunnerModes(str, Enum):
    """Supported parallel running modes for VotingEnsemble"""

    array = "array"  # running one by one
    process = "process"  # running in separated processes
    thread = "thread"  # running in separated threads

    @staticmethod
    def all():
        return [
            ParallelRunnerModes.thread,
            ParallelRunnerModes.process,
            ParallelRunnerModes.array,
        ]


class VotingTypes(str, Enum):
    """Supported voting types for VotingEnsemble"""

    classification = "classification"
    regression = "regression"


class OperationTypes(str, Enum):
    """Supported opreations for VotingEnsemble"""

    infer = "infer"
    predict = "predict"
    explain = "explain"


class ParallelRun(BaseModelRouter):
    # TODO: change name to ParallelRunModelRouter
    # To consider because ParallelRun inherits from BaseModelRouter
    # Didn't changed yet because ParallelRun is not only for models
    def __init__(
        self,
        context=None,
        name: typing.Optional[str] = None,
        routes=None,
        protocol: typing.Optional[str] = None,
        url_prefix: typing.Optional[str] = None,
        health_prefix: typing.Optional[str] = None,
        extend_event=None,
        executor_type: Union[ParallelRunnerModes, str] = ParallelRunnerModes.thread,
        **kwargs,
    ):
        """Process multiple steps (child routes) in parallel and merge the results

        By default the results dict from each step are merged (by key), when setting the `extend_event`
        the results will start from the event body dict (values can be overwritten)

        Users can overwrite the merger() method to implement custom merging logic.

        Example::

            # create a function with a parallel router and 3 children
            fn = mlrun.new_function("parallel", kind="serving")
            graph = fn.set_topology(
                "router",
                mlrun.serving.routers.ParallelRun(
                    extend_event=True, executor_type=executor
                ),
            )
            graph.add_route("child1", class_name="Cls1")
            graph.add_route("child2", class_name="Cls2", my_arg={"c": 7})
            graph.add_route("child3", handler="my_handler")
            server = fn.to_mock_server()
            resp = server.test("", {"x": 8})


        :param context:       for internal use (passed in init)
        :param name:          step name
        :param routes:        for internal use (routes passed in init)
        :param protocol:      serving API protocol (default "v2")
        :param url_prefix:    url prefix for the router (default /v2/models)
        :param health_prefix: health api url prefix (default /v2/health)
        :param executor_type: Parallelism mechanism,  Have 3 option :
                              * array - running one by one
                              * process - running in separated process
                              * thread - running in separated threads
                              by default `threads`
        :param extend_event:  True will add the event body to the result
        :param kwargs:        extra arguments
        """
        super().__init__(
            context=context,
            name=name,
            routes=routes,
            protocol=protocol,
            url_prefix=url_prefix,
            health_prefix=health_prefix,
            **kwargs,
        )
        self.name = name or "ParallelRun"
        self.extend_event = extend_event
        self.executor_type = ParallelRunnerModes(executor_type)
        self._pool: typing.Optional[
            Union[
                concurrent.futures.ProcessPoolExecutor,
                concurrent.futures.ThreadPoolExecutor,
            ]
        ] = None

    def _apply_logic(self, results: dict, event=None):
        """
        Apply merge logic on results.

        :param results: A list of sample results by models e.g. results[model][prediction]
        :param event: Response event

        :return: Dictionary of results
        """
        if not self.extend_event:
            event.body = {}
        return self.merger(event.body, results)

    def merger(self, body, results):
        """Merging logic

        input the event body and a dict of route results and returns a dict with merged results
        """
        for result in results.values():
            body.update(result)
        return body

    def do_event(self, event, *args, **kwargs):
        # Handle and verify the request
        original_body = event.body
        event.body = _extract_input_data(self._input_path, event.body)
        event = self.preprocess(event)
        event = self._pre_handle_event(event)

        # Should we terminate the event?
        if hasattr(event, "terminated") and event.terminated:
            event.body = _update_result_body(
                self._result_path, original_body, event.body
            )
            self._shutdown_pool()
            return event

        response = copy.copy(event)
        results = self._parallel_run(event)
        self._apply_logic(results, response)
        response = self.postprocess(response)

        event.body = _update_result_body(
            self._result_path, original_body, response.body if response else None
        )
        return event

    def _init_pool(
        self,
    ) -> Union[
        concurrent.futures.ProcessPoolExecutor, concurrent.futures.ThreadPoolExecutor
    ]:
        """

        Get the tasks pool of this runner. If the pool is `None`,
        a new pool will be initialized according to `executor_type`.

        :return: The tasks pool
        """
        if self._pool is None:
            if self.executor_type == ParallelRunnerModes.process:
                # init the context and route on the worker side (cannot be pickeled)
                server = self.context.server.to_dict()
                routes = {}
                for key, route in self.routes.items():
                    step = copy.copy(route)
                    step.context = None
                    step._parent = None
                    if step._object:
                        step._object.context = None
                        if hasattr(step._object, "_kwargs"):
                            step._object._kwargs["graph_step"] = None
                    routes[key] = step
                executor_class = concurrent.futures.ProcessPoolExecutor
                self._pool = executor_class(
                    max_workers=len(self.routes),
                    initializer=ParallelRun.init_pool,
                    initargs=(server, routes, self.context.is_mock),
                )
            elif self.executor_type == ParallelRunnerModes.thread:
                executor_class = concurrent.futures.ThreadPoolExecutor
                self._pool = executor_class(max_workers=len(self.routes))

        return self._pool

    def _shutdown_pool(self):
        """
        Shutdowns the pool and updated self._pool to None
        """
        if self._pool is not None:
            if self.executor_type == ParallelRunnerModes.process:
                global local_routes
                del local_routes
            self._pool.shutdown()
            self._pool = None

    def _parallel_run(self, event: dict):
        """
        Execute parallel run

        :param event: event to run in parallel

        :return: All the results of the runs
        """
        results = {}
        if self.executor_type == ParallelRunnerModes.array:
            results = {
                model_name: model.run(copy.copy(event)).body
                for model_name, model in self.routes.items()
            }
            return results
        futures = []
        executor = self._init_pool()
        for route in self.routes.keys():
            if self.executor_type == ParallelRunnerModes.process:
                future = executor.submit(
                    ParallelRun._wrap_step, route, copy.copy(event)
                )
            elif self.executor_type == ParallelRunnerModes.thread:
                step = self.routes[route]
                future = executor.submit(
                    ParallelRun._wrap_method,
                    route,
                    step.run,
                    copy.copy(event),
                )

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                key, result = future.result()
                results[key] = result.body
            except Exception as exc:
                logger.error(traceback.format_exc())
                print(f"child route generated an exception: {exc}")
        self.context.logger.debug(f"Collected results from children: {results}")
        return results

    @staticmethod
    def init_pool(server_spec, routes, is_mock):
        server = mlrun.serving.GraphServer.from_dict(server_spec)
        server.init_states(None, None, is_mock=is_mock)
        global local_routes
        for route in routes.values():
            route.context = server.context
            if route._object:
                route._object.context = server.context
        local_routes = routes

    @staticmethod
    def _wrap_step(route, event):
        global local_routes
        if local_routes is None:
            return None, None
        return route, local_routes[route].run(event)

    @staticmethod
    def _wrap_method(route, handler, event):
        return route, handler(event)


class VotingEnsemble(ParallelRun):
    def __init__(
        self,
        context=None,
        name: typing.Optional[str] = None,
        routes=None,
        protocol: typing.Optional[str] = None,
        url_prefix: typing.Optional[str] = None,
        health_prefix: typing.Optional[str] = None,
        vote_type: typing.Optional[str] = None,
        weights: typing.Optional[dict[str, float]] = None,
        executor_type: Union[ParallelRunnerModes, str] = ParallelRunnerModes.thread,
        format_response_with_col_name_flag: bool = False,
        prediction_col_name: str = "prediction",
        shard_by_endpoint: typing.Optional[bool] = None,
        **kwargs,
    ):
        """Voting Ensemble

        The `VotingEnsemble` class enables you to apply prediction logic on top of
        the different added models.

        You can use it by calling:

        - `<prefix>/<model>[/versions/<ver>]/operation`
            Sends the event to the specific <model>[/versions/<ver>]
        - `<prefix>/operation`
            Sends the event to all models and applies `vote(self, event)`

        The `VotingEnsemble` applies the following logic:
        Incoming Event -> Router Preprocessing -> Send to model/s ->
        Apply all model/s logic (Preprocessing -> Prediction -> Postprocessing) ->
        Router Voting logic -> Router Postprocessing -> Response

        This enables you to do the general preprocessing and postprocessing steps
        once on the router level, with only model-specific adjustments at the
        model level.

            When enabling model tracking via `set_tracking()` the ensemble logic
            predictions will appear with model name as the given VotingEnsemble name
            or "VotingEnsemble" by default.

        Example::

            # Define a serving function
            # Note: You can point the function to a file containing you own Router or Classifier Model class
            #       this basic class supports sklearn based models (with `<model>.predict()` api)
            fn = mlrun.code_to_function(name='ensemble',
                                        kind='serving',
                                        filename='model-server.py'
                                        image='mlrun/mlrun')

            # Set the router class
            # You can set your own classes by simply changing the `class_name`
            fn.set_topology(class_name='mlrun.serving.routers.VotingEnsemble')

            # Add models
            fn.add_model(<model_name>, <model_path>, <model_class_name>)
            fn.add_model(<model_name>, <model_path>, <model_class_name>)

        How to extend the VotingEnsemble:

        The VotingEnsemble applies its logic using the `logic(predictions)` function.
        The `logic()` function receives an array of (# samples, # predictors) which you
        can then use to apply whatever logic you may need.

        If we use this `VotingEnsemble` as an example, the `logic()` function tries to figure
        out whether you are trying to do a **classification** or a **regression** prediction by
        the prediction type or by the given `vote_type` parameter.  Then we apply the appropriate
        `max_vote()` or `mean_vote()` which calculates the actual prediction result and returns it
        as the VotingEnsemble's prediction.


        :param context:       for internal use (passed in init)
        :param name:          step name
        :param routes:        for internal use (routes passed in init)
        :param protocol:      serving API protocol (default "v2")
        :param url_prefix:    url prefix for the router (default /v2/models)
        :param health_prefix: health api url prefix (default /v2/health)
        :param input_path:    when specified selects the key/path in the event to use as body
                              this require that the event body will behave like a dict, example:
                              event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means request body will be 7
        :param result_path:   selects the key/path in the event to write the results to
                              this require that the event body will behave like a dict, example:
                              event: {"x": 5} , result_path="resp" means the returned response will be written
                              to event["y"] resulting in {"x": 5, "resp": <result>}
        :param vote_type:     Voting type to be used (from `VotingTypes`).
                              by default will try to self-deduct upon the first event:
                              - float prediction type: regression
                              - int prediction type: classification
        :param weights        A dictionary ({"<model_name>": <model_weight>}) that specified each model weight,
                              if there is a model that didn't appear in the dictionary his
                              weight will be count as a zero. None means that all the models have the same weight.
        :param executor_type: Parallelism mechanism, out of `ParallelRunnerModes`, by default `threads`
        :param format_response_with_col_name_flag: If this flag is True the model's responses output format is
                                                     `{id: <id>, model_name: <name>, outputs:
                                                     {..., prediction: [<predictions>], ...}}`
                                                   Else
                                                      `{id: <id>, model_name: <name>, outputs: [<predictions>]}`
        :param prediction_col_name: The dict key for the predictions column in the model's responses output.
                              Example: If the model returns
                              `{id: <id>, model_name: <name>, outputs: {..., prediction: [<predictions>], ...}}`
                              the prediction_col_name should be `prediction`.
                              by default, `prediction`
        :param shard_by_endpoint: whether to use the endpoint as the partition/sharding key when writing to model
                                  monitoring stream. Defaults to True.
        :param kwargs:        extra arguments
        """
        super().__init__(
            context=context,
            name=name,
            routes=routes,
            protocol=protocol,
            url_prefix=url_prefix,
            health_prefix=health_prefix,
            executor_type=executor_type,
            **kwargs,
        )
        self.name = name or "VotingEnsemble"
        self.vote_type = vote_type
        self.vote_flag = True if self.vote_type is not None else False
        self.weights = weights
        self.log_router = True
        self.prediction_col_name = prediction_col_name or "prediction"
        self.format_response_with_col_name_flag = format_response_with_col_name_flag
        self.model_endpoint_uid = kwargs.get("model_endpoint_uid", None)
        self.shard_by_endpoint = shard_by_endpoint
        self._model_logger = None
        self.initialized = False

    def post_init(self, mode="sync", **kwargs):
        self._update_weights(self.weights)

    def _lazy_init(self, event):
        if event and isinstance(event, dict):
            background_task_state = event.get("background_task_state", None)
            if (
                background_task_state
                == mlrun.common.schemas.BackgroundTaskState.succeeded
            ):
                self._model_logger = (
                    _ModelLogPusher(self, self.context)
                    if self.context
                    and self.context.stream.enabled
                    and self.model_endpoint_uid
                    else None
                )
                self.initialized = True

    def _resolve_route(self, body, urlpath):
        """Resolves the appropriate model to send the event to.
        Supports:
        - <prefix>/<model>[/versions/<ver>]/operation
        Sends the event to the specific <model>[/versions/<ver>]

        - <prefix>/operation
        Sends the event to all models

        Args:
            body (dict): event body
            urlpath (string): url path

        Raises:
            ValueError: model does't exist in the model registry

        Returns:
            model_name (string): name of the selected model
            route (Selected Model's Class): actual selected model from the registry
            subpath: contains the operator for the model
        """
        subpath = None
        model = ""
        if urlpath and not urlpath == "/":
            # process the url <prefix>/<model>[/versions/<ver>]/operation
            subpath = ""
            urlpath = urlpath[len(self.url_prefix) :].strip("/")

            # Test if Only `self.url_prefix/` was given
            if not urlpath:
                return "", None, ""
            segments = urlpath.split("/")

            # Test for router level `/operation`
            if len(segments) == 1:
                # Path =  <prefix>/<segment>
                # Are we looking at a router level operation?
                try:
                    operation = OperationTypes(segments[0])
                # Unrecognized operation was given, probably a model name
                except ValueError:
                    model = segments[0]
                else:
                    self.log_router = True
                    return self.name, None, operation

            # Test for `self.url_prefix/<model>/versions/<version>/operation`
            if len(segments) > 2 and segments[1] == "versions":
                # Add versioning to the model as: <model>:<version>
                model = f"{segments[0]}:{segments[2]}"

                # Prune handled URI parts
                segments = segments[2:]
            else:
                model = segments[0]
            if len(segments) > 1:
                subpath = "/".join(segments[1:])

        # accepting route information from body as well
        # to support streaming protocols (e.g. Kafka).
        if isinstance(body, dict):
            model = model or self.name
            subpath = body.get("operation", subpath)

        # Set default subpath (operation) if needed
        if subpath is None:
            subpath = "infer"

        # Test if the given model is one of our registered models
        if model in self.routes:
            # Turn off unnecessary router logging for simple event passing
            self.log_router = False
            return model, self.routes[model], subpath

        # Test if it's our voting ensemble name
        elif model != self.name:
            # The given model is not the `VotingEnsemble.name` nor is it
            # any of our registered models.
            models = " | ".join(self.routes.keys())
            raise ValueError(
                f"model {model} doesnt exist, available models: "
                f"{models} | {self.name} or an operation alone for ensemble operation"
            )
        return model, None, subpath

    def _majority_vote(self, all_predictions: list[list[int]], weights: list[float]):
        """
        Returns most predicted class for each event

        :param all_predictions: The predictions from all models, per event
        :param weights: models weights in the prediction order

        :return: A list with the most predicted class by all models, per event
        """
        all_predictions = np.array(all_predictions)
        # Create 3d matrix (n,c,m) - m the number of models,
        # c the number of classes and n the number of samples
        one_hot_representation = np.transpose(
            (np.arange(all_predictions.max() + 1) == all_predictions[..., None]).astype(
                int
            ),
            (0, 2, 1),
        )
        # Each 2d matrix multiply by the weights, and
        # we get matrix (n,c) such that each row
        # represent the prediction to each sample.
        weighted_res = one_hot_representation @ weights
        return np.argmax(weighted_res, axis=1).tolist()

    def _mean_vote(self, all_predictions: list[list[float]], weights: list[float]):
        """
        Returns weighted mean of the predictions

        :param all_predictions: The predictions from all models, per event
        :param weights: models weights in the prediction order

        :return: A list of the mean of predictions from all models, per event
        """
        return (np.array(all_predictions) @ weights).tolist()

    def _is_int(self, value):
        return float(value).is_integer()

    def logic(self, predictions: list[list[Union[int, float]]], weights: list[float]):
        """
        Returns the final prediction of all the models after applying the desire logic

        :param predictions: The predictions from all models, per event
        :param weights: models weights in the prediction order

        :return: List of the resulting voted predictions
        """
        # Infer voting type if not given (Classification or recommendation) (once)
        if not self.vote_flag:
            # Are we dealing with an All-Int predictions
            # e.g. Classification
            if all(
                [
                    all(response)
                    for response in [
                        list(map(self._is_int, prediction_array))
                        for prediction_array in predictions
                    ]
                ]
            ):
                self.vote_type = VotingTypes.classification
            # Do we have `float` predictions
            # e.g. Regression
            else:
                self.vote_type = VotingTypes.regression

            # set flag to not infer this again
            self.vote_flag = True
        # Apply voting logic
        if self.vote_type == VotingTypes.classification:
            int_predictions = [
                list(map(int, sample_predictions)) for sample_predictions in predictions
            ]
            self.context.logger.debug(f"Applying max logic vote on {predictions}")
            votes = self._majority_vote(int_predictions, weights)
        else:
            self.context.logger.debug(f"Applying majority logic vote on {predictions}")
            votes = self._mean_vote(predictions, weights)

        return votes

    def _apply_logic(self, results: dict, event=None):
        """
        Reduces a list of k predictions from n models to k predictions according to voting logic

        :param results: A list of sample predictions by models e.g. predictions[model][prediction]
        :param event: Response event
        :return: List of the resulting voted predictions
        """
        flattened_predictions = np.array(
            list(
                map(
                    lambda dictionary: (
                        dictionary["outputs"][self.prediction_col_name]
                        if self.format_response_with_col_name_flag
                        else dictionary["outputs"]
                    ),
                    results.values(),
                )
            )
        ).T
        weights = [self._weights[model_name] for model_name in results.keys()]
        return self.logic(flattened_predictions, np.array(weights))

    def do_event(self, event, *args, **kwargs):
        """
        Handles incoming requests.

        Parameters
        ----------
        event : nuclio.Event
            Incoming request as a nuclio.Event.

        Returns
        -------
        Response
            Event response after running the requested logic
        """
        start = now_date()
        # Handle and verify the request
        original_body = event.body
        event.body = _extract_input_data(self._input_path, event.body)
        event = self.preprocess(event)
        event = self._pre_handle_event(event)
        if not self.initialized:
            self._lazy_init(event.body)

        # Should we terminate the event?
        if hasattr(event, "terminated") and event.terminated:
            event.body = _update_result_body(
                self._result_path, original_body, event.body
            )

            self._shutdown_pool()
            return event

        # Extract route information
        name, route, subpath = self._resolve_route(event.body, event.path)
        self.context.logger.debug(f"router run model {name}, op={subpath}")
        event.path = subpath

        # Return the correct response
        # If no model name was given and no operation
        if not name and route is None:
            # Return model list
            setattr(event, "terminated", True)
            event.body = {
                "models": list(self.routes.keys()) + [self.name],
                "weights": self.weights,
            }
            event.body = _update_result_body(
                self._result_path, original_body, event.body
            )
            return event
        else:
            # Verify we use the V2 protocol
            request = self.validate(event.body, event.method)

            # If this is a Router Operation
            if name == self.name and event.method != "GET":
                predictions = self._parallel_run(event)
                votes = self._apply_logic(predictions)
                # Format the prediction response like the regular
                # model's responses
                if self.format_response_with_col_name_flag:
                    votes = {self.prediction_col_name: votes}
                response = copy.copy(event)
                response_body = {
                    "id": event.id,
                    "model_name": self.name,
                    "outputs": votes,
                }
                if self.model_endpoint_uid:
                    response_body["model_endpoint_uid"] = self.model_endpoint_uid
                response.body = response_body
            elif name == self.name and event.method == "GET" and not subpath:
                response = copy.copy(event)
                response_body = {
                    "name": self.name,
                    "model_endpoint_uid": self.model_endpoint_uid or "",
                    "inputs": [],
                    "outputs": [],
                }
                for route in self.routes.values():
                    response_random_route = route.run(copy.copy(event))
                    response_body["inputs"] = (
                        response_body["inputs"] or response_random_route.body["inputs"]
                    )
                    response_body["outputs"] = (
                        response_body["outputs"]
                        or response_random_route.body["outputs"]
                    )
                    if response_body["inputs"] and response_body["outputs"]:
                        break
                response.body = response_body
            # A specific model event
            else:
                response = route.run(event)

        response = self.postprocess(response)

        if self._model_logger and self.log_router:
            if "id" not in request:
                request["id"] = response.body["id"]
            partition_key = (
                self.model_endpoint_uid if self.shard_by_endpoint is not False else None
            )
            self._model_logger.push(
                start, request, response.body, partition_key=partition_key
            )
        event.body = _update_result_body(
            self._result_path, original_body, response.body if response else None
        )
        return event

    def extract_results_from_response(self, response):
        """Extracts the prediction from the model response.
        This function is used to allow multiple model return types. and allow for easy
        extension to the user's ensemble and models best practices.

        Parameters
        ----------
        response : Union[List, Dict]
            The model response's `output` field.

        Returns
        -------
        List
            The model's predictions
        """
        if isinstance(response, (list, numpy.ndarray)):
            return response
        try:
            self.format_response_with_col_name_flag = True
            return response[self.prediction_col_name]
        except KeyError:
            raise ValueError(
                f"The given `prediction_col_name` ({self.prediction_col_name}) does not exist "
                f"in the model's response ({response.keys()})"
            )

    def validate(self, request: dict, method: str):
        """
        Validate the event body (after preprocessing)

        :param request: Event body.
        :param method: Event method.


        :return: The given Event body (request).

        :raise Exception: If validation failed.
        """
        if self.protocol == "v2" and method != "GET":
            if "inputs" not in request:
                raise Exception('Expected key "inputs" in request body')

            if not isinstance(request["inputs"], list):
                raise Exception('Expected "inputs" to be a list')
        return request

    def _normalize_weights(self, weights_dict: dict[str, float]):
        """
        Normalized all the weights such that abs(weights_sum - 1.0) <= 0.001
        and adding 0 weight to all the routes that doesn't appear in the dict.
        If weights_dict is None the function returns equal weights to all the routes.

        :param weights_dict: weights dictionary {<model_name>: <wight>}

        :return: Normalized weights dictionary
        """
        if weights_dict is None:
            num_of_models = len(self.routes)
            return dict(zip(self.routes.keys(), [1 / num_of_models] * num_of_models))
        weights_values = [*weights_dict.values()]
        weights_sum = np.sum(weights_values)
        if 1.0 - weights_sum <= 1e-5:
            return weights_dict
        new_weights_values = (np.array(weights_dict.values()) / weights_sum).tolist()
        return dict(zip(weights_dict.keys(), new_weights_values))

    def _update_weights(self, weights_dict):
        """
        Updated self._weights

        :param weights_dict: weights dictionary {<model_name>: <wight>}
        """
        self._weights = self._normalize_weights(weights_dict)
        for model in self.routes.keys():
            if model not in self._weights.keys():
                self._weights[model] = 0


class EnrichmentModelRouter(ModelRouter):
    """
    Model router with feature enrichment and imputing
    """

    def __init__(
        self,
        context=None,
        name: typing.Optional[str] = None,
        routes=None,
        protocol: typing.Optional[str] = None,
        url_prefix: typing.Optional[str] = None,
        health_prefix: typing.Optional[str] = None,
        feature_vector_uri: str = "",
        impute_policy: typing.Optional[dict] = None,
        **kwargs,
    ):
        """
        Model router with feature enrichment (from the feature store)

        The `EnrichmentModelRouter` class enrich the incoming event with real-time features
        read from a feature vector (in MLRun feature store) and forwards the enriched event to the child models

        The feature vector is specified using the `feature_vector_uri`, in addition an imputing policy
        can be specified to substitute None/NaN values with pre defines constant or stats.

        :param feature_vector_uri:  feature vector uri in the form: [project/]name[:tag]
        :param impute_policy: value imputing (substitute NaN/Inf values with statistical or constant value),
            you can set the `impute_policy` parameter with the imputing policy, and specify which constant or
            statistical value will be used instead of NaN/Inf value, this can be defined per column or
            for all the columns ("*"). The replaced value can be fixed number for constants or $mean, $max, $min, $std,
            $count for statistical values.
            “*” is used to specify the default for all features, example: impute_policy={"*": "$mean", "age": 33}
        :param context:       for internal use (passed in init)
        :param name:          step name
        :param routes:        for internal use (routes passed in init)
        :param protocol:      serving API protocol (default "v2")
        :param url_prefix:    url prefix for the router (default /v2/models)
        :param health_prefix: health api url prefix (default /v2/health)
        :param input_path:    when specified selects the key/path in the event to use as body this require that the
            event body will behave like a dict, example: event: {"data": {"a": 5, "b": 7}}, input_path="data.b"
            means request body will be 7.
        :param result_path:   selects the key/path in the event to write the results to this require that the event body
            will behave like a dict, example: event: {"x": 5} , result_path="resp" means the returned response will be
            written to event["y"] resulting in {"x": 5, "resp": <result>}
        :param kwargs:        extra arguments
        """
        super().__init__(
            context,
            name,
            routes,
            protocol,
            url_prefix,
            health_prefix,
            **kwargs,
        )

        self.feature_vector_uri = feature_vector_uri
        self.impute_policy = impute_policy or {}

        self._feature_service = None

    def post_init(self, mode="sync", **kwargs):
        from ..feature_store import get_feature_vector

        super().post_init(mode)
        self._feature_service = get_feature_vector(
            self.feature_vector_uri
        ).get_online_feature_service(
            impute_policy=self.impute_policy,
        )

    def preprocess(self, event):
        """Turn an entity identifier (source) to a Feature Vector"""
        if isinstance(event.body, (str, bytes)):
            event.body = json.loads(event.body)
        event.body["inputs"] = self._feature_service.get(
            event.body["inputs"], as_list=True
        )
        return event


class EnrichmentVotingEnsemble(VotingEnsemble):
    """
    Voting Ensemble with feature enrichment (from the feature store)
    """

    def __init__(
        self,
        context=None,
        name: typing.Optional[str] = None,
        routes=None,
        protocol=None,
        url_prefix: typing.Optional[str] = None,
        health_prefix: typing.Optional[str] = None,
        vote_type: typing.Optional[str] = None,
        executor_type: Union[ParallelRunnerModes, str] = ParallelRunnerModes.thread,
        prediction_col_name: typing.Optional[str] = None,
        feature_vector_uri: str = "",
        impute_policy: typing.Optional[dict] = None,
        **kwargs,
    ):
        """
        Voting Ensemble with feature enrichment (from the feature store)

        The `EnrichmentVotingEnsemble` class enables to enrich the incoming event with real-time features
        read from a feature vector (in MLRun feature store) and apply prediction logic on top of
        the different added models.

        You can use it by calling:

        - `<prefix>/<model>[/versions/<ver>]/operation`
            Sends the event to the specific <model>[/versions/<ver>]
        - `<prefix>/operation`
            Sends the event to all models and applies `vote(self, event)`

        The `VotingEnsemble` applies the following logic:
        Incoming Event -> Feature enrichment -> Send to model/s ->
        Apply all model/s logic (Preprocessing -> Prediction -> Postprocessing) ->
        Router Voting logic -> Router Postprocessing -> Response

        The feature vector is specified using the `feature_vector_uri`, in addition an imputing policy
        can be specified to substitute None/NaN values with pre defines constant or stats.

        When enabling model tracking via `set_tracking()` the ensemble logic
        predictions will appear with model name as the given VotingEnsemble name
        or "VotingEnsemble" by default.

        Example::

            # Define a serving function
            # Note: You can point the function to a file containing you own Router or Classifier Model class
            # this basic class supports sklearn based models (with `<model>.predict()` api)
            fn = mlrun.code_to_function(
                name='ensemble',
                kind='serving',
                filename='model-server.py',
                image='mlrun/mlrun')


            # Set the router class
            # You can set your own classes by simply changing the `class_name`
            fn.set_topology(
                class_name='mlrun.serving.routers.EnrichmentVotingEnsemble',
                feature_vector_uri="transactions-fraud",
                impute_policy={"*": "$mean"})

            # Add models
            fn.add_model(<model_name>, <model_path>, <model_class_name>)
            fn.add_model(<model_name>, <model_path>, <model_class_name>)

        How to extend the VotingEnsemble
        --------------------------------
        The VotingEnsemble applies its logic using the `logic(predictions)` function.
        The `logic()` function receives an array of (# samples, # predictors) which you
        can then use to apply whatever logic you may need.

        If we use this `VotingEnsemble` as an example, the `logic()` function tries to figure
        out whether you are trying to do a **classification** or a **regression** prediction by
        the prediction type or by the given `vote_type` parameter.  Then we apply the appropriate
        `max_vote()` or `mean_vote()` which calculates the actual prediction result and returns it
        as the VotingEnsemble's prediction.


        :param context:       for internal use (passed in init)
        :param name:          step name
        :param routes:        for internal use (routes passed in init)
        :param protocol:      serving API protocol (default `v2`)
        :param url_prefix:    url prefix for the router (default `/v2/models`)
        :param health_prefix: health api url prefix (default `/v2/health`)
        :param feature_vector_uri:  feature vector uri in the form `[project/]name[:tag]`
        :param impute_policy: value imputing (substitute NaN/Inf values with statistical or constant value),
            you can set the `impute_policy` parameter with the imputing policy, and specify which constant or
            statistical value will be used instead of NaN/Inf value, this can be defined per column or for all
            the columns ("*"). The replaced value can be fixed number for constants or $mean, $max, $min, $std, $count
            for statistical values. “*” is used to specify the default for all features,
            example: impute_policy={"*": "$mean", "age": 33}
        :param input_path:    when specified selects the key/path in the event to use as body this require that
            the event body will behave like a dict, example: event: {"data": {"a": 5, "b": 7}}, input_path="data.b"
            means request body will be 7.
        :param result_path:   selects the key/path in the event to write the results to this require that the event body
            will behave like a dict, example: event: {"x": 5} , result_path="resp" means the returned response will be
            written to event["y"] resulting in {"x": 5, "resp": <result>}.
        :param vote_type: Voting type to be used (from `VotingTypes`). by default will try to self-deduct upon the
                    first event:
                    * float prediction type: regression
                    * int prediction type: classification
        :param executor_type: Parallelism mechanism, out of `ParallelRunnerModes`, by default `threads`
        :param prediction_col_name: The dict key for the predictions column in the model's responses output.
            Example:
            If the model returns `{id: <id>, model_name: <name>, outputs: {..., prediction: [<predictions>], ...}}`,
            the prediction_col_name should be `prediction`. By default, `prediction`.
        :param kwargs:  extra arguments
        """
        super().__init__(
            context=context,
            name=name,
            routes=routes,
            protocol=protocol,
            url_prefix=url_prefix,
            health_prefix=health_prefix,
            vote_type=vote_type,
            executor_type=executor_type,
            prediction_col_name=prediction_col_name,
            **kwargs,
        )

        self.feature_vector_uri = feature_vector_uri
        self.impute_policy = impute_policy or {}

        self._feature_service = None

    def post_init(self, mode="sync", **kwargs):
        from ..feature_store import get_feature_vector

        super().post_init(mode)
        self._feature_service = get_feature_vector(
            self.feature_vector_uri
        ).get_online_feature_service(
            impute_policy=self.impute_policy,
        )

    def preprocess(self, event):
        """
        Turn an entity identifier (source) to a Feature Vector
        """
        if isinstance(event.body, (str, bytes)):
            event.body = json.loads(event.body)
        event.body["inputs"] = self._feature_service.get(
            event.body["inputs"], as_list=True
        )
        return event
