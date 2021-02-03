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
import mlrun

from .v2_serving import _ModelLogPusher

from io import BytesIO
from numpy.core.fromnumeric import mean
from datetime import datetime
import copy
import concurrent

from enum import Enum


class BaseModelRouter:
    """base model router class"""

    def __init__(
        self,
        context,
        name,
        routes=None,
        protocol=None,
        url_prefix=None,
        health_prefix=None,
        **kwargs,
    ):
        self.name = name
        self.context = context
        self.routes = routes
        self.protocol = protocol or "v2"
        self.url_prefix = url_prefix or f"/{self.protocol}/models"
        self.health_prefix = health_prefix or f"/{self.protocol}/health"
        self.inputs_key = "instances" if self.protocol == "v1" else "inputs"
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

        except Exception as e:
            #  if images convert to bytes
            content_type = getattr(event, "content_type") or ""
            if content_type.startswith("image/"):
                sample = BytesIO(event.body)
                parsed_event[self.inputs_key] = [sample]
            else:
                raise ValueError("Unrecognized request format: %s" % e)

        return parsed_event

    def post_init(self, mode="sync"):
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
        return event

    def do_event(self, event, *args, **kwargs):
        """handle incoming events, event is nuclio event class"""

        event = self.preprocess(event)
        event = self._pre_handle_event(event)
        if hasattr(event, "terminated") and event.terminated:
            return event
        return self.postprocess(self._handle_event(event))

    def _handle_event(self, event):
        return event

    def preprocess(self, event):
        """run tasks before processing the event"""
        return event

    def postprocess(self, event):
        """run tasks after processing the event"""
        return event


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

    array = "array"
    thread = "thread"


class VotingTypes(str, Enum):
    """Supported voting types for VotingEnsemble"""

    classification = "classification"
    regression = "regression"


class VotingEnsemble(BaseModelRouter):
    """Voting Ensemble class

        The `VotingEnsemble` class enables you to apply prediction logic on top of
        the different added models.

        You can use it by calling:
        - <prefix>/<model>[/versions/<ver>]/operation
            Sends the event to the specific <model>[/versions/<ver>]
        - <prefix>/operation
            Sends the event to all models and applies `vote(self, event)`

        The `VotingEnsemble` applies the following logic:
        Incoming Event -> Router Preprocessing -> Send to model/s ->
        Apply all model/s logic (Preprocessing -> Prediction -> Postprocessing) ->
        Router Voting logic -> Router Postprocessing -> Response

        This enables you to do the general preprocessing and postprocessing steps
        once on the router level, with only model-specific adjustments at the
        model level.

        * When enabling model tracking via `set_tracking()` the ensemble logic
        predictions will appear with model name as the given VotingEnsemble name
        or "VotingEnsemble" by default.
    """

    def __init__(
        self,
        context,
        name,
        routes=None,
        protocol=None,
        url_prefix=None,
        health_prefix=None,
        vote_type=None,
        executor_type=None,
        **kwargs,
    ):
        super().__init__(
            context, name, routes, protocol, url_prefix, health_prefix, **kwargs
        )
        self.name = name or "VotingEnsemble"
        self.vote_type = vote_type
        self.vote_flag = True if self.vote_type is not None else False
        self.executor_type = executor_type
        self._model_logger = (
            _ModelLogPusher(self, context) if context.stream.enabled else None
        )
        self.version = kwargs.get("version", "v1")
        self.log_router = True

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
            if not urlpath:
                # Only self.url_prefix was given
                return "", None, ""
            segments = urlpath.split("/")
            if len(segments) == 1:
                # Are we looking at a router level operation?
                if segments[0] in ["infer", "predict", "explain"]:
                    # We are given an operation
                    self.log_router = True
                    return self.name, None, segments[0]
            model = segments[0]
            if len(segments) > 2 and segments[1] == "versions":
                model = model + ":" + segments[2]
                segments = segments[2:]
            if len(segments) > 1:
                subpath = "/".join(segments[1:])

        if isinstance(body, dict):
            # accepting route information from body as well
            # to support streaming protocols (e.g. Kafka).
            model = model or self.name
            subpath = body.get("operation", subpath)
        if subpath is None:
            subpath = "infer"

        if model in self.routes:
            self.log_router = False
        elif model != self.name:
            models = " | ".join(self.routes.keys())
            raise ValueError(
                f"model {model} doesnt exist, available models: {models} or an operation alone for ensemble operation"
            )
        return model, self.routes[model], subpath

    def _max_vote(self, all_predictions):
        """Returns most predicted class for each event

        Args:
            all_predictions (List[List[Int]]): The predictions from all models, per event

        Returns:
            List[Int]: The most predicted class by all models, per event
        """
        return [
            max(predictions, key=predictions.count) for predictions in all_predictions
        ]

    def _mean_vote(self, all_predictions):
        """Returns mean of the predictions

        Args:
            all_predictions (List[List[float]]): The predictions from all models, per event

        Returns:
            List[Float]: The mean of predictions from all models, per event
        """
        return [mean(predictions) for predictions in all_predictions]

    def _is_int(self, value):
        return float(value).is_integer()

    def _vote(self, events):
        if "outputs" in events.body:
            # Dealing with a specific model prediction
            return events
        predictions = [model.body["outputs"] for _, model in events.body.items()]

        flattened_predictions = [
            [predictions[j][i] for j in range(len(predictions))]
            for i in range(len(predictions[0]))
        ]
        if not self.vote_flag:
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
            else:
                self.vote_type = VotingTypes.regression
            self.vote_flag = True
        if self.vote_type == VotingTypes.classification:
            flattened_predictions = [
                list(map(int, predictions)) for predictions in flattened_predictions
            ]
            result = self._max_vote(flattened_predictions)
        else:
            result = self._mean_vote(flattened_predictions)

        event = {
            "model_name": self.name,
            "outputs": result,
            "id": events.body[list(events.body.keys())[0]].body["id"],
        }

        events.body = event
        return events

    def preprocess(self, event):
        try:
            event.body = json.loads(event.body)
            return event
        except Exception:
            return event

    def do_event(self, event, *args, **kwargs):
        """handle incoming events

        Args:
            event (nuclio.Event): Incoming nuclio event

        Raises:
            ValueError: Illeagel prefix from URI

        Returns:
            Response: Event response after running the event processing logic
        """
        start = datetime.now()
        event = self.preprocess(event)
        request = event.body
        request = self.validate(request)
        event = self._pre_handle_event(event)
        if hasattr(event, "terminated") and event.terminated:
            return event
        else:
            response = self.postprocess(self._vote(self._handle_event(event)))
            if self._model_logger and self.log_router:
                if "id" not in request:
                    request["id"] = response.body["id"]
                self._model_logger.push(start, request, response.body)
            return response

    def _parallel_run(self, event, mode: str = ParallelRunnerModes.thread):
        """Executes the processing logic in parallel

        Args:
            event (nuclio.Event): Incoming event after router preprocessing
            mode (str, optional): Parallel processing method. Defaults to "thread".

        Returns:
            dict[str, nuclio.Event]: {model_name: model_response} for selected all models the registry
        """
        if mode == ParallelRunnerModes.array:
            results = {
                model_name: model.run(copy.deepcopy(event))
                for model_name, model in self.routes.items()
            }
        elif mode == ParallelRunnerModes.thread:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.routes))
            with pool as executor:
                results = []
                futures = {
                    executor.submit(self.routes[model].run, copy.copy(event)): model
                    for model in self.routes.keys()
                }
                for future in concurrent.futures.as_completed(futures):
                    child = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print("%r generated an exception: %s" % (child.fullname, exc))
                results = {event.body["model_name"]: event for event in results}
        else:
            raise ValueError(
                f"{mode} is not a supported parallel run mode, please select from "
                f"{[mode.value for mode in list(ParallelRunnerModes)]}"
            )
        return results

    def validate(self, request):
        """validate the event body (after preprocess)"""
        if self.protocol == "v2":
            if "inputs" not in request:
                raise Exception('Expected key "inputs" in request body')

            if not isinstance(request["inputs"], list):
                raise Exception('Expected "inputs" to be a list')
        return request

    def _handle_event(self, event):
        name, route, subpath = self._resolve_route(event.body, event.path)
        if not route and name != self.name:
            # if model wasn't specified return model list
            setattr(event, "terminated", True)
            event.body = {"models": list(self.routes.keys())}
            return event

        self.context.logger.debug(f"router run model {name}, op={subpath}")
        event.path = subpath
        if name == self.name:
            response = self._parallel_run(event)
            event.body = response
        else:
            response = route.run(event)
            event.body = response.body if response else None
        return event
