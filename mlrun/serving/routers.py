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

import concurrent
import copy
import json
import traceback
from enum import Enum
from io import BytesIO

from numpy.core.fromnumeric import mean

import mlrun
from mlrun.utils import logger, now_date, parse_versioned_object_uri

from ..api.schemas import (
    ModelEndpoint,
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from ..config import config
from ..utils.model_monitoring import EndpointType
from .utils import RouterToDict, _extract_input_data, _update_result_body
from .v2_serving import _ModelLogPusher


class BaseModelRouter(RouterToDict):
    """base model router class"""

    def __init__(
        self,
        context=None,
        name: str = None,
        routes=None,
        protocol: str = None,
        url_prefix: str = None,
        health_prefix: str = None,
        input_path: str = None,
        result_path: str = None,
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
                raise ValueError(f"Unrecognized request format: {exc}")

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


class OperationTypes(str, Enum):
    """Supported opreations for VotingEnsemble"""

    infer = "infer"
    predict = "predict"
    explain = "explain"


class VotingEnsemble(BaseModelRouter):
    def __init__(
        self,
        context=None,
        name: str = None,
        routes=None,
        protocol: str = None,
        url_prefix: str = None,
        health_prefix: str = None,
        vote_type=None,
        executor_type=None,
        prediction_col_name=None,
        **kwargs,
    ):
        """Voting Ensemble

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

        Example::

            # Define a serving function
            # Note: You can point the function to a file containing you own Router or Classifier Model class
            #       this basic class supports sklearn based models (with `<model>.predict()` api)
            fn = mlrun.code_to_function(name='ensemble',
                                        kind='serving',
                                        filename='model-server.py'
                                        image='mlrun/ml-models')

            # Set the router class
            # You can set your own classes by simply changing the `class_name`
            fn.set_topology(class_name='mlrun.serving.routers.VotingEnsemble')

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
        :param executor_type: Parallelism mechanism, out of `ParallelRunnerModes`, by default `threads`
        :param prediction_col_name: The dict key for the predictions column in the model's responses output.
                              Example: If the model returns
                                       {id: <id>, model_name: <name>, outputs: {..., prediction: [<predictions>], ...}}
                                       the prediction_col_name should be `prediction`.
                              by default, `prediction`
        :param kwargs:        extra arguments
        """
        super().__init__(
            context, name, routes, protocol, url_prefix, health_prefix, **kwargs
        )
        self.name = name or "VotingEnsemble"
        self.vote_type = vote_type
        self.vote_flag = True if self.vote_type is not None else False
        self.executor_type = executor_type
        self._model_logger = (
            _ModelLogPusher(self, context)
            if context and context.stream.enabled
            else None
        )
        self.version = kwargs.get("version", "v1")
        self.log_router = True
        self.prediction_col_name = prediction_col_name or "prediction"
        self.format_response_with_col_name_flag = False
        self.model_endpoint_uid = None

    def post_init(self, mode="sync"):
        server = getattr(self.context, "_server", None) or getattr(
            self.context, "server", None
        )
        if not server:
            logger.warn("GraphServer not initialized for VotingEnsemble instance")
            return

        if not self.context.is_mock or self.context.server.track_models:
            self.model_endpoint_uid = _init_endpoint_record(server, self)

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

    def logic(self, predictions):
        self.context.logger.debug(f"Applying logic to {predictions}")
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
            votes = self._max_vote(int_predictions)
        else:
            votes = self._mean_vote(predictions)

        return votes

    def _apply_logic(self, predictions):
        """Reduces a list of k predictions from n models to k predictions according to voting logic

        Parameters
        ----------
        predictions : List[List]
            A list of sample predictions by models
            e.g. predictions[model][prediction]

        Returns
        -------
        List
            List of the resulting voted predictions
        """

        # Flatten predictions by sample instead of by model as received
        flattened_predictions = [
            [predictions[j][i] for j in range(len(predictions))]
            for i in range(len(predictions[0]))
        ]

        return self.logic(flattened_predictions)

    def do_event(self, event, *args, **kwargs):
        """Handles incoming requests.

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

        # Should we terminate the event?
        if hasattr(event, "terminated") and event.terminated:
            event.body = _update_result_body(
                self._result_path, original_body, event.body
            )
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
            event.body = {"models": list(self.routes.keys()) + [self.name]}
            event.body = _update_result_body(
                self._result_path, original_body, event.body
            )
            return event
        else:
            # Verify we use the V2 protocol
            request = self.validate(event.body)

            # If this is a Router Operation
            if name == self.name:
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
                if self.version:
                    response_body["model_version"] = self.version
                response.body = response_body
            # A specific model event
            else:
                response = route.run(event)

        response = self.postprocess(response)

        if self._model_logger and self.log_router:
            if "id" not in request:
                request["id"] = response.body["id"]
            self._model_logger.push(start, request, response.body)
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
        if type(response) == list:
            return response
        try:
            self.format_response_with_col_name_flag = True
            return response[self.prediction_col_name]
        except KeyError:
            raise ValueError(
                f"The given `prediction_col_name` ({self.prediction_col_name}) does not exist "
                f"in the model's response ({response.keys()})"
            )

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
                model_name: model.run(copy.copy(event))
                for model_name, model in self.routes.items()
            }
        elif mode == ParallelRunnerModes.thread:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.routes))
            with pool as executor:
                results = []
                futures = [
                    executor.submit(self.routes[model].run, copy.copy(event))
                    for model in self.routes.keys()
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print(f"child route generated an exception: {exc}")
                results = [
                    self.extract_results_from_response(event.body["outputs"])
                    for event in results
                ]
                self.context.logger.debug(f"Collected results from models: {results}")
        else:
            raise ValueError(
                f"{mode} is not a supported parallel run mode, please select from "
                f"{[mode.value for mode in list(ParallelRunnerModes)]}"
            )
        return results

    def validate(self, request):
        """Validate the event body (after preprocessing)

        Parameters
        ----------
        request : dict
            Event body.

        Returns
        -------
        dict
            Event body after validation

        Raises
        ------
        Exception
            `inputs` key not found in `request`
        Exception
            `inputs` should be of type List
        """
        if self.protocol == "v2":
            if "inputs" not in request:
                raise Exception('Expected key "inputs" in request body')

            if not isinstance(request["inputs"], list):
                raise Exception('Expected "inputs" to be a list')
        return request


def _init_endpoint_record(graph_server, voting_ensemble: VotingEnsemble):
    logger.info("Initializing endpoint records")

    endpoint_uid = None

    try:
        project, uri, tag, hash_key = parse_versioned_object_uri(
            graph_server.function_uri
        )

        if voting_ensemble.version:
            versioned_model_name = f"{voting_ensemble.name}:{voting_ensemble.version}"
        else:
            versioned_model_name = f"{voting_ensemble.name}:latest"

        children_uids = []
        for _, c in voting_ensemble.routes.items():
            if hasattr(c, "endpoint_uid"):
                children_uids.append(c.endpoint_uid)

        model_endpoint = ModelEndpoint(
            metadata=ModelEndpointMetadata(project=project),
            spec=ModelEndpointSpec(
                function_uri=graph_server.function_uri,
                model=versioned_model_name,
                model_class=voting_ensemble.__class__.__name__,
                stream_path=config.model_endpoint_monitoring.store_prefixes.default.format(
                    project=project, kind="stream"
                ),
                active=True,
            ),
            status=ModelEndpointStatus(
                children=list(voting_ensemble.routes.keys()),
                endpoint_type=EndpointType.ROUTER,
                children_uids=children_uids,
            ),
        )
        endpoint_uid = model_endpoint.metadata.uid

        db = mlrun.get_run_db()

        db.create_or_patch_model_endpoint(
            project=project,
            endpoint_id=model_endpoint.metadata.uid,
            model_endpoint=model_endpoint,
        )

        for model_endpoint in children_uids:
            # here to update that it is a node now
            current_endpoint = db.get_model_endpoint(
                project=project, endpoint_id=model_endpoint
            )
            current_endpoint.status.endpoint_type = EndpointType.LEAF_EP

            db.create_or_patch_model_endpoint(
                project=project,
                endpoint_id=model_endpoint,
                model_endpoint=current_endpoint,
            )

    except Exception as exc:
        logger.warning(
            "Failed creating model endpoint record",
            exc=exc,
            traceback=traceback.format_exc(),
        )
    return endpoint_uid


class EnrichmentModelRouter(ModelRouter):
    """model router with feature enrichment and imputing"""

    def __init__(
        self,
        context=None,
        name: str = None,
        routes=None,
        protocol: str = None,
        url_prefix: str = None,
        health_prefix: str = None,
        feature_vector_uri: str = "",
        impute_policy: dict = {},
        **kwargs,
    ):
        """Model router with feature enrichment (from the feature store)

        The `EnrichmentModelRouter` class enrich the incoming event with real-time features
        read from a feature vector (in MLRun feature store) and forwards the enriched event to the child models

        The feature vector is specified using the `feature_vector_uri`, in addition an imputing policy
        can be specified to substitute None/NaN values with pre defines constant or stats.

        :param feature_vector_uri :  feature vector uri in the form: [project/]name[:tag]
        :param impute_policy : value imputing (substitute NaN/Inf values with statistical or constant value),
                              you can set the `impute_policy` parameter with the imputing policy, and specify which
                              constant or statistical value will be used instead of NaN/Inf value, this can be defined
                              per column or for all the columns ("*"). the replaced value can be fixed number for
                              constants or $mean, $max, $min, $std, $count for statistical values.
                              “*” is used to specify the default for all features, example:
                              impute_policy={"*": "$mean", "age": 33}
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
        super().__init__(
            context, name, routes, protocol, url_prefix, health_prefix, **kwargs,
        )

        self.feature_vector_uri = feature_vector_uri
        self.impute_policy = impute_policy

        self._feature_service = None

    def post_init(self, mode="sync"):
        super().post_init(mode)
        self._feature_service = mlrun.feature_store.get_online_feature_service(
            feature_vector=self.feature_vector_uri, impute_policy=self.impute_policy,
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
    """Voting Ensemble with feature enrichment (from the feature store)"""

    def __init__(
        self,
        context=None,
        name: str = None,
        routes=None,
        protocol=None,
        url_prefix: str = None,
        health_prefix: str = None,
        vote_type: str = None,
        executor_type=None,
        prediction_col_name=None,
        feature_vector_uri: str = "",
        impute_policy: dict = {},
        **kwargs,
    ):
        """Voting Ensemble with feature enrichment (from the feature store)

        The `EnrichmentVotingEnsemble` class enables to enrich the incoming event with real-time features
        read from a feature vector (in MLRun feature store) and apply prediction logic on top of
        the different added models.

        You can use it by calling:
        - <prefix>/<model>[/versions/<ver>]/operation
            Sends the event to the specific <model>[/versions/<ver>]
        - <prefix>/operation
            Sends the event to all models and applies `vote(self, event)`

        The `VotingEnsemble` applies the following logic:
        Incoming Event -> Feature enrichment -> Send to model/s ->
        Apply all model/s logic (Preprocessing -> Prediction -> Postprocessing) ->
        Router Voting logic -> Router Postprocessing -> Response

        The feature vector is specified using the `feature_vector_uri`, in addition an imputing policy
        can be specified to substitute None/NaN values with pre defines constant or stats.

        * When enabling model tracking via `set_tracking()` the ensemble logic
        predictions will appear with model name as the given VotingEnsemble name
        or "VotingEnsemble" by default.

        Example::

            # Define a serving function
            # Note: You can point the function to a file containing you own Router or Classifier Model class
            #       this basic class supports sklearn based models (with `<model>.predict()` api)
            fn = mlrun.code_to_function(name='ensemble',
                                        kind='serving',
                                        filename='model-server.py'
                                        image='mlrun/ml-models')

            # Set the router class
            # You can set your own classes by simply changing the `class_name`
            fn.set_topology(class_name='mlrun.serving.routers.EnrichmentVotingEnsemble',
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
        :param protocol:      serving API protocol (default "v2")
        :param url_prefix:    url prefix for the router (default /v2/models)
        :param health_prefix: health api url prefix (default /v2/health)
        :param feature_vector_uri :  feature vector uri in the form: [project/]name[:tag]
        :param impute_policy : value imputing (substitute NaN/Inf values with statistical or constant value),
                              you can set the `impute_policy` parameter with the imputing policy, and specify which
                              constant or statistical value will be used instead of NaN/Inf value, this can be defined
                              per column or for all the columns ("*").
                              the replaced value can be fixed number for constants or $mean, $max, $min, $std, $count
                              for statistical values. “*” is used to specify the default for all features, example:
                              impute_policy={"*": "$mean", "age": 33}
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
        :param executor_type: Parallelism mechanism, out of `ParallelRunnerModes`, by default `threads`
        :param prediction_col_name: The dict key for the predictions column in the model's responses output.
                              Example: If the model returns
                                       {id: <id>, model_name: <name>, outputs: {..., prediction: [<predictions>], ...}}
                                       the prediction_col_name should be `prediction`.
                              by default, `prediction`
        :param kwargs:        extra arguments
        """
        super().__init__(
            context,
            name,
            routes,
            protocol,
            url_prefix,
            health_prefix,
            vote_type,
            executor_type,
            prediction_col_name,
            **kwargs,
        )

        self.feature_vector_uri = feature_vector_uri
        self.impute_policy = impute_policy

        self._feature_service = None

    def post_init(self, mode="sync"):
        super().post_init(mode)
        self._feature_service = mlrun.feature_store.get_online_feature_service(
            feature_vector=self.feature_vector_uri, impute_policy=self.impute_policy,
        )

    def preprocess(self, event):
        """Turn an entity identifier (source) to a Feature Vector"""
        if isinstance(event.body, (str, bytes)):
            event.body = json.loads(event.body)
        event.body["inputs"] = self._feature_service.get(
            event.body["inputs"], as_list=True
        )
        return event
