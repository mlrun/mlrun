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

import random
import threading
import time
import traceback
from typing import Optional

import mlrun.artifacts
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.model_monitoring
from mlrun.utils import logger, now_date

from .utils import StepToDict, _extract_input_data, _update_result_body


class V2ModelServer(StepToDict):
    def __init__(
        self,
        context=None,
        name: Optional[str] = None,
        model_path: Optional[str] = None,
        model=None,
        protocol=None,
        input_path: Optional[str] = None,
        result_path: Optional[str] = None,
        shard_by_endpoint: Optional[bool] = None,
        **kwargs,
    ):
        """base model serving class (v2), using similar API to KFServing v2 and Triton

        The class is initialized automatically by the model server and can run locally
        as part of a nuclio serverless function, or as part of a real-time pipeline
        default model url is: /v2/models/<model>[/versions/<ver>]/operation

        You need to implement two mandatory methods:
          load()     - download the model file(s) and load the model into memory
          predict()  - accept request payload and return prediction/inference results

        you can override additional methods : preprocess, validate, postprocess, explain
        you can add custom api endpoint by adding method op_xx(event), will be invoked by
        calling the <model-url>/xx (operation = xx)

        model server classes are subclassed (subclass implements the `load()` and `predict()` methods)
        the subclass can be added to a serving graph or to a model router

        defining a sub class::

            class MyClass(V2ModelServer):
                def load(self):
                    # load and initialize the model and/or other elements
                    model_file, extra_data = self.get_model(suffix=".pkl")
                    self.model = load(open(model_file, "rb"))

                def predict(self, request):
                    events = np.array(request["inputs"])
                    dmatrix = xgb.DMatrix(events)
                    result: xgb.DMatrix = self.model.predict(dmatrix)
                    return {"outputs": result.tolist()}

        usage example::

            # adding a model to a serving graph using the subclass MyClass
            # MyClass will be initialized with the name "my", the model_path, and an arg called my_param
            graph = fn.set_topology("router")
            fn.add_model("my", class_name="MyClass", model_path="<model-uri>>", my_param=5)

        :param context:    for internal use (passed in init)
        :param name:       step name
        :param model_path: model file/dir or artifact path
        :param model:      model object (for local testing)
        :param protocol:   serving API protocol (default "v2")
        :param input_path:    when specified selects the key/path in the event to use as body
                              this require that the event body will behave like a dict, example:
                              event: {"data": {"a": 5, "b": 7}}, input_path="data.b" means request body will be 7
        :param result_path:   selects the key/path in the event to write the results to
                              this require that the event body will behave like a dict, example:
                              event: {"x": 5} , result_path="resp" means the returned response will be written
                              to event["y"] resulting in {"x": 5, "resp": <result>}
        :param shard_by_endpoint: whether to use the endpoint as the partition/sharding key when writing to model
                                  monitoring stream. Defaults to True.
        :param kwargs:     extra arguments (can be accessed using self.get_param(key))
        """
        self.name = name
        self.context = context
        self.ready = False
        self.error = ""
        self.protocol = protocol or "v2"
        self.model_path = model_path
        self.model_spec: Optional[mlrun.artifacts.ModelArtifact] = None
        self._input_path = input_path
        self._result_path = result_path
        self._kwargs = kwargs  # for to_dict()
        self._params = kwargs
        self.metrics = {}
        self.labels = {}
        self.model = None
        if model:
            self.model = model
            self.ready = True
        self.model_endpoint_uid = kwargs.get("model_endpoint_uid", None)
        self.shard_by_endpoint = shard_by_endpoint
        self._model_logger = None
        self.initialized = False
        self.output_schema = kwargs.get("outputs", [])

    def _load_and_update_state(self):
        try:
            self.load()
        except Exception as exc:
            self.error = exc
            self.context.logger.error(traceback.format_exc())
            raise RuntimeError(f"failed to load model {self.name}") from exc
        self.ready = True
        self.context.logger.info(f"model {self.name} was loaded")

    def post_init(self, mode="sync", **kwargs):
        """sync/async model loading, for internal use"""
        if not self.ready:
            if mode == "async":
                t = threading.Thread(target=self._load_and_update_state)
                t.start()
                self.context.logger.info(f"started async model loading for {self.name}")
            else:
                self._load_and_update_state()

        if self.ready and not self.context.is_mock and not self.model_spec:
            self.get_model()

        if self.model_spec:
            self.output_schema = self.output_schema or [
                feature.name for feature in self.model_spec.outputs
            ]

        if (
            kwargs.get("endpoint_type", mlrun.common.schemas.EndpointType.LEAF_EP)
            == mlrun.common.schemas.EndpointType.NODE_EP
        ):
            self._initialize_model_logger()

    def _lazy_init(self, event):
        if event and isinstance(event, dict) and not self.initialized:
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

    def get_param(self, key: str, default=None):
        """get param by key (specified in the model or the function)"""
        if key in self._params:
            return self._params.get(key)
        return self.context.get_param(key, default=default)

    def set_metric(self, name: str, value):
        """set real time metric (for model monitoring)"""
        self.metrics[name] = value

    def get_model(self, suffix=""):
        """get the model file(s) and metadata from model store

        the method returns a path to the model file and the extra data (dict of dataitem objects)
        it also loads the model metadata into the self.model_spec attribute, allowing direct access
        to all the model metadata attributes.

        get_model is usually used in the model .load() method to init the model
        Examples
        --------
        ::

            def load(self):
                model_file, extra_data = self.get_model(suffix=".pkl")
                self.model = load(open(model_file, "rb"))
                categories = extra_data["categories"].as_df()

        Parameters
        ----------
        suffix : str
            optional, model file suffix (when the model_path is a directory)

        Returns
        -------
        str
            (local) model file
        dict
            extra dataitems dictionary

        """
        if self.model_path:
            model_file, self.model_spec, extra_dataitems = mlrun.artifacts.get_model(
                self.model_path, suffix
            )
            if self.model_spec and self.model_spec.parameters:
                for key, value in self.model_spec.parameters.items():
                    self._params[key] = value
            return model_file, extra_dataitems
        return None, None

    def load(self):
        """model loading function, see also .get_model() method"""
        if not self.ready and not self.model:
            raise ValueError("please specify a load method or a model object")

    def _check_readiness(self, event):
        if self.ready:
            return
        if not event.trigger or event.trigger.kind in ["http", ""]:
            raise RuntimeError(f"model {self.name} is not ready yet")
        self.context.logger.info(f"waiting for model {self.name} to load")
        for i in range(50):  # wait up to 4.5 minutes
            time.sleep(5)
            if self.ready:
                return
        raise RuntimeError(f"model {self.name} is not ready {self.error}")

    def _pre_event_processing_actions(self, event, event_body, op):
        self._check_readiness(event)
        if "_dict" in op:
            event_body = self._inputs_to_list(event_body)
        request = self.preprocess(event_body, op)
        return self.validate(request, op)

    def do_event(self, event, *args, **kwargs):
        """main model event handler method"""
        if not self.initialized:
            self._lazy_init(event.body)
        start = now_date()
        original_body = event.body
        event_body = _extract_input_data(self._input_path, event.body)
        event_id = event.id
        op = event.path.strip("/")

        partition_key = (
            self.model_endpoint_uid if self.shard_by_endpoint is not False else None
        )

        if event_body and isinstance(event_body, dict):
            op = op or event_body.get("operation")
            event_id = event_body.get("id", event_id)
        if not op and event.method != "GET":
            op = "infer"

        if (
            op == "predict"
            or op == "infer"
            or op == "infer_dict"
            or op == "predict_dict"
        ):
            # predict operation
            request = self._pre_event_processing_actions(event, event_body, op)
            try:
                outputs = self.predict(request)
            except Exception as exc:
                request["id"] = event_id
                if self._model_logger:
                    self._model_logger.push(
                        start,
                        request,
                        op=op,
                        error=exc,
                        partition_key=partition_key,
                    )
                raise exc

            response = {
                "id": event_id,
                "model_name": self.name.split(":")[0],
                "outputs": outputs,
                "timestamp": start.isoformat(sep=" ", timespec="microseconds"),
            }
            if self.model_endpoint_uid:
                response["model_endpoint_uid"] = self.model_endpoint_uid
        elif op == "ready" and event.method == "GET":
            # get model health operation
            setattr(event, "terminated", True)
            if self.ready:
                # Generate a response, confirming that the model is ready
                event.body = self.context.Response(
                    status_code=200,
                    body=bytes(
                        f"Model {self.name} is ready (event_id = {event_id})",
                        encoding="utf-8",
                    ),
                )

            else:
                event.body = self.context.Response(
                    status_code=408, body=b"model not ready"
                )

            return event

        elif op == "" and event.method == "GET":
            # get model metadata operation
            setattr(event, "terminated", True)
            event_body = {
                "name": self.name.split(":")[0],
                "model_endpoint_uid": self.model_endpoint_uid or "",
                "inputs": [],
                "outputs": [],
            }
            if self.model_spec:
                event_body["inputs"] = self.model_spec.inputs.to_dict()
                event_body["outputs"] = self.model_spec.outputs.to_dict()
            event.body = _update_result_body(
                self._result_path, original_body, event_body
            )
            return event

        elif op == "explain":
            # explain operation
            request = self._pre_event_processing_actions(event, event_body, op)
            try:
                outputs = self.explain(request)
            except Exception as exc:
                request["id"] = event_id
                if self._model_logger:
                    self._model_logger.push(
                        start,
                        request,
                        op=op,
                        error=exc,
                        partition_key=partition_key,
                    )
                raise exc

            response = {
                "id": event_id,
                "model_name": self.name,
                "outputs": outputs,
            }
            if self.model_endpoint_uid:
                response["model_endpoint_uid"] = self.model_endpoint_uid

        elif hasattr(self, "op_" + op):
            # custom operation (child methods starting with "op_")
            response = getattr(self, "op_" + op)(event)
            event.body = _update_result_body(self._result_path, original_body, response)
            return event

        else:
            raise ValueError(f"illegal model operation {op}, method={event.method}")

        response = self.postprocess(response)
        if self._model_logger:
            inputs, outputs = self.logged_results(request, response, op)
            if inputs is None and outputs is None:
                self._model_logger.push(
                    start, request, response, op, partition_key=partition_key
                )
            else:
                track_request = {"id": event_id, "inputs": inputs or []}
                track_response = {"outputs": outputs or []}
                # TODO : check dict/list
                self._model_logger.push(
                    start,
                    track_request,
                    track_response,
                    op,
                    partition_key=partition_key,
                )
        event.body = _update_result_body(self._result_path, original_body, response)
        return event

    def logged_results(self, request: dict, response: dict, op: str):
        """hook for controlling which results are tracked by the model monitoring

        this hook allows controlling which input/output data is logged by the model monitoring
        allow filtering out columns or adding custom values, can also be used to monitor derived metrics
        for example in image classification calculate and track the RGB values vs the image bitmap

        the request["inputs"] holds a list of input values/arrays, the response["outputs"] holds a list of
        corresponding output values/arrays (the schema of the input/output fields is stored in the model object),
        this method should return lists of alternative inputs and outputs which will be monitored

        :param request:   predict/explain request, see model serving docs for details
        :param response:  result from the model predict/explain (after postprocess())
        :param op:        operation (predict/infer or explain)
        :returns: the input and output lists to track
        """
        return None, None

    def validate(self, request, operation):
        """validate the event body (after preprocess)"""
        if self.protocol == "v2":
            if "inputs" not in request:
                raise Exception('Expected key "inputs" in request body')

            if not isinstance(request["inputs"], list):
                raise Exception('Expected "inputs" to be a list')

        return request

    def preprocess(self, request: dict, operation) -> dict:
        """preprocess the event body before validate and action"""
        return request

    def postprocess(self, request: dict) -> dict:
        """postprocess, before returning response"""
        return request

    def predict(self, request: dict) -> list:
        """model prediction operation
        :return: list with the model prediction results (can be multi-port) or list of lists for multiple predictions
        """
        raise NotImplementedError()

    def explain(self, request: dict) -> dict:
        """model explain operation"""
        raise NotImplementedError()

    def _inputs_to_list(self, request: dict) -> dict:
        """
        Convert the inputs from list of dictionary / dictionary to list of lists / list
        where the internal list order is according to the ArtifactModel inputs.

        :param request: event
        :return: evnet body converting the inputs to be list of lists
        """
        if self.model_spec and self.model_spec.inputs:
            input_order = [feature.name for feature in self.model_spec.inputs]
        else:
            raise mlrun.MLRunInvalidArgumentError(
                "In order to use predict_dict or infer_dict operation you have to provide `model_path` "
                "to the model server and to load it by `load()` function"
            )
        inputs = request.get("inputs")
        try:
            if isinstance(inputs, list) and all(
                isinstance(item, dict) for item in inputs
            ):
                new_inputs = [
                    [input_dict[key] for key in input_order] for input_dict in inputs
                ]
            elif isinstance(inputs, dict):
                new_inputs = [inputs[key] for key in input_order]
            else:
                raise mlrun.MLRunInvalidArgumentError(
                    "When using predict_dict or infer_dict operation the inputs must be "
                    "of type `list[dict]` or `dict`"
                )
        except KeyError:
            raise mlrun.MLRunInvalidArgumentError(
                f"Input dictionary don't contain all the necessary input keys : {input_order}"
            )
        request["inputs"] = new_inputs
        return request

    def _initialize_model_logger(self):
        server: mlrun.serving.GraphServer = getattr(
            self.context, "_server", None
        ) or getattr(self.context, "server", None)
        if not self.context.is_mock or self.context.monitoring_mock:
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
                    if (
                        background_task.status.state
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

                else:  # in progress
                    logger.debug(
                        f"Model endpoint creation task is still in progress with the current state: "
                        f"{background_task.status.state}.",
                        name=self.name,
                    )
            else:
                logger.debug(
                    "Model endpoint creation task name not provided",
                )


class _ModelLogPusher:
    def __init__(self, model: V2ModelServer, context, output_stream=None):
        self.model = model
        self.verbose = context.verbose
        self.hostname = context.stream.hostname
        self.function_uri = context.stream.function_uri
        self.sampling_percentage = float(context.get_param("sampling_percentage", 100))
        self.output_stream = output_stream or context.stream.output_stream
        self._worker = context.worker_id

    def base_data(self):
        base_data = {
            "class": self.model.__class__.__name__,
            "worker": self._worker,
            "model": self.model.name,
            "host": self.hostname,
            "function_uri": self.function_uri,
            "endpoint_id": self.model.model_endpoint_uid,
            "sampling_percentage": self.sampling_percentage,
        }
        if getattr(self.model, "labels", None):
            base_data["labels"] = self.model.labels
        return base_data

    def push(self, start, request, resp=None, op=None, error=None, partition_key=None):
        start_str = start.isoformat(sep=" ", timespec="microseconds")
        if error:
            data = self.base_data()
            data["request"] = request
            data["op"] = op
            data["when"] = start_str
            message = str(error)
            if self.verbose:
                message = f"{message}\n{traceback.format_exc()}"
            data["error"] = message
            self.output_stream.push([data], partition_key=partition_key)
            return

        if self.output_stream:
            # Ensure that the inputs are a list of lists
            request["inputs"] = (
                request["inputs"]
                if not any(not isinstance(req, list) for req in request["inputs"])
                else [request["inputs"]]
            )
            microsec = (now_date() - start).microseconds

            if self.sampling_percentage != 100:
                # Randomly select a subset of the requests based on the percentage
                num_of_inputs = len(request["inputs"])
                sampled_requests_indices = self._pick_random_requests(
                    num_of_inputs, self.sampling_percentage
                )
                if not sampled_requests_indices:
                    # No events were selected for sampling
                    return

                request["inputs"] = [
                    request["inputs"][i] for i in sampled_requests_indices
                ]

                if resp and "outputs" in resp and isinstance(resp["outputs"], list):
                    resp["outputs"] = [
                        resp["outputs"][i] for i in sampled_requests_indices
                    ]
                if self.model.output_schema and len(self.model.output_schema) != len(
                    resp["outputs"][0]
                ):
                    logger.info(
                        "The number of outputs returned by the model does not match the number of outputs "
                        "specified in the model endpoint.",
                        model_endpoint=self.model.name,
                        model_endpoint_id=self.model.model_endpoint_uid,
                        output_len=len(resp["outputs"][0]),
                        schema_len=len(self.model.output_schema),
                    )

            data = self.base_data()
            data["request"] = request
            data["op"] = op
            data["resp"] = resp
            data["when"] = start_str
            data["microsec"] = microsec
            if getattr(self.model, "metrics", None):
                data["metrics"] = self.model.metrics
            data["effective_sample_count"] = len(request["inputs"])
            self.output_stream.push([data], partition_key=partition_key)

    @staticmethod
    def _pick_random_requests(num_of_reqs: int, percentage: float) -> list[int]:
        """
        Randomly selects indices of requests to sample based on the given percentage

        :param num_of_reqs: Number of requests to select from
        :param percentage: Sample percentage for each request
        :return: A list containing the indices of the selected requests
        """

        return [
            req for req in range(num_of_reqs) if random.random() < (percentage / 100)
        ]
