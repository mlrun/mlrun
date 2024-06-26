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

import threading
import time
import traceback
from typing import Dict, Union

import mlrun.common.model_monitoring
import mlrun.common.schemas.model_monitoring
from mlrun.artifacts import ModelArtifact  # noqa: F401
from mlrun.config import config
from mlrun.utils import logger, now_date

from ..common.helpers import parse_versioned_object_uri
from .server import GraphServer
from .utils import StepToDict, _extract_input_data, _update_result_body


class V2ModelServer(StepToDict):
    """base model serving class (v2), using similar API to KFServing v2 and Triton"""

    def __init__(
        self,
        context=None,
        name: str = None,
        model_path: str = None,
        model=None,
        protocol=None,
        input_path: str = None,
        result_path: str = None,
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
                    model_file, extra_data = self.get_model(suffix='.pkl')
                    self.model = load(open(model_file, "rb"))

                def predict(self, request):
                    events = np.array(request['inputs'])
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
        :param kwargs:     extra arguments (can be accessed using self.get_param(key))
        """
        self.name = name
        self.version = ""
        if name and ":" in name:
            self.name, self.version = name.split(":", 1)
        self.context = context
        self.ready = False
        self.error = ""
        self.protocol = protocol or "v2"
        self.model_path = model_path
        self.model_spec: mlrun.artifacts.ModelArtifact = None
        self._input_path = input_path
        self._result_path = result_path
        self._kwargs = kwargs  # for to_dict()
        self._params = kwargs
        self._model_logger = (
            _ModelLogPusher(self, context)
            if context and context.stream.enabled
            else None
        )

        self.metrics = {}
        self.labels = {}
        self.model = None
        if model:
            self.model = model
            self.ready = True
        self.model_endpoint_uid = None

    def _load_and_update_state(self):
        try:
            self.load()
        except Exception as exc:
            self.error = exc
            self.context.logger.error(traceback.format_exc())
            raise RuntimeError(f"failed to load model {self.name}") from exc
        self.ready = True
        self.context.logger.info(f"model {self.name} was loaded")

    def post_init(self, mode="sync"):
        """sync/async model loading, for internal use"""
        if not self.ready:
            if mode == "async":
                t = threading.Thread(target=self._load_and_update_state)
                t.start()
                self.context.logger.info(f"started async model loading for {self.name}")
            else:
                self._load_and_update_state()

        server = getattr(self.context, "_server", None) or getattr(
            self.context, "server", None
        )
        if not server:
            logger.warn("GraphServer not initialized for VotingEnsemble instance")
            return

        if not self.context.is_mock or self.context.server.track_models:
            self.model_endpoint_uid = _init_endpoint_record(
                graph_server=server, model=self
            )

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
                model_file, extra_data = self.get_model(suffix='.pkl')
                self.model = load(open(model_file, "rb"))
                categories = extra_data['categories'].as_df()

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
        model_file, self.model_spec, extra_dataitems = mlrun.artifacts.get_model(
            self.model_path, suffix
        )
        if self.model_spec and self.model_spec.parameters:
            for key, value in self.model_spec.parameters.items():
                self._params[key] = value
        return model_file, extra_dataitems

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
        request = self.preprocess(event_body, op)
        return self.validate(request, op)

    def do_event(self, event, *args, **kwargs):
        """main model event handler method"""
        start = now_date()
        original_body = event.body
        event_body = _extract_input_data(self._input_path, event.body)
        event_id = event.id
        op = event.path.strip("/")
        if event_body and isinstance(event_body, dict):
            op = op or event_body.get("operation")
            event_id = event_body.get("id", event_id)
        if not op and event.method != "GET":
            op = "infer"

        if op == "predict" or op == "infer":
            # predict operation
            request = self._pre_event_processing_actions(event, event_body, op)
            try:
                outputs = self.predict(request)
            except Exception as exc:
                request["id"] = event_id
                if self._model_logger:
                    self._model_logger.push(start, request, op=op, error=exc)
                raise exc

            response = {
                "id": event_id,
                "model_name": self.name,
                "outputs": outputs,
            }
            if self.version:
                response["model_version"] = self.version

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
                "name": self.name,
                "version": self.version,
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
                    self._model_logger.push(start, request, op=op, error=exc)
                raise exc

            response = {
                "id": event_id,
                "model_name": self.name,
                "outputs": outputs,
            }
            if self.version:
                response["model_version"] = self.version

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
                self._model_logger.push(start, request, response, op)
            else:
                track_request = {"id": event_id, "inputs": inputs or []}
                track_response = {"outputs": outputs or []}
                self._model_logger.push(start, track_request, track_response, op)
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

    def preprocess(self, request: Dict, operation) -> Dict:
        """preprocess the event body before validate and action"""
        return request

    def postprocess(self, request: Dict) -> Dict:
        """postprocess, before returning response"""
        return request

    def predict(self, request: Dict) -> Dict:
        """model prediction operation"""
        raise NotImplementedError()

    def explain(self, request: Dict) -> Dict:
        """model explain operation"""
        raise NotImplementedError()


class _ModelLogPusher:
    def __init__(self, model, context, output_stream=None):
        self.model = model
        self.verbose = context.verbose
        self.hostname = context.stream.hostname
        self.function_uri = context.stream.function_uri
        self.stream_path = context.stream.stream_uri
        self.stream_batch = int(context.get_param("log_stream_batch", 1))
        self.stream_sample = int(context.get_param("log_stream_sample", 1))
        self.output_stream = output_stream or context.stream.output_stream
        self._worker = context.worker_id
        self._sample_iter = 0
        self._batch_iter = 0
        self._batch = []

    def base_data(self):
        base_data = {
            "class": self.model.__class__.__name__,
            "worker": self._worker,
            "model": self.model.name,
            "version": self.model.version,
            "host": self.hostname,
            "function_uri": self.function_uri,
        }
        if getattr(self.model, "labels", None):
            base_data["labels"] = self.model.labels
        return base_data

    def push(self, start, request, resp=None, op=None, error=None):
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
            self.output_stream.push([data])
            return

        self._sample_iter = (self._sample_iter + 1) % self.stream_sample
        if self.output_stream and self._sample_iter == 0:
            microsec = (now_date() - start).microseconds

            if self.stream_batch > 1:
                if self._batch_iter == 0:
                    self._batch = []
                self._batch.append(
                    [request, op, resp, str(start), microsec, self.model.metrics]
                )
                self._batch_iter = (self._batch_iter + 1) % self.stream_batch

                if self._batch_iter == 0:
                    data = self.base_data()
                    data["headers"] = [
                        "request",
                        "op",
                        "resp",
                        "when",
                        "microsec",
                        "metrics",
                    ]
                    data["values"] = self._batch
                    self.output_stream.push([data])
            else:
                data = self.base_data()
                data["request"] = request
                data["op"] = op
                data["resp"] = resp
                data["when"] = start_str
                data["microsec"] = microsec
                if getattr(self.model, "metrics", None):
                    data["metrics"] = self.model.metrics
                self.output_stream.push([data])


def _init_endpoint_record(
    graph_server: GraphServer, model: V2ModelServer
) -> Union[str, None]:
    """
    Initialize model endpoint record and write it into the DB. In general, this method retrieve the unique model
    endpoint ID which is generated according to the function uri and the model version. If the model endpoint is
    already exist in the DB, we skip the creation process. Otherwise, it writes the new model endpoint record to the DB.

    :param graph_server: A GraphServer object which will be used for getting the function uri.
    :param model:        Base model serving class (v2). It contains important details for the model endpoint record
                         such as model name, model path, and model version.

    :return: Model endpoint unique ID.
    """

    logger.info("Initializing endpoint records")

    # Generate required values for the model endpoint record
    try:
        # Getting project name from the function uri
        project, uri, tag, hash_key = parse_versioned_object_uri(
            graph_server.function_uri
        )
    except Exception as e:
        logger.error("Failed to parse function URI", exc=e)
        return None

    # Generating version model value based on the model name and model version
    if model.version:
        versioned_model_name = f"{model.name}:{model.version}"
    else:
        versioned_model_name = f"{model.name}:latest"

    # Generating model endpoint ID based on function uri and model version
    uid = mlrun.common.model_monitoring.create_model_endpoint_uid(
        function_uri=graph_server.function_uri, versioned_model=versioned_model_name
    ).uid

    # If model endpoint object was found in DB, skip the creation process.
    try:
        mlrun.get_run_db().get_model_endpoint(project=project, endpoint_id=uid)

    except mlrun.errors.MLRunNotFoundError:
        logger.info("Creating a new model endpoint record", endpoint_id=uid)

        try:
            model_endpoint = mlrun.common.schemas.ModelEndpoint(
                metadata=mlrun.common.schemas.ModelEndpointMetadata(
                    project=project, labels=model.labels, uid=uid
                ),
                spec=mlrun.common.schemas.ModelEndpointSpec(
                    function_uri=graph_server.function_uri,
                    model=versioned_model_name,
                    model_class=model.__class__.__name__,
                    model_uri=model.model_path,
                    stream_path=config.model_endpoint_monitoring.store_prefixes.default.format(
                        project=project, kind="stream"
                    ),
                    active=True,
                    monitoring_mode=mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled
                    if model.context.server.track_models
                    else mlrun.common.schemas.model_monitoring.ModelMonitoringMode.disabled,
                ),
                status=mlrun.common.schemas.ModelEndpointStatus(
                    endpoint_type=mlrun.common.schemas.model_monitoring.EndpointType.NODE_EP
                ),
            )

            db = mlrun.get_run_db()

            db.create_model_endpoint(
                project=project,
                endpoint_id=uid,
                model_endpoint=model_endpoint.dict(),
            )

        except Exception as e:
            logger.error("Failed to create endpoint record", exc=e)

    except Exception as e:
        logger.error("Failed to retrieve model endpoint object", exc=e)

    return uid
