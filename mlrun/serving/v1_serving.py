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
from copy import deepcopy
from io import BytesIO
from typing import Dict
from urllib.request import urlopen
from datetime import datetime
import nuclio

import mlrun
from mlrun.platforms.iguazio import OutputStream
from mlrun.runtimes import RemoteRuntime

serving_handler = "handler"


def new_v1_model_server(
    name,
    model_class: str,
    models: dict = None,
    filename="",
    protocol="",
    image="",
    endpoint="",
    workers=8,
    canary=None,
):
    f = RemoteRuntime()
    if not image:
        name, spec, code = nuclio.build_file(
            filename, name=name, handler=serving_handler, kind="serving"
        )
        f.spec.base_spec = spec

    f.metadata.name = name

    if models:
        for k, v in models.items():
            f.set_env("SERVING_MODEL_{}".format(k), v)

    if protocol:
        f.set_env("TRANSPORT_PROTOCOL", protocol)
    if model_class:
        f.set_env("MODEL_CLASS", model_class)
    f.with_http(workers, host=endpoint, canary=canary)
    f.spec.function_kind = "serving"

    if image:
        f.from_image(image)

    return f


class MLModelServer:
    def __init__(self, name: str, model_dir: str = None, model=None):
        self.name = name
        self.ready = False
        self.model_dir = model_dir
        self.model_spec: mlrun.artifacts.ModelArtifact = None
        self._params = {}
        self.metrics = {}
        self.labels = {}
        if model:
            self.model = model
            self.ready = True

    def get_param(self, key: str, default=None):
        return self._params.get(key, default)

    def get_model(self, suffix=""):
        model_file, self.model_spec, extra_dataitems = mlrun.artifacts.get_model(
            self.model_dir, suffix
        )
        if self.model_spec and self.model_spec.parameters:
            for key, value in self.model_spec.parameters.items():
                self._params[key] = value
        return model_file, extra_dataitems

    def load(self):
        if not self.ready and not self.model:
            raise ValueError("please specify a load method or a model object")

    def preprocess(self, request: Dict) -> Dict:
        return request

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict:
        raise NotImplementedError

    def explain(self, request: Dict) -> Dict:
        raise NotImplementedError


def nuclio_serving_init(context, data):
    model_prefix = "SERVING_MODEL_"
    params_prefix = "SERVING_PARAMS"

    # Initialize models from environment variables
    # Using the {model_prefix}_{model_name} = {model_path} syntax
    model_paths = {
        k[len(model_prefix) :]: v
        for k, v in os.environ.items()
        if k.startswith(model_prefix)
    }
    model_class = os.environ.get("MODEL_CLASS", "MLModelServer")
    fhandler = data[model_class]
    models = {
        name: fhandler(name=name, model_dir=path) for name, path in model_paths.items()
    }

    params = os.environ.get(params_prefix)
    if params:
        params = json.loads(params)

    for name, model in models.items():
        if params:
            setattr(model, "_params", deepcopy(params))
        if not model.ready:
            model.load()
            model.ready = True

    # Verify that models are loaded
    assert len(models) > 0, (
        "No models were loaded!\n Please load a model by using the environment variable "
        "SERVING_MODEL_{model_name} = model_path"
    )

    context.logger.info(f"Loaded {list(models.keys())}")

    # Initialize route handlers
    hostname = socket.gethostname()
    server_context = _ServerInfo(context, hostname, model_class)
    predictor = PredictHandler(models).with_context(server_context)
    explainer = ExplainHandler(models).with_context(server_context)
    router = {"predict": predictor.post, "explain": explainer.post}

    # Define handle
    setattr(context, "mlrun_handler", nuclio_serving_handler)
    setattr(context, "models", models)
    setattr(context, "router", router)


err_string = (
    "Got path: {} \n Path must be <model-name>/<action> \nactions: {} \nmodels: {}"
)


def nuclio_serving_handler(context, event):

    # check if valid route & model
    try:
        if hasattr(event, "trigger") and event.trigger.kind != "http":
            # non http triggers (i.e. stream) are directed to the first model
            # todo: take model name and action from json is specified
            model_name = next(iter(context.models))
            route = "predict"
        else:
            model_name, route = event.path.strip("/").split("/")
        route = context.router[route]
    except Exception:
        return context.Response(
            body=err_string.format(
                event.path,
                "|".join(context.router.keys()),
                "|".join(context.models.keys()),
            ),
            content_type="text/plain",
            status_code=404,
        )

    return route(context, model_name, event)


class _ServerInfo:
    def __init__(self, context, hostname, model_class):
        self.context = context
        self.worker = context.worker_id
        self.model_class = model_class
        self.hostname = hostname
        self.output_stream = None
        out_stream = os.environ.get("INFERENCE_STREAM", "")
        self.stream_sample = int(os.environ.get("INFERENCE_STREAM_SAMPLE", "1"))
        self.stream_batch = int(os.environ.get("INFERENCE_STREAM_BATCH", "1"))
        if out_stream:
            self.output_stream = OutputStream(out_stream)


class HTTPHandler:
    kind = ""

    def __init__(self, models: Dict, server: _ServerInfo = None):
        self.models = models
        self.srvinfo = server
        self.context = None
        self._sample_iter = 0
        self._batch_iter = 0
        self._batch = []

    def with_context(self, server: _ServerInfo):
        self.srvinfo = server
        self.context = server.context
        return self

    def get_model_class(self, name: str):
        model = self.models[name]
        if not model.ready:
            model.load()
            model.ready = True
        setattr(model, "context", self.srvinfo.context)
        return model

    def parse_event(self, event):
        parsed_event = {"instances": []}
        try:
            if not isinstance(event.body, dict):
                body = json.loads(event.body)
            else:
                body = event.body
            self.context.logger.info(f"event.body: {event.body}")
            if "data_url" in body:
                # Get data from URL
                url = body["data_url"]
                self.context.logger.debug_with(f"downloading data url={url}")
                data = urlopen(url).read()
                sample = BytesIO(data)
                parsed_event["instances"].append(sample)
            else:
                parsed_event = body

        except Exception as e:
            if event.content_type.startswith("image/"):
                sample = BytesIO(event.body)
                parsed_event["instances"].append(sample)
                parsed_event["content_type"] = event.content_type
            else:
                raise Exception("Unrecognized request format: %s" % e)

        return parsed_event

    def validate(self, request):
        if "instances" not in request:
            raise Exception('Expected key "instances" in request body')

        if not isinstance(request["instances"], list):
            raise Exception('Expected "instances" to be a list')

        return request

    def push_to_stream(self, start, request, resp, model):
        def base_data():
            data = {
                "op": self.kind,
                "class": self.srvinfo.model_class,
                "worker": self.srvinfo.worker,
                "model": model.name,
                "host": self.srvinfo.hostname,
            }
            if getattr(model, "labels", None):
                data["labels"] = model.labels
            return data

        self._sample_iter = (self._sample_iter + 1) % self.srvinfo.stream_sample
        if self.srvinfo.output_stream and self._sample_iter == 0:
            microsec = (datetime.now() - start).microseconds

            if self.srvinfo.stream_batch > 1:
                if self._batch_iter == 0:
                    self._batch = []
                self._batch.append([request, resp, str(start), microsec, model.metrics])
                self._batch_iter = (self._batch_iter + 1) % self.srvinfo.stream_batch

                if self._batch_iter == 0:
                    data = base_data()
                    data["headers"] = ["request", "resp", "when", "microsec", "metrics"]
                    data["values"] = self._batch
                    self.srvinfo.output_stream.push([data])
            else:
                data = base_data()
                data["request"] = request
                data["resp"] = resp
                data["when"] = str(start)
                data["microsec"] = microsec
                if getattr(model, "metrics", None):
                    data["metrics"] = model.metrics
                self.srvinfo.output_stream.push([data])


class PredictHandler(HTTPHandler):
    kind = "predict"

    def post(self, context, name: str, event):
        if name not in self.models:
            return context.Response(
                body=f"Model with name {name} does not exist, please try to list the models",
                content_type="text/plain",
                status_code=404,
            )

        model = self.get_model_class(name)
        context.logger.debug("event: {}".format(type(event.body)))
        start = datetime.now()
        body = self.parse_event(event)
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.predict(request)
        response = model.postprocess(response)
        self.push_to_stream(start, request, response, model)

        return context.Response(
            body=json.dumps(response), content_type="application/json", status_code=200
        )


class ExplainHandler(HTTPHandler):
    kind = "explain"

    def post(self, context, name: str, event):
        if name not in self.models:
            return context.Response(
                body=f"Model with name {name} does not exist, please try to list the models",
                content_type="text/plain",
                status_code=404,
            )

        model = self.get_model_class(name)
        try:
            body = json.loads(event.body)
        except json.decoder.JSONDecodeError as e:
            return context.Response(
                body="Unrecognized request format: %s" % e,
                content_type="text/plain",
                status_code=400,
            )

        start = datetime.now()
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.explain(request)
        response = model.postprocess(response)
        self.push_to_stream(start, request, response, model)

        return context.Response(
            body=json.dumps(response), content_type="application/json", status_code=200
        )
