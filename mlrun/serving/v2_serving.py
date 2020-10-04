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
import time
from typing import Dict
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

from mlrun.artifacts import ModelArtifact, get_model


class V2ModelServer:
    def __init__(self, context, name: str, model_dir: str = None, model=None, **kwargs):
        self.name = name
        self.version = ""
        if ":" in name:
            self.name, self.version = name.split(":", 1)
        self.context = context
        self.ready = False
        self.loading = False
        self.model_dir = model_dir
        self.model_spec: ModelArtifact = None
        self._params = kwargs
        self._model_logger = None
        self._params = context.add_root_params(self._params)
        self._model_logger = _ModelLogPusher(self, context)

        self.metrics = {}
        self.labels = {}
        if model:
            self.model = model
            self.ready = True

    def async_load(self):
        if not self.ready:
            self.load()
            self.ready = True

    def get_param(self, key: str, default=None):
        """get param by key (specified in the model or the function)"""
        return self._params.get(key, default)

    def get_model(self, suffix=""):
        """get the model file(s) and metadata from model store

        the method returns a path to the model file and the extra data (dict of dataitem objects)
        it also loads the model metadata into the self.model_spec attribute, allowing direct access
        to all the model metadata attributes.

        get_model is usually used in the model .load() method to init the model

        example:
            def load(self):
                model_file, extra_data = self.get_model(suffix='.pkl')
                model = load(open(model_file, "rb"))
                categories = extra_data['categories'].as_df()

        :param  suffix:  optional, model file suffix (when the model_path is a directory)
        :return (local) model file, extra dataitems dictionary
        """
        model_file, self.model_spec, extra_dataitems = get_model(self.model_dir, suffix)
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
        if not event.trigger or event.trigger == "http":
            raise RuntimeError(f"model {self.name} is not ready yet")
        self.context.logger.info(f"waiting for model {self.name} to load")
        for i in range(50):  # wait up to 4.5 minutes
            time.sleep(5)
            if self.ready:
                return
        raise RuntimeError(f"model {self.name} did not become ready")

    def do_event(self, event, *args, **kwargs):
        """main model event handler method"""
        start = datetime.now()
        op = event.path.strip("/")

        if op == "predict" or op == "infer":
            # predict operation
            self._check_readiness(event)
            request = self.preprocess(event.body, op)
            request = self.validate(request, op)
            response = self.predict(request)

        elif op == "ready" and event.method == "GET":
            # get model health operation
            setattr(event, "terminated", True)
            if self.ready:
                event.body = self.context.Response()
            else:
                event.body = self.context.Response(
                    status_code=408, body=b"model not ready"
                )
            return event

        elif op == "" and event.method == "GET":
            # get model metadata operation
            setattr(event, "terminated", True)
            event.body = {
                "name": self.name,
                "version": self.version,
                "inputs": [],
                "outputs": [],
            }
            if self.model_spec:
                event.body["inputs"] = self.model_spec.inputs
                event.body["outputs"] = self.model_spec.outputs
            return event

        elif op == "explain":
            # explain operation
            self._check_readiness(event)
            request = self.preprocess(event.body, op)
            request = self.validate(request, op)
            response = self.explain(request)

        elif hasattr(self, "op_" + op):
            # custom operation (child methods starting with "op_")
            response = getattr(self, "op_" + op)(event)
            event.body = response
            return event

        else:
            raise ValueError(f"illegal model operation {op}, method={event.method}")

        response = self.postprocess(response)
        if self._model_logger:
            self._model_logger.push(start, request, response)
        event.body = response
        return event

    def validate(self, request, operation):
        """validate the event body (after preprocess)"""
        if "data" not in request:
            raise Exception('Expected key "data" in request body')

        if not isinstance(request["data"], list):
            raise Exception('Expected "data" to be a list')

        return request

    def preprocess(self, request: Dict, operation) -> Dict:
        """preprocess the event body before validate and action"""
        return request

    def postprocess(self, request: Dict) -> Dict:
        """postprocess, before returning response"""
        return request

    def predict(self, request: Dict) -> Dict:
        """model prediction operation"""
        raise NotImplementedError

    def explain(self, request: Dict) -> Dict:
        """model explain operation"""
        raise NotImplementedError


class _ModelLogPusher:
    def __init__(self, model, context, output_stream=None):
        self.model = model
        self.hostname = context.server.hostname
        self.stream_batch = context.server.stream_batch
        self.stream_sample = context.server.stream_sample
        self.output_stream = output_stream or context.server.output_stream
        self._worker = context.worker_id
        self._sample_iter = 0
        self._batch_iter = 0
        self._batch = []

    def base_data(self):
        base_data = {
            "class": self.model.__class__.__name__,
            "worker": self.worker,
            "model": self.model.name,
            "version": self.model.version,
            "host": self.hostname,
        }
        if getattr(self.model, "labels", None):
            base_data["labels"] = self.model.labels
        return base_data

    def push(self, start, request, resp):
        self._sample_iter = (self._sample_iter + 1) % self.stream_sample
        if self.output_stream and self._sample_iter == 0:
            microsec = (datetime.now() - start).microseconds

            if self.stream_batch > 1:
                if self._batch_iter == 0:
                    self._batch = []
                self._batch.append(
                    [request, resp, str(start), microsec, self.model.metrics]
                )
                self._batch_iter = (self._batch_iter + 1) % self.stream_batch

                if self._batch_iter == 0:
                    data = self.base_data()
                    data["headers"] = ["request", "resp", "when", "microsec", "metrics"]
                    data["values"] = self._batch
                    self.output_stream.push([data])
            else:
                data = self.base_data()
                data["request"] = request
                data["resp"] = resp
                data["when"] = str(start)
                data["microsec"] = microsec
                if getattr(self.model, "metrics", None):
                    data["metrics"] = self.model.metrics
                self.output_stream.push([data])


http_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
)


class HttpTransport:
    """class for calling remote models"""

    def __init__(self, url):
        self.url = url
        self.format = "json"
        self._session = requests.Session()
        self._session.mount("http://", http_adapter)
        self._session.mount("https://", http_adapter)

    def do(self, event):
        kwargs = {}
        kwargs["headers"] = event.headers or {}
        method = event.method or "POST"
        if method != "GET":
            if isinstance(event.body, (str, bytes)):
                kwargs["data"] = event.body
            else:
                kwargs["json"] = event.body

        url = self.url.strip("/") + event.path
        try:
            resp = self._session.request(method, url, verify=False, **kwargs)
        except OSError as err:
            raise OSError(f"error: cannot run function at url {url}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.text}")

        data = resp.content
        if self.format == "json" or resp.headers["content-type"] == "application/json":
            data = json.loads(data)
        event.body = data
        return event
