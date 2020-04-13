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
from io import BytesIO
from tempfile import mktemp
from typing import Dict
from urllib.request import urlopen
from datetime import datetime

from ..datastore import StoreManager
from ..platforms.iguazio import OutputStream


class MLModelServer:

    def __init__(self, name: str, model_dir: str = None, model=None):
        self.name = name
        self.ready = False
        self.model_dir = model_dir
        self._stores = StoreManager()
        if model:
            self.model = model
            self.ready = True

    def get_model_file(self, suffix=''):
        model_file = ''
        suffix = suffix or '.pkl'
        if self.model_dir.endswith(suffix):
            model_file = self.model_dir
        else:
            for file in self._stores.object(self.model_dir).listdir():
                if file.endswith(suffix):
                    model_file = os.path.join(self.model_dir, file)
                    break
        if not model_file:
            raise ValueError('cant resolve model file for {} suffix{}'.format(
                self.model_dir, suffix))
        obj = self._stores.object(model_file)
        if obj.kind == 'file':
            return model_file, obj.meta

        tmp = mktemp(suffix)
        obj.download(tmp)
        return tmp, obj.meta

    def load(self):
        if not self.ready and not self.model:
            raise ValueError('please specify a load method or a model object')

    def preprocess(self, request: Dict) -> Dict:
        return request

    def postprocess(self, request: Dict) -> Dict:
        return request

    def predict(self, request: Dict) -> Dict:
        raise NotImplementedError

    def explain(self, request: Dict) -> Dict:
        raise NotImplementedError


def nuclio_serving_init(context, data):
    model_prefix = 'SERVING_MODEL_'

    # Initialize models from environment variables
    # Using the {model_prefix}_{model_name} = {model_path} syntax
    model_paths = {k[len(model_prefix):]: v for k, v in os.environ.items() if
                   k.startswith(model_prefix)}
    model_class = os.environ.get('MODEL_CLASS', 'MLModel')
    fhandler = data[model_class]
    models = {name: fhandler(name=name, model_dir=path) for name, path in
              model_paths.items()}

    # Verify that models are loaded
    assert len(
        models) > 0, "No models were loaded!\n Please load a model by using the environment variable SERVING_MODEL_{model_name} = model_path"
    context.logger.info(f'Loaded {list(models.keys())}')

    # Initialize route handlers
    hostname = socket.gethostname()
    server_context = _ServerContext(context, hostname)
    predictor = PredictHandler(models).with_context(server_context)
    explainer = ExplainHandler(models).with_context(server_context)
    router = {
        'predict': predictor.post,
        'explain': explainer.post
    }

    ## Define handle
    setattr(context, 'mlrun_handler', nuclio_serving_handler)
    setattr(context, 'models', models)
    setattr(context, 'router', router)


err_string = 'Got path: {} \n Path must be <model-name>/<action> \nactions: {} \nmodels: {}'


def nuclio_serving_handler(context, event):
    # check if valid route & model
    try:
        model_name, route = event.path.strip('/').split('/')
        route = context.router[route]
    except:
        return context.Response(
            body=err_string.format(event.path, '|'.join(context.router.keys()), '|'.join(context.models.keys())),
            content_type='text/plain',
            status_code=404)

    return route(context, model_name, event)


class _ServerContext:
    def __init__(self, context, hostname):
        self.context = context
        self.hostname = hostname
        self.output_stream = None
        out_stream = os.environ.get('INFERENCE_STREAM', '')
        if out_stream:
            self.output_stream = OutputStream(out_stream)


class HTTPHandler:
    def __init__(self, models: Dict, context: _ServerContext = None):
        self.models = models # pylint:disable=attribute-defined-outside-init
        self.srv_context = context

    def with_context(self, context: _ServerContext):
        self.srv_context = context
        return self

    def get_model(self, name: str):
        if name not in self.models:
            return self.context.Response(
                body=f'Model with name {name} does not exist, please try to list the models',
                content_type='text/plain',
                status_code=404)

        model = self.models[name]
        if not model.ready:
            model.load()
        setattr(model, 'context', self.srv_context.context)
        return model

    def parse_event(self, event):
        parsed_event = {'instances': []}
        try:
            if not isinstance(event.body, dict):
                body = json.loads(event.body)
            else:
                body = event.body
            self.context.logger.info(f'event.body: {event.body}')
            if 'data_url' in body:
                # Get data from URL
                url = body['data_url']
                self.context.logger.debug_with('downloading data', url=url)
                data = urlopen(url).read()
                sample = BytesIO(data)
                parsed_event['instances'].append(sample)
            else:
                parsed_event = body

        except Exception as e:
            if event.content_type.startswith('image/'):
                sample = BytesIO(event.body)
                parsed_event['instances'].append(sample)
                parsed_event['content_type'] = event.content_type
            else:
                raise Exception("Unrecognized request format: %s" % e)
                
        return parsed_event

    def validate(self, request):
        if "instances" not in request:
            raise Exception("Expected key \"instances\" in request body")

        if not isinstance(request["instances"], list):
            raise Exception("Expected \"instances\" to be a list")

        return request


class PredictHandler(HTTPHandler):
    def post(self, context, name: str, event):
        model = self.get_model(name)
        context.logger.debug('event: {}'.format(type(event.body)))
        start = datetime.now()
        body = self.parse_event(event)
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.predict(request)
        response = model.postprocess(response)
        if self.srv_context.output_stream:
            data = {'kind': 'predict',
                    'request': request, 'resp': response,
                    'model': model.name,
                    'host': self.srv_context.hostname,
                    'when': str(start),
                    'microsec': (datetime.now() - start).microseconds}

            self.srv_context.output_stream.push(data)

        return context.Response(body=json.dumps(response),
                                content_type='application/json',
                                status_code=200)


class ExplainHandler(HTTPHandler):
    def post(self, context, name: str, event):
        model = self.get_model(name)
        try:
            body = json.loads(event.body)
        except json.decoder.JSONDecodeError as e:
            return context.Response(body="Unrecognized request format: %s" % e,
                                    content_type='text/plain',
                                    status_code=400)

        start = datetime.now()
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.explain(request)
        response = model.postprocess(response)
        if self.srv_context.output_stream:
            data = {'kind': 'explain',
                    'request': request, 'resp': response,
                    'model': model.name,
                    'host': self.srv_context.hostname,
                    'when': str(start),
                    'microsec': (datetime.now() - start).microseconds}

            self.srv_context.output_stream.push(data)

        return context.Response(body=json.dumps(response),
                                content_type='application/json',
                                status_code=200)
