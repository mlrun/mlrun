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
from io import BytesIO
from typing import Dict
from urllib.request import urlopen


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
    predictor = PredictHandler(models).with_context(context)
    explainer = ExplainHandler(models).with_context(context)
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


class HTTPHandler:
    def __init__(self, models: Dict):
        self.models = models # pylint:disable=attribute-defined-outside-init
        self.context = None

    def with_context(self, context):
        self.context = context
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
        setattr(model, 'context', self.context)
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
        context.logger.info('event type: {}'.format(type(event.body)))
        body = self.parse_event(event)
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.predict(request)
        response = model.postprocess(response)

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

        request = model.preprocess(body)
        request = self.validate(request)
        response = model.explain(request)
        response = model.postprocess(response)

        return context.Response(body=json.dumps(response),
                                content_type='application/json',
                                status_code=200)