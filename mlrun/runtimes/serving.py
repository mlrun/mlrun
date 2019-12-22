import json
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

import os
from typing import Dict


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
    predictor = PredictHandler(models)
    explainer = ExplainHandler(models)
    router = {
        'predict': predictor.post,
        'explain': explainer.post
    }

    ## Define handle
    setattr(context, 'mlrun_handler', nuclio_serving_handler)
    setattr(context, 'models', models)
    setattr(context, 'router', router)


err_string = 'Got path: {} \n Path must be <version>/<host>/<model-name>/<action> \nactions: {} \nmodels: {}'


def nuclio_serving_handler(context, event):
    # check if valid route & model
    try:
        api_ver, section, model_name, route = event.path.strip('/').split('/')

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
        return model

    def validate(self, request):
        if "instances" not in request:
            return self.context.Response(
                body="Expected key \"instances\" in request body",
                content_type='text/plain',
                status_code=400)

        if not isinstance(request["instances"], list):
            return self.context.Response(
                body="Expected \"instances\" to be a list",
                content_type='text/plain',
                status_code=400)

        return request


class PredictHandler(HTTPHandler):
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