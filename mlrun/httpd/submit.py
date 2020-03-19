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

import ast
import tempfile
from datetime import datetime
from http import HTTPStatus
from os import remove

from flask import jsonify, request
from kfp import Client as kfclient

from ..config import config
from . import app
from . import _submit


# curl -d@/path/to/job.json http://localhost:8080/submit
@app.route('/api/submit', methods=['POST'])
@app.route('/api/submit/', methods=['POST'])
@app.route('/api/submit_job', methods=['POST'])
@app.route('/api/submit_job/', methods=['POST'])
@app.catch_err
def submit_job():
    try:
        data: dict = request.get_json(force=True)
    except ValueError:
        return app.json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    app.logger.info('submit_job: {}'.format(data))
    return _submit.submit(data)


# curl -d@/path/to/pipe.yaml http://localhost:8080/submit_pipeline
@app.route('/api/submit_pipeline', methods=['POST'])
@app.route('/api/submit_pipeline/', methods=['POST'])
@app.catch_err
def submit_pipeline():
    namespace = request.args.get('namespace', config.namespace)
    experiment_name = request.args.get('experiment', 'Default')
    run_name = request.args.get('run', '')
    run_name = run_name or \
        experiment_name + ' ' + datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    arguments = {}
    arguments_data = request.headers.get('pipeline-arguments')
    if arguments_data:
        arguments = ast.literal_eval(arguments_data)
        app.logger.info('pipeline arguments {}'.format(arguments_data))

    ctype = request.content_type
    if '/yaml' in ctype:
        ctype = '.yaml'
    elif ' /zip' in ctype:
        ctype = '.zip'
    else:
        return app.json_error(
            HTTPStatus.BAD_REQUEST,
            reason='unsupported pipeline type {}'.format(ctype))

    app.logger.info('writing file {}'.format(ctype))
    if not request.data:
        return app.json_error(
            HTTPStatus.BAD_REQUEST, reason='post data is empty')

    print(str(request.data))
    pipe_tmp = tempfile.mktemp(suffix=ctype)
    with open(pipe_tmp, 'wb') as fp:
        fp.write(request.data)

    try:
        client = kfclient(namespace=namespace)
        experiment = client.create_experiment(name=experiment_name)
        run_info = client.run_pipeline(experiment.id, run_name, pipe_tmp,
                                       params=arguments)
    except Exception as e:
        remove(pipe_tmp)
        return app.json_error(
            HTTPStatus.BAD_REQUEST, reason='kfp err: {}'.format(e))

    remove(pipe_tmp)
    return jsonify(ok=True, id=run_info.run_id,
                   name=run_info.run_info.name)
