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

import traceback
from distutils.util import strtobool
from http import HTTPStatus

from flask import Response, jsonify, request

from ..builder import build_runtime
from ..config import config
from ..run import new_function
from ..runtimes import runtime_resources_map
from ..utils import get_in, parse_function_uri, update_in
from .app import app, catch_err, db, json_error, k8s, logger


# curl -d@/path/to/job.json http://localhost:8080/build/function
@app.route('/api/build/function', methods=['POST'])
@app.route('/api/build/function/', methods=['POST'])
@catch_err
def build_function():
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.info('build_function:\n{}'.format(data))
    function = data.get('function')
    with_mlrun = strtobool(data.get('with_mlrun', 'on'))
    ready = False

    try:
        fn = new_function(runtime=function)
        fn.set_db_connection(db)
        fn.save(versioned=False)

        ready = build_runtime(fn, with_mlrun)
        fn.save(versioned=False)
        logger.info('Fn:\n %s', fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: {}'.format(err),
        )

    return jsonify(ok=True, data=fn.to_dict(), ready=ready)


# curl -d@/path/to/job.json http://localhost:8080/start/function
@app.route('/api/start/function', methods=['POST'])
@app.route('/api/start/function/', methods=['POST'])
@catch_err
def start_function():
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.info('start_function:\n{}'.format(data))
    url = data.get('functionUrl')
    if not url:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: functionUrl not specified',
        )

    project, name, tag = parse_function_uri(url)
    runtime = db.get_function(name, project, tag)
    if not runtime:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: function {} not found'.format(url),
        )

    fn = new_function(runtime=runtime)
    resource = runtime_resources_map.get(fn.kind)
    if 'start' not in resource:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: "start" not supported by this runtime',
        )

    try:
        fn.set_db_connection(db)
        #  resp = resource['start'](fn)  # TODO: handle resp?
        resource['start'](fn)
        fn.save(versioned=False)
        logger.info('Fn:\n %s', fn.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: {}'.format(err),
        )

    return jsonify(ok=True, data=fn.to_dict())


# curl -d@/path/to/job.json http://localhost:8080/status/function
@app.route('/api/status/function', methods=['POST'])
@app.route('/api/status/function/', methods=['POST'])
@catch_err
def function_status():
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.info('function_status:\n{}'.format(data))
    selector = data.get('selector')
    kind = data.get('kind')
    if not selector or not kind:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: selector or runtime kind not specified',
        )

    resource = runtime_resources_map.get(kind)
    if 'status' not in resource:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: "status" not supported by this runtime',
        )

    try:
        resp = resource['status'](selector)
        logger.info('status: %s', resp)
    except Exception as err:
        logger.error(traceback.format_exc())
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: {}'.format(err),
        )

    return jsonify(ok=True, data=resp)


# curl -d@/path/to/job.json http://localhost:8080/build/status
@app.route('/api/build/status', methods=['GET'])
@app.route('/api/build/status/', methods=['GET'])
@catch_err
def build_status():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    tag = request.args.get('tag', '')
    offset = int(request.args.get('offset', '0'))
    logs = strtobool(request.args.get('logs', 'on'))

    fn = db.get_function(name, project, tag)
    if not fn:
        return json_error(HTTPStatus.NOT_FOUND, name=name,
                          project=project, tag=tag)

    state = get_in(fn, 'status.state', '')
    pod = get_in(fn, 'status.build_pod', '')
    image = get_in(fn, 'spec.build.image', '')
    out = b''
    if not pod:
        return Response(out, mimetype='text/plain',
                        headers={"function_status": state,
                                 "function_image": image,
                                 "builder_pod": pod})

    logger.info('get pod {} status'.format(pod))
    state = k8s.get_pod_status(pod)
    logger.info('pod state={}'.format(state))

    if state == 'succeeded':
        logger.info('build completed successfully')
        state = 'ready'
    if state in ['failed', 'error']:
        logger.error('build {}, watch the build pod logs: {}'.format(
            state, pod))

    if logs and state != 'pending':
        resp = k8s.logs(pod)
        if resp:
            out = resp.encode()[offset:]

    update_in(fn, 'status.state', state)
    if state == 'ready':
        update_in(fn, 'spec.image', image)

    db.store_function(fn, name, project, tag)

    return Response(out, mimetype='text/plain',
                    headers={"function_status": state,
                             "function_image": image,
                             "builder_pod": pod})

# curl -d@/path/to/func.json http://localhost:8080/func/prj/7?tag=0.3.2
@app.route('/api/func/<project>/<name>', methods=['POST'])
@catch_err
def store_function(project, name):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.debug(data)
    tag = request.args.get('tag', '')

    db.store_function(data, name, project, tag)
    return jsonify(ok=True)


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@app.route('/api/func/<project>/<name>', methods=['GET'])
@catch_err
def get_function(project, name):
    tag = request.args.get('tag', '')
    func = db.get_function(name, project, tag)
    return jsonify(ok=True, func=func)


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@app.route('/api/funcs', methods=['GET'])
@catch_err
def list_functions():
    name = request.args.get('name') or None
    project = request.args.get('project', config.default_project)
    tag = request.args.get('tag') or None
    labels = request.args.getlist('label')

    out = db.list_functions(name, project, tag, labels)
    return jsonify(
        ok=True,
        funcs=list(out),
    )
