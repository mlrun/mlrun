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


from distutils.util import strtobool
from http import HTTPStatus

from flask import jsonify, request

from .app import app, catch_err, db, json_error, logger


# curl -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@app.route('/api/run/<project>/<uid>', methods=['POST'])
@catch_err
def store_run(project, uid):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.debug(data)
    iter = int(request.args.get('iter', '0'))
    db.store_run(data, uid, project, iter=iter)
    app.logger.info('store run: {}'.format(data))
    return jsonify(ok=True)


# curl -X PATCH -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@app.route('/api/run/<project>/<uid>', methods=['PATCH'])
@catch_err
def update_run(project, uid):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.debug(data)
    iter = int(request.args.get('iter', '0'))
    db.update_run(data, uid, project, iter=iter)
    app.logger.info('update run: {}'.format(data))
    return jsonify(ok=True)


# curl http://localhost:8080/run/p1/3
@app.route('/api/run/<project>/<uid>', methods=['GET'])
@catch_err
def read_run(project, uid):
    iter = int(request.args.get('iter', '0'))
    data = db.read_run(uid, project, iter=iter)
    return jsonify(ok=True, data=data)


# curl -X DELETE http://localhost:8080/run/p1/3
@app.route('/api/run/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_run(project, uid):
    iter = int(request.args.get('iter', '0'))
    db.del_run(uid, project, iter=iter)
    return jsonify(ok=True)


# curl http://localhost:8080/runs?project=p1&name=x&label=l1&label=l2&sort=no
@app.route('/api/runs', methods=['GET'])
@catch_err
def list_runs():
    name = request.args.get('name')
    uid = request.args.get('uid')
    project = request.args.get('project')
    labels = request.args.getlist('label')
    state = request.args.get('state')
    sort = strtobool(request.args.get('sort', 'on'))
    iter = strtobool(request.args.get('iter', 'on'))
    last = int(request.args.get('last', '0'))

    runs = db.list_runs(
        name=name or None,
        uid=uid or None,
        project=project or None,
        labels=labels,
        state=state or None,
        sort=sort,
        last=last,
        iter=iter,
    )
    return jsonify(ok=True, runs=runs)

# curl -X DELETE http://localhost:8080/runs?project=p1&name=x&days_ago=3
@app.route('/api/runs', methods=['DELETE'])
@catch_err
def del_runs():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    labels = request.args.getlist('label')
    state = request.args.get('state', '')
    days_ago = int(request.args.get('days_ago', '0'))

    db.del_runs(name, project, labels, state, days_ago)
    return jsonify(ok=True)
