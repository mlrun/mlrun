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

from http import HTTPStatus

from flask import jsonify, request

from ..config import config
from .app import app, catch_err, db, json_error, logger


# curl -d@/path/to/artifcat http://localhost:8080/artifact/p1/7&key=k
@app.route('/api/artifact/<project>/<uid>/<path:key>', methods=['POST'])
@catch_err
def store_artifact(project, uid, key):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.debug(data)
    tag = request.args.get('tag', '')
    iter = int(request.args.get('iter', '0'))
    db.store_artifact(key, data, uid, iter=iter, tag=tag, project=project)
    return jsonify(ok=True)


# curl http://localhost:8080/artifact/p1/tags
@app.route('/api/projects/<project>/artifact-tags', methods=['GET'])
@catch_err
def list_artifact_tags(project):
    return jsonify(
        ok=True,
        project=project,
        tags=db.list_artifact_tags(project),
    )


# curl http://localhost:8080/artifact/p1/tag/key
@app.route('/api/artifact/<project>/<tag>/<path:key>', methods=['GET'])
@catch_err
def read_artifact(project, tag, key):
    iter = int(request.args.get('iter', '0'))
    data = db.read_artifact(key, tag=tag, iter=iter, project=project)
    return data

# curl -X DELETE http://localhost:8080/artifact/p1&key=k&tag=t
@app.route('/api/artifact/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_artifact(project, uid):
    key = request.args.get('key')
    if not key:
        return json_error(HTTPStatus.BAD_REQUEST, reason='missing data')

    tag = request.args.get('tag', '')
    db.del_artifact(key, tag, project)
    return jsonify(ok=True)

# curl http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/api/artifacts', methods=['GET'])
@catch_err
def list_artifacts():
    name = request.args.get('name') or None
    project = request.args.get('project', config.default_project)
    tag = request.args.get('tag') or None
    labels = request.args.getlist('label')

    artifacts = db.list_artifacts(name, project, tag, labels)
    return jsonify(ok=True, artifacts=artifacts)

# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/api/artifacts', methods=['DELETE'])
@catch_err
def del_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    db.del_artifacts(name, project, tag, labels)
    return jsonify(ok=True)
