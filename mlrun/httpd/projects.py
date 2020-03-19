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
from operator import attrgetter

from flask import jsonify, request

from ..db.sqldb import to_dict as db2dict
from .app import app, catch_err, db, json_error


# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' \
#   http://localhost:8080/project
@app.route('/api/project', methods=['POST'])
@catch_err
def add_project():
    data = request.get_json(force=True)
    for attr in ('name', 'owner'):
        if attr not in data:
            return json_error(error=f'missing {attr!r}')

    project_id = db.add_project(data)
    return jsonify(
        ok=True,
        id=project_id,
        name=data['name'],
    )

# curl -d '{"name": "p1", "description": "desc", "users": ["u1", "u2"]}' \
#   -X UPDATE http://localhost:8080/project
@app.route('/api/project/<name>', methods=['POST'])
@catch_err
def update_project(name):
    data = request.get_json(force=True)
    db.update_project(name, data)
    return jsonify(ok=True)

# curl http://localhost:8080/project/<name>
@app.route('/api/project/<name>', methods=['GET'])
@catch_err
def get_project(name):
    project = db.get_project(name)
    if not project:
        return json_error(error=f'project {name!r} not found')

    resp = {
        'name': project.name,
        'description': project.description,
        'owner': project.owner,
        'source': project.source,
        'users': [u.name for u in project.users],
    }

    return jsonify(ok=True, project=resp)


# curl http://localhost:8080/projects?full=true
@app.route('/api/projects', methods=['GET'])
@catch_err
def list_projects():
    full = strtobool(request.args.get('full', 'no'))
    fn = db2dict if full else attrgetter('name')
    projects = []
    for p in db.list_projects():
        if isinstance(p, str):
            projects.append(p)
        else:
            projects.append(fn(p))

    return jsonify(ok=True, projects=projects)
