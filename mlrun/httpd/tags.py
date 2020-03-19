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

from ..db.sqldb import table2cls
from ..db.sqldb import to_dict as db2dict
from .app import app, catch_err, db, json_error


@app.route('/api/<project>/tag/<name>', methods=['POST'])
@catch_err
def tag_objects(project, name):
    try:
        data: dict = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    objs = []
    for typ, query in data.items():
        cls = table2cls(typ)
        if cls is None:
            err = f'unknown type - {typ}'
            return json_error(HTTPStatus.BAD_REQUEST, reason=err)
        # {'name': 'bugs'} -> [Function.name=='bugs']
        db_query = [
            getattr(cls, key) == value for key, value in query.items()
        ]
        # TODO: Change _query to query?
        # TODO: Not happy about exposing db internals to API
        objs.extend(db.session.query(cls).filter(*db_query))
    db.tag_objects(objs, project, name)
    return jsonify(ok=True, project=project, name=name, count=len(objs))


@app.route('/api/<project>/tag/<name>', methods=['DELETE'])
@catch_err
def del_tag(project, name):
    count = db.del_tag(project, name)
    return jsonify(ok=True, project=project, name=name, count=count)


@app.route('/api/<project>/tags', methods=['GET'])
@catch_err
def list_tags(project):
    return jsonify(
        ok=True,
        project=project,
        tags=list(db.list_tags(project)),
    )


@app.route('/api/<project>/tag/<name>', methods=['GET'])
@catch_err
def get_tagged(project, name):
    objs = db.find_tagged(project, name)
    return jsonify(
        ok=True,
        project=project,
        tag=name,
        objects=[db2dict(obj) for obj in objs],
    )
