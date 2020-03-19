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

from .app import app, catch_err, json_error
from flask import request, jsonify, Response
from os import environ
from http import HTTPStatus
from mlrun.datastore import get_object, get_object_stat
import mimetypes
from ..config import config

# curl http://localhost:8080/api/files?schema=s3&path=mybucket/a.txt
@app.route('/api/files', methods=['GET'])
@catch_err
def get_files():
    schema = request.args.get('schema', '')
    path = request.args.get('path', '')
    size = int(request.args.get('size', '0'))
    offset = int(request.args.get('offset', '0'))

    _, filename = path.split(path)

    path = get_obj_path(schema, path)
    if not path:
        return json_error(HTTPStatus.NOT_FOUND, path=path,
                          err='illegal path prefix or schema')

    try:
        body = get_object(path, size, offset)
    except FileNotFoundError as e:
        return json_error(HTTPStatus.NOT_FOUND, path=path, err=str(e))
    if body is None:
        return json_error(HTTPStatus.NOT_FOUND, path=path)

    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        ctype = 'application/octet-stream'

    return Response(
        body, mimetype=ctype, headers={"x-suggested-filename": filename})


# curl http://localhost:8080/api/filestat?schema=s3&path=mybucket/a.txt
@app.route('/api/filestat', methods=['GET'])
@catch_err
def get_filestat():
    schema = request.args.get('schema', '')
    path = request.args.get('path', '')

    _, filename = path.split(path)

    path = get_obj_path(schema, path)
    if not path:
        return json_error(HTTPStatus.NOT_FOUND, path=path,
                          err='illegal path prefix or schema')

    try:
        stat = get_object_stat(path)
    except FileNotFoundError as e:
        return json_error(HTTPStatus.NOT_FOUND, path=path, err=str(e))

    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        ctype = 'application/octet-stream'

    return jsonify(ok=True, size=stat.size,
                   modified=stat.modified,
                   mimetype=ctype)


def get_obj_path(schema, path):
    if schema:
        return schema + '://' + path
    elif path.startswith('/User/'):
        user = environ.get('V3IO_USERNAME', 'admin')
        return 'v3io:///users/' + user + path[5:]
    elif config.httpdb.data_volume and \
            path.startswith(config.httpdb.data_volume):
        if config.httpdb.real_path:
            path = config.httpdb.real_path + \
                   path[len(config.httpdb.data_volume)-1:]
        return path
    return None
