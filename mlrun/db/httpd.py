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
"""mlrun database HTTP server"""
import mimetypes
from base64 import b64decode
from distutils.util import strtobool
from functools import wraps
from http import HTTPStatus
from os import environ, path

from flask import Flask, jsonify, request, Response

from mlrun.datastore import get_object, get_object_stat
from mlrun.db import RunDBError
from mlrun.db.filedb import FileRunDB
from mlrun.utils import logger, parse_function_uri, get_in
from mlrun.config import config
from mlrun.runtimes import RunError
from mlrun.run import new_function, import_function

_file_db: FileRunDB = None
app = Flask(__name__)
basic_prefix = 'Basic '
bearer_prefix = 'Bearer '


def json_error(status=HTTPStatus.BAD_REQUEST, **kw):
    kw.setdefault('ok', False)
    logger.error(str(kw))
    reply = jsonify(**kw)
    reply.status_code = status
    return reply


def parse_basic_auth(header):
    """
    >>> parse_basic_auth('Basic YnVnczpidW5ueQ==')
    ['bugs', 'bunny']
    """
    b64value = header[len(basic_prefix):]
    value = b64decode(b64value).decode()
    return value.split(':', 1)


class AuthError(Exception):
    pass


def basic_auth_required(cfg):
    return cfg.user or cfg.password


def bearer_auth_required(cfg):
    return cfg.token


@app.before_request
def check_auth():
    if request.path == '/api/healthz':
        return

    cfg = config.httpdb

    header = request.headers.get('Authorization', '')
    try:
        if basic_auth_required(cfg):
            if not header.startswith(basic_prefix):
                raise AuthError('missing basic auth')
            user, passwd = parse_basic_auth(header)
            if user != cfg.user or passwd != cfg.password:
                raise AuthError('bad basic auth')
        elif bearer_auth_required(cfg):
            if not header.startswith(bearer_prefix):
                raise AuthError('missing bearer auth')
            token = header[len(bearer_prefix):]
            if token != cfg.token:
                raise AuthError('bad bearer auth')
    except AuthError as err:
        resp = jsonify(ok=False, error=str(err))
        resp.status_code = HTTPStatus.UNAUTHORIZED
        return resp


def catch_err(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        try:
            return fn(*args, **kw)
        except RunDBError as err:
            return json_error(
                HTTPStatus.INTERNAL_SERVER_ERROR, ok=False, reason=str(err))

    return wrapper


# curl -d@/path/to/job.json http://localhost:8080/submit
@app.route('/api/submit', methods=['POST'])
@app.route('/api/submit/', methods=['POST'])
@app.route('/api/submit/<path:func>', methods=['POST'])
@catch_err
def submit_job(func=''):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.info('submit_job: func %s', func)
    task = data.get('task')
    function = data.get('function')
    url = data.get('functionUrl')
    if not url and task:
        url = get_in(task, 'spec.function')
    if not (function or url) or not task:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='bad JSON, need to include function/url and task objects',
        )

    # TODO: block exec for function['kind'] in ['', 'local]  (must be a
    # remote/container runtime)

    try:
        if function:
            resp = new_function(runtime=function).run(task)
        else:
            if '://' in url:
                resp = import_function(url=url).run(task)
            else:
                project, name, tag = parse_function_uri(url)
                resp = import_function(
                    project=project, name=name, tag=tag).run(task)

        logger.info('resp: %s', resp.to_yaml())
    except RunError as err:
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: {}'.format(err),
        )

    return jsonify(ok=True, data=resp.to_dict())


# curl http://localhost:8080/api/files?schema=s3&path=mybucket/a.txt
@app.route('/api/files', methods=['GET'])
@catch_err
def get_files():
    schema = request.args.get('schema', '')
    path = request.args.get('path', '')
    size = int(request.args.get('size', '0'))
    offset = int(request.args.get('offset', '0'))

    _, filename = path.split(path)

    if schema:
        path = schema + '://' + path
    elif path.startswith('/User/'):
        user = environ.get('V3IO_USERNAME', 'admin')
        path = 'v3io:///users/' + user + path[5:]

    try:
        body = get_object(path, size, offset)
    except FileNotFoundError as e:
        return json_error(HTTPStatus.NOT_FOUND, path=path, err=str(e))
    if body is None:
        return json_error(HTTPStatus.NOT_FOUND, path=path)

    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        ctype = 'application/octet-stream'

    return Response(body, mimetype=ctype, headers={"x-suggested-filename": filename})


# curl http://localhost:8080/api/filestat?schema=s3&path=mybucket/a.txt
@app.route('/api/filestat', methods=['GET'])
@catch_err
def get_filestat():
    schema = request.args.get('schema', '')
    path = request.args.get('path', '')

    _, filename = path.split(path)

    if schema:
        path = schema + '://' + path
    elif path.startswith('/User/'):
        user = environ.get('V3IO_USERNAME', 'admin')
        path = 'v3io:///users/' + user + path[5:]

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


# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@app.route('/api/log/<project>/<uid>', methods=['POST'])
@catch_err
def store_log(project, uid):
    append = strtobool(request.args.get('append', 'no'))
    body = request.get_data()  # TODO: Check size
    _file_db.store_log(uid, project, body, append)
    return jsonify(ok=True)


# curl http://localhost:8080/log/prj/7
@app.route('/api/log/<project>/<uid>', methods=['GET'])
def get_log(project, uid):
    data = _file_db.get_log(uid, project)
    if data is None:
        data = _file_db.read_run(uid, project)
        if not data:
            return json_error(HTTPStatus.NOT_FOUND,
                              project=project, uid=uid)
        msg = 'No logs, {}'.format(get_in(data, 'status.error', 'no error'))
        return msg.encode()

    return data

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
    _file_db.store_run(data, uid, project, iter=iter)
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

    iter = int(request.args.get('iter', '0'))
    _file_db.update_run(data, uid, project, iter=iter)
    app.logger.info('update run: {}'.format(data))
    return jsonify(ok=True)


# curl http://localhost:8080/run/p1/3
@app.route('/api/run/<project>/<uid>', methods=['GET'])
@catch_err
def read_run(project, uid):
    iter = int(request.args.get('iter', '0'))
    data = _file_db.read_run(uid, project, iter=iter)
    return jsonify(ok=True, data=data)


# curl -X DELETE http://localhost:8080/run/p1/3
@app.route('/api/run/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_run(project, uid):
    iter = int(request.args.get('iter', '0'))
    _file_db.del_run(uid, project, iter=iter)
    return jsonify(ok=True)


# curl http://localhost:8080/runs?project=p1&name=x&label=l1&label=l2&sort=no
@app.route('/api/runs', methods=['GET'])
@catch_err
def list_runs():
    name = request.args.get('name', '')
    uid = request.args.get('uid', '')
    project = request.args.get('project', 'default')
    labels = request.args.getlist('label')
    state = request.args.get('state', '')
    sort = strtobool(request.args.get('sort', 'on'))
    iter = strtobool(request.args.get('iter', 'on'))
    last = int(request.args.get('last', '0'))

    runs = _file_db.list_runs(
        name=name,
        uid=uid,
        project=project,
        labels=labels,
        state=state,
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

    _file_db.del_runs(name, project, labels, state, days_ago)
    return jsonify(ok=True)


# curl -d@/path/to/artifcat http://localhost:8080/artifact/p1/7&key=k
@app.route('/api/artifact/<project>/<uid>/<path:key>', methods=['POST'])
@catch_err
def store_artifact(project, uid, key):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    tag = request.args.get('tag', '')
    _file_db.store_artifact(key, data, uid, tag, project)
    return jsonify(ok=True)


# curl http://localhost:8080/artifact/p1/tag/key
@app.route('/api/artifact/<project>/<tag>/<path:key>', methods=['GET'])
@catch_err
def read_artifact(project, tag, key):
    data = _file_db.read_artifact(key, tag, project)
    return data

# curl -X DELETE http://localhost:8080/artifact/p1&key=k&tag=t
@app.route('/api/artifact/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_artifact(project, uid):
    key = request.args.get('key')
    if not key:
        return json_error(HTTPStatus.BAD_REQUEST, reason='missing data')

    tag = request.args.get('tag', '')
    _file_db.del_artifact(key, tag, project)
    return jsonify(ok=True)

# curl http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/api/artifacts', methods=['GET'])
@catch_err
def list_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', 'default')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    artifacts = _file_db.list_artifacts(name, project, tag, labels)
    return jsonify(ok=True, artifacts=artifacts)

# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/api/artifacts', methods=['DELETE'])
@catch_err
def del_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    _file_db.del_artifacts(name, project, tag, labels)
    return jsonify(ok=True)

# curl -d@/path/to/func.json http://localhost:8080/func/prj/7?tag=0.3.2
@app.route('/api/func/<project>/<name>', methods=['POST'])
@catch_err
def store_function(project, name):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    tag = request.args.get('tag', '')

    _file_db.store_function(data, name, project, tag)
    return jsonify(ok=True)


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@app.route('/api/func/<project>/<name>', methods=['GET'])
@catch_err
def get_function(project, name):
    tag = request.args.get('tag', '')
    func = _file_db.get_function(name, project, tag)
    return jsonify(ok=True, func=func)


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@app.route('/api/funcs', methods=['GET'])
@catch_err
def list_functions():
    name = request.args.get('name', '')
    project = request.args.get('project', 'default')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    out = _file_db.list_functions(name, project, tag, labels)
    return jsonify(
        ok=True,
        funcs=list(out),
    )


@app.route('/api/healthz', methods=['GET'])
def health():
    return 'OK\n'


@app.before_first_request
def init_app():
    global _file_db

    from mlrun.config import config

    logger.info('configuration dump\n%s', config.dump_yaml())
    _file_db = FileRunDB(config.httpdb.dirpath, '.yaml')
    _file_db.connect()


# Don't remove this function, it's an entry point in setup.py
def main():
    app.run(
        host='0.0.0.0',
        port=config.httpdb.port,
        debug=config.httpdb.debug,
    )


if __name__ == '__main__':
    main()
