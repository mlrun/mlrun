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
from os import environ
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, Response

from mlrun.builder import build_runtime
from mlrun.datastore import get_object, get_object_stat
from mlrun.db import RunDBError, RunDBInterface, periodic
from mlrun.db.sqldb import SQLDB
from mlrun.db.filedb import FileRunDB
from mlrun.utils import logger, parse_function_uri, get_in, update_in
from mlrun.config import config
from mlrun.run import new_function, import_function
from mlrun.k8s_utils import K8sHelper
from mlrun.runtimes import runtime_resources_map
from mlrun.scheduler import Scheduler

_scheduler: Scheduler = None
_db: RunDBInterface
_k8s: K8sHelper = None
_logs_dir = None
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
def submit_job():
    try:
        data: dict = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    logger.info('submit_job: {}'.format(data))
    return _submit(data)


def _submit(data):
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
            fn = new_function(runtime=function)
        else:
            if '://' in url:
                fn = import_function(url=url)
            else:
                project, name, tag = parse_function_uri(url)
                runtime = _db.get_function(name, project, tag)
                if not runtime:
                    return json_error(
                        HTTPStatus.BAD_REQUEST,
                        reason='runtime error: function {} not found'.format(url),
                    )
                fn = new_function(runtime=runtime)

        fn.set_db_connection(_db, True)
        logger.info('func:\n{}'.format(fn.to_yaml()))
        # fn.spec.rundb = 'http://mlrun-api:8080'
        schedule = data.get('schedule')
        if schedule:
            args = (task, )
            job_id = _scheduler.add(schedule, fn, args)
            _db.save_schedule(data)
            resp = {'schedule': schedule, 'id': job_id}
        else:
            resp = fn.run(task, watch=False)

        logger.info('resp: %s', resp.to_yaml())
    except Exception as err:
        logger.error(traceback.format_exc())
        return json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: {}'.format(err),
        )

    if not isinstance(resp, dict):
        resp = resp.to_dict()
    return jsonify(ok=True, data=resp)


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
        fn.set_db_connection(_db)
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
    runtime = _db.get_function(name, project, tag)
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
        fn.set_db_connection(_db)
        resp = resource['start'](fn)
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

    fn = _db.get_function(name, project, tag)
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
    state = _k8s.get_pod_status(pod)
    logger.info('pod state={}'.format(state))

    if state == 'succeeded':
        logger.info('build completed successfully')
        state = 'ready'
    if state in ['failed', 'error']:
        logger.error('build {}, watch the build pod logs: {}'.format(
            state, pod))

    if logs and state != 'pending':
        resp = _k8s.logs(pod)
        if resp:
            out = resp.encode()[offset:]

    update_in(fn, 'status.state', state)
    if state == 'ready':
        update_in(fn, 'spec.image', image)

    _db.store_function(fn, name, project, tag)

    return Response(out, mimetype='text/plain',
                    headers={"function_status": state,
                             "function_image": image,
                             "builder_pod": pod})


def get_obj_path(schema, path):
    if schema:
        return schema + '://' + path
    elif path.startswith('/User/'):
        user = environ.get('V3IO_USERNAME', 'admin')
        return 'v3io:///users/' + user + path[5:]
    elif config.httpdb.files_path and \
            path.startswith(config.httpdb.files_path):
        return path
    return None


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


def log_path(project, uid) -> Path:
    return _logs_dir / project / uid

# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@app.route('/api/log/<project>/<uid>', methods=['POST'])
@catch_err
def store_log(project, uid):
    append = strtobool(request.args.get('append', 'no'))
    log_file = log_path(project, uid)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    body = request.get_data()  # TODO: Check size
    mode = 'ab' if append else 'wb'
    with log_file.open(mode) as fp:
        fp.write(body)
    return jsonify(ok=True)


# curl http://localhost:8080/log/prj/7
@app.route('/api/log/<project>/<uid>', methods=['GET'])
def get_log(project, uid):
    size = int(request.args.get('size', '-1'))
    offset = int(request.args.get('offset', '0'))

    out = b''
    log_file = log_path(project, uid)
    if log_file.exists():
        with log_file.open('rb') as fp:
            fp.seek(offset)
            out = fp.read(size)
        status = ''
    else:
        data = _db.read_run(uid, project)
        if not data:
            return json_error(HTTPStatus.NOT_FOUND,
                              project=project, uid=uid)

        status = get_in(data, 'status.state', '')
        if _k8s:
            pods = _k8s.get_logger_pods(uid)
            if pods:
                pod, new_status = list(pods.items())[0]
                new_status = new_status.lower()

                # TODO: handle in cron/tracking
                if new_status != 'pending':
                    resp = _k8s.logs(pod)
                    if resp:
                        out = resp.encode()[offset:]
                    if status == 'running':
                        now = str(datetime.now())
                        update_in(data, 'status.last_update', now)
                        if new_status == 'failed':
                            update_in(data, 'status.state', 'error')
                            update_in(
                                data, 'status.error', 'error, check logs')
                            _db.store_run(data, uid, project)
                        if new_status == 'succeeded':
                            update_in(data, 'status.state', 'completed')
                            _db.store_run(data, uid, project)
                status = new_status
            elif status == 'running':
                update_in(data, 'status.state', 'error')
                update_in(
                    data, 'status.error', 'pod not found, maybe terminated')
                _db.store_run(data, uid, project)
                status = 'failed'

    return Response(out, mimetype='text/plain',
                    headers={"pod_status": status})


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
    _db.store_run(data, uid, project, iter=iter)
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
    _db.update_run(data, uid, project, iter=iter)
    app.logger.info('update run: {}'.format(data))
    return jsonify(ok=True)


# curl http://localhost:8080/run/p1/3
@app.route('/api/run/<project>/<uid>', methods=['GET'])
@catch_err
def read_run(project, uid):
    iter = int(request.args.get('iter', '0'))
    data = _db.read_run(uid, project, iter=iter)
    return jsonify(ok=True, data=data)


# curl -X DELETE http://localhost:8080/run/p1/3
@app.route('/api/run/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_run(project, uid):
    iter = int(request.args.get('iter', '0'))
    _db.del_run(uid, project, iter=iter)
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

    runs = _db.list_runs(
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

    _db.del_runs(name, project, labels, state, days_ago)
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
    _db.store_artifact(key, data, uid, tag, project)
    return jsonify(ok=True)


# curl http://localhost:8080/artifact/p1/tag/key
@app.route('/api/artifact/<project>/<tag>/<path:key>', methods=['GET'])
@catch_err
def read_artifact(project, tag, key):
    data = _db.read_artifact(key, tag, project)
    return data

# curl -X DELETE http://localhost:8080/artifact/p1&key=k&tag=t
@app.route('/api/artifact/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_artifact(project, uid):
    key = request.args.get('key')
    if not key:
        return json_error(HTTPStatus.BAD_REQUEST, reason='missing data')

    tag = request.args.get('tag', '')
    _db.del_artifact(key, tag, project)
    return jsonify(ok=True)

# curl http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/api/artifacts', methods=['GET'])
@catch_err
def list_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', 'default')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    artifacts = _db.list_artifacts(name, project, tag, labels)
    return jsonify(ok=True, artifacts=artifacts)

# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/api/artifacts', methods=['DELETE'])
@catch_err
def del_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    _db.del_artifacts(name, project, tag, labels)
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

    _db.store_function(data, name, project, tag)
    return jsonify(ok=True)


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@app.route('/api/func/<project>/<name>', methods=['GET'])
@catch_err
def get_function(project, name):
    tag = request.args.get('tag', '')
    func = _db.get_function(name, project, tag)
    return jsonify(ok=True, func=func)


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@app.route('/api/funcs', methods=['GET'])
@catch_err
def list_functions():
    name = request.args.get('name', '')
    project = request.args.get('project', 'default')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    out = _db.list_functions(name, project, tag, labels)
    return jsonify(
        ok=True,
        funcs=list(out),
    )


@app.route('/api/healthz', methods=['GET'])
def health():
    return 'OK\n'


@app.before_first_request
def init_app():
    global _db, _logs_dir, _k8s, _scheduler

    logger.info('configuration dump\n%s', config.dump_yaml())
    if config.httpdb.db_type == 'sqldb':
        logger.info('using SQLDB')
        _db = SQLDB(config.httpdb.dsn)
    else:
        logger.info('using FileRunDB')
        _db = FileRunDB(config.httpdb.dirpath)
    _db.connect()
    _logs_dir = Path(config.httpdb.logs_path)

    try:
        _k8s = K8sHelper()
    except Exception:
        pass

    # @yaronha - Initialize here
    task = periodic.Task()
    periodic.schedule(task, 60)

    _scheduler = Scheduler()
    for data in _db.list_schedules():
        if 'schedule' not in data:
            logger.warning('bad scheduler data - %s', data)
            continue
        _submit(data)


# Don't remove this function, it's an entry point in setup.py
def main():
    app.run(
        host='0.0.0.0',
        port=config.httpdb.port,
        debug=config.httpdb.debug,
    )


if __name__ == '__main__':
    main()
