from distutils.util import strtobool
from functools import wraps
from http import HTTPStatus

from flask import Flask, request, jsonify

from mlrun.db.filedb import FileRunDB
from mlrun.db import RunDBError

_file_db: FileRunDB = None


app = Flask(__name__)


def json_error(status=HTTPStatus.BAD_REQUEST, **kw):
    kw.setdefault('ok', False)
    reply = jsonify(**kw)
    reply.status_code = status
    return reply


def catch_err(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        try:
            return fn(*args, **kw)
        except RunDBError as err:
            return json_error(
                HTTPStatus.INTERNAL_SERVER_ERROR, ok=False, reason=str(err))

    return wrapper


# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@app.route('/log/<project>/<uid>', methods=['POST'])
@catch_err
def store_log(project, uid):
    append = strtobool(request.args.get('append', 'no'))
    body = request.get_data()  # TODO: Check size
    _file_db.store_log(uid, project, body, append)
    return jsonify(ok=True)


# curl http://localhost:8080/log/prj/7
@app.route('/log/<project>/<uid>', methods=['GET'])
def get_log(project, uid):
    data = _file_db.get_log(uid, project)
    if data is None:
        return json_error(HTTPStatus.NOT_FOUND, project=project, uid=uid)

    return data

# curl -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@app.route('/run/<project>/<uid>', methods=['POST'])
@catch_err
def store_run(project, uid):
    commit = strtobool(request.args.get('commit', 'no'))
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    _file_db.store_run(data, uid, project, commit)
    return jsonify(ok=True)


# curl -X PATCH -d @/path/to/run.json http://localhost:8080/run/p1/3?commit=yes
@app.route('/run/<project>/<uid>', methods=['PATCH'])
@catch_err
def update_run(project, uid):
    try:
        data = request.get_json(force=True)
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason='bad JSON body')

    _file_db.update_run(data, uid, project)
    return jsonify(ok=True)


# curl http://localhost:8080/run/p1/3
@app.route('/run/<project>/<uid>', methods=['GET'])
@catch_err
def read_run(project, uid):
    data = _file_db.read_run(uid, project)
    return jsonify(ok=True, data=data)


# curl -X DELETE http://localhost:8080/run/p1/3
@app.route('/run/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_run(project, uid):
    _file_db.del_run(uid, project)
    return jsonify(ok=True)


# curl http://localhost:8080/runs?project=p1&name=x&label=l1&label=l2&sort=no
@app.route('/runs', methods=['GET'])
@catch_err
def list_runs(project):
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    labels = request.args.getlist('label')
    state = request.args.get('state', '')
    sort = strtobool(request.args.get('sort', 'on'))
    last = int(request.args.get('last', '30'))

    runs = _file_db.list_runs(name, project, labels, state, sort, last)
    return jsonify(ok=True, runs=runs)

# curl -X DELETE http://localhost:8080/runs?project=p1&name=x&days_ago=3
@app.route('/runs', methods=['DELETE'])
@catch_err
def del_runs(project):
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    labels = request.args.getlist('label')
    state = request.args.get('state', '')
    days_ago = int(request.args.get('days_ago', '0'))

    _file_db.del_runs(name, project, labels, state, days_ago)
    return jsonify(ok=True)


# curl -d@/path/to/artifcat http://localhost:8080/artifact/p1/7&key=k
@app.route('/artifact/<project>/<uid>', methods=['POST'])
@catch_err
def store_artifact(project, uid):
    artifact = request.get_data()
    key = request.args.get('key')
    if (not artifact) or (not key):
        return json_error(HTTPStatus.BAD_REQUEST, reason='missing data')

    tag = request.args.get('tag', '')
    _file_db.store_artifact(key, artifact, uid, tag, project)
    return jsonify(ok=True)

# curl http://localhost:8080/artifact/p1&key=k&tag=t
@app.route('/artifact/<project>/<uid>', methods=['GET'])
@catch_err
def read_artifact(project, uid):
    key = request.args.get('key')
    if not key:
        return json_error(HTTPStatus.BAD_REQUEST, reason='missing data')

    tag = request.args.get('tag', '')
    data = _file_db.read_artifact(key, tag, project)
    return data

# curl -X DELETE http://localhost:8080/artifact/p1&key=k&tag=t
@app.route('/artifact/<project>/<uid>', methods=['DELETE'])
@catch_err
def del_artifact(project, uid):
    key = request.args.get('key')
    if not key:
        return json_error(HTTPStatus.BAD_REQUEST, reason='missing data')

    tag = request.args.get('tag', '')
    _file_db.del_artifact(key, tag, project)
    return jsonify(ok=True)

# curl http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/artifacts', methods=['GET'])
@catch_err
def list_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    artifacts = _file_db.list_artifacts(name, project, tag, labels)
    return jsonify(ok=True, artifacts=artifacts)

# curl -X DELETE http://localhost:8080/artifacts?project=p1?label=l1
@app.route('/artifacts', methods=['DELETE'])
@catch_err
def del_artifacts():
    name = request.args.get('name', '')
    project = request.args.get('project', '')
    tag = request.args.get('tag', '')
    labels = request.args.getlist('label')

    _file_db.del_artifacts(name, project, tag, labels)
    return jsonify(ok=True)


@app.route('/healthz', methods=['GET'])
def health():
    return 'OK\n'


if __name__ == '__main__':
    from os import environ
    from os.path import expanduser

    default_dirpath = expanduser('~/.mlrun/db')
    dirpath = environ.get('MLRUN_HTTPDB_DIRPATH', default_dirpath)
    port = int(environ.get('MLRUN_HTTPDB_PORT', 8080))
    _file_db = FileRunDB(dirpath, '.yaml')
    _file_db.connect()
    app.run(
        host='0.0.0.0',
        port=port,
        debug='MLRUN_HTTPDB_DEBUG' in environ,
    )
