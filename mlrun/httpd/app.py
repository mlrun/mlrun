# Copyright 2020 Iguazio
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

from datetime import date
from functools import wraps
from http import HTTPStatus
from json import JSONEncoder
from pathlib import Path

from flask import Flask, jsonify

from ..config import config
from ..db import SQLDB, FileRunDB, RunDBError
from ..k8s_utils import K8sHelper
from ..scheduler import Scheduler
from ..utils import logger
from . import periodic
from . import _submit

db = None
scheduler = None
k8s: K8sHelper = None
logs_dir = None


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, date):
                return obj.isoformat()
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)


app = Flask(__name__)
app.json_encoder = CustomJSONEncoder


def catch_err(fn):
    @wraps(fn)
    def wrapper(*args, **kw):
        try:
            return fn(*args, **kw)
        except RunDBError as err:
            return json_error(
                HTTPStatus.INTERNAL_SERVER_ERROR, ok=False, reason=str(err))

    return wrapper


def json_error(status=HTTPStatus.BAD_REQUEST, **kw):
    kw.setdefault('ok', False)
    logger.error(str(kw))
    reply = jsonify(**kw)
    reply.status_code = status
    return reply


@app.route('/api/healthz', methods=['GET'])
def health():
    return jsonify(ok=True, version=config.version)


@app.before_first_request
def init_app():
    global db, logs_dir, k8s, scheduler

    logger.info('configuration dump\n%s', config.dump_yaml())
    if config.httpdb.db_type == 'sqldb':
        logger.info('using SQLDB')
        db = SQLDB(config.httpdb.dsn)
    else:
        logger.info('using FileRunDB')
        db = FileRunDB(config.httpdb.dirpath)
    db.connect()
    logs_dir = Path(config.httpdb.logs_path)

    try:
        k8s = K8sHelper()
    except Exception:
        pass

    # @yaronha - Initialize here
    task = periodic.Task()
    periodic.schedule(task, 60)

    scheduler = Scheduler()
    for data in db.list_schedules():
        if 'schedule' not in data:
            logger.warning('bad scheduler data - %s', data)
            continue
        _submit.submit(data)
