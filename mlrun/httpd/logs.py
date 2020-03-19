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
from pathlib import Path

from flask import Response, jsonify, request

from ..utils import get_in, now_date, update_in
from . import app


def log_path(project, uid) -> Path:
    return app.logs_dir / project / uid

# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@app.route('/api/log/<project>/<uid>', methods=['POST'])
@app.catch_err
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
        data = app.db.read_run(uid, project)
        if not data:
            return app.json_error(
                HTTPStatus.NOT_FOUND, project=project, uid=uid)

        status = get_in(data, 'status.state', '')
        if app.k8s:
            pods = app.k8s.get_logger_pods(uid)
            if pods:
                pod, new_status = list(pods.items())[0]
                new_status = new_status.lower()

                # TODO: handle in cron/tracking
                if new_status != 'pending':
                    resp = app.k8s.logs(pod)
                    if resp:
                        out = resp.encode()[offset:]
                    if status == 'running':
                        now = now_date().isoformat()
                        update_in(data, 'status.last_update', now)
                        if new_status == 'failed':
                            update_in(data, 'status.state', 'error')
                            update_in(
                                data, 'status.error', 'error, check logs')
                            app.db.store_run(data, uid, project)
                        if new_status == 'succeeded':
                            update_in(data, 'status.state', 'completed')
                            app.db.store_run(data, uid, project)
                status = new_status
            elif status == 'running':
                update_in(data, 'status.state', 'error')
                update_in(
                    data, 'status.error', 'pod not found, maybe terminated')
                app.db.store_run(data, uid, project)
                status = 'failed'

    return Response(out, mimetype='text/plain',
                    headers={"pod_status": status})
