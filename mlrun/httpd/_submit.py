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

# This module exists to avoid circular dependencies

import traceback
from http import HTTPStatus

from flask import jsonify

from ..run import import_function, new_function, parse_function_uri
from ..utils import get_in
from . import app


def submit(data):
    task = data.get('task')
    function = data.get('function')
    url = data.get('functionUrl')
    if not url and task:
        url = get_in(task, 'spec.function')
    if not (function or url) or not task:
        return app.json_error(
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
                runtime = app.db.get_function(name, project, tag)
                if not runtime:
                    return app.json_error(
                        HTTPStatus.BAD_REQUEST,
                        reason='runtime error: function {} not found'.format(
                            url),
                    )
                fn = new_function(runtime=runtime)

        fn.set_db_connection(app.db, True)
        app.logger.info('func:\n{}'.format(fn.to_yaml()))
        # fn.spec.rundb = 'http://mlrun-api:8080'
        schedule = data.get('schedule')
        if schedule:
            args = (task, )
            job_id = app.scheduler.add(schedule, fn, args)
            app.db.store_schedule(data)
            resp = {'schedule': schedule, 'id': job_id}
        else:
            resp = fn.run(task, watch=False)

        app.logger.info('resp: %s', resp.to_yaml())
    except Exception as err:
        app.logger.error(traceback.format_exc())
        return app.json_error(
            HTTPStatus.BAD_REQUEST,
            reason='runtime error: {}'.format(err),
        )

    if not isinstance(resp, dict):
        resp = resp.to_dict()
    return jsonify(ok=True, data=resp)
