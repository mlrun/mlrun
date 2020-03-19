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

from .app import app, catch_err, db
from flask import jsonify


# curl http://localhost:8080/schedules
@app.route('/api/schedules', methods=['GET'])
@catch_err
def list_schedules():
    return jsonify(
        ok=True,
        schedules=list(db.list_schedules())
    )
