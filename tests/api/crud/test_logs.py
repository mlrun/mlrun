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
#
import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.crud


class TestLogs:
    @staticmethod
    def test_legacy_log_mechanism(
        db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        project = "project-name"
        uid = "m33"
        data1, data2 = b"ab", b"cd"
        mlrun.api.crud.Runs().store_run(
            db,
            {"metadata": {"name": "run-name"}, "some-run-data": "blabla"},
            uid,
            project=project,
        )
        mlrun.api.crud.Logs().store_log(data1, project, uid)
        log = mlrun.api.crud.Logs()._get_logs_legacy_method(db, project, uid)
        assert data1 == log, "get log 1"

        mlrun.api.crud.Logs().store_log(data2, project, uid, append=True)
        log = mlrun.api.crud.Logs()._get_logs_legacy_method(db, project, uid)
        assert data1 + data2 == log, "get log 2"

        mlrun.api.crud.Logs().store_log(data1, project, uid, append=False)
        log = mlrun.api.crud.Logs()._get_logs_legacy_method(db, project, uid)
        assert data1 == log, "get log append=False"
