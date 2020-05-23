# Copyright 2019 Iguazio
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

from mlrun.api.db.base import DBError
from mlrun.api.db.sqldb.db import SQLDB as SQLAPIDB
from mlrun.api.db.sqldb.session import create_session
from .base import RunDBInterface, RunDBError


# This class is a proxy for the real implementation that sits under mlrun.api.db.sqldb
# The runtime objects (which manages the resources that do the real logic, like Nuclio functions, Dask jobs, etc...)
# require a RunDB to manage their state, when a user run them locally this db will either be the
# local filedb or the remote httpdb (we decided that we don't want to support SQLDB as an optional RunDB).
# When the user submits something to run (task, function etc...) this runtime managers actually runs inside the api
# service, in order to prevent the api from calling itself several times for each submission request (since the runDB
# will be httpdb to that same api service) we have this class which is kind of a proxy between the RunDB interface to
# the api service's DB interface
class SQLDB(RunDBInterface):
    def __init__(self, dsn, session=None, projects=None):
        self.session = session
        self.dsn = dsn
        self.projects = projects
        self.db = None

    def connect(self, secrets=None):
        if not self.session:
            self.session = create_session()
        self.db = SQLAPIDB(self.dsn, self.projects)
        return self

    def store_log(self, uid, project='', body=b'', append=False):
        return self._transform_db_error(self.db.store_log, self.session, uid, project, body, append)

    def get_log(self, uid, project='', offset=0, size=0):
        return self._transform_db_error(self.db.get_log, self.session, uid, project, offset, size)

    def store_run(self, struct, uid, project='', iter=0):
        return self._transform_db_error(self.db.store_run, self.session, struct, uid, project, iter)

    def update_run(self, updates: dict, uid, project='', iter=0):
        return self._transform_db_error(self.db.update_run, self.session, updates, uid, project, iter)

    def read_run(self, uid, project=None, iter=None):
        return self._transform_db_error(self.db.read_run, self.session, uid, project, iter)

    def list_runs(
            self, name=None, uid=None, project=None, labels=None,
            state=None, sort=True, last=0, iter=None):
        return self._transform_db_error(self.db.list_runs, self.session, name, uid, project, labels, state, sort, last,
                                        iter)

    def del_run(self, uid, project=None, iter=None):
        return self._transform_db_error(self.db.del_run, self.session, uid, project, iter)

    def del_runs(
            self, name=None, project=None, labels=None,
            state=None, days_ago=0):
        return self._transform_db_error(self.db.del_runs, self.session, name, project, labels, state, days_ago)

    def store_artifact(
            self, key, artifact, uid, iter=None, tag='', project=''):
        return self._transform_db_error(self.db.store_artifact, self.session, key, artifact, uid, iter, tag, project)

    def read_artifact(self, key, tag='', iter=None, project=''):
        return self._transform_db_error(self.db.read_artifact, self.session, key, tag, iter, project)

    def list_artifacts(
            self, name=None, project=None, tag=None, labels=None,
            since=None, until=None):
        return self._transform_db_error(self.db.list_artifacts, self.session, name, project, tag, labels, since, until)

    def del_artifact(self, key, tag='', project=''):
        return self._transform_db_error(self.db.del_artifact, self.session, key, tag, project)

    def del_artifacts(
            self, name='', project='', tag='', labels=None):
        return self._transform_db_error(self.db.del_artifacts, self.session, name, project, tag, labels)

    def store_function(self, function, name, project='', tag='', versioned=False):
        return self._transform_db_error(self.db.store_function, self.session, function, name, project, tag, versioned)

    def get_function(self, name, project='', tag='', hash_key=''):
        return self._transform_db_error(self.db.get_function, self.session, name, project, tag, hash_key)

    def list_functions(
            self, name, project=None, tag=None, labels=None):
        return self._transform_db_error(self.db.list_functions, self.session, name, project, tag, labels)

    def list_artifact_tags(self, project):
        return self._transform_db_error(self.db.list_artifact_tags, self.session, project)

    def store_schedule(self, data):
        return self._transform_db_error(self.db.store_schedule, self.session, data)

    def list_schedules(self):
        return self._transform_db_error(self.db.list_schedules, self.session)

    def tag_objects(self, objs, project: str, name: str):
        return self._transform_db_error(self.db.tag_objects, self.session, objs, project, name)

    def del_tag(self, project: str, name: str):
        return self._transform_db_error(self.db.del_tag, self.session, project, name)

    def find_tagged(self, project: str, name: str):
        return self._transform_db_error(self.db.find_tagged, self.session, project, name)

    def list_tags(self, project: str):
        return self._transform_db_error(self.db.list_tags, self.session, project)

    def add_project(self, project: dict):
        return self._transform_db_error(self.db.add_project, self.session, project)

    def update_project(self, name, data: dict):
        return self._transform_db_error(self.db.update_project, self.session, name, data)

    def get_project(self, name=None, project_id=None):
        return self._transform_db_error(self.db.get_project, self.session, name, project_id)

    def list_projects(self, owner=None):
        return self._transform_db_error(self.db.list_projects, self.session, owner)

    @staticmethod
    def _transform_db_error(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DBError as exc:
            raise RunDBError(exc.args)
