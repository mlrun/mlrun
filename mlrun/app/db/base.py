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

import warnings
from abc import ABC, abstractmethod


class DBError(Exception):
    pass


class DBInterface(ABC):

    @abstractmethod
    def initialize(self, session):
        pass

    @abstractmethod
    def store_log(self, session, uid, project="", body=None, append=False):
        pass

    @abstractmethod
    def get_log(self, session, uid, project="", offset=0, size=0):
        pass

    @abstractmethod
    def store_run(self, session, struct, uid, project="", iter=0):
        pass

    @abstractmethod
    def update_run(self, session, updates: dict, uid, project="", iter=0):
        pass

    @abstractmethod
    def read_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def list_runs(
            self, session, name="", uid=None, project="", labels=None,
            state="", sort=True, last=0, iter=False):
        pass

    @abstractmethod
    def del_run(self, session, uid, project="", iter=0):
        pass

    @abstractmethod
    def del_runs(self, session, name="", project="", labels=None, state="", days_ago=0):
        pass

    @abstractmethod
    def store_artifact(self, session, key, artifact, uid, iter=None, tag="", project=""):
        pass

    @abstractmethod
    def read_artifact(self, session, key, tag="", iter=None, project=""):
        pass

    @abstractmethod
    def list_artifacts(
            self, session, name="", project="", tag="", labels=None,
            since=None, until=None):
        pass

    @abstractmethod
    def del_artifact(self, session, key, tag="", project=""):
        pass

    @abstractmethod
    def del_artifacts(
            self, session, name="", project="", tag="", labels=None):
        pass

    # TODO: Make these abstract once filedb implements them
    def store_metric(
            self, session, uid, project="", keyvals=None, timestamp=None, labels=None):
        warnings.warn("store_metric not implemented yet")

    def read_metric(self, session, keys, project="", query=""):
        warnings.warn("store_metric not implemented yet")

    @abstractmethod
    def store_function(self, session, func, name, project="", tag=""):
        pass

    @abstractmethod
    def get_function(self, session, name, project="", tag=""):
        pass

    @abstractmethod
    def list_functions(self, session, name, project="", tag="", labels=None):
        pass

    @abstractmethod
    def store_schedule(self, session, data):
        pass

    @abstractmethod
    def list_schedules(self, session):
        pass

    def list_projects(self, session):
        return []

    def list_artifact_tags(self, session, project):
        return []
