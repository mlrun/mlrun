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

from abc import ABC, abstractmethod
import warnings


class RunDBError(Exception):
    pass


class RunDBInterface(ABC):
    kind = ''

    @abstractmethod
    def connect(self, secrets=None):
        return self

    @abstractmethod
    def store_log(self, uid, project='', body=None, append=True):
        pass

    @abstractmethod
    def get_log(self, uid, project=''):
        pass

    @abstractmethod
    def store_run(self, struct, uid, project='', commit=False):
        pass

    @abstractmethod
    def update_run(self, updates: dict, uid, project=''):
        pass

    @abstractmethod
    def read_run(self, uid, project=''):
        pass

    @abstractmethod
    def list_runs(
            self, name='', uid=None, project='', labels=None,
            state='', sort=True, last=0):
        pass

    @abstractmethod
    def del_run(self, uid, project=''):
        pass

    @abstractmethod
    def del_runs(self, name='', project='', labels=None, state='', days_ago=0):
        pass

    @abstractmethod
    def store_artifact(self, key, artifact, uid, tag='', project=''):
        pass

    @abstractmethod
    def read_artifact(self, key, tag='', project=''):
        pass

    @abstractmethod
    def list_artifacts(self, name='', project='', tag='', labels=None):
        pass

    @abstractmethod
    def del_artifact(self, key, tag='', project=''):
        pass

    @abstractmethod
    def del_artifacts(
            self, name='', project='', tag='', labels=None, days_ago=0):
        pass

    # TODO: Make these abstract once filedb implements them
    def store_metric(
            self, uid, project='', keyvals=None, timestamp=None, labels=None):
        warnings.warn('store_metric not implemented yet')

    def read_metric(self, keys, project='', query=''):
        warnings.warn('store_metric not implemented yet')
