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

import pandas as pd
from ..utils import get_in, match_labels, dict_to_yaml, flatten
from ..render import run_to_html, runs_to_html, artifacts_to_html


class RunDBError(Exception):
    pass


class RunDBInterface:
    kind = ''

    def connect(self, secrets=None):
        return self

    def store_log(self, uid, project='', body=None, append=True):
        pass

    def get_log(self, uid, project=''):
        pass

    def store_run(self, struct, uid, project='', commit=False):
        pass

    def update_run(self, updates: dict, uid, project=''):
        pass

    def read_run(self, uid, project=''):
        pass

    def list_runs(self, name='', project='', labels=[],
                  state='', sort=True, last=0):
        pass

    def del_run(self, uid, project=''):
        pass

    def del_runs(self, name='', project='', labels=[], state='', days_ago=0):
        pass

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        pass

    def read_artifact(self, key, tag='', project=''):
        pass

    def list_artifacts(self, name='', project='', tag='', labels=[]):
        pass

    def del_artifact(self, key, tag='', project=''):
        pass

    def del_artifacts(self, name='', project='', tag='', labels=[], days_ago=0):
        pass

    def store_metric(self, uid, project='', keyvals={}, timestamp=None, labels={}):
        pass

    def read_metric(self, keys, project='', query=''):
        pass

