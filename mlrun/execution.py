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

from copy import deepcopy
import yaml, json
from datetime import datetime

from .artifacts import ArtifactManager
from .datastore import StoreManager
from .secrets import SecretsStore
from .rundb import get_run_db
from .utils import uxjoin, run_keys


class MLClientCtx(object):
    """Execution Client Context"""

    def __init__(self, name, uid, rundb: '', autocommit=False, tmp=''):
        self.uid = uid
        self.name = name
        self._project = ''
        self._tag = ''
        self._secrets_manager = SecretsStore()
        self._data_stores = StoreManager(self._secrets_manager)

        # runtime db service interfaces
        self._rundb = None
        if rundb:
            self._rundb = get_run_db(rundb)
            self._rundb.connect(self._secrets_manager)
        self._tmpfile = tmp
        self._artifacts_manager = ArtifactManager(
            self._data_stores, self, db=self._rundb)

        self._logger = None
        self._matrics_db = None
        self._autocommit = autocommit

        self._labels = {}
        self._annotations = {}

        self._runtime = {}
        self._parameters = {}
        self._in_path = ''
        self._out_path = ''
        self._objects = {}

        self._outputs = {}
        self._state = 'created'
        self._start_time = datetime.now()
        self._last_update = datetime.now()

    def get_meta(self):
        return {'name': self.name,
                'labels': self.labels,
                'start_time': str(self._start_time),
                'project': self._project,
                'uid': self.uid}

    def from_dict(self, attrs={}):
        meta = attrs.get('metadata')
        if meta:
            self.uid = meta.get('uid', self.uid)
            self.name = meta.get('name', self.name)
            self._project = meta.get('project', self._project)
            self._tag = meta.get('tag', self._tag)
            self._annotations = meta.get('annotations', self._annotations)
            self._labels = meta.get('labels', self._labels)
        spec = attrs.get('spec')
        if spec:
            self._secrets_manager.from_dict(spec)
            self._runtime = spec.get('runtime', self._runtime)
            self._parameters = spec.get('parameters', self._parameters)
            self._out_path = spec.get(run_keys.output_path, self._out_path)
            self._in_path = spec.get(run_keys.input_path, self._in_path)
            in_list = spec.get(run_keys.input_objects)
            if in_list and isinstance(in_list, list):
                for item in in_list:
                    self._set_object(item['key'], item.get('path'))

            self._data_stores.from_dict(spec)
            self._artifacts_manager.from_dict(spec)

    def _set_from_json(self, data):
        attrs = json.loads(data)
        self.from_dict(attrs)

    @property
    def project(self):
        return self._project

    @property
    def tag(self):
        return self._tag or self.uid

    @property
    def parameters(self):
        return deepcopy(self._parameters)

    @property
    def in_path(self):
        return self._in_path

    @property
    def out_path(self):
        return self._out_path

    @property
    def labels(self):
        return deepcopy(self._labels)

    @property
    def annotations(self):
        return deepcopy(self._annotations)

    def get_param(self, key, default=None):
        if key not in self._parameters:
            self._parameters[key] = default
            self._update_db()
            return default
        return self._parameters[key]

    def get_secret(self, key):
        if self._secrets_manager:
            return self._secrets_manager.get(key)
        return None

    def _set_object(self, key, realpath=''):
        if not realpath:
            realpath = uxjoin(self._in_path, key)
        object = self._data_stores.object(key, realpath)
        self._objects[key] = object
        return object

    def get_object(self, key, realpath=''):
        if key not in self._objects:
            return self._set_object(key, realpath)
        else:
            return self._objects[key]

    def log_output(self, key, value):
        self._outputs[key] = value
        self._update_db()

    def log_outputs(self, outputs={}):
        for p in outputs.keys():
            self._outputs[p] = outputs[p]
        self._update_db()

    def log_metric(self, key, value, timestamp=None, labels={}):
        if not timestamp:
            timestamp = datetime.now()
        if self._rundb:
            self._rundb.store_metric({key: value}, timestamp, labels)

    def log_metrics(self, keyvals={}, timestamp=None, labels={}):
        if not timestamp:
            timestamp = datetime.now()
        if self._rundb:
            self._rundb.store_metric(keyvals, timestamp, labels)

    def log_artifact(self, item, body=None, target_path='', src_path='',
                     tag='', viewer='', upload=True):
        self._artifacts_manager.log_artifact(item, body=body,
                                             target_path=target_path,
                                             src_path=src_path,
                                             tag=tag or self._tag,
                                             viewer=viewer,
                                             upload=upload)
        self._update_db()

    def commit(self, message=''):
        self._update_db(commit=True, message=message)

    def to_dict(self):
        struct = {
            'metadata':
                {'name': self.name,
                 'uid': self.uid,
                 'project': self._project,
                 'tag': self._tag,
                 'labels': self._labels,
                 'annotations': self._annotations},
            'spec':
                {'runtime': self._runtime,
                 'parameters': self._parameters,
                 run_keys.input_objects: [item.to_dict() for item in self._objects.values()],
                 },
            'status':
                {'state': self._state,
                 'outputs': self._outputs,
                 'start_time': str(self._start_time),
                 'last_update': str(self._last_update)},
            }
        self._data_stores.to_dict(struct['spec'])
        self._artifacts_manager.to_dict(struct)
        return struct

    def to_yaml(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self):
        return json.dumps(self.to_dict())

    def _update_db(self, state='', elements=[], commit=False, message=''):
        self.last_update = datetime.now()
        self._state = state or 'running'
        if self._tmpfile:
            data = self.to_json()
            with open(self._tmpfile, 'w') as fp:
                fp.write(data)
                fp.close()

        if commit or self._autocommit:
            if self._rundb:
                self._rundb.store_run(self, elements, commit)


class MLMetric(object):

    def __init__(self, labels={}):
        self.labels = labels
        self.xvalues = []
        self.yvalues = []

    def to_dict(self):
        return {
            'labels': self.labels,
            'xvalues': self.xvalues,
            'yvalues': self.yvalues,
        }

