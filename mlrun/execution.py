from copy import deepcopy
from os import path

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
        self._objects = {}

        self._outputs = {}
        self._metrics = {}
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
            self._in_path = spec.get('default_input_path', self._in_path)
            in_list = spec.get('input_objects')
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

    def log_metric(self, key, value, timestamp=None):
        self._log_metric(key, value, timestamp)
        self._update_db()

    def _log_metric(self, key, value, timestamp=None):
        if key not in self._metrics:
            self._metrics[key] = MLMetric()
        if not timestamp:
            timestamp = datetime.now()
        self._metrics[key].xvalues.append(str(timestamp))
        self._metrics[key].yvalues.append(value)

    def log_metrics(self, keyvals={}, timestamp=None):
        if not timestamp:
            timestamp = datetime.now()
        for k, v in keyvals.items():
            self.log_metric(k, v, timestamp)
        self._update_db()

    def log_artifact(self, item, body=None, target_path=''):
        self._artifacts_manager.log_artifact(item, body, target_path, self._tag)
        self._update_db()

    def commit(self, message=''):
        self._update_db(commit=True)

    def to_dict(self):
        metrics = {k: v.to_dict() for (k, v) in self._metrics.items()}
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
                 'metrics': metrics,
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

    def _update_db(self, state='', elements=[], commit=False):
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

