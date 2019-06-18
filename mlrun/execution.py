from copy import deepcopy
from os import path

import yaml, json
from datetime import datetime

from .artifacts import ArtifactManager
from .datastore import StoreManager
from .secrets import SecretsStore
from .rundb import get_run_db


class MLClientCtx(object):
    """Execution Client Context"""

    def __init__(self, name, uid, rundb: '', autocommit=False):
        self.uid = uid
        self.name = name
        self.project = ''
        self._secrets_manager = SecretsStore()
        self._data_stores = StoreManager(self._secrets_manager)
        self._artifacts_manager = ArtifactManager(self._data_stores, self._get_meta)

        # runtime db service interfaces
        self._rundb = None
        if rundb:
            self._rundb = get_run_db(rundb)
            self._rundb.connect(self._secrets_manager)

        self._logger = None
        self._matrics_db = None
        self._autocommit = autocommit

        self._labels = {}
        self._annotations = {}

        self._parameters = {}

        self._outputs = {}
        self._metrics = {}
        self._state = 'created'
        self._start_time = datetime.now()
        self._last_update = datetime.now()

    def _get_meta(self):
        return {'name': self.name, 'type': self.parent_type, 'parent': self.parent, 'uid': self.uid}

    def from_dict(self, attrs={}):
        meta = attrs.get('metadata')
        if meta:
            self.uid = meta.get('uid', self.uid)
            self.name = meta.get('name', self.name)
            self.project = meta.get('project', self.project)
            self._annotations = meta.get('annotations', self._annotations)
            self._labels = meta.get('labels', self._labels)
        spec = attrs.get('spec')
        if spec:
            self._parameters = spec.get('parameters', self._parameters)
            self._secrets_manager.from_dict(spec)
            self._data_stores.from_dict(spec)
            self._artifacts_manager.from_dict(spec)

    def _set_from_json(self, data):
        attrs = json.loads(data)
        self._set_from_dict(attrs)

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

    def input_artifact(self, key):
        return self._artifacts_manager.get_input_artifact(key)

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

    def log_artifact(self, key, body=None, target_path='', atype='', source_path=''):
        self._artifacts_manager.log_artifact(key, target_path, body, atype, source_path)
        self._update_db()

    def commit(self, message=''):
        self._update_db(commit=True)

    def to_dict(self):
        metrics = {k: v.to_dict() for (k, v) in self._metrics.items()}
        struct = {
            'metadata':
                {'name': self.name,
                 'uid': self.uid,
                 'project': self.project,
                 'labels': self._labels,
                 'annotations': self._annotations},
            'spec':
                {'parameters': self._parameters},
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

