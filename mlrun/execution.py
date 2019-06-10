from os import path

import yaml, json
from datetime import datetime

from .data.artifacts import ArtifactManager
from .data.datastore import StoreManager
from .secrets import SecretsStore


class MLClientCtx(object):
    """Execution Client Context"""

    def __init__(self, name, uid, parent_type='', parent=''):
        self.uid = uid
        self.name = name
        self.parent = parent
        self.parent_type = parent_type
        self._secrets_manager = SecretsStore()
        self._data_stores = StoreManager(self._secrets_manager)
        self._artifacts_manager = ArtifactManager(self._data_stores, self.uid)
        self._reset_attrs()

        # runtime db service interfaces
        self._rundb = None
        self._logger = None
        self._matrics_db = None

    def _reset_attrs(self):
        self._project = ''
        self.owner = ''
        self._labels = {}
        self._annotations = {}

        self._parameters = {}

        self._outputs = {}
        self._metrics = {}
        self._state = 'created'
        self._start_time = datetime.now()
        self._last_update = datetime.now()

    def from_dict(self, attrs={}):
        meta = attrs.get('metadata')
        if meta:
            self.parent = meta.get('parent', self.parent)
            self.parent_type = meta.get('parent_type', self.parent_type)
            self.owner = meta.get('owner', self.owner)
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
        return self._parameters

    @property
    def labels(self):
        return self._labels

    @property
    def annotations(self):
        return self._annotations

    def get_or_set_param(self, key, default=None):
        if key not in self._parameters:
            self._parameters[key] = default
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

    def log_outputs(self, outputs={}):
        for p in outputs.keys():
            self._outputs[p] = outputs[p]

    def log_metric(self, key, value, timestamp=None):
        if key not in self._metrics:
            self._metrics[key] = KFPMetric()
        if not timestamp:
            timestamp = datetime.now()
        self._metrics[key].xvalues.append(str(timestamp))
        self._metrics[key].yvalues.append(value)

    def log_metrics(self, keyvals={}, timestamp=None):
        if not timestamp:
            timestamp = datetime.now()
        for k, v in keyvals.items():
            self.log_metric(k, v, timestamp)

    def log_artifact(self, key, body=None, target_path='', atype=''):
        self._artifacts_manager.log_artifact(key, target_path, body, atype)

    def commit(self, message=''):
        pass

    def to_dict(self):
        metrics = {k: v.to_dict() for (k, v) in self._metrics.items()}
        struct = {
            'metadata':
                {'name': self.name,
                 'uid': self.uid,
                 'parent': self.parent,
                 'parent_type': self.parent_type,
                 'owner': self.owner,
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
        self._artifacts_manager.to_dict(struct['spec'])
        return struct

    def to_yaml(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self):
        return json.dumps(self.to_dict())

    def _update_db(self, state='', outputs={}, artifacts={}, metrics={}):
        self.last_update = datetime.now()
        # TBD call external DB/API


class KFPMetric(object):

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

