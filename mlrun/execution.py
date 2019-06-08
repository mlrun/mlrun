from os import path

import yaml, json
from datetime import datetime

from mlrun import stores


class KFPClientCtx(object):
    """Execution Client Context"""

    def __init__(self, uid, name, parent_type='', parent=''):
        self.uid = uid
        self.name = name
        self.parent = parent
        self.parent_type = parent_type
        self._reset_attrs()

    def _reset_attrs(self):
        self._project = ''
        self.owner = ''
        self._labels = {}
        self._annotations = {}

        self._parameters = {}
        self._input_artifacts = {}
        self._default_in_path = ''
        self._default_out_path = ''

        # runtime db/storage service interfaces
        self._secrets_store = None
        self._rundb = None
        self._logger = None
        self._matrics_db = None
        self._data_stores = {}

        self._outputs = {}
        self._output_artifacts = {}
        self._metrics = {}
        self._state = 'created'
        self._start_time = datetime.now()
        self._last_update = datetime.now()

    def _set_from_dict(self, attrs={}):
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
            self._input_artifacts = spec.get('input_artifacts', self._input_artifacts)
            self._data_out_path = spec.get('_data_out_path', self._data_out_path)

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
        if self._secrets_store:
            return self._secrets_store.get(key)
        return None

    def input_artifact(self, key):
        if key not in self._input_artifacts:
            url = path.join(self._default_in_path, key)
            self._input_artifacts[key] = url
        else:
            url = self._input_artifacts[key]
        repo = stores.url2repo(url, self._secrets_store)
        return repo

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

    def log_artifact(self, key, localpath='', body=None, target_path='', atype=''):
        artifact = KFPArtifact(self, localpath, body, target_path, atype)
        # TBD uploads
        self._output_artifacts[key] = artifact

    def to_dict(self):
        metrics = {k: v.to_dict() for (k, v) in self._metrics.items()}
        artifacts = {k: v.to_dict() for k, v in self._output_artifacts.items()}
        return {'metadata':
                   {'name': self.name,
                    'uid': self.uid,
                    'parent': self.parent,
                    'parent_type': self.parent_type,
                    'owner': self.owner,
                    'labels': self._labels,
                    'annotations': self._annotations},
                'spec':
                   {'parameters': self._parameters,
                    'input_artifacts': self._input_artifacts,
                    'data_in_path': self._data_out_path,
                    'data_out_path': self._data_out_path},
        'status':
                   {'state': self._state,
                    'outputs': self._outputs,
                    'output_artifacts': artifacts,
                    'metrics': metrics,
                    'start_time': str(self._start_time),
                    'last_update': str(self.last_update)},
                }

    def to_yaml(self):
        return yaml.dump(self.to_dict())

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


class KFPArtifact(object):

    def __init__(self, run, localpath='', body=None, target_path='', atype=''):
        self.run = run
        self.localpath = localpath
        if target_path:
            self.target_path = target_path
        else:
            self.target_path = path.join(run._data_out_path, localpath)
        self.atype = atype

    def upload(self, secrets_func=None):
        repo = stores.url2repo(self.target_path, self._secrets_store)
        repo.upload(self.localpath)

    def to_dict(self):
        return {
            'localpath': self.localpath,
            'target_path': self.target_path,
            'atype': self.atype,
        }

