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
import yaml
import json
from datetime import datetime
import uuid


from .artifacts import ArtifactManager
from .datastore import StoreManager
from .secrets import SecretsStore
from .db import get_run_db
from .utils import uxjoin, run_keys, get_in, dict_to_yaml, logger, dict_to_json


class MLCtxValueError(Exception):
    pass


class MLClientCtx(object):
    """ML Execution Client Context

    the context is generated using the get_or_create_ctx call (see its doc)
    and provides an interface to use run params, metadata, inputs, and outputs

    base metadata include: uid, name, project, and iteration (for hyper params)
    users can set labels and annotations using set_labels(), set_annotation()
    access parameters and secrets using get_param(), get_secret()
    access input data objects using get_input()
    store results, artifacts, and real-time metrics using log_xx methods

    see doc for the individual params and methods
    """

    def __init__(self, autocommit=False, tmp=''):
        self._uid = ''
        self.name = ''
        self._iteration = 0
        self._project = ''
        self._tag = ''
        self._secrets_manager = SecretsStore()

        # runtime db service interfaces
        self._rundb = None
        self._tmpfile = tmp
        self._logger = logger
        self._log_level = 'info'
        self._matrics_db = None
        self._autocommit = autocommit

        self._labels = {}
        self._annotations = {}

        self._runtime = {}
        self._parameters = {}
        #self._hyper_parameters = {}
        self._in_path = ''
        self._out_path = ''
        self._inputs = {}
        self._outputs = []

        self._results = {}
        self._state = 'created'
        self._error = None
        self._commit = ''
        self._host = None
        self._start_time = datetime.now()
        self._last_update = datetime.now()
        self._iteration_results = None

    def _init_dbs(self, rundb):
        if rundb:
            if isinstance(rundb, str):
                self._rundb = get_run_db(rundb)
                self._rundb.connect(self._secrets_manager)
            else:
                self._rundb = rundb
        self._data_stores = StoreManager(self._secrets_manager)
        self._artifacts_manager = ArtifactManager(
            self._data_stores, db=self._rundb, out_path=self._out_path)

    def get_meta(self):
        """Reserved for internal use"""
        resp = {'name': self.name,
                'kind': 'run',
                'uri': f'{self._project}/{self.uid}' if self._project else self.uid,
                'owner': get_in(self._labels, 'owner')}
        if 'workflow' in self._labels:
            resp['workflow'] = self._labels['workflow']
        return resp

    @classmethod
    def from_dict(cls, attrs: dict, rundb='', autocommit=False, tmp='',
                  host=None):

        self = cls(autocommit=autocommit, tmp=tmp)

        meta = attrs.get('metadata')
        if meta:
            self._uid = meta.get('uid', self._uid or uuid.uuid4().hex)
            self._iteration = meta.get('iteration', self._iteration)
            self.name = meta.get('name', self.name)
            self._project = meta.get('project', self._project)
            self._annotations = meta.get('annotations', self._annotations)
            self._labels = meta.get('labels', self._labels)
        spec = attrs.get('spec')
        if spec:
            self._secrets_manager = SecretsStore.from_dict(spec)
            self._log_level = spec.get('log_level', self._log_level)
            self._runtime = spec.get('runtime', self._runtime)
            self._parameters = spec.get('parameters', self._parameters)
            self._outputs = spec.get('outputs', self._outputs)
            self._out_path = spec.get(run_keys.output_path, self._out_path)
            self._in_path = spec.get(run_keys.input_path, self._in_path)
            inputs = spec.get(run_keys.inputs)

        self._init_dbs(rundb)

        if spec:
            # init data related objects (require DB & Secrets to be set first)
            self._data_stores.from_dict(spec)
            if inputs and isinstance(inputs, dict):
                for k, v in inputs.items():
                    self._set_input(k, v)

        if host:
            self._host = host
            self._state = 'running'
            self.set_label('host', host)

        self._update_db(commit=True)
        return self

    @property
    def uid(self):
        """Unique run id"""
        if self._iteration:
            return f'{self._uid}-{self._iteration}'
        return self._uid

    @property
    def tag(self):
        """run tag (uid or workflow id if exists)"""
        return self._labels.get('workflow', self._uid)

    @property
    def iteration(self):
        """child iteration index, for hyper parameters """
        return self._iteration

    @property
    def project(self):
        """project name, runs can be categorized by projects"""
        return self._project

    @property
    def logger(self):
        """built-in logger interface"""
        return self._logger

    @property
    def log_level(self):
        """get the logging level, e.g. 'debug', 'info', 'error'"""
        return self._log_level

    @log_level.setter
    def log_level(self, value: str):
        """set the logging level, e.g. 'debug', 'info', 'error'"""
        self._log_level = value
        print(f'changed log level to: {value}')

    @property
    def parameters(self):
        """dictionary of run parameters (read-only)"""
        return deepcopy(self._parameters)

    @property
    def in_path(self):
        """default input path for data objects"""
        return self._in_path

    @property
    def out_path(self):
        """default output path for artifacts"""
        return self._out_path

    @property
    def labels(self):
        """dictionary with labels (read-only)"""
        return deepcopy(self._labels)

    def set_label(self, key: str, value, replace: bool = True):
        """set/record a specific label"""
        if replace or not self._labels.get(key):
            self._labels[key] = str(value)

    @property
    def annotations(self):
        """dictionary with annotations (read-only)"""
        return deepcopy(self._annotations)

    def set_annotation(self, key: str, value, replace: bool = True):
        """set/record a specific annotation"""
        if replace or not self._annotations.get(key):
            self._annotations[key] = str(value)

    def get_param(self, key: str, default=None):
        """get a run parameter, or use the provided default if not set"""
        if key not in self._parameters:
            self._parameters[key] = default
            self._update_db()
            return default
        return self._parameters[key]

    def get_secret(self, key: str):
        """get a key based secret e.g. DB password from the context
        secrets can be specified when invoking a run through files, env, ..
        """
        if self._secrets_manager:
            return self._secrets_manager.get(key)
        return None

    def _set_input(self, key, url=''):
        if not url:
            url = key
        if self.in_path and not (url.startswith('/') or '://' in url):
            url = uxjoin(self._in_path, key)
        obj = self._data_stores.object(key, url)
        self._inputs[key] = obj
        return obj

    def get_input(self, key: str, url: str = ''):
        """get an input data object, data objects have methods such as
         .get(), .download(), .url, .. to access the actual data"""
        if key not in self._inputs:
            return self._set_input(key, url)
        else:
            return self._inputs[key]

    def log_result(self, key: str, value):
        """log a scalar result value"""
        self._results[str(key)] = value
        self._update_db()

    def log_results(self, results: dict):
        """log a set of scalar result values"""
        if not isinstance(results, dict):
            raise MLCtxValueError('(multiple) results must be in the form of dict')

        for p in results.keys():
            self._results[str(p)] = results[p]
        self._update_db()

    def log_iteration_results(self, best, summary: list, task: dict, commit=False):
        """Reserved for internal use"""

        if best:
            self._results['best_iteration'] = best
            for k, v in get_in(task, ['status', 'results'], {}).items():
                self._results[k] = v
            for a in get_in(task, ['status', run_keys.artifacts], []):
                self._artifacts_manager.artifacts[a['key']] = a

        self._iteration_results = summary
        if commit:
            self._update_db(commit=True)

    def log_metric(self, key: str, value, timestamp=None, labels={}):
        """TBD, log a real-time time-series metric"""
        if not timestamp:
            timestamp = datetime.now()
        if self._rundb:
            self._rundb.store_metric({key: value}, timestamp, labels)

    def log_metrics(self, keyvals: dict, timestamp=None, labels={}):
        """TBD, log a set of real-time time-series metrics"""
        if not timestamp:
            timestamp = datetime.now()
        if self._rundb:
            self._rundb.store_metric(keyvals, timestamp, labels)

    def log_artifact(self, item, body=None, target_path='', src_path=None,
                     tag='', viewer=None, upload=True, labels=None):
        """log an output artifact and optionally upload it"""
        self._artifacts_manager.log_artifact(self, item, body=body,
                                             target_path=target_path,
                                             src_path=src_path,
                                             tag=tag,
                                             viewer=viewer,
                                             upload=upload,
                                             labels=labels)
        self._update_db()

    def commit(self, message: str = ''):
        """save run state and add a commit message"""
        self._annotations['message'] = message
        self._update_db(commit=True, message=message)

    def set_state(self, state: str = None, error: str = None):
        """modify and store the run state or mark an error"""
        updates = {'status.last_update': str(datetime.now())}

        if error:
            self._state = 'error'
            self._error = str(error)
            updates['status.state'] = 'error'
            updates['status.error'] = error
        elif state and state != self._state and self._state != 'error':
            self._state = state
            updates['status.state'] = 'completed'

        if self._rundb:
            self._rundb.update_run(updates, self.uid, self.project)

    def set_hostname(self, host: str):
        """update the hostname"""
        self._host = host
        if self._rundb:
            updates = {'status.host': host}
            self._rundb.update_run(updates, self.uid, self.project)

    def to_dict(self):
        """convert the run context to a dictionary"""

        def set_if_valid(struct, key, val):
            if val:
                struct[key] = val

        struct = {
            'metadata':
                {'name': self.name,
                 'uid': self._uid,
                 'iteration': self._iteration,
                 'project': self._project,
                 'labels': self._labels,
                 'annotations': self._annotations},
            'spec':
                {'runtime': self._runtime,
                 'log_level': self._log_level,
                 'parameters': self._parameters,
                 'outputs': self._outputs,
                 run_keys.output_path: self._out_path,
                 run_keys.inputs: {k: v.url for k, v in self._inputs.items()},
                 },
            'status':
                {'state': self._state,
                 'results': self._results,
                 'start_time': str(self._start_time),
                 'last_update': str(self._last_update)},
            }

        set_if_valid(struct['status'], 'error', self._error)
        set_if_valid(struct['status'], 'commit', self._commit)
        set_if_valid(struct['status'], 'host', self._host)

        if self._iteration_results:
            struct['status']['iterations'] = self._iteration_results
        self._data_stores.to_dict(struct['spec'])
        self._artifacts_manager.to_dict(struct['status'])
        return struct

    def to_yaml(self):
        """convert the run context to a yaml buffer"""
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        """convert the run context to a json buffer"""
        return dict_to_json(self.to_dict())

    def _update_db(self, state='', commit=False, message=''):
        self.last_update = datetime.now()
        self._state = state or 'running'
        if self._tmpfile:
            data = self.to_json()
            with open(self._tmpfile, 'w') as fp:
                fp.write(data)
                fp.close()

        if commit or self._autocommit:
            self._commit = message
            if self._rundb:
                self._rundb.store_run(self.to_dict(), self.uid, self.project, commit)

