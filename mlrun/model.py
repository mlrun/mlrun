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

import inspect
from copy import deepcopy
from os import environ

from .db import get_run_db
from .utils import dict_to_yaml, get_in, dict_to_json, get_artifact_target
from .config import config


class ModelObj:
    _dict_fields = []

    @staticmethod
    def _verify_list(param, name):
        if not isinstance(param, list):
            raise ValueError(f'parameter {name} must be a list')

    @staticmethod
    def _verify_dict(param, name, new_type=None):
        if (
          param is not None and
          not isinstance(param, dict) and
          not hasattr(param, 'to_dict')):
            raise ValueError(f'parameter {name} must be a dict or object')
        if new_type and (isinstance(param, dict) or param is None):
            return new_type.from_dict(param)
        return param

    def to_dict(self, fields=None, exclude=None):
        struct = {}
        fields = fields or self._dict_fields
        if not fields:
            fields = list(inspect.signature(self.__init__).parameters.keys())
        for t in fields:
            if not exclude or t not in exclude:
                val = getattr(self, t, None)
                if val is not None and not (isinstance(val, dict) and not val):
                    if hasattr(val, 'to_dict'):
                        val = val.to_dict()
                        if val:
                            struct[t] = val
                    else:
                        struct[t] = val
        return struct

    @classmethod
    def from_dict(cls, struct=None, fields=None):
        struct = {} if struct is None else struct
        fields = fields or cls._dict_fields
        if not fields:
            fields = list(inspect.signature(cls.__init__).parameters.keys())
        new_obj = cls()
        if struct:
            for key, val in struct.items():
                if key in fields:
                    setattr(new_obj, key, val)
        return new_obj

    def to_yaml(self):
        return dict_to_yaml(self.to_dict())

    def to_json(self):
        return dict_to_json(self.to_dict())

    def to_str(self):
        return '{}'.format(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        return deepcopy(self)


class BaseMetadata(ModelObj):
    def __init__(self, name=None, tag=None, hash=None, namespace=None,
                 project=None, labels=None, annotations=None,
                 categories=None, updated=None):
        self.name = name
        self.tag = tag
        self.hash = hash
        self.namespace = namespace
        self.project = project or config.default_project
        self.labels = labels or {}
        self.categories = categories or []
        self.annotations = annotations or {}
        self.updated = updated


class ImageBuilder(ModelObj):
    """An Image builder"""
    def __init__(
        self, functionSourceCode=None, source=None, image=None,
            base_image=None, commands=None, secret=None,
            code_origin=None, registry=None):
        self.functionSourceCode = functionSourceCode  #: functionSourceCode
        self.codeEntryType = ''  #: codeEntryType
        self.source = source  #: course
        self.code_origin = code_origin  #: code_origin
        self.image = image  #: image
        self.base_image = base_image  #: base_image
        self.commands = commands or []  #: commands
        self.secret = secret  #: secret
        self.registry = registry  #: registry
        self.build_pod = None


class RunMetadata(ModelObj):
    """Run metadata"""
    def __init__(
        self, uid=None, name=None, project=None, labels=None,
            annotations=None, iteration=None):
        self.uid = uid
        self._iteration = iteration
        self.name = name
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}

    @property
    def iteration(self):
        return self._iteration or 0

    @iteration.setter
    def iteration(self, iteration):
        self._iteration = iteration


class RunSpec(ModelObj):
    """Run specification"""
    def __init__(self, parameters=None, hyperparams=None, param_file=None,
                 selector=None, handler=None, inputs=None, outputs=None,
                 input_path=None, output_path=None, function=None,
                 secret_sources=None, data_stores=None,
                 tuning_strategy=None, verbose=None):

        self.parameters = parameters or {}
        self.hyperparams = hyperparams or {}
        self.param_file = param_file
        self.tuning_strategy = tuning_strategy
        self.selector = selector
        self.handler = handler
        self._inputs = inputs
        self._outputs = outputs
        self.input_path = input_path
        self.output_path = output_path
        self.function = function
        self._secret_sources = secret_sources or []
        self._data_stores = data_stores
        self.verbose = verbose

    def to_dict(self, fields=None, exclude=None):
        struct = super().to_dict(fields, exclude=['handler'])
        if self.handler and isinstance(self.handler, str):
            struct['handler'] = self.handler
        return struct

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = self._verify_dict(inputs, 'inputs')

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._verify_list(outputs, 'outputs')
        self._outputs = outputs

    @property
    def secret_sources(self):
        return self._secret_sources

    @secret_sources.setter
    def secret_sources(self, secret_sources):
        self._verify_list(secret_sources, 'secret_sources')
        self._secret_sources = secret_sources

    @property
    def data_stores(self):
        return self._data_stores

    @data_stores.setter
    def data_stores(self, data_stores):
        self._verify_list(data_stores, 'data_stores')
        self._data_stores = data_stores

    @property
    def handler_name(self):
        if self.handler:
            if inspect.isfunction(self.handler):
                return self.handler.__name__
            else:
                return str(self.handler)
        return ''


class RunStatus(ModelObj):
    """Run status"""
    def __init__(self, state=None, error=None, host=None, commit=None,
                 status_text=None, results=None, artifacts=None,
                 start_time=None, last_update=None, iterations=None):
        self.state = state or 'created'
        self.status_text = status_text
        self.error = error
        self.host = host
        self.commit = commit
        self.results = results
        self.artifacts = artifacts
        self.start_time = start_time
        self.last_update = last_update
        self.iterations = iterations


class RunTemplate(ModelObj):
    """Run template"""
    def __init__(self, spec: RunSpec = None,
                 metadata: RunMetadata = None):
        self._spec = None
        self._metadata = None
        self.spec = spec
        self.metadata = metadata

    @property
    def spec(self) -> RunSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', RunSpec)

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, 'metadata', RunMetadata)

    def with_params(self, **kwargs):
        self.spec.parameters = kwargs
        return self

    def with_input(self, key, path):
        if not self.spec.inputs:
            self.spec.inputs = {}
        self.spec.inputs[key] = path
        return self

    def with_hyper_params(self, hyperparams, selector=None, strategy=None):
        self.spec.hyperparams = hyperparams
        self.spec.selector = selector
        self.spec.tuning_strategy = strategy
        return self

    def with_param_file(self, param_file, selector=None, strategy=None):
        self.spec.param_file = param_file
        self.spec.selector = selector
        self.spec.tuning_strategy = strategy
        return self

    def with_secrets(self, kind, source):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows, e.g.

        proj.with_secrets('file', 'file.txt')
        proj.with_secrets('inline', {'key': 'val'})
        proj.with_secrets('env', 'ENV1,ENV2')

        :param kind:   secret type (file, inline, env)
        :param source: secret data or link (see example)

        :returns: project object
        """
        self.spec.secret_sources.append({'kind': kind, 'source': source})
        return self

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    def to_env(self):
        environ['MLRUN_EXEC_CONFIG'] = self.to_json()


class RunObject(RunTemplate):
    """A run"""
    def __init__(self, spec: RunSpec = None,
                 metadata: RunMetadata = None,
                 status: RunStatus = None):
        super().__init__(spec, metadata)
        self._status = None
        self.status = status

    @classmethod
    def from_template(cls, template: RunTemplate):
        return cls(template.spec, template.metadata)

    @property
    def status(self) -> RunStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, 'status', RunStatus)

    def output(self, key):
        if self.status.results and key in self.status.results:
            return self.status.results.get(key)
        artifact = self.artifact(key)
        if artifact:
            return get_artifact_target(artifact, self.metadata.project)
        return None

    @property
    def outputs(self):
        outputs = {}
        if self.status.results:
            outputs = {k: v for k, v in self.status.results.items()}
        if self.status.artifacts:
            for a in self.status.artifacts:
                outputs[a['key']] = get_artifact_target(
                    a, self.metadata.project)
        return outputs

    def artifact(self, key):
        if self.status.artifacts:
            for a in self.status.artifacts:
                if a['key'] == key:
                    return a
        return None

    def uid(self):
        return self.metadata.uid

    def state(self):
        db = get_run_db().connect()
        run = db.read_run(uid=self.metadata.uid,
                          project=self.metadata.project,
                          iter=self.metadata.iteration)
        if run:
            return get_in(run, 'status.state', 'unknown')

    def show(self):
        db = get_run_db().connect()
        db.list_runs(
            uid=self.metadata.uid, project=self.metadata.project).show()

    def logs(self, watch=True, db=None):
        if not db:
            db = get_run_db().connect()
        if not db:
            print('DB is not configured, cannot show logs')
            return None

        if db.kind == 'http':
            state = db.watch_log(self.metadata.uid,
                                 self.metadata.project,
                                 watch=watch)
        else:
            state, text = db.get_log(self.metadata.uid,
                                     self.metadata.project)
            if text:
                print(text.decode())

        if state:
            print('final state: {}'.format(state))
        return state


def NewTask(name=None, project=None, handler=None,
            params=None, hyper_params=None, param_file=None, selector=None,
            tuning_strategy=None, inputs=None, outputs=None,
            in_path=None, out_path=None, artifact_path=None,
            secrets=None, base=None):
    """Create new task"""

    if base:
        run = deepcopy(base)
    else:
        run = RunTemplate()
    run.metadata.name = name or run.metadata.name
    run.metadata.project = project or run.metadata.project
    run.spec.handler = handler or run.spec.handler
    run.spec.parameters = params or run.spec.parameters
    run.spec.hyperparams = hyper_params or run.spec.hyperparams
    run.spec.param_file = param_file or run.spec.param_file
    run.spec.tuning_strategy = tuning_strategy or run.spec.tuning_strategy
    run.spec.selector = selector or run.spec.selector
    run.spec.inputs = inputs or run.spec.inputs
    run.spec.outputs = outputs or run.spec.outputs or []
    run.spec.input_path = in_path or run.spec.input_path
    run.spec.output_path = artifact_path or out_path or run.spec.output_path
    run.spec.secret_sources = secrets or run.spec.secret_sources or []
    return run
