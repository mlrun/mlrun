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
import json
from copy import deepcopy
from os import environ
from pprint import pformat
from .utils import dict_to_yaml


class ModelObj:
    _dict_fields = []

    @staticmethod
    def _verify_list(param, name):
        if not isinstance(param, list):
            raise ValueError(f'parameter {name} must be a list')

    @staticmethod
    def _verify_dict(param, name, new_type=None):
        if param is not None and not isinstance(param, dict) and not hasattr(param, 'to_dict'):
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
    def from_dict(cls, struct={}):
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
        return json.dumps(self.to_dict())

    def to_str(self):
        return pformat(self.to_dict())

    def __str__(self):
        return self.to_str()

    def copy(self):
        return deepcopy(self)


class BaseMetadata(ModelObj):
    def __init__(self, name=None, namespace=None, labels=None, annotations=None):
        self.name = name
        self.namespace = namespace
        self.labels = labels or {}
        self.annotations = annotations or {}


class ImageBuilder(ModelObj):
    def __init__(self, inline_code=None, source=None, image=None, base_image=None,
                 commands=None, secret=None, registry=None):
        self.inline_code = inline_code
        self.source = source
        self.image = image
        self.base_image = base_image
        self.commands = commands or []
        self.secret = secret
        self.registry = registry
        self.interactive = True


class RunMetadata(ModelObj):
    def __init__(self, uid=None, name=None, project=None, labels=None, annotations=None, iteration=None):
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
    def __init__(self, parameters=None, hyperparams=None, param_file=None,
                 selector=None, handler=None, inputs=None, output_artifacts=None,
                 input_path=None, output_path=None,
                 secret_sources=None, data_stores=None):

        self.parameters = parameters or {}
        self.hyperparams = hyperparams or {}
        self.param_file = param_file
        self.selector = selector
        self.handler = handler
        self._inputs = inputs
        self._output_artifacts = output_artifacts
        self.input_path = input_path
        self.output_path = output_path
        self._secret_sources = secret_sources or []
        self._data_stores = data_stores

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
    def output_artifacts(self):
        return self._output_artifacts

    @output_artifacts.setter
    def output_artifacts(self, output_artifacts):
        self._verify_list(output_artifacts, 'output_artifacts')
        self._output_artifacts = output_artifacts

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


class RunStatus(ModelObj):
    def __init__(self, state=None, error=None, host=None, commit=None,
                 outputs=None, output_artifacts=None,
                 start_time=None, last_update=None, iterations=None):
        self.state = state or 'created'
        self.error = error
        self.host = host
        self.commit = commit
        self.outputs = outputs
        self.output_artifacts = output_artifacts
        self.start_time = start_time
        self.last_update = last_update
        self.iterations = iterations


class RunTemplate(ModelObj):
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

    def with_params(self, params):
        self.spec.parameters = params
        return self

    def with_hyper_params(self, hyperparams, selector=None):
        self.spec.hyperparams = hyperparams
        self.spec.selector = selector
        return self

    def with_param_file(self, param_file, selector=None):
        self.spec.param_file = param_file
        self.spec.selector = selector
        return self

    def with_secrets(self, kind, source):
        self.spec.secret_sources.append({'kind': kind, 'source': source})
        return self

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    def to_env(self):
        environ['MLRUN_EXEC_CONFIG'] = self.to_json()


class RunObject(RunTemplate):
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
        if self.status.outputs and key in self.status.outputs:
            return self.status.outputs.get(key)
        artifact = self.artifact(key)
        if artifact:
            return artifact.get('target_path')
        return None

    def artifact(self, key):
        if self.status.output_artifacts:
            for a in self.status.output_artifacts:
                if a['key'] == key:
                    return a
        return None


def NewRun(name=None, project=None, handler=None,
           params=None, hyper_params=None, param_file=None, selector=None,
           in_path=None, out_path=None, secrets=None):

    run = RunTemplate()
    run.metadata.name = name
    run.metadata.project = project
    run.spec.handler = handler
    run.spec.parameters = params
    run.spec.hyperparams = hyper_params
    run.spec.param_file = param_file
    run.spec.selector = selector
    run.spec.input_path = in_path
    run.spec.output_path = out_path
    run.spec.secret_sources = secrets or []
    return run
