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

from .utils import ModelObj


class BaseMetadata(ModelObj):
    def __init__(self, name=None, namespace=None, labels=None, annotations=None):
        self.name = name
        self.namespace = namespace
        self.labels = labels or {}
        self.annotations = annotations or {}


class RunRuntime(ModelObj):
    def __init__(self, kind=None, command=None, args=None, metadata=None, spec=None):
        self.kind = kind or ''
        self.command = command or ''
        self.args = args or []
        self._metadata = None
        self._spec = None
        self.metadata = metadata
        self.spec = spec

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec')

    @property
    def metadata(self) -> BaseMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, 'metadata', BaseMetadata)


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
    def __init__(self, runtime: RunRuntime = None,
                 parameters=None, hyperparams=None, param_file=None,
                 input_objects=None, output_artifacts=None,
                 input_path=None, output_path=None,
                 secret_sources=None, data_stores=None):

        self._runtime = None
        self.runtime = runtime
        self.parameters = parameters or {}
        self.hyperparams = hyperparams or {}
        self.param_file = param_file
        self._input_objects = input_objects
        self._output_artifacts = output_artifacts
        self.input_path = input_path
        self.output_path = output_path
        self._secret_sources = secret_sources
        self._data_stores = data_stores

    @property
    def runtime(self) -> RunRuntime:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime):
        self._runtime = self._verify_dict(runtime, 'runtime', RunRuntime)

    @property
    def input_objects(self):
        return self._input_objects

    @input_objects.setter
    def input_objects(self, input_objects):
        self._verify_list(input_objects, 'input_objects')
        self._input_objects = input_objects

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
    def __init__(self, state=None, outputs=None, output_artifacts=None,
                 start_time=None, last_update=None, iterations=None):
        self.state = state
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


