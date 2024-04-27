# Copyright 2024 Iguazio
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
#

from kfp.dsl import PipelineTask
from mlrun_pipelines.common.helpers import FlexibleMapper

# class pointer for type checking on the main MLRun codebase
PipelineNodeWrapper = PipelineTask


class PipelineManifest(FlexibleMapper):
    """
    A Pipeline Manifest might have been created by an 1.8 SDK regardless of coming from a 2.0 API,
    so this class tries to account for that
    """

    def get_schema_version(self) -> str:
        raise NotImplementedError

    def is_argo_compatible(self) -> bool:
        raise NotImplementedError

    def get_executors(self):
        raise NotImplementedError


class PipelineRun(FlexibleMapper):
    @property
    def id(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @name.setter
    def name(self, name):
        raise NotImplementedError

    @property
    def status(self):
        raise NotImplementedError

    @status.setter
    def status(self, status):
        raise NotImplementedError

    @property
    def description(self):
        raise NotImplementedError

    @description.setter
    def description(self, description):
        raise NotImplementedError

    @property
    def created_at(self):
        raise NotImplementedError

    @created_at.setter
    def created_at(self, created_at):
        raise NotImplementedError

    @property
    def scheduled_at(self):
        raise NotImplementedError

    @scheduled_at.setter
    def scheduled_at(self, scheduled_at):
        raise NotImplementedError

    @property
    def finished_at(self):
        raise NotImplementedError

    @finished_at.setter
    def finished_at(self, finished_at):
        raise NotImplementedError

    @property
    def workflow_manifest(self) -> PipelineManifest:
        raise NotImplementedError


class PipelineExperiment(FlexibleMapper):
    @property
    def id(self):
        raise NotImplementedError
