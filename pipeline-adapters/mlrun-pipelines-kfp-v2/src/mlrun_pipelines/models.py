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
        try:
            return self._external_data["schemaVersion"]
        except KeyError:
            return self._external_data["apiVersion"]

    def is_argo_compatible(self) -> bool:
        if self.get_schema_version().startswith("argoproj.io"):
            return True
        return False

    def get_executors(self):
        if self.is_argo_compatible():
            yield from [
                (t.get("name"), t) for t in self._external_data["spec"]["templates"]
            ]
        else:
            yield from self._external_data["deploymentSpec"]["executors"].items()


class PipelineRun(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["run_id"]

    @property
    def name(self):
        return self._external_data["display_name"]

    @name.setter
    def name(self, name):
        self._external_data["display_name"] = name

    @property
    def status(self):
        return self._external_data["state"]

    @status.setter
    def status(self, status):
        self._external_data["state"] = status

    @property
    def description(self):
        return self._external_data["description"]

    @description.setter
    def description(self, description):
        self._external_data["description"] = description

    @property
    def created_at(self):
        return self._external_data["created_at"]

    @created_at.setter
    def created_at(self, created_at):
        self._external_data["created_at"] = created_at

    @property
    def scheduled_at(self):
        return self._external_data["scheduled_at"]

    @scheduled_at.setter
    def scheduled_at(self, scheduled_at):
        self._external_data["scheduled_at"] = scheduled_at

    @property
    def finished_at(self):
        return self._external_data["finished_at"]

    @finished_at.setter
    def finished_at(self, finished_at):
        self._external_data["finished_at"] = finished_at

    def workflow_manifest(self) -> PipelineManifest:
        return PipelineManifest(
            self._external_data["pipeline_spec"],
        )


class PipelineExperiment(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["experiment_id"]
