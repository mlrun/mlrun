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

import typing

from mlrun_pipelines.common.helpers import FlexibleMapper
from mlrun_pipelines.imports import PipelineTask

# class pointer for type checking on the main MLRun codebase
PipelineNodeWrapper = PipelineTask


class PipelineStep(FlexibleMapper):
    @property
    def step_type(self):
        raise NotImplementedError

    @property
    def node_name(self):
        raise NotImplementedError

    @property
    def phase(self):
        raise NotImplementedError

    @property
    def skipped(self):
        raise NotImplementedError

    @property
    def display_name(self):
        raise NotImplementedError

    def get_annotation(self, annotation_name: str):
        raise NotImplementedError


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
        # TODO: make sure this is compatible with KFP 2. The schema version in kfp 2 is a semver string,
        #       but since this code supports kfp 1.8 as well, where it considers the api version as the schema version
        #       we need to check if the schema version starts with "argoproj.io". Either way, for now this check is
        #       good enough and won't break whether the schema version is a semver string or not.
        schema_version_split = self.get_schema_version().split("/")[0]
        return schema_version_split == "argoproj.io"

    def get_executors(self):
        if self.is_argo_compatible():
            yield from [
                (t.get("name"), t) for t in self._external_data["spec"]["templates"]
            ]
        else:
            yield from self._external_data["deploymentSpec"]["executors"].items()

    def get_steps(self) -> typing.Generator[PipelineStep, None, None]:
        raise NotImplementedError


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

    def experiment_id(self) -> str:
        raise NotImplementedError

    def workflow_manifest(self) -> PipelineManifest:
        return PipelineManifest(
            self._external_data["pipeline_spec"],
        )


class PipelineExperiment(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["experiment_id"]
