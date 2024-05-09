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

import json
from typing import Any, Union

from kfp.dsl import ContainerOp
from kfp_server_api.models.api_run_detail import ApiRunDetail
from mlrun_pipelines.common.helpers import FlexibleMapper

# class pointer for type checking on the main MLRun codebase
PipelineNodeWrapper = ContainerOp


class PipelineManifest(FlexibleMapper):
    def __init__(
        self, workflow_manifest: Union[str, dict] = "{}", pipeline_manifest: str = "{}"
    ):
        try:
            main_manifest = json.loads(workflow_manifest)
        except TypeError:
            main_manifest = workflow_manifest
        if pipeline_manifest:
            pipeline_manifest = json.loads(pipeline_manifest)
            main_manifest["status"] = pipeline_manifest.get("status", {})
        super().__init__(main_manifest)


class PipelineRun(FlexibleMapper):
    _workflow_manifest: PipelineManifest

    def __init__(self, external_data: Any):
        if isinstance(external_data, ApiRunDetail):
            super().__init__(external_data.run)
            self._workflow_manifest = PipelineManifest(
                self._external_data.get("pipeline_spec", {}).get("workflow_manifest"),
                external_data.pipeline_runtime.workflow_manifest,
            )
        else:
            super().__init__(external_data)
            pipeline_spec = self._external_data.get("pipeline_spec", None) or {}
            workflow_manifest = pipeline_spec.get("workflow_manifest", None) or {}
            self._workflow_manifest = PipelineManifest(workflow_manifest)

    @property
    def id(self):
        return self._external_data["id"]

    @property
    def name(self):
        return self._external_data["name"]

    @name.setter
    def name(self, name):
        self._external_data["name"] = name

    @property
    def status(self):
        return self._external_data["status"]

    @status.setter
    def status(self, status):
        self._external_data["status"] = status

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
        return self._workflow_manifest


class PipelineExperiment(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["id"]
