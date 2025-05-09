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

import json
import typing
from enum import IntEnum
from typing import Any, Union

from kfp_server_api.models.api_run_detail import ApiRunDetail

from mlrun_pipelines.common.helpers import FlexibleMapper
from mlrun_pipelines.imports import ContainerOp

# class pointer for type checking on the main MLRun codebase
PipelineNodeWrapper = ContainerOp


class PipelineStep(FlexibleMapper):
    def __init__(self, step_type, node_name, node, node_template):
        data = {
            "step_type": step_type,
            "node_name": node_name,
            "node": node,
            "node_template": node_template,
        }
        super().__init__(data)

    @property
    def step_type(self):
        return self._external_data["step_type"]

    @property
    def node_name(self):
        return self._external_data["node_name"]

    @property
    def phase(self):
        return self._external_data["node"]["phase"]

    @property
    def skipped(self):
        return self._external_data["node"]["type"] == "Skipped"

    @property
    def display_name(self):
        return self._external_data["node"]["displayName"]

    def get_annotation(self, annotation_name: str):
        return self._external_data["node_template"]["metadata"]["annotations"].get(
            annotation_name
        )


class PipelineManifest(FlexibleMapper):
    def __init__(
        self, workflow_manifest: Union[str, dict] = "{}", pipeline_manifest: str = "{}"
    ):
        try:
            main_manifest = json.loads(workflow_manifest)
        except TypeError:
            main_manifest = workflow_manifest
        if pipeline_manifest != "{}":
            pipeline_manifest = json.loads(pipeline_manifest)
            main_manifest["status"] = pipeline_manifest.get("status", {})
        super().__init__(main_manifest)

    def get_steps(self) -> typing.Generator[PipelineStep, None, None]:
        nodes = sorted(
            self._external_data["status"]["nodes"].items(),
            key=lambda _node: _node[1]["finishedAt"],
        )
        for node_name, node in nodes:
            if node["type"] == "DAG":
                # Skip the parent DAG node
                continue

            node_template = next(
                template
                for template in self._external_data["spec"]["templates"]
                if template["name"] == node["templateName"]
            )
            step_type = node_template["metadata"]["annotations"].get(
                "mlrun/pipeline-step-type"
            )
            yield PipelineStep(step_type, node_name, node, node_template)


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

    @property
    def experiment_id(self) -> str:
        # If the PipelineRun object was created from another PipelineRun object/format,
        # the experiment_id is already available
        experiment_id = self._external_data.get("experiment_id")
        if experiment_id:
            return experiment_id

        for reference in self._external_data.get("resource_references") or []:
            data = reference.get("key", {})
            if (
                data.get("type", "") == "EXPERIMENT"
                and reference.get("relationship", "") == "OWNER"
                and reference.get("name", "") != "Default"
            ):
                return data.get("id", "")
        return ""

    def workflow_manifest(self) -> PipelineManifest:
        return self._workflow_manifest


class PipelineExperiment(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["id"]


class FilterOperations(IntEnum):
    UNKNOWN = 0
    EQUALS = 1
    NOT_EQUALS = 2
    GREATER_THAN = 3
    GREATER_THAN_EQUALS = 5
    LESS_THAN = 6
    LESS_THAN_EQUALS = 7
