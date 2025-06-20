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

import mlrun
from mlrun_pipelines.common.helpers import PROJECT_ANNOTATION
from mlrun_pipelines.common.models import RunStatuses


class PipelineProviderMixin:
    def resolve_project_from_workflow_manifest(self, workflow_manifest):
        templates = workflow_manifest.get("spec", {}).get("templates", [])
        for template in templates:
            project_from_annotation = (
                template.get("metadata", {})
                .get("annotations", {})
                .get(PROJECT_ANNOTATION)
            )
            if project_from_annotation:
                return project_from_annotation
            command = template.get("container", {}).get("command", [])
            action = None
            for index, argument in enumerate(command):
                if argument == "mlrun" and index + 1 < len(command):
                    action = command[index + 1]
                    break
            if action:
                if action == "deploy":
                    project = self._resolve_project_from_command(
                        command,
                        hyphen_p_is_also_project=True,
                        has_func_url_flags=True,
                        has_runtime_flags=False,
                    )
                    if project:
                        return project
                elif action == "run":
                    project = self._resolve_project_from_command(
                        command,
                        hyphen_p_is_also_project=False,
                        has_func_url_flags=True,
                        has_runtime_flags=True,
                    )
                    if project:
                        return project
                elif action == "build":
                    project = self._resolve_project_from_command(
                        command,
                        hyphen_p_is_also_project=False,
                        has_func_url_flags=False,
                        has_runtime_flags=True,
                    )
                    if project:
                        return project
                else:
                    raise NotImplementedError(f"Unknown action: {action}")

        raise mlrun.errors.MLRunMissingProjectError()

    @staticmethod
    def resolve_error_from_pipeline(pipeline):
        if pipeline.run.status in [RunStatuses.error, RunStatuses.failed]:
            # status might not be available just yet
            workflow_status = json.loads(
                pipeline.pipeline_runtime.workflow_manifest
            ).get("status", {})
            for node in workflow_status.get("nodes", {}).values():
                # The "DAG" node is the parent node of the pipeline so we skip it for getting the detailed error
                if node["type"] not in ["DAG", "Skipped"]:
                    if message := node.get("message"):
                        return message
