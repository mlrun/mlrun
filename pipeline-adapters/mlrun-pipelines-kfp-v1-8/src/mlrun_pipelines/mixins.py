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

import kfp
from mlrun_pipelines.common.helpers import PROJECT_ANNOTATION
from mlrun_pipelines.common.models import RunStatuses
from mlrun_pipelines.utils import apply_kfp

import mlrun

# Disable the warning about reusing components
kfp.dsl.ContainerOp._DISABLE_REUSABLE_COMPONENT_WARNING = True


class KfpAdapterMixin:
    def apply(self, modify):
        """
        Apply a modifier to the runtime which is used to change the runtimes k8s object's spec.
        Modifiers can be either KFP modifiers or MLRun modifiers (which are compatible with KFP). All modifiers accept
        a `kfp.dsl.ContainerOp` object, apply some changes on its spec and return it so modifiers can be chained
        one after the other.

        :param modify: a modifier runnable object
        :return: the runtime (self) after the modifications
        """

        # Kubeflow pipeline have a hook to add the component to the DAG on ContainerOp init
        # we remove the hook to suppress kubeflow op registration and return it after the apply()
        old_op_handler = kfp.dsl._container_op._register_op_handler
        kfp.dsl._container_op._register_op_handler = lambda x: self.metadata.name
        cop = kfp.dsl.ContainerOp("name", "image")
        kfp.dsl._container_op._register_op_handler = old_op_handler

        return apply_kfp(modify, cop, self)


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

        return mlrun.mlconf.default_project

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
