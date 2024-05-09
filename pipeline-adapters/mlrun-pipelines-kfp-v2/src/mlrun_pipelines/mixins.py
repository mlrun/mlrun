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

from mlrun_pipelines.common.helpers import PROJECT_ANNOTATION

import mlrun


class KfpAdapterMixin:
    def apply(self, modify):
        """
        Apply a modifier to the runtime which is used to change the runtimes k8s object's spec.
        Modifiers can be either KFP modifiers or MLRun modifiers (which are compatible with KFP)

        :param modify: a modifier runnable object
        :return: the runtime (self) after the modifications
        """
        return modify(self)


class PipelineProviderMixin:
    def resolve_project_from_workflow_manifest(self, workflow_manifest):
        for _, executor in workflow_manifest.get_executors():
            project_from_annotation = (
                executor.get("metadata", {})
                .get("annotations", {})
                .get(PROJECT_ANNOTATION)
            )
            if project_from_annotation:
                return project_from_annotation
            command = executor.get("container", {}).get("command", [])
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
