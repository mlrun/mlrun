# Copyright 2023 Iguazio
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

from mlrun.api.runtime_handlers.base import BaseRuntimeHandler
from mlrun.runtimes.base import RuntimeClassMode


class KubeRuntimeHandler(BaseRuntimeHandler):
    kind = "job"
    class_modes = {RuntimeClassMode.run: "job", RuntimeClassMode.build: "build"}

    @staticmethod
    def _expect_pods_without_uid() -> bool:
        """
        builder pods are handled as part of this runtime handler - they are not coupled to run object, therefore they
        don't have the uid in their labels
        """
        return True

    @staticmethod
    def _are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"mlrun/uid={object_id}"


class DatabricksRuntimeHandler(KubeRuntimeHandler):
    kind = "databricks"
    class_modes = {RuntimeClassMode.run: "databricks"}
    pod_grace_period_seconds = 60
