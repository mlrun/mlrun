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

import mlrun.common.constants as mlrun_constants
import mlrun.runtimes
from mlrun.runtimes.base import RuntimeClassMode
from server.api.runtime_handlers.kubejob import KubeRuntimeHandler


class RemoteSparkRuntimeHandler(KubeRuntimeHandler):
    kind = "remote-spark"
    class_modes = {RuntimeClassMode.run: "remote-spark"}

    def run(
        self,
        runtime: mlrun.runtimes.RemoteSparkRuntime,
        run: mlrun.run.RunObject,
        execution: mlrun.execution.MLClientCtx,
    ):
        runtime.spec.image = runtime.spec.image or runtime.default_image
        super().run(runtime=runtime, run=run, execution=execution)

    @staticmethod
    def are_resources_coupled_to_run_object() -> bool:
        return True

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f"{mlrun_constants.MLRunInternalLabels.uid}={object_id}"
