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
import mlrun.runtimes


class DeploymentRuntime(mlrun.runtimes.KubeResource):
    kind = "deployment"

    def __init__(self, spec=None, metadata=None):
        super().__init__(metadata, spec)
        self._reverse_proxy = None

    def _init_reverse_proxy(self):
        reverse_proxy = mlrun.new_function(
            name=self.metadata.name,
            project=self.metadata.project,
            kind=mlrun.runtimes.RuntimeKinds.remote,
            command="./something/here",
            image="mlrun/mlrun",
        )
        self._reverse_proxy = reverse_proxy
