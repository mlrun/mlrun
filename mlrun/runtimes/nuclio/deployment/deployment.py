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
from mlrun.common.schemas import AuthInfo
from mlrun.runtimes import RemoteRuntime


class DeploymentRuntime(RemoteRuntime):
    kind = "deployment"

    def __init__(self, spec=None, metadata=None):
        super().__init__(metadata, spec)
        # TODO: verify min_nuclio_versions
        self.internal_app_port = 8080

    def set_internal_app_port(self, port):
        # TODO: move to spec?
        self.internal_app_port = port

    def deploy(
        self,
        dashboard="",
        project="",
        tag="",
        verbose=False,
        auth_info: AuthInfo = None,
        builder_env: dict = None,
        force_build: bool = False,
    ):
        self._init_sidecar()
        super().deploy(
            dashboard,
            project,
            tag,
            verbose,
            auth_info,
            builder_env,
            force_build,
        )

    def _init_sidecar(self):
        # TODO: avoid double init
        self.with_sidecar(
            f"{self.metadata.name}-sidecar", self.spec.image, self.internal_app_port
        )
        self.set_env("SERVING_PORT", self.internal_app_port)
        self.spec.image = ""
        # TODO: move other fields
