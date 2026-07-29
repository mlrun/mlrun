# Copyright 2026 Iguazio
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

import pytest

import mlrun
import mlrun.common.schemas
import mlrun.errors
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestBuildahParity(tests.system.base.TestMLRunSystem):
    """ML-12889 system tier: rootless build under PodSecurity + end-to-end push, and the
    pod-Failed -> function-error invariant, against a real Iguazio system.

    Which builder backend actually runs is a server-side setting
    (``config.httpdb.builder.builder_backend``), not something a system test can flip per run -
    see the dedicated buildah-test-{azure,gcp} clusters kept patched with ``builder_backend=buildah``
    for this purpose. Parity is established by running this same suite twice: once against a
    Buildah-configured lab, once against a Kaniko-configured lab - both runs must pass with the
    same assertions.
    """

    project_name = "buildah-parity-system-test"
    image: str = "mlrun/mlrun"

    def test_build_and_push_succeeds_under_pod_security(self):
        # a real end-to-end build + push under whatever PodSecurity policy the target namespace
        # enforces - Buildah's caps-rootless model (BuildahBackend) must build and push under the
        # same admission constraints Kaniko already runs under, and the resulting image must run.
        code_path = str(self.assets_path / "kubejob_function.py")
        function = mlrun.code_to_function(
            name="buildah-parity-rootless",
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        function.build_config(
            base_image=self.image, commands=["echo buildah-parity-rootless"]
        )
        function.deploy()
        run = function.run()
        assert run.state() == "completed", f"Unexpected state: {run.state()}"

    def test_failed_build_drives_function_to_error_state(self):
        # the failure-contract invariant (ML-12889): a build that fails inside the pod must
        # leave the pod Failed -> function state error, and deploy() must raise - independent
        # of which builder backend produced the failure.
        code_path = str(self.assets_path / "kubejob_function.py")
        function = mlrun.code_to_function(
            name="buildah-parity-failed-build",
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        function.build_config(base_image=self.image, commands=["exit 1"])
        with pytest.raises(mlrun.errors.MLRunRuntimeError):
            function.deploy()
        assert function.status.state == mlrun.common.schemas.FunctionState.error
