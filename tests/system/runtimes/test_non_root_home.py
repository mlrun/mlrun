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
import kubernetes.client as k8s_client
import pytest

import mlrun
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import tests.system.base

NON_ROOT_UID = 12345


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNonRootHome(tests.system.base.TestMLRunSystem):
    project_name = "non-root-home-system-test"
    reuse_project_across_tests = True

    image: str = "mlrun/mlrun"

    @pytest.mark.collect_pod_logs
    def test_kubejob_non_root_home_is_writable(self):
        function = mlrun.code_to_function(
            name="non-root-home-job",
            kind="job",
            project=self.project_name,
            filename=str(self.assets_path / "non_root_home.py"),
            image=self.image,
        )
        self._set_non_root_security_context(function)

        run = function.run(handler="probe")

        assert run.status.state == mlrun.common.runtimes.constants.RunStates.completed
        self._assert_probe_result(run.outputs)

    @pytest.mark.collect_pod_logs
    def test_nuclio_non_root_home_is_writable(self):
        function = mlrun.code_to_function(
            name="non-root-home-nuclio",
            kind="nuclio",
            project=self.project_name,
            filename=str(self.assets_path / "non_root_home.py"),
            image=self.image,
            handler="handler",
        )
        self._set_non_root_security_context(function)
        function.deploy()

        result = function.invoke("/")

        self._assert_probe_result(result)

    def _set_non_root_security_context(self, function):
        # explicit context is allowed only when enrichment is disabled; otherwise
        # the server supplies the UID
        enrichment_mode = mlrun.mlconf.function.spec.security_context.enrichment_mode
        self._logger.info(
            "Security context enrichment mode", enrichment_mode=enrichment_mode
        )
        if (
            enrichment_mode
            == mlrun.common.schemas.SecurityContextEnrichmentModes.disabled.value
        ):
            function.with_security_context(
                k8s_client.V1SecurityContext(
                    run_as_user=NON_ROOT_UID, run_as_group=NON_ROOT_UID
                )
            )

    def _assert_probe_result(self, result: dict):
        # enrichment mode controls the effective UID, so don't assert a specific UID
        self._logger.info(
            "Non-root HOME probe result", uid=result["uid"], home=result["home"]
        )
        assert result["uid"] != 0
        assert result["home"] != "/"
        assert result["writable"] is True
