# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import kubernetes.client as k8s_client

import mlrun
import mlrun.runtimes.pod
import mlrun_pipelines.ops as ops


class _FakeTask:
    def __init__(self):
        self.env = {}

    def set_env_variable(self, name, value):
        self.env[name] = value


def _non_root_job():
    function = mlrun.new_function("test-home", project="p", kind="job")
    function.with_security_context(
        k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345)
    )
    return function


def test_sync_environment_variables_does_not_sync_generated_home():
    """ML-13134: a KFP v2 step of a non-local function only runs the `mlrun run` client, the user code runs
    in the function pod built by the server, which gets the generated non-root HOME default. The step itself
    gets no MLRun-managed security context, so syncing the default to it is not needed."""
    function = _non_root_job()
    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {"HOME": "/tmp"}

    task = _FakeTask()
    ops.sync_environment_variables(function, task)

    assert "HOME" not in task.env


def test_local_function_step_has_no_non_root_home():
    """ML-13134: a local function's code runs inside the KFP v2 step, but local runtimes have no security
    context and MLRun sets none on v2 steps, so the non-root HOME default never applies to them."""
    function = mlrun.new_function("test-home", project="p", kind="local")
    assert not isinstance(function, mlrun.runtimes.pod.KubeResource)
    runtime_env, _ = function._generate_runtime_env()
    assert "HOME" not in runtime_env


def test_sync_environment_variables_syncs_explicit_home():
    function = _non_root_job()
    function.set_env("HOME", "/home/user")

    task = _FakeTask()
    ops.sync_environment_variables(function, task)

    assert task.env["HOME"] == "/home/user"
