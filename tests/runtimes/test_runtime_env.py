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
import base64
import copy
import json

import kubernetes.client as k8s_client
import pytest

import mlrun
import mlrun.common.schemas
import mlrun.config
import mlrun.runtimes.pod

NON_ROOT_HOME = "/tmp"


def _new_job() -> mlrun.runtimes.KubejobRuntime:
    return mlrun.new_function("test-home", project="p", kind="job", image="mlrun/mlrun")


@pytest.mark.parametrize(
    "security_context, expected_env",
    [
        (None, {}),
        (k8s_client.V1SecurityContext(), {}),
        (k8s_client.V1SecurityContext(run_as_user=0), {}),
        (k8s_client.V1SecurityContext(run_as_group=12345), {}),
        (k8s_client.V1SecurityContext(run_as_user=0, run_as_non_root=False), {}),
        (
            k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345),
            {"HOME": NON_ROOT_HOME},
        ),
        (k8s_client.V1SecurityContext(run_as_non_root=True), {"HOME": NON_ROOT_HOME}),
        ({"runAsUser": 12345}, {"HOME": NON_ROOT_HOME}),
        ({"runAsNonRoot": True}, {"HOME": NON_ROOT_HOME}),
        ({"runAsUser": 0}, {}),
        ({}, {}),
    ],
)
def test_get_non_root_home_env_by_security_context(security_context, expected_env):
    function = _new_job()
    function.spec.security_context = security_context
    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == expected_env


def test_get_non_root_home_env_unconverted_dict_security_context():
    # security context can bypass the setter and remain a camelCase dictionary
    function = _new_job()
    function.spec._security_context = {"runAsUser": 12345, "runAsGroup": 12345}
    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {
        "HOME": NON_ROOT_HOME
    }


def test_non_root_home_ignores_enrichment_mode():
    # the effective security context, not enrichment mode, determines the default
    assert (
        mlrun.mlconf.function.spec.security_context.enrichment_mode
        == mlrun.common.schemas.SecurityContextEnrichmentModes.disabled.value
    )
    function = _new_job()
    function.with_security_context(
        k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345)
    )
    runtime_env, _ = function._generate_runtime_env()
    assert runtime_env["HOME"] == NON_ROOT_HOME


def test_non_root_home_from_mlconf_default_security_context():
    mlrun.mlconf.function.spec.security_context.default = base64.b64encode(
        json.dumps({"runAsUser": 12345, "runAsGroup": 12345}).encode("utf-8")
    ).decode("utf-8")
    function = _new_job()
    runtime_env, _ = function._generate_runtime_env()
    assert runtime_env["HOME"] == NON_ROOT_HOME


def test_no_home_for_root_function():
    function = _new_job()
    runtime_env, external_source_env = function._generate_runtime_env()
    assert "HOME" not in runtime_env
    assert "HOME" not in external_source_env


@pytest.mark.parametrize("set_home_first", [True, False])
@pytest.mark.parametrize(
    "home_kwargs",
    [
        {"value": "/home/user"},
        {
            "value_from": k8s_client.V1EnvVarSource(
                config_map_key_ref=k8s_client.V1ConfigMapKeySelector(
                    name="cm", key="home"
                )
            )
        },
    ],
)
def test_explicit_home_wins(set_home_first, home_kwargs):
    function = _new_job()
    security_context = k8s_client.V1SecurityContext(
        run_as_user=12345, run_as_group=12345
    )
    if set_home_first:
        function.set_env("HOME", **home_kwargs)
        function.with_security_context(security_context)
    else:
        function.with_security_context(security_context)
        function.set_env("HOME", **home_kwargs)

    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {}

    # serialized environment entries become dictionaries
    function = mlrun.new_function(runtime=function.to_dict())
    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {}
    runtime_env, _ = function._generate_runtime_env()
    assert "HOME" not in runtime_env


def test_non_root_home_disabled_by_config():
    mlrun.mlconf.function.spec.security_context.non_root_home = ""
    function = _new_job()
    function.with_security_context(
        k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345)
    )
    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {}
    runtime_env, _ = function._generate_runtime_env()
    assert "HOME" not in runtime_env


def test_non_root_home_custom_path():
    mlrun.mlconf.function.spec.security_context.non_root_home = "/scratch"
    function = _new_job()
    function.with_security_context(k8s_client.V1SecurityContext(run_as_user=12345))
    assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {
        "HOME": "/scratch"
    }


def test_non_root_home_after_client_spec_sync():
    # match the security context installed by HTTPRunDB.connect()
    mlrun.mlconf.function.spec.security_context = {
        "default": "e30=",
        "enrichment_mode": mlrun.common.schemas.SecurityContextEnrichmentModes.disabled.value,
        "enrichment_group_id": 65534,
        "pipelines": {"kfp_pod_user_unix_id": 5},
    }
    try:
        function = _new_job()
        function.with_security_context(k8s_client.V1SecurityContext(run_as_user=12345))
        assert mlrun.runtimes.pod.get_non_root_home_env(function.spec) == {
            "HOME": NON_ROOT_HOME
        }
    finally:
        # reloading config does not restore keys omitted by the synced context
        mlrun.mlconf.function.spec.security_context = copy.deepcopy(
            mlrun.config.default_config["function"]["spec"]["security_context"]
        )


def test_non_root_home_does_not_mutate_spec():
    function = _new_job()
    function.set_env("OTHER", "value")
    function.with_security_context(
        k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345)
    )
    original = copy.deepcopy(function.to_dict())

    mlrun.runtimes.pod.get_non_root_home_env(function.spec)
    function._generate_runtime_env()

    assert function.to_dict() == original
    assert not function.is_env_exists("HOME")


def test_nuclio_explicit_home_not_overridden_by_generated_env():
    # generated Nuclio environment values otherwise override spec.env
    function = mlrun.new_function("test-home", project="p", kind="nuclio")
    function.with_security_context(
        k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345)
    )
    function.set_env("HOME", "/home/user")
    env_dict, external_source_env_dict = function._get_nuclio_config_spec_env()
    assert env_dict["HOME"] == "/home/user"
    assert "HOME" not in external_source_env_dict


def test_nuclio_non_root_gets_home():
    function = mlrun.new_function("test-home", project="p", kind="nuclio")
    function.with_security_context(
        k8s_client.V1SecurityContext(run_as_user=12345, run_as_group=12345)
    )
    env_dict, _ = function._get_nuclio_config_spec_env()
    assert env_dict["HOME"] == NON_ROOT_HOME
