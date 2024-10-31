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

import os
import pathlib
import subprocess
import sys
import unittest.mock
from contextlib import contextmanager
from os import environ
from tempfile import NamedTemporaryFile

import deepdiff
import dotenv
import pytest
import requests_mock as requests_mock_package
import yaml

import mlrun
import mlrun.errors
import mlrun.projects.project
from mlrun.common.schemas import SecurityContextEnrichmentModes
from mlrun.db.httpdb import HTTPRunDB
from tests.conftest import out_path

assets_path = pathlib.Path(__file__).parent / "assets"

namespace_env_key = f"{mlrun.config.env_prefix}NAMESPACE"
default_function_pod_resources_env_key = (
    f"{mlrun.config.env_prefix}DEFAULT_FUNCTION_POD_RESOURCES__"
)
default_function_pod_resources_request_gpu_env_key = (
    f"{default_function_pod_resources_env_key}REQUESTS__GPU"
)
default_function_pod_resources_limits_gpu_env_key = (
    f"{default_function_pod_resources_env_key}LIMITS__GPU"
)
default_function_pod_resources_request_cpu_env_key = (
    f"{default_function_pod_resources_env_key}REQUESTS__CPU"
)
default_function_pod_resources_request_memory_env_key = (
    f"{default_function_pod_resources_env_key}REQUESTS__MEMORY"
)
default_function_pod_resources_limits_cpu_env_key = (
    f"{default_function_pod_resources_env_key}LIMITS__CPU"
)


@pytest.fixture
def config():
    old = mlrun.config.config
    mlrun.config.config = mlrun.config.Config.from_dict(mlrun.config.default_config)
    mlrun.config._loaded = False

    yield mlrun.config.config

    mlrun.config.config = old
    mlrun.config._loaded = False


@contextmanager
def patch_env(kw):
    old, new = [], []
    for key, value in kw.items():
        old_val = environ.get(key)
        if old_val:
            old.append((key, old_val))
        else:
            new.append(key)
        environ[key] = value

    yield

    for key, value in old:
        environ[key] = value
    for key in new:
        del environ[key]


def test_nothing(config):
    expected = mlrun.config.default_config["namespace"]
    assert config.namespace == expected, "namespace changed"


def create_yaml_config(**kw):
    tmp = NamedTemporaryFile(mode="wt", suffix=".yml", delete=False)
    yaml.safe_dump(kw, tmp, default_flow_style=False)
    tmp.flush()
    return tmp.name


def test_file(config):
    ns = "banana"
    config_path = create_yaml_config(namespace=ns)

    with patch_env({mlrun.config.env_file_key: config_path}):
        mlrun.mlconf.reload()

    assert config.namespace == ns, "not populated from file"


@pytest.mark.parametrize(
    "mlrun_dbpath,v3io_api,v3io_framesd,expected_v3io_api,expected_v3io_framesd",
    (
        (
            "http://mlrun-api:8080",
            "",
            "",
            "http://v3io-webapi:8081",
            "http://framesd:8080",
        ),
        (
            "http://mlrun-api:8080",
            "http://v3io-webapi:8081",
            "",
            "http://v3io-webapi:8081",
            "http://framesd:8080",
        ),
        (
            "https://mlrun-api.default-tenant.app.somedev.cluster.amzn.com",
            "",
            "",
            "https://webapi.default-tenant.app.somedev.cluster.amzn.com",
            "https://framesd.default-tenant.app.somedev.cluster.amzn.com",
        ),
        (
            "https://mlrun-api.default-tenant.app.somedev.cluster.amzn.com",
            "https://webapi.default-tenant.app.somedev.cluster.amzn.com",
            "",
            "https://webapi.default-tenant.app.somedev.cluster.amzn.com",
            "https://framesd.default-tenant.app.somedev.cluster.amzn.com",
        ),
        (
            "https://mlrun-api.default-tenant.app.somedev.cluster.amzn.com",
            "",
            "https://framesd.default-tenant.app.somedev.cluster.amzn.com",
            "https://webapi.default-tenant.app.somedev.cluster.amzn.com",
            "https://framesd.default-tenant.app.somedev.cluster.amzn.com",
        ),
    ),
)
def test_v3io_api_and_framesd_enrichment_from_dbpath(
    config,
    mlrun_dbpath,
    v3io_api,
    v3io_framesd,
    expected_v3io_api,
    expected_v3io_framesd,
    monkeypatch,
):
    with unittest.mock.patch.object(mlrun.db, "get_run_db", return_value=None):
        env = {
            "MLRUN_DBPATH": mlrun_dbpath,
            "V3IO_API": v3io_api,
            "V3IO_FRAMESD": v3io_framesd,
        }
        with patch_env(env):
            mlrun.mlconf.reload()

            assert config.v3io_api == expected_v3io_api
            assert config.v3io_framesd == expected_v3io_framesd


def test_env(config):
    ns = "orange"
    with patch_env({namespace_env_key: ns}):
        mlrun.mlconf.reload()

    assert config.namespace == ns, "not populated from env"


def test_env_override(config):
    env_ns = "daffy"
    config_ns = "bugs"

    config_path = create_yaml_config(namespace=config_ns)
    env = {
        mlrun.config.env_file_key: config_path,
        namespace_env_key: env_ns,
    }

    with patch_env(env):
        mlrun.mlconf.reload()

    assert config.namespace == env_ns, "env did not override"


def test_decode_base64_config_and_load_to_object():
    encoded_dict_attribute = "eyJlbmNvZGVkIjogImF0dHJpYnV0ZSJ9"
    expected_decoded_dict_output = {"encoded": "attribute"}

    encoded_list = "W3sidGVzdCI6IHsidGVzdF9kaWN0IjogMX19LCAxLCAyXQ=="
    expected_decoded_list_output = [{"test": {"test_dict": 1}}, 1, 2]

    # Non-hierarchical attribute loading with passing of expected type
    mlrun.mlconf.encoded_attribute = encoded_dict_attribute
    decoded_output = mlrun.mlconf.decode_base64_config_and_load_to_object(
        "encoded_attribute", dict
    )
    assert isinstance(decoded_output, dict)
    assert decoded_output == expected_decoded_dict_output

    # Hierarchical attribute loading without passing of expected type
    mlrun.mlconf.for_test = {"encoded_attribute": encoded_dict_attribute}
    decoded_output = mlrun.mlconf.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute"
    )
    assert isinstance(decoded_output, dict)
    assert decoded_output == expected_decoded_dict_output

    # Not defined attribute without passing of expected type
    mlrun.mlconf.for_test = {"encoded_attribute": None}
    decoded_output = mlrun.mlconf.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute"
    )
    assert isinstance(decoded_output, dict)
    assert decoded_output == {}

    # Not defined attribute with passing of expected type
    mlrun.mlconf.for_test = {"encoded_attribute": None}
    decoded_output = mlrun.mlconf.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute", list
    )
    assert isinstance(decoded_output, list)
    assert decoded_output == []

    # Non existing attribute loading
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlrun.mlconf.decode_base64_config_and_load_to_object("non_existing_attribute")

    # Attribute defined but not encoded
    mlrun.mlconf.for_test = {"encoded_attribute": "notencoded"}
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError):
        mlrun.mlconf.decode_base64_config_and_load_to_object(
            "for_test.encoded_attribute"
        )

    # list attribute loading
    mlrun.mlconf.for_test = {"encoded_attribute": encoded_list}
    decoded_list_output = mlrun.mlconf.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute", list
    )
    assert isinstance(decoded_list_output, list)
    assert decoded_list_output == expected_decoded_list_output


def test_with_gpu_option_get_default_function_pod_resources(config):
    requests_cpu = "30mi"
    limits_cpu = "4"
    requests_memory = "1M"
    requests_gpu = "2"
    limits_gpu = "2"
    env = {
        default_function_pod_resources_request_cpu_env_key: requests_cpu,
        default_function_pod_resources_limits_cpu_env_key: limits_cpu,
        default_function_pod_resources_request_memory_env_key: requests_memory,
        default_function_pod_resources_request_gpu_env_key: requests_gpu,
        default_function_pod_resources_limits_gpu_env_key: limits_gpu,
    }
    with patch_env(env):
        mlrun.mlconf.reload()

        for test_case in [
            {
                "with_gpu_requests": True,
                "with_gpu_limits": True,
                "expected_resources": {
                    "requests": {
                        "cpu": requests_cpu,
                        "memory": requests_memory,
                        "nvidia.com/gpu": requests_gpu,
                    },
                    "limits": {
                        "cpu": limits_cpu,
                        "memory": None,
                        "nvidia.com/gpu": limits_gpu,
                    },
                },
            },
            {
                "with_gpu_requests": False,
                "with_gpu_limits": True,
                "expected_resources": {
                    "requests": {"cpu": requests_cpu, "memory": requests_memory},
                    "limits": {
                        "cpu": limits_cpu,
                        "memory": None,
                        "nvidia.com/gpu": limits_gpu,
                    },
                },
            },
            {
                "with_gpu_requests": True,
                "with_gpu_limits": False,
                "expected_resources": {
                    "requests": {
                        "cpu": requests_cpu,
                        "memory": requests_memory,
                        "nvidia.com/gpu": requests_gpu,
                    },
                    "limits": {
                        "cpu": limits_cpu,
                        "memory": None,
                    },
                },
            },
            {
                "with_gpu_requests": False,
                "with_gpu_limits": False,
                "expected_resources": {
                    "requests": {"cpu": requests_cpu, "memory": requests_memory},
                    "limits": {"cpu": limits_cpu, "memory": None},
                },
            },
        ]:
            with_requests_gpu = test_case.get("with_gpu_requests")
            with_gpu_limits = test_case.get("with_gpu_limits")
            resources = config.get_default_function_pod_resources(
                with_requests_gpu, with_gpu_limits
            )
            assert (
                deepdiff.DeepDiff(
                    resources,
                    test_case.get("expected_resources"),
                    ignore_order=True,
                )
                == {}
            )


def test_get_default_function_pod_requirement_resources(config):
    requests_gpu = "2"
    limits_gpu = "2"
    env = {
        default_function_pod_resources_request_gpu_env_key: requests_gpu,
        default_function_pod_resources_limits_gpu_env_key: limits_gpu,
    }
    expected_resources_without_gpu = {
        "requests": {"cpu": None, "memory": None},
        "limits": {"cpu": None, "memory": None},
    }
    expected_resources_with_gpu = {
        "requests": {"cpu": None, "memory": None, "nvidia.com/gpu": requests_gpu},
        "limits": {"cpu": None, "memory": None, "nvidia.com/gpu": limits_gpu},
    }
    with patch_env(env):
        mlrun.mlconf.reload()
        requests = config.get_default_function_pod_requirement_resources(
            "requests", with_gpu=True
        )
        assert (
            deepdiff.DeepDiff(
                requests,
                expected_resources_with_gpu["requests"],
                ignore_order=True,
            )
            == {}
        )
        limits = config.get_default_function_pod_requirement_resources(
            "limits", with_gpu=True
        )
        assert (
            deepdiff.DeepDiff(
                limits,
                expected_resources_with_gpu["limits"],
                ignore_order=True,
            )
            == {}
        )
        requests = config.get_default_function_pod_requirement_resources(
            "requests", with_gpu=False
        )
        assert (
            deepdiff.DeepDiff(
                requests,
                expected_resources_without_gpu["requests"],
                ignore_order=True,
            )
            == {}
        )
        limits = config.get_default_function_pod_requirement_resources(
            "limits", with_gpu=False
        )
        assert (
            deepdiff.DeepDiff(
                limits,
                expected_resources_without_gpu["limits"],
                ignore_order=True,
            )
            == {}
        )


def test_gpu_validation(config):
    # when gpu requests and gpu limits are not equal
    requests_gpu = "3"
    limits_gpu = "2"
    env = {
        default_function_pod_resources_request_gpu_env_key: requests_gpu,
        default_function_pod_resources_limits_gpu_env_key: limits_gpu,
    }
    with patch_env(env):
        with pytest.raises(mlrun.errors.MLRunConflictError):
            mlrun.mlconf.reload()

    # when only gpu request is set
    requests_gpu = "3"
    env = {default_function_pod_resources_request_gpu_env_key: requests_gpu}
    with patch_env(env):
        with pytest.raises(mlrun.errors.MLRunConflictError):
            mlrun.mlconf.reload()

    # when gpu requests and gpu limits are equal
    requests_gpu = "2"
    limits_gpu = "2"
    env = {
        default_function_pod_resources_request_gpu_env_key: requests_gpu,
        default_function_pod_resources_limits_gpu_env_key: limits_gpu,
    }
    with patch_env(env):
        mlrun.mlconf.reload()
    assert config.default_function_pod_resources.requests.gpu == requests_gpu
    assert config.default_function_pod_resources.limits.gpu == limits_gpu

    # None of the requests and limits gpu are set
    env = {}
    with patch_env(env):
        mlrun.mlconf.reload()
    assert config.default_function_pod_resources.requests.gpu is None
    assert config.default_function_pod_resources.limits.gpu is None


##################################
# Unit Test Memory Sharing Tests #
##################################
#
# These tests are no longer relevant, since we run unit tests with pytest-fork.
# pytest-fork creates a new process for each test, so the memory space is not shared between tests.
# Each test receives its own memory page, so changes made in one test do not affect the memory of another test.
#
# old_config_value = None
# new_config_value = "blabla"
#
#
# def test_overriding_config_not_remain_for_next_tests_setter():
#     global old_config_value, new_config_value
#     old_config_value = mlrun.mlconf.igz_version
#     mlrun.mlconf.igz_version = new_config_value
#     mlrun.mlconf.httpdb.data_volume = new_config_value
#
#
# def test_overriding_config_not_remain_for_next_tests_tester():
#     global old_config_value
#     assert old_config_value == mlrun.mlconf.igz_version
#     assert old_config_value == mlrun.mlconf.httpdb.data_volume
#####################################
# EO Unit Test Memory Sharing Tests #
#####################################


def test_get_parsed_igz_version():
    # open source - version not set
    mlrun.mlconf.igz_version = None
    assert mlrun.mlconf.get_parsed_igz_version() is None

    # 3.2 (or after) - semver compatible
    mlrun.mlconf.igz_version = "3.2.0-b26.20210904121245"
    igz_version = mlrun.mlconf.get_parsed_igz_version()
    assert igz_version.major == 3
    assert igz_version.minor == 2
    assert igz_version.patch == 0

    # 3.0 (or before) - non semver compatible
    mlrun.mlconf.igz_version = "3.0_b154_20210326104738"
    igz_version = mlrun.mlconf.get_parsed_igz_version()
    assert igz_version.major == 3
    assert igz_version.minor == 0
    assert igz_version.patch == 0


def test_get_default_function_node_selector():
    mlrun.mlconf.default_function_node_selector = None
    assert mlrun.mlconf.get_default_function_node_selector() == {}

    mlrun.mlconf.default_function_node_selector = ""
    assert mlrun.mlconf.get_default_function_node_selector() == {}

    mlrun.mlconf.default_function_node_selector = "e30="
    assert mlrun.mlconf.get_default_function_node_selector() == {}

    mlrun.mlconf.default_function_node_selector = "bnVsbA=="
    assert mlrun.mlconf.get_default_function_node_selector() == {}


def test_setting_dbpath_trigger_connect(requests_mock: requests_mock_package.Mocker):
    api_url = "http://mlrun-api-url:8080"
    remote_host = "some-namespace"
    response_body = {
        "version": "some-version",
        "remote_host": remote_host,
    }
    requests_mock.get(
        f"{api_url}/{HTTPRunDB.get_api_path_prefix()}/client-spec",
        json=response_body,
    )
    assert "" == mlrun.mlconf.remote_host
    mlrun.mlconf.dbpath = api_url
    assert remote_host == mlrun.mlconf.remote_host


def test_verify_security_context_enrichment_mode_is_allowed_success():
    mlrun.mlconf.verify_security_context_enrichment_mode_is_allowed()

    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    mlrun.mlconf.igz_version = "3.5.1-b1"
    mlrun.mlconf.verify_security_context_enrichment_mode_is_allowed()

    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    mlrun.mlconf.igz_version = "3.6.0-b1"
    mlrun.mlconf.verify_security_context_enrichment_mode_is_allowed()


def test_verify_security_context_enrichment_mode_is_allowed_failure():
    igz_version = "3.5.0-b1"
    mlrun.mlconf.igz_version = igz_version
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlrun.mlconf.verify_security_context_enrichment_mode_is_allowed()
    assert (
        f"Security context enrichment mode enabled (override/retain) "
        f"is not allowed for iguazio version: {igz_version} < 3.5.1" in str(exc.value)
    )

    igz_version = "3.4.2-b1"
    mlrun.mlconf.igz_version = igz_version
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlrun.mlconf.verify_security_context_enrichment_mode_is_allowed()
    assert (
        f"Security context enrichment mode enabled (override/retain) "
        f"is not allowed for iguazio version: {igz_version} < 3.5.1" in str(exc.value)
    )

    mlrun.mlconf.igz_version = ""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlrun.mlconf.verify_security_context_enrichment_mode_is_allowed()
    assert (
        "Unable to determine if security context enrichment mode is allowed. Missing iguazio version"
        in str(exc.value)
    )


def test_set_environment_cred():
    old_key = os.environ.get("V3IO_ACCESS_KEY")
    old_user = os.environ.get("V3IO_USERNAME")
    artifact_path = mlrun.mlconf.artifact_path or "./"
    mlrun.set_environment(access_key="xyz", username="joe", artifact_path=artifact_path)
    assert os.environ["V3IO_ACCESS_KEY"] == "xyz"
    assert os.environ["V3IO_USERNAME"] == "joe"

    if old_key:
        os.environ["V3IO_ACCESS_KEY"] = old_key
    else:
        del os.environ["V3IO_ACCESS_KEY"]
    if old_user:
        os.environ["V3IO_USERNAME"] = old_user
    else:
        del os.environ["V3IO_USERNAME"]


def test_env_from_file():
    env_path = str(assets_path / "envfile")
    env_dict = mlrun.set_env_from_file(env_path, return_dict=True)
    assert env_dict == {"ENV_ARG1": "123", "ENV_ARG2": "abc", "MLRUN_KFP_TTL": "12345"}
    assert mlrun.mlconf.kfp_ttl == 12345
    for key, value in env_dict.items():
        assert os.environ[key] == value
    for key in env_dict.keys():
        del os.environ[key]

    # test setting env_file using set_environment()
    artifact_path = mlrun.mlconf.artifact_path or "./"
    mlrun.set_environment(env_file=env_path, artifact_path=artifact_path)
    for key, value in env_dict.items():
        assert os.environ[key] == value
    for key in env_dict.keys():
        del os.environ[key]


def test_mock_functions():
    mock_nuclio_config = mlrun.mlconf.mock_nuclio_deployment
    local_config = mlrun.mlconf.force_run_local

    # test setting env_file using set_environment()
    artifact_path = mlrun.mlconf.artifact_path or "./"
    mlrun.set_environment(mock_functions=True, artifact_path=artifact_path)
    assert mlrun.mlconf.mock_nuclio_deployment == "1"
    assert mlrun.mlconf.force_run_local == "1"

    mlrun.set_environment(mock_functions="auto", artifact_path=artifact_path)
    assert mlrun.mlconf.mock_nuclio_deployment == "auto"
    assert mlrun.mlconf.force_run_local == "auto"

    mlrun.mlconf.mock_nuclio_deployment = mock_nuclio_config
    mlrun.mlconf.force_run_local = local_config


def test_bad_env_files():
    bad_envs = ["badenv1"]

    for env in bad_envs:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            mlrun.set_env_from_file(str(assets_path / env))


def test_env_file_does_not_exist():
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlrun.set_env_from_file("some-nonexistent-path")


def test_auto_load_env_file():
    os.environ["MLRUN_ENV_FILE"] = str(assets_path / "envfile")
    mlrun.mlconf.reload()
    assert mlrun.mlconf.kfp_ttl == 12345
    expected = {"ENV_ARG1": "123", "ENV_ARG2": "abc", "MLRUN_KFP_TTL": "12345"}
    count = 0
    for key in os.environ.keys():
        if key in expected:
            assert os.environ[key] == expected[key]
            count += 1
    assert count == 3
    for key in expected.keys():
        del os.environ[key]


def test_deduct_v3io_paths():
    cluster = ".default-tenant.app.xxx.iguazio-cd1.com"
    conf = mlrun.config.read_env({"MLRUN_DBPATH": "https://mlrun-api" + cluster})
    assert conf["v3io_api"] == "https://webapi" + cluster
    assert conf["v3io_framesd"] == "https://framesd" + cluster


def test_set_config():
    env_path = f"{out_path}/env/myenv.env"
    api = "http://localhost:8080"
    pathlib.Path(env_path).parent.mkdir(parents=True, exist_ok=True)
    _exec_mlrun(
        f"config set -f {env_path} -a {api} -u joe -k mykey -p /c/y -e XXX=myvar"
    )

    expected = {
        "MLRUN_DBPATH": api,
        "MLRUN_ARTIFACT_PATH": "/c/y",
        "V3IO_USERNAME": "joe",
        "V3IO_ACCESS_KEY": "mykey",
        "XXX": "myvar",
    }
    env_vars = dotenv.dotenv_values(env_path)
    assert len(env_vars) == 5
    for key, val in expected.items():
        assert env_vars[key] == val


def test_set_and_load_default_config():
    env_path = os.path.expanduser(mlrun.config.default_env_file)
    env_body = None
    if os.path.isfile(env_path):
        with open(env_path) as fp:
            env_body = fp.read()

    # set two config (mlrun and custom vars) and read/verify the default .env file
    _exec_mlrun("config set -e YYYY=myvar -e MLRUN_KFP_TTL=12345")
    env_vars = dotenv.dotenv_values(env_path)
    assert env_vars["YYYY"] == "myvar"
    assert env_vars["MLRUN_KFP_TTL"] == "12345"

    # verify the new env impact mlrun config
    mlrun.mlconf.reload()
    assert mlrun.mlconf.kfp_ttl == 12345

    _exec_mlrun("config clear")
    assert not os.path.isfile(env_path), "config file was not deleted"

    # write back old content and del env vars
    if env_body:
        with open(env_path, "w") as fp:
            fp.write(env_body)
    print(os.environ)
    if "YYYY" in os.environ:
        del os.environ["YYYY"]
    if "MLRUN_KFP_TTL" in os.environ:
        del os.environ["MLRUN_KFP_TTL"]


def _exec_mlrun(cmd, cwd=None):
    cmd = [sys.executable, "-m", "mlrun"] + cmd.split()
    out = subprocess.run(cmd, capture_output=True, cwd=cwd)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"), file=sys.stderr)
        print(out.stdout.decode("utf-8"), file=sys.stderr)
        raise Exception(out.stderr.decode("utf-8"))
    return out.stdout.decode("utf-8")
