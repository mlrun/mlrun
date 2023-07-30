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

import unittest.mock
from contextlib import contextmanager
from os import environ
from tempfile import NamedTemporaryFile

import deepdiff
import pytest
import requests_mock as requests_mock_package
import yaml

import mlrun.errors
from mlrun import config as mlconf
from mlrun.common.schemas import SecurityContextEnrichmentModes
from mlrun.db.httpdb import HTTPRunDB

namespace_env_key = f"{mlconf.env_prefix}NAMESPACE"
default_function_pod_resources_env_key = (
    f"{mlconf.env_prefix}DEFAULT_FUNCTION_POD_RESOURCES__"
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
    old = mlconf.config
    mlconf.config = mlconf.Config.from_dict(mlconf.default_config)
    mlconf._loaded = False

    yield mlconf.config

    mlconf.config = old
    mlconf._loaded = False


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
    expected = mlconf.default_config["namespace"]
    assert config.namespace == expected, "namespace changed"


def create_yaml_config(**kw):
    tmp = NamedTemporaryFile(mode="wt", suffix=".yml", delete=False)
    yaml.dump(kw, tmp, default_flow_style=False)
    tmp.flush()
    return tmp.name


def test_file(config):
    ns = "banana"
    config_path = create_yaml_config(namespace=ns)

    with patch_env({mlconf.env_file_key: config_path}):
        mlconf.config.reload()

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
            mlconf.config.reload()

            assert config.v3io_api == expected_v3io_api
            assert config.v3io_framesd == expected_v3io_framesd


def test_env(config):
    ns = "orange"
    with patch_env({namespace_env_key: ns}):
        mlconf.config.reload()

    assert config.namespace == ns, "not populated from env"


def test_env_override(config):
    env_ns = "daffy"
    config_ns = "bugs"

    config_path = create_yaml_config(namespace=config_ns)
    env = {
        mlconf.env_file_key: config_path,
        namespace_env_key: env_ns,
    }

    with patch_env(env):
        mlconf.config.reload()

    assert config.namespace == env_ns, "env did not override"


def test_decode_base64_config_and_load_to_object():
    encoded_dict_attribute = "eyJlbmNvZGVkIjogImF0dHJpYnV0ZSJ9"
    expected_decoded_dict_output = {"encoded": "attribute"}

    encoded_list = "W3sidGVzdCI6IHsidGVzdF9kaWN0IjogMX19LCAxLCAyXQ=="
    expected_decoded_list_output = [{"test": {"test_dict": 1}}, 1, 2]

    # Non-hierarchical attribute loading with passing of expected type
    mlconf.config.encoded_attribute = encoded_dict_attribute
    decoded_output = mlconf.config.decode_base64_config_and_load_to_object(
        "encoded_attribute", dict
    )
    assert type(decoded_output) == dict
    assert decoded_output == expected_decoded_dict_output

    # Hierarchical attribute loading without passing of expected type
    mlconf.config.for_test = {"encoded_attribute": encoded_dict_attribute}
    decoded_output = mlconf.config.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute"
    )
    assert type(decoded_output) == dict
    assert decoded_output == expected_decoded_dict_output

    # Not defined attribute without passing of expected type
    mlconf.config.for_test = {"encoded_attribute": None}
    decoded_output = mlconf.config.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute"
    )
    assert type(decoded_output) == dict
    assert decoded_output == {}

    # Not defined attribute with passing of expected type
    mlconf.config.for_test = {"encoded_attribute": None}
    decoded_output = mlconf.config.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute", list
    )
    assert type(decoded_output) == list
    assert decoded_output == []

    # Non existing attribute loading
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlconf.config.decode_base64_config_and_load_to_object("non_existing_attribute")

    # Attribute defined but not encoded
    mlconf.config.for_test = {"encoded_attribute": "notencoded"}
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError):
        mlconf.config.decode_base64_config_and_load_to_object(
            "for_test.encoded_attribute"
        )

    # list attribute loading
    mlconf.config.for_test = {"encoded_attribute": encoded_list}
    decoded_list_output = mlconf.config.decode_base64_config_and_load_to_object(
        "for_test.encoded_attribute", list
    )
    assert type(decoded_list_output) == list
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
        mlconf.config.reload()

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
        mlconf.config.reload()
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
            mlconf.config.reload()

    # when only gpu request is set
    requests_gpu = "3"
    env = {default_function_pod_resources_request_gpu_env_key: requests_gpu}
    with patch_env(env):
        with pytest.raises(mlrun.errors.MLRunConflictError):
            mlconf.config.reload()

    # when gpu requests and gpu limits are equal
    requests_gpu = "2"
    limits_gpu = "2"
    env = {
        default_function_pod_resources_request_gpu_env_key: requests_gpu,
        default_function_pod_resources_limits_gpu_env_key: limits_gpu,
    }
    with patch_env(env):
        mlconf.config.reload()
    assert config.default_function_pod_resources.requests.gpu == requests_gpu
    assert config.default_function_pod_resources.limits.gpu == limits_gpu

    # None of the requests and limits gpu are set
    env = {}
    with patch_env(env):
        mlconf.config.reload()
    assert config.default_function_pod_resources.requests.gpu is None
    assert config.default_function_pod_resources.limits.gpu is None


old_config_value = None
new_config_value = "blabla"


def test_overriding_config_not_remain_for_next_tests_setter():
    global old_config_value, new_config_value
    old_config_value = mlconf.config.igz_version
    mlconf.config.igz_version = new_config_value
    mlconf.config.httpdb.data_volume = new_config_value


def test_overriding_config_not_remain_for_next_tests_tester():
    global old_config_value
    assert old_config_value == mlconf.config.igz_version
    assert old_config_value == mlconf.config.httpdb.data_volume


def test_resolve_kfp_url():
    mlconf.config.igz_version = ""
    mlconf.config.namespace = ""

    # URL configured - return it
    mlconf.config.kfp_url = "http://ml-pipeline.custom_namespace.svc.cluster.local:8888"
    assert mlconf.config.resolve_kfp_url() == mlconf.config.kfp_url

    # igz configured and less than 3.6.0 without namespace - explode
    mlconf.config.kfp_url = ""
    mlconf.config.igz_version = "1.2.3"
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlconf.config.resolve_kfp_url()

    # igz configured and less than 3.6.0 with namespace - resolve
    mlconf.config.namespace = "default-tenant"
    assert (
        mlconf.config.resolve_kfp_url()
        == "http://ml-pipeline.default-tenant.svc.cluster.local:8888"
    )

    # igz configured and over 3.6.0 - return None (after 3.6.0 kfp_url should be configured)
    mlconf.config.igz_version = "4.0.0"
    assert mlconf.config.resolve_kfp_url() is None

    # nothing configured - return None
    mlconf.config.igz_version = ""
    assert mlconf.config.resolve_kfp_url() is None


def test_get_hub_url():
    # full path configured - no edits
    mlconf.config.hub_url = (
        "https://raw.githubusercontent.com/mlrun/functions/{tag}/{name}/function.yaml"
    )
    assert mlconf.config.get_hub_url() == mlconf.config.hub_url
    # partial path configured + http - edit with tag
    mlconf.config.hub_url = "https://raw.githubusercontent.com/some-fork/functions"
    assert (
        mlconf.config.get_hub_url()
        == f"{mlconf.config.hub_url}/{{tag}}/{{name}}/function.yaml"
    )
    # partial path configured + http - edit without tag
    mlconf.config.hub_url = "v3io://users/admin/mlrun/function-hub"
    assert (
        mlconf.config.get_hub_url() == f"{mlconf.config.hub_url}/{{name}}/function.yaml"
    )


def test_get_parsed_igz_version():
    # open source - version not set
    mlconf.config.igz_version = None
    assert mlconf.config.get_parsed_igz_version() is None

    # 3.2 (or after) - semver compatible
    mlconf.config.igz_version = "3.2.0-b26.20210904121245"
    igz_version = mlconf.config.get_parsed_igz_version()
    assert igz_version.major == 3
    assert igz_version.minor == 2
    assert igz_version.patch == 0

    # 3.0 (or before) - non semver compatible
    mlconf.config.igz_version = "3.0_b154_20210326104738"
    igz_version = mlconf.config.get_parsed_igz_version()
    assert igz_version.major == 3
    assert igz_version.minor == 0
    assert igz_version.patch == 0


def test_get_default_function_node_selector():
    mlconf.config.default_function_node_selector = None
    assert mlconf.config.get_default_function_node_selector() == {}

    mlconf.config.default_function_node_selector = ""
    assert mlconf.config.get_default_function_node_selector() == {}

    mlconf.config.default_function_node_selector = "e30="
    assert mlconf.config.get_default_function_node_selector() == {}

    mlconf.config.default_function_node_selector = "bnVsbA=="
    assert mlconf.config.get_default_function_node_selector() == {}


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
    assert "" == mlconf.config.remote_host
    mlconf.config.dbpath = api_url
    assert remote_host == mlconf.config.remote_host


def test_verify_security_context_enrichment_mode_is_allowed_success():
    mlconf.config.verify_security_context_enrichment_mode_is_allowed()

    mlconf.config.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    mlconf.config.igz_version = "3.5.1-b1"
    mlconf.config.verify_security_context_enrichment_mode_is_allowed()

    mlconf.config.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    mlconf.config.igz_version = "3.6.0-b1"
    mlconf.config.verify_security_context_enrichment_mode_is_allowed()


def test_verify_security_context_enrichment_mode_is_allowed_failure():
    igz_version = "3.5.0-b1"
    mlconf.config.igz_version = igz_version
    mlconf.config.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlconf.config.verify_security_context_enrichment_mode_is_allowed()
    assert (
        f"Security context enrichment mode enabled (override/retain) "
        f"is not allowed for iguazio version: {igz_version} < 3.5.1" in str(exc.value)
    )

    igz_version = "3.4.2-b1"
    mlconf.config.igz_version = igz_version
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlconf.config.verify_security_context_enrichment_mode_is_allowed()
    assert (
        f"Security context enrichment mode enabled (override/retain) "
        f"is not allowed for iguazio version: {igz_version} < 3.5.1" in str(exc.value)
    )

    mlconf.config.igz_version = ""
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        mlconf.config.verify_security_context_enrichment_mode_is_allowed()
    assert (
        "Unable to determine if security context enrichment mode is allowed. Missing iguazio version"
        in str(exc.value)
    )
