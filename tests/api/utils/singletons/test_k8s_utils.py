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
from contextlib import nullcontext as does_not_raise

import pytest
from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.runtimes
import server.api.runtime_handlers.mpijob
import server.api.utils.singletons.k8s


@pytest.fixture
def k8s_helper():
    k8s_helper = server.api.utils.singletons.k8s.K8sHelper(
        "test-namespace", silent=True
    )
    k8s_helper.v1api = unittest.mock.MagicMock()
    k8s_helper.crdapi = unittest.mock.MagicMock()
    return k8s_helper


@pytest.mark.parametrize(
    "run_type,mpi_version,extra_selector",
    [
        ("job", "", ""),
        ("spark", "", "spark-role=driver"),
        (
            "mpijob",
            "v1",
            f"{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher",
        ),
        (
            "mpijob",
            "v1alpha1",
            f"{mlrun_constants.MLRunInternalLabels.mpi_role_type}=launcher",
        ),
    ],
)
def test_get_logger_pods_label_selector(
    k8s_helper, monkeypatch, run_type, mpi_version, extra_selector
):
    monkeypatch.setattr(
        server.api.runtime_handlers.mpijob,
        "cached_mpijob_crd_version",
        mpi_version or mlrun.common.runtimes.constants.MPIJobCRDVersions.default(),
    )
    uid = "test-uid"
    project = "test-project"
    selector = (
        f"{mlrun_constants.MLRunInternalLabels.mlrun_class},"
        f"{mlrun_constants.MLRunInternalLabels.project}={project},"
        f"{mlrun_constants.MLRunInternalLabels.uid}={uid}"
    )
    if extra_selector:
        selector += f",{extra_selector}"

    k8s_helper.list_pods = unittest.mock.MagicMock()

    k8s_helper.get_logger_pods(project, uid, run_type)
    k8s_helper.list_pods.assert_called_once_with(
        k8s_helper.namespace, selector=selector
    )


@pytest.mark.parametrize(
    "secret_data,secrets,expected_data,expected_result",
    [
        # we want to ensure that if the data is None, the function doesn't raise an exception
        (None, {}, {}, None),
        (None, None, {}, None),
        # regular case
        (
            {"a": "b"},
            {"a": "c"},
            {"a": "Yw=="},
            mlrun.common.schemas.SecretEventActions.updated,
        ),
        (
            None,
            {"a": "b"},
            {"a": "Yg=="},
            mlrun.common.schemas.SecretEventActions.updated,
        ),
    ],
)
def test_store_secret(k8s_helper, secret_data, secrets, expected_data, expected_result):
    k8s_helper.v1api.read_namespaced_secret.return_value = unittest.mock.MagicMock(
        data=secret_data
    )
    k8s_helper.v1api.replace_namespaced_secret = unittest.mock.MagicMock()
    result = k8s_helper.store_secrets("my-secret", secrets)
    assert result == expected_result
    if expected_data:
        data = k8s_helper.v1api.replace_namespaced_secret.call_args.args[2].data
        assert data == expected_data


@pytest.mark.parametrize(
    "k8s_secret_data, secrets_data, expected_action, expected_secret_data",
    [
        (
            {"key1": "value1", "key2": "value2"},
            [],
            None,
            {"key1": "value1", "key2": "value2"},
        ),
        (
            {"key1": "value1", "key2": "value2"},
            None,  # delete all secrets
            mlrun.common.schemas.SecretEventActions.deleted,
            {},
        ),
        (
            {"key1": "value1", "key2": "value2"},
            ["key3"],
            None,
            {"key1": "value1", "key2": "value2"},
        ),
        (None, ["key1"], mlrun.common.schemas.SecretEventActions.deleted, {}),
        ({}, ["key1"], mlrun.common.schemas.SecretEventActions.deleted, {}),
        (
            {"key1": "value1"},
            ["key1"],
            mlrun.common.schemas.SecretEventActions.deleted,
            {},
        ),
        (
            {"key1": "value1", "key2": "value2"},
            ["key1"],
            mlrun.common.schemas.SecretEventActions.updated,
            {"key2": "value2"},
        ),
    ],
)
def test_delete_secrets(
    k8s_helper, k8s_secret_data, secrets_data, expected_action, expected_secret_data
):
    k8s_secret_mock = unittest.mock.MagicMock(data=k8s_secret_data)
    k8s_helper.v1api.read_namespaced_secret.return_value = k8s_secret_mock

    result = k8s_helper.delete_secrets("my-secret", secrets_data)
    assert result == expected_action

    k8s_helper.v1api.read_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace
    )

    if expected_action == mlrun.common.schemas.SecretEventActions.updated:
        data = k8s_helper.v1api.replace_namespaced_secret.call_args.args[2].data
        assert data == expected_secret_data


@pytest.mark.parametrize(
    "side_effect, expectation, expected_result",
    [
        (
            [
                ApiException(status=410),
                ApiException(status=410),
                k8s_client.V1PodList(
                    items=[],
                    metadata=k8s_client.V1ListMeta(),
                ),
            ],
            does_not_raise(),
            [],
        ),
        (
            [
                ApiException(status=410),
                ApiException(status=410),
                ApiException(status=410),
                ApiException(status=410),
            ],
            pytest.raises(mlrun.errors.MLRunHTTPError),
            None,
        ),
        (
            [
                ApiException(status=400),
                k8s_client.V1PodList(
                    items=[],
                    metadata=k8s_client.V1ListMeta(),
                ),
            ],
            pytest.raises(mlrun.errors.MLRunBadRequestError),
            None,
        ),
    ],
)
def test_list_paginated_pods_retry(
    k8s_helper, side_effect, expectation, expected_result
):
    k8s_helper.v1api.list_namespaced_pod.side_effect = side_effect
    with expectation:
        result = list(k8s_helper.list_pods_paginated("my-ns"))
        if expected_result is not None:
            assert result == expected_result


@pytest.mark.parametrize(
    "side_effect, expectation, expected_result",
    [
        (
            [
                ApiException(status=410),
                ApiException(status=410),
                {"items": [], "metadata": {"continue": None}},
            ],
            does_not_raise(),
            [],
        ),
        (
            [
                ApiException(status=410),
                ApiException(status=410),
                ApiException(status=410),
                ApiException(status=410),
            ],
            pytest.raises(mlrun.errors.MLRunHTTPError),
            None,
        ),
        (
            [
                ApiException(status=400),
                {},
            ],
            pytest.raises(mlrun.errors.MLRunBadRequestError),
            None,
        ),
        # Ignoring not found - should not raise
        (
            [
                ApiException(status=404),
            ],
            does_not_raise(),
            [],
        ),
    ],
)
def test_list_paginated_crds_retry(
    k8s_helper, side_effect, expectation, expected_result
):
    k8s_helper.crdapi.list_namespaced_custom_object.side_effect = side_effect
    with expectation:
        result = list(k8s_helper.list_crds_paginated("group", "v1", "objects", "my-ns"))
        if expected_result is not None:
            assert result == expected_result
