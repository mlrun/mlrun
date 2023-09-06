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
#
import base64
import json
import unittest.mock
from http import HTTPStatus

import kubernetes.client
import pytest
from deepdiff import DeepDiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.iguazio
import mlrun.common.schemas
import mlrun.k8s_utils
import mlrun.runtimes.pod
import tests.api.api.utils
import tests.api.conftest
from mlrun.api.api.utils import (
    _generate_function_and_task_from_submit_run_body,
    _mask_v3io_access_key_env_var,
    _mask_v3io_volume_credentials,
    ensure_function_has_auth_set,
    ensure_function_security_context,
    get_scheduler,
)
from mlrun.common.schemas import SecurityContextEnrichmentModes
from mlrun.utils import logger

# Want to use k8s_secrets_mock for all tests in this module. It is needed since
# _generate_function_and_task_from_submit_run_body looks for project secrets for secret-account validation.
pytestmark = pytest.mark.usefixtures("k8s_secrets_mock")
PROJECT = "some-project"


def test_submit_run_sync(db: Session, client: TestClient):
    auth_info = mlrun.common.schemas.AuthInfo()
    tests.api.api.utils.create_project(client, PROJECT)
    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "schedule": "0 * * * *",
        "task": {
            "spec": {
                "function": f"{project}/{function_name}:{function_tag}",
            },
            "metadata": {"name": "sometask", "project": project},
        },
        "function": {
            "metadata": {"credentials": {"access_key": "some-access-key-override"}},
        },
    }
    _, _, _, response_data = mlrun.api.api.utils.submit_run_sync(
        db, auth_info, submit_job_body
    )
    assert response_data["data"]["action"] == "created"

    # submit again, make sure it was modified
    submit_job_body["schedule"] = "0 1 * * *"  # change schedule
    _, _, _, response_data = mlrun.api.api.utils.submit_run_sync(
        db, auth_info, submit_job_body
    )
    assert response_data["data"]["action"] == "modified"

    updated_schedule = get_scheduler().get_schedule(
        db, project, submit_job_body["task"]["metadata"]["name"]
    )
    assert (
        updated_schedule.cron_trigger.to_crontab() == "0 1 * * *"
    ), "schedule was not updated"


def test_generate_function_and_task_from_submit_run_body_body_override_values(
    db: Session, client: TestClient
):
    task_name = "task_name"
    tests.api.api.utils.create_project(client, PROJECT)

    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": PROJECT},
        },
        "function": {
            "metadata": {"credentials": {"access_key": "some-access-key-override"}},
            "spec": {
                "preemption_mode": "prevent",
                "volumes": [
                    {
                        "name": "override-volume-name",
                        "flexVolume": {
                            "driver": "v3io/fuse",
                            "options": {
                                "container": "users",
                                "accessKey": "4dbc1521-f6f2-4b28-aeac-29073413b9ae",
                                "subPath": "/pipelines/.mlrun",
                            },
                        },
                    },
                    {
                        "name": "new-volume-name",
                        "secret": {"secretName": "secret-name"},
                    },
                ],
                "volume_mounts": [
                    {
                        "name": "old-volume-name",
                        "mountPath": "/v3io/old/volume/mount/path",
                    },
                    {
                        "name": "override-volume-name",
                        "mountPath": "/v3io/volume/mount/path",
                    },
                    {
                        "name": "new-volume-name",
                        "mountPath": "/secret/volume/mount/path",
                    },
                ],
                "env": [
                    {"name": "OVERRIDE_ENV_VAR_KEY", "value": "new-env-var-value"},
                    {"name": "NEW_ENV_VAR_KEY", "value": "env-var-value"},
                    {
                        "name": "CURRENT_NODE_IP",
                        "valueFrom": {
                            "fieldRef": {
                                "apiVersion": "v1",
                                "fieldPath": "status.hostIP",
                            }
                        },
                    },
                ],
                "resources": {
                    "limits": {"cpu": "250m", "memory": "64Mi", "nvidia.com/gpu": "2"},
                    "requests": {"cpu": "200m", "memory": "32Mi"},
                },
                "image_pull_policy": "Always",
                "replicas": "3",
                "node_name": "k8s-node1",
                "node_selector": {"kubernetes.io/hostname": "k8s-node1"},
                "affinity": {
                    "nodeAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "preference": {
                                    "matchExpressions": [
                                        {
                                            "key": "some_node_label",
                                            "operator": "In",
                                            "values": [
                                                "possible-label-value-1",
                                                "possible-label-value-2",
                                            ],
                                        }
                                    ]
                                },
                                "weight": 1,
                            }
                        ],
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "some_node_label",
                                            "operator": "In",
                                            "values": [
                                                "required-label-value-1",
                                                "required-label-value-2",
                                            ],
                                        }
                                    ]
                                }
                            ]
                        },
                    },
                    "podAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "labelSelector": {
                                    "matchLabels": {
                                        "some-pod-label-key": "some-pod-label-value"
                                    }
                                },
                                "namespaces": ["namespace-a", "namespace-b"],
                                "topologyKey": "key-1",
                            }
                        ]
                    },
                    "podAntiAffinity": {
                        "preferredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "podAffinityTerm": {
                                    "labelSelector": {
                                        "matchExpressions": [
                                            {
                                                "key": "some_pod_label",
                                                "operator": "NotIn",
                                                "values": [
                                                    "forbidden-label-value-1",
                                                    "forbidden-label-value-2",
                                                ],
                                            }
                                        ]
                                    },
                                    "namespaces": ["namespace-c"],
                                    "topologyKey": "key-2",
                                },
                                "weight": 1,
                            }
                        ]
                    },
                },
                "tolerations": [
                    {
                        "key": "key1",
                        "operator": "value1",
                        "effect": "NoSchedule",
                        "tolerationSeconds": 3600,
                    }
                ],
            },
        },
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, submit_job_body
    )
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == project
    assert parsed_function_object.metadata.tag == function_tag
    assert (
        DeepDiff(
            parsed_function_object.metadata.credentials.to_dict(),
            submit_job_body["function"]["metadata"]["credentials"],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            parsed_function_object.spec.resources,
            submit_job_body["function"]["spec"]["resources"],
            ignore_order=True,
        )
        == {}
    )
    assert (
        parsed_function_object.spec.image_pull_policy
        == submit_job_body["function"]["spec"]["image_pull_policy"]
    )
    assert (
        parsed_function_object.spec.replicas
        == submit_job_body["function"]["spec"]["replicas"]
    )

    _assert_volumes_and_volume_mounts(
        parsed_function_object, submit_job_body, original_function
    )
    _assert_env_vars(parsed_function_object, submit_job_body, original_function)
    assert (
        parsed_function_object.spec.node_name
        == submit_job_body["function"]["spec"]["node_name"]
    )
    assert (
        DeepDiff(
            parsed_function_object.spec.node_selector,
            submit_job_body["function"]["spec"]["node_selector"],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            mlrun.runtimes.pod.get_sanitized_attribute(
                parsed_function_object.spec, "affinity"
            ),
            submit_job_body["function"]["spec"]["affinity"],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            mlrun.runtimes.pod.get_sanitized_attribute(
                parsed_function_object.spec, "tolerations"
            ),
            submit_job_body["function"]["spec"]["tolerations"],
            ignore_order=True,
        )
        == {}
    )
    assert (
        parsed_function_object.spec.preemption_mode
        == submit_job_body["function"]["spec"]["preemption_mode"]
    )


def test_generate_function_and_task_from_submit_run_with_preemptible_nodes_and_tolerations(
    db: Session, client: TestClient
):
    k8s_api = kubernetes.client.ApiClient()
    task_name = "task_name"
    tests.api.api.utils.create_project(client, PROJECT)

    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    node_selector = {"label-1": "val1"}
    mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
        json.dumps(node_selector).encode("utf-8")
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": PROJECT},
        },
        "function": {"spec": {"preemption_mode": "prevent"}},
    }
    expected_anti_affinity = kubernetes.client.V1Affinity(
        node_affinity=kubernetes.client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=kubernetes.client.V1NodeSelector(
                node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_anti_affinity_terms(),
            ),
        ),
    )
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, submit_job_body
    )
    assert (
        parsed_function_object.spec.preemption_mode
        == submit_job_body["function"]["spec"]["preemption_mode"]
    )
    assert parsed_function_object.spec.affinity == expected_anti_affinity
    assert parsed_function_object.spec.tolerations is None

    preemptible_tolerations = [
        kubernetes.client.V1Toleration(
            effect="NoSchedule",
            key="test1",
            operator="Exists",
        )
    ]
    serialized_tolerations = k8s_api.sanitize_for_serialization(preemptible_tolerations)
    mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
        json.dumps(serialized_tolerations).encode("utf-8")
    )

    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": PROJECT},
        },
        "function": {"spec": {"preemption_mode": "constrain"}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, submit_job_body
    )
    expected_affinity = kubernetes.client.V1Affinity(
        node_affinity=kubernetes.client.V1NodeAffinity(
            required_during_scheduling_ignored_during_execution=kubernetes.client.V1NodeSelector(
                node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_affinity_terms(),
            ),
        ),
    )

    assert (
        parsed_function_object.spec.preemption_mode
        == submit_job_body["function"]["spec"]["preemption_mode"]
    )
    assert parsed_function_object.spec.affinity == expected_affinity
    assert parsed_function_object.spec.tolerations == preemptible_tolerations


def test_generate_function_and_task_from_submit_run_body_keep_resources(
    db: Session, client: TestClient
):
    task_name = "task_name"
    tests.api.api.utils.create_project(client, PROJECT)

    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": PROJECT},
        },
        "function": {"spec": {"resources": {"limits": {}, "requests": {}}}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, submit_job_body
    )
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == PROJECT
    assert parsed_function_object.metadata.tag == function_tag
    assert (
        DeepDiff(
            parsed_function_object.spec.resources,
            submit_job_body["function"]["spec"]["resources"],
            ignore_order=True,
        )
        != {}
    )
    assert (
        DeepDiff(
            parsed_function_object.spec.resources,
            original_function["spec"]["resources"],
            ignore_order=True,
        )
        == {}
    )


def test_generate_function_and_task_from_submit_run_body_keep_credentials(
    db: Session, client: TestClient
):
    task_name = "task_name"
    tests.api.api.utils.create_project(client, PROJECT)

    access_key = "original-function-access-key"
    project, function_name, function_tag, original_function = _mock_original_function(
        client, access_key
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": PROJECT},
        },
        "function": {"metadata": {"credentials": None}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, submit_job_body
    )
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == project
    assert parsed_function_object.metadata.tag == function_tag
    assert parsed_function_object.metadata.credentials.access_key == access_key


def test_ensure_function_has_auth_set(
    db: Session, client: TestClient, k8s_secrets_mock: tests.api.conftest.K8sSecretsMock
):
    tests.api.api.utils.create_project(client, PROJECT)

    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )

    logger.info("Local function, nothing should be changed")
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.local
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_has_auth_set(function, mlrun.common.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info("Generate keyword, secret should be created, env should reference it")
    username = "username"
    access_key = "generated-access-key"
    _, _, _, original_function_dict = _generate_original_function(
        access_key=mlrun.model.Credentials.generate_access_key,
        kind=mlrun.runtimes.RuntimeKinds.job,
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key = (
        unittest.mock.Mock(return_value=access_key)
    )
    ensure_function_has_auth_set(
        function, mlrun.common.schemas.AuthInfo(username=username)
    )
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            # ignore order with exclude path of specific list index ends up with errors
            ignore_order_func=lambda level: "env" not in level.path(),
            exclude_paths=[
                "root['metadata']['credentials']['access_key']",
                f"root['spec']['env'][{len(function.spec.env)-1}]",
            ],
        )
        == {}
    )
    secret_name = k8s_secrets_mock.resolve_auth_secret_name(username, access_key)
    assert (
        function.metadata.credentials.access_key
        == f"{mlrun.model.Credentials.secret_reference_prefix}{secret_name}"
    )
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    _assert_env_var_from_secret(
        function,
        mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session,
        secret_name,
        mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )

    logger.info("No access key - explode")
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"(.*)Function access key must be set(.*)",
    ):
        ensure_function_has_auth_set(function, mlrun.common.schemas.AuthInfo())

    logger.info("Access key without username - explode")
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job, access_key="some-access-key"
    )
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match=r"(.*)Username is missing(.*)"
    ):
        ensure_function_has_auth_set(function, mlrun.common.schemas.AuthInfo())

    logger.info("Access key ref provided - env should be set")
    secret_name = "some-access-key-secret-name"
    access_key = f"{mlrun.model.Credentials.secret_reference_prefix}{secret_name}"
    _, _, _, original_function_dict = _generate_original_function(
        access_key=access_key,
        kind=mlrun.runtimes.RuntimeKinds.job,
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_has_auth_set(function, mlrun.common.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            # ignore order with exclude path of specific list index ends up with errors
            ignore_order_func=lambda level: "env" not in level.path(),
            exclude_paths=[f"root['spec']['env'][{len(function.spec.env)-1}]"],
        )
        == {}
    )
    _assert_env_var_from_secret(
        function,
        mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session,
        secret_name,
        mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )

    logger.info(
        "Raw access key provided - secret should be created, env should reference it"
    )
    access_key = "some-access-key"
    username = "some-username"
    _, _, _, original_function_dict = _generate_original_function(
        access_key=access_key,
        kind=mlrun.runtimes.RuntimeKinds.job,
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_has_auth_set(
        function, mlrun.common.schemas.AuthInfo(username=username)
    )
    secret_name = k8s_secrets_mock.resolve_auth_secret_name(username, access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
            # ignore order with exclude path of specific list index ends up with errors
            ignore_order_func=lambda level: "env" not in level.path(),
            exclude_paths=[
                "root['metadata']['credentials']['access_key']",
                f"root['spec']['env'][{len(function.spec.env)-1}]",
            ],
        )
        == {}
    )
    assert (
        function.metadata.credentials.access_key
        == f"{mlrun.model.Credentials.secret_reference_prefix}{secret_name}"
    )
    _assert_env_var_from_secret(
        function,
        mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session,
        secret_name,
        mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )


def test_mask_v3io_access_key_env_var(
    db: Session, client: TestClient, k8s_secrets_mock: tests.api.conftest.K8sSecretsMock
):
    tests.api.api.utils.create_project(client, PROJECT)

    logger.info("Mask function without access key, nothing should be changed")
    _, _, _, original_function_dict = _generate_original_function()
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _mask_v3io_access_key_env_var(function, mlrun.common.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info(
        "Mask function with access key without username when iguazio auth on - explode"
    )
    v3io_access_key = "some-v3io-access-key"
    _, _, _, original_function_dict = _generate_original_function(
        v3io_access_key=v3io_access_key
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"(.*)Username is missing(.*)",
    ):
        _mask_v3io_access_key_env_var(function, mlrun.common.schemas.AuthInfo())

    logger.info(
        "Mask function with access key without username when iguazio auth off - skip"
    )
    _, _, _, original_function_dict = _generate_original_function(
        v3io_access_key=v3io_access_key
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=False)
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _mask_v3io_access_key_env_var(function, mlrun.common.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info(
        "Happy flow - mask function with access key with username from env var - secret should be "
        "created, env should reference it"
    )
    username = "some-username"
    _, _, _, original_function_dict = _generate_original_function(
        v3io_access_key=v3io_access_key, v3io_username=username
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function: mlrun.runtimes.pod.KubeResource = mlrun.new_function(
        runtime=original_function_dict
    )
    _mask_v3io_access_key_env_var(function, mlrun.common.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            # ignore order with exclude path of specific list index ends up with errors
            ignore_order_func=lambda level: "env" not in level.path(),
            exclude_paths=[f"root['spec']['env'][{len(function.spec.env)-1}]"],
        )
        == {}
    )
    secret_name = k8s_secrets_mock.resolve_auth_secret_name(username, v3io_access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, v3io_access_key)
    _assert_env_var_from_secret(
        function,
        "V3IO_ACCESS_KEY",
        secret_name,
        mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )

    logger.info(
        "mask same function again, access key is already a reference - nothing should change"
    )
    original_function = mlrun.new_function(runtime=function)
    _mask_v3io_access_key_env_var(function, mlrun.common.schemas.AuthInfo())
    mlrun.api.crud.Secrets().store_auth_secret = unittest.mock.Mock()
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
        )
        == {}
    )
    # assert we're not trying to store unneeded-ly
    assert mlrun.api.crud.Secrets().store_auth_secret.call_count == 0

    logger.info(
        "mask same function again, access key is already a reference, but this time a dict - nothing "
        "should change"
    )
    function.spec.env.append(function.spec.env.pop().to_dict())
    original_function = mlrun.new_function(runtime=function)
    _mask_v3io_access_key_env_var(
        function, mlrun.common.schemas.AuthInfo(username=username)
    )
    mlrun.api.crud.Secrets().store_auth_secret = unittest.mock.Mock()
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
        )
        == {}
    )
    # assert we're not trying to store unneeded-ly
    assert mlrun.api.crud.Secrets().store_auth_secret.call_count == 0


@pytest.mark.parametrize("use_structs", [True, False])
def test_mask_v3io_volume_credentials(
    db: Session,
    client: TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    use_structs: bool,
):
    username = "volume-username"
    access_key = "volume-access-key"
    v3io_volume = mlrun.platforms.iguazio.v3io_to_vol(
        "some-v3io-volume-name", "", access_key, user=username
    )
    v3io_volume_mount = kubernetes.client.V1VolumeMount(
        mount_path="some-v3io-mount-path",
        sub_path=f"users/{username}",
        name=v3io_volume["name"],
    )
    conflicting_v3io_volume_mount = kubernetes.client.V1VolumeMount(
        mount_path="some-other-mount-path",
        sub_path="users/another-username",
        name=v3io_volume["name"],
    )
    no_matching_mount_v3io_volume = mlrun.platforms.iguazio.v3io_to_vol(
        "no-matching-mount-v3io-volume-name", "", access_key
    )
    regular_volume = kubernetes.client.V1Volume(
        name="regular-volume-name", empty_dir=kubernetes.client.V1EmptyDirVolumeSource()
    )
    regular_volume_mount = kubernetes.client.V1VolumeMount(
        mount_path="regular-mount-path", name=regular_volume.name
    )
    no_name_volume_mount = kubernetes.client.V1VolumeMount(
        name="",
        mount_path="some-mount-path",
    )
    no_access_key_v3io_volume = mlrun.platforms.iguazio.v3io_to_vol(
        "no-access-key-v3io-volume-name", "", ""
    )
    no_name_v3io_volume = mlrun.platforms.iguazio.v3io_to_vol("", "", access_key)
    k8s_api_client = kubernetes.client.ApiClient()
    if not use_structs:
        v3io_volume["flexVolume"] = k8s_api_client.sanitize_for_serialization(
            v3io_volume["flexVolume"]
        )
        no_access_key_v3io_volume[
            "flexVolume"
        ] = k8s_api_client.sanitize_for_serialization(
            no_access_key_v3io_volume["flexVolume"]
        )
        no_name_v3io_volume["flexVolume"] = k8s_api_client.sanitize_for_serialization(
            no_name_v3io_volume["flexVolume"]
        )
        no_matching_mount_v3io_volume[
            "flexVolume"
        ] = k8s_api_client.sanitize_for_serialization(
            no_matching_mount_v3io_volume["flexVolume"]
        )
        v3io_volume_mount = k8s_api_client.sanitize_for_serialization(v3io_volume_mount)
        conflicting_v3io_volume_mount = k8s_api_client.sanitize_for_serialization(
            conflicting_v3io_volume_mount
        )
        regular_volume = k8s_api_client.sanitize_for_serialization(regular_volume)
        regular_volume_mount = k8s_api_client.sanitize_for_serialization(
            regular_volume_mount
        )
        no_name_volume_mount = k8s_api_client.sanitize_for_serialization(
            no_name_volume_mount
        )
    tests.api.api.utils.create_project(client, PROJECT)

    logger.info("Mask function without v3io volume, nothing should be changed")
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[regular_volume], volume_mounts=[regular_volume_mount]
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _mask_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info("Mask several edge cases, nothing should be changed")
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[
            no_access_key_v3io_volume,
            no_name_v3io_volume,
            v3io_volume,
            no_matching_mount_v3io_volume,
        ],
        volume_mounts=[
            no_name_volume_mount,
            v3io_volume_mount,
            conflicting_v3io_volume_mount,
        ],
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _mask_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info(
        "Happy flow, username resolved from volume mount, masking should be done, secret should be "
        "created, volume should reference it"
    )
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[v3io_volume], volume_mounts=[v3io_volume_mount]
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _mask_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
            exclude_paths=["root['spec']['volumes'][0]['flexVolume']"],
        )
        == {}
    )
    secret_name = k8s_secrets_mock.resolve_auth_secret_name(username, access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    assert "accessKey" not in function.spec.volumes[0]["flexVolume"]["options"]
    assert function.spec.volumes[0]["flexVolume"]["secretRef"]["name"] == secret_name

    logger.info(
        "Happy flow, username resolved from env var, masking should be done, secret should be "
        "created, volume should reference it"
    )
    k8s_secrets_mock.reset_mock()
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[v3io_volume], v3io_username=username
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _mask_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
            exclude_paths=["root['spec']['volumes'][0]['flexVolume']"],
        )
        == {}
    )
    secret_name = k8s_secrets_mock.resolve_auth_secret_name(username, access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    assert "accessKey" not in function.spec.volumes[0]["flexVolume"]["options"]
    assert function.spec.volumes[0]["flexVolume"]["secretRef"]["name"] == secret_name


def test_ensure_function_security_context_no_enrichment(
    db: Session, client: TestClient
):
    tests.api.api.utils.create_project(client, PROJECT)
    auth_info = mlrun.common.schemas.AuthInfo(user_unix_id=1000)
    mlrun.mlconf.igz_version = "3.6"

    logger.info("Enrichment mode is disabled, nothing should be changed")
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.disabled.value
    )
    _, _, _, original_function_dict_job_kind = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )
    original_function = mlrun.new_function(runtime=original_function_dict_job_kind)
    function = mlrun.new_function(runtime=original_function_dict_job_kind)
    ensure_function_security_context(function, auth_info)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info("Local function, nothing should be changed")
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    _, _, _, original_function_dict_local_kind = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.local
    )
    original_function = mlrun.new_function(runtime=original_function_dict_local_kind)
    function = mlrun.new_function(runtime=original_function_dict_local_kind)
    ensure_function_security_context(function, auth_info)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info("Not running on iguazio, nothing should be changed")
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    _, _, _, original_function_dict_job_kind = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )
    original_function = mlrun.new_function(runtime=original_function_dict_job_kind)
    function = mlrun.new_function(runtime=original_function_dict_job_kind)
    ensure_function_security_context(function, mlrun.common.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )


def test_ensure_function_security_context_override_enrichment_mode(
    db: Session, client: TestClient
):
    tests.api.api.utils.create_project(client, PROJECT)
    mlrun.mlconf.igz_version = "3.6"
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )

    logger.info("Enrichment mode is override, security context should be enriched")
    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id = unittest.mock.Mock()
    auth_info = mlrun.common.schemas.AuthInfo(user_unix_id=1000)
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )
    original_function = mlrun.new_function(runtime=original_function_dict)

    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_security_context(function, auth_info)

    # assert user unix id was not fetched from iguazio
    assert mlrun.api.utils.clients.iguazio.Client.get_user_unix_id.called == 0

    # assert function was changed
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        != {}
    )

    # the enrichment that should be done
    original_function.spec.security_context = kubernetes.client.V1SecurityContext(
        run_as_user=auth_info.user_unix_id,
        run_as_group=int(
            mlrun.mlconf.function.spec.security_context.enrichment_group_id
        ),
    )
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )


def test_ensure_function_security_context_enrichment_group_id(
    db: Session, client: TestClient
):
    tests.api.api.utils.create_project(client, PROJECT)
    mlrun.mlconf.igz_version = "3.6"
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override.value
    )
    auth_info = mlrun.common.schemas.AuthInfo(user_unix_id=1000)
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )

    logger.info("Change enrichment group id and validate it is being enriched")
    group_id = 2000
    mlrun.mlconf.function.spec.security_context.enrichment_group_id = group_id
    original_function = mlrun.new_function(runtime=original_function_dict)
    original_function.spec.security_context = kubernetes.client.V1SecurityContext(
        run_as_user=auth_info.user_unix_id,
        run_as_group=group_id,
    )

    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_security_context(function, auth_info)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info("Enrichment group id is -1, user unix id should be used as group id")
    mlrun.mlconf.function.spec.security_context.enrichment_group_id = -1
    original_function = mlrun.new_function(runtime=original_function_dict)
    original_function.spec.security_context = kubernetes.client.V1SecurityContext(
        run_as_user=auth_info.user_unix_id,
        run_as_group=auth_info.user_unix_id,
    )

    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_security_context(function, auth_info)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )


def test_ensure_function_security_context_unknown_enrichment_mode(
    db: Session, client: TestClient
):
    tests.api.api.utils.create_project(client, PROJECT)
    mlrun.mlconf.igz_version = "3.6"
    mlrun.mlconf.function.spec.security_context.enrichment_mode = "not a real mode"
    auth_info = mlrun.common.schemas.AuthInfo(user_unix_id=1000)
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )

    logger.info("Unknown enrichment mode, should fail")
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        ensure_function_security_context(function, auth_info)
    assert (
        f"Invalid security context enrichment mode {mlrun.mlconf.function.spec.security_context.enrichment_mode}"
        in str(exc.value)
    )


def test_ensure_function_security_context_missing_control_plane_session_tag(
    db: Session, client: TestClient
):
    tests.api.api.utils.create_project(client, PROJECT)
    mlrun.mlconf.igz_version = "3.6"
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override
    )
    auth_info = mlrun.common.schemas.AuthInfo(
        planes=[mlrun.api.utils.clients.iguazio.SessionPlanes.data]
    )
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )

    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id = unittest.mock.Mock(
        side_effect=mlrun.errors.MLRunHTTPError()
    )
    logger.info(
        "Session missing control plane, and it is actually only a data plane session, expected to fail"
    )
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(mlrun.errors.MLRunUnauthorizedError) as exc:
        ensure_function_security_context(function, auth_info)
    assert "Were unable to enrich user unix id" in str(exc.value)
    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id.assert_called_once()

    user_unix_id = 1000
    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id = unittest.mock.Mock(
        return_value=user_unix_id
    )
    auth_info = mlrun.common.schemas.AuthInfo(planes=[])
    logger.info(
        "Session missing control plane, but actually just because it wasn't enriched, expected to succeed"
    )
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_security_context(function, auth_info)
    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id.assert_called_once()
    assert auth_info.planes == [mlrun.api.utils.clients.iguazio.SessionPlanes.control]


def test_ensure_function_security_context_get_user_unix_id(
    db: Session, client: TestClient
):
    tests.api.api.utils.create_project(client, PROJECT)
    mlrun.mlconf.igz_version = "3.6"
    user_unix_id = 1000
    mlrun.mlconf.function.spec.security_context.enrichment_mode = (
        SecurityContextEnrichmentModes.override
    )

    # set auth info with control plane and without user unix id so that it will be fetched
    auth_info = mlrun.common.schemas.AuthInfo(
        planes=[mlrun.api.utils.clients.iguazio.SessionPlanes.control]
    )
    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id = unittest.mock.Mock(
        return_value=user_unix_id
    )

    logger.info("No user unix id in headers, should fetch from iguazio")
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    original_function.spec.security_context = kubernetes.client.V1SecurityContext(
        run_as_user=user_unix_id,
        run_as_group=mlrun.mlconf.function.spec.security_context.enrichment_group_id,
    )

    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_security_context(function, auth_info)
    mlrun.api.utils.clients.iguazio.Client.get_user_unix_id.assert_called_once()
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )


def test_generate_function_and_task_from_submit_run_body_imported_function_project_assignment(
    db: Session, client: TestClient, monkeypatch
):
    task_name = "task_name"
    tests.api.api.utils.create_project(client, PROJECT)

    _mock_import_function(monkeypatch)
    submit_job_body = {
        "task": {
            "spec": {"function": "hub://gen-class-data"},
            "metadata": {"name": task_name, "project": PROJECT},
        },
        "function": {"spec": {"resources": {"limits": {}, "requests": {}}}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, submit_job_body
    )
    assert parsed_function_object.metadata.project == PROJECT


def test_get_obj_path(db: Session, client: TestClient):
    cases = [
        {
            "path": "/local/path",
            "schema": "v3io",
            "expected_path": "v3io:///local/path",
        },
        {"path": "/User/my/path", "expected_path": "v3io:///users/admin/my/path"},
        {
            "path": "/User/my/path",
            "schema": "v3io",
            "expected_path": "v3io:///users/admin/my/path",
        },
        {
            "path": "/User/my/path",
            "user": "hedi",
            "expected_path": "v3io:///users/hedi/my/path",
        },
        {
            "path": "/v3io/projects/my-proj/my/path",
            "expected_path": "v3io:///projects/my-proj/my/path",
        },
        {
            "path": "/v3io/projects/my-proj/my/path",
            "schema": "v3io",
            "expected_path": "v3io:///projects/my-proj/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data",
            "real_path": "/root",
            "expected_path": "/root/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data/",
            "real_path": "/root",
            "expected_path": "/root/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data/",
            "real_path": "/root/",
            "expected_path": "/root/my/path",
        },
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data",
            "real_path": "/root/",
            "expected_path": "/root/my/path",
        },
        {"path": "/local/path", "expect_error": True},
        {
            "path": "/home/jovyan/data/my/path",
            "data_volume": "/home/jovyan/data",
            "expect_error": True,
        },
        {
            "path": "gcs://bucket/and/path",
            "allowed_paths": "http://, gcs:// ",
            "expected_path": "gcs://bucket/and/path",
        },
        {
            "path": "bucket/and/path",
            "schema": "gs",
            "allowed_paths": " gs://, gcs:// ",
            "expected_path": "gs://bucket/and/path",
        },
        {
            "path": "gcs://bucket/and/path",
            "allowed_paths": "s3://",
            "expect_error": True,
        },
        {
            "path": "/local/file/security/breach",
            "allowed_paths": "/local",
            "expect_error": True,
        },
    ]
    for case in cases:
        logger.info("Testing case", case=case)
        old_real_path = mlrun.mlconf.httpdb.real_path
        old_data_volume = mlrun.mlconf.httpdb.data_volume
        old_allowed_file_paths = mlrun.mlconf.httpdb.allowed_file_paths
        if case.get("real_path"):
            mlrun.mlconf.httpdb.real_path = case["real_path"]
        if case.get("data_volume"):
            mlrun.mlconf.httpdb.data_volume = case["data_volume"]
        if case.get("allowed_paths"):
            mlrun.mlconf.httpdb.allowed_file_paths = case["allowed_paths"]
        if case.get("expect_error"):
            with pytest.raises(
                mlrun.errors.MLRunAccessDeniedError, match="Unauthorized path"
            ):
                mlrun.api.api.utils.get_obj_path(
                    case.get("schema"), case.get("path"), case.get("user")
                )
        else:
            result_path = mlrun.api.api.utils.get_obj_path(
                case.get("schema"), case.get("path"), case.get("user")
            )
            assert result_path == case["expected_path"]
            if case.get("real_path"):
                mlrun.mlconf.httpdb.real_path = old_real_path
            if case.get("data_volume"):
                mlrun.mlconf.httpdb.data_volume = old_data_volume
            if case.get("allowed_paths"):
                mlrun.mlconf.httpdb.allowed_file_paths = old_allowed_file_paths


def _mock_import_function(monkeypatch):
    def _mock_import_function_to_dict(*args, **kwargs):
        _, _, _, original_function = _generate_original_function()
        return original_function

    def _mock_extend_hub_uri_if_needed(*args, **kwargs):
        return "some-url", True

    monkeypatch.setattr(
        mlrun.run, "import_function_to_dict", _mock_import_function_to_dict
    )

    monkeypatch.setattr(
        mlrun.run, "extend_hub_uri_if_needed", _mock_extend_hub_uri_if_needed
    )


def _mock_original_function(
    client, access_key=None, kind=mlrun.runtimes.RuntimeKinds.job
):
    (
        project,
        function_name,
        function_tag,
        original_function,
    ) = _generate_original_function(access_key=access_key, kind=kind)
    resp = client.post(
        f"func/{project}/{function_name}",
        json=original_function,
        params={"tag": function_tag},
    )
    assert resp.status_code == HTTPStatus.OK.value
    return project, function_name, function_tag, original_function


def _generate_original_function(
    access_key=None,
    kind=mlrun.runtimes.RuntimeKinds.job,
    v3io_username=None,
    v3io_access_key=None,
    volumes=None,
    volume_mounts=None,
):
    function_name = "function-name"
    project = "some-project"
    function_tag = "function_tag"
    original_function = {
        "kind": kind,
        "metadata": {"name": function_name, "tag": function_tag, "project": project},
        "spec": {
            "volumes": [
                {
                    "name": "old-volume-name",
                    "flexVolume": {
                        "driver": "v3io/fuse",
                        "options": {
                            "container": "bigdata",
                            "accessKey": "1acf6fa2-f3b3-4c37-a9c6-759e555e0018",
                            "subPath": "/admin/data",
                        },
                    },
                },
                {
                    "name": "override-volume-name",
                    "flexVolume": {
                        "driver": "v3io/fuse",
                        "options": {
                            "container": "bigdata",
                            "accessKey": "c7f736ec-567b-42eb-b7c0-1aea8a66f880",
                            "subPath": "/iguazio/.db",
                        },
                    },
                },
            ],
            "volume_mounts": [
                {
                    "name": "old-volume-name",
                    "mountPath": "/v3io/old/volume/mount/path",
                },
                {
                    "name": "override-volume-name",
                    "mountPath": "/v3io/override/volume/mount/path",
                },
            ],
            "resources": {
                "limits": {"cpu": "40m", "memory": "128Mi", "nvidia.com/gpu": "7"},
                "requests": {"cpu": "15m", "memory": "86Mi"},
            },
            "env": [
                {"name": "OLD_ENV_VAR_KEY", "value": "old-env-var-value"},
                {"name": "OVERRIDE_ENV_VAR_KEY", "value": "override-env-var-value"},
            ],
            "image_pull_policy": "IfNotPresent",
            "replicas": "1",
        },
    }
    if access_key:
        original_function["metadata"]["credentials"] = {
            "access_key": access_key,
        }
    if v3io_username:
        original_function["spec"]["env"].append(
            {
                "name": "V3IO_USERNAME",
                "value": v3io_username,
            }
        )
    if v3io_access_key:
        original_function["spec"]["env"].append(
            {
                "name": "V3IO_ACCESS_KEY",
                "value": v3io_access_key,
            }
        )
    if volumes:
        original_function["spec"]["volumes"] = volumes
    if volume_mounts:
        original_function["spec"]["volume_mounts"] = volume_mounts
    return project, function_name, function_tag, original_function


def _assert_volumes_and_volume_mounts(
    parsed_function_object, submit_job_body, original_function
):
    """
    expected volumes and volume mounts:
    0: old volume from original function (the first one there)
    1: volume that was in original function but was overridden with body (the first one in the body)
    2: new volume from the body (the second in the body)
    """
    assert (
        DeepDiff(
            original_function["spec"]["volumes"][0],
            parsed_function_object.spec.volumes[0],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            original_function["spec"]["volumes"][1],
            parsed_function_object.spec.volumes[1],
            ignore_order=True,
        )
        != {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volumes"][0],
            parsed_function_object.spec.volumes[1],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volumes"][1],
            parsed_function_object.spec.volumes[2],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            original_function["spec"]["volume_mounts"][0],
            parsed_function_object.spec.volume_mounts[0],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            original_function["spec"]["volume_mounts"][1],
            parsed_function_object.spec.volume_mounts[1],
            ignore_order=True,
        )
        != {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volume_mounts"][0],
            parsed_function_object.spec.volume_mounts[0],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volume_mounts"][1],
            parsed_function_object.spec.volume_mounts[1],
            ignore_order=True,
        )
        == {}
    )
    assert (
        DeepDiff(
            submit_job_body["function"]["spec"]["volume_mounts"][2],
            parsed_function_object.spec.volume_mounts[2],
            ignore_order=True,
        )
        == {}
    )


def _assert_env_vars(parsed_function_object, submit_job_body, original_function):
    """
    expected env vars:
    0: old env var from original function (the first one there)
    1: env var that was in original function but was overridden with body (the first one in the body)
    2: new env var from the body (the second in the body)
    3: new env var (with valueFrom) from the body (the third in the body)
    """
    assert (
        original_function["spec"]["env"][0]["name"]
        == parsed_function_object.spec.env[0]["name"]
    )
    assert (
        original_function["spec"]["env"][0]["value"]
        == parsed_function_object.spec.env[0]["value"]
    )

    assert (
        original_function["spec"]["env"][1]["name"]
        == parsed_function_object.spec.env[1].name
    )
    assert (
        submit_job_body["function"]["spec"]["env"][0]["name"]
        == parsed_function_object.spec.env[1].name
    )
    assert (
        original_function["spec"]["env"][1]["value"]
        != parsed_function_object.spec.env[1].value
    )
    assert (
        submit_job_body["function"]["spec"]["env"][0]["value"]
        == parsed_function_object.spec.env[1].value
    )

    assert (
        submit_job_body["function"]["spec"]["env"][1]["name"]
        == parsed_function_object.spec.env[2].name
    )
    assert (
        submit_job_body["function"]["spec"]["env"][1]["value"]
        == parsed_function_object.spec.env[2].value
    )

    assert (
        submit_job_body["function"]["spec"]["env"][2]["name"]
        == parsed_function_object.spec.env[3].name
    )
    assert (
        submit_job_body["function"]["spec"]["env"][2]["valueFrom"]
        == parsed_function_object.spec.env[3].value_from
    )


def _assert_env_var_from_secret(
    function: mlrun.runtimes.pod.KubeResource,
    name: str,
    secret_name: str,
    secret_key: str,
):
    env_var_value = function.get_env(name)
    if env_var_value is None:
        pytest.fail(f"Env var {name} not found")
    if isinstance(env_var_value, str):
        pytest.fail(
            f"Env var {name} value is string. expected to be reference to secret. value={env_var_value}"
        )
    assert env_var_value.secret_key_ref.name == secret_name
    assert env_var_value.secret_key_ref.key == secret_key
