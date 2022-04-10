import unittest.mock
from http import HTTPStatus

import kubernetes.client
import pytest
from deepdiff import DeepDiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.runtimes.pod
import tests.api.conftest
from mlrun.api.api.utils import (
    _generate_function_and_task_from_submit_run_body,
    _obfuscate_v3io_access_key_env_var,
    _obfuscate_v3io_volume_credentials,
    ensure_function_has_auth_set,
)
from mlrun.utils import logger

# Want to use k8s_secrets_mock for all tests in this module. It is needed since
# _generate_function_and_task_from_submit_run_body looks for project secrets for secret-account validation.
pytestmark = pytest.mark.usefixtures("k8s_secrets_mock")


def test_generate_function_and_task_from_submit_run_body_body_override_values(
    db: Session, client: TestClient
):
    task_name = "task_name"
    task_project = "task-project"
    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {
            "metadata": {"credentials": {"access_key": "some-access-key-override"}},
            "spec": {
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
        db, mlrun.api.schemas.AuthInfo(), submit_job_body
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


def test_generate_function_and_task_from_submit_run_body_keep_resources(
    db: Session, client: TestClient
):
    task_name = "task_name"
    task_project = "task-project"
    project, function_name, function_tag, original_function = _mock_original_function(
        client
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {"spec": {"resources": {"limits": {}, "requests": {}}}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, mlrun.api.schemas.AuthInfo(), submit_job_body
    )
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == project
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
    task_project = "task-project"
    access_key = "original-function-access-key"
    project, function_name, function_tag, original_function = _mock_original_function(
        client, access_key
    )
    submit_job_body = {
        "task": {
            "spec": {"function": f"{project}/{function_name}:{function_tag}"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {"metadata": {"credentials": None}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, mlrun.api.schemas.AuthInfo(), submit_job_body
    )
    assert parsed_function_object.metadata.name == function_name
    assert parsed_function_object.metadata.project == project
    assert parsed_function_object.metadata.tag == function_tag
    assert parsed_function_object.metadata.credentials.access_key == access_key


def test_ensure_function_has_auth_set(
    db: Session, client: TestClient, k8s_secrets_mock: tests.api.conftest.K8sSecretsMock
):
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=True)
    )

    # local function so nothing should be changed
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.local
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_has_auth_set(function, mlrun.api.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    # generate access key - secret should be created, env should reference it
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
        function, mlrun.api.schemas.AuthInfo(username=username)
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
    secret_name = k8s_secrets_mock.get_auth_secret_name(username, access_key)
    assert (
        function.metadata.credentials.access_key
        == f"{mlrun.model.Credentials.secret_reference_prefix}{secret_name}"
    )
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    _assert_env_var_from_secret(
        function,
        mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session,
        secret_name,
        mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )

    # no access key - explode
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job
    )
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"(.*)Function access key must be set(.*)",
    ):
        ensure_function_has_auth_set(function, mlrun.api.schemas.AuthInfo())

    # access key without username - explode
    _, _, _, original_function_dict = _generate_original_function(
        kind=mlrun.runtimes.RuntimeKinds.job, access_key="some-access-key"
    )
    function = mlrun.new_function(runtime=original_function_dict)
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match=r"(.*)Username is missing(.*)"
    ):
        ensure_function_has_auth_set(function, mlrun.api.schemas.AuthInfo())

    # access key ref provided - env should be set
    secret_name = "some-access-key-secret-name"
    access_key = f"{mlrun.model.Credentials.secret_reference_prefix}{secret_name}"
    _, _, _, original_function_dict = _generate_original_function(
        access_key=access_key,
        kind=mlrun.runtimes.RuntimeKinds.job,
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_has_auth_set(function, mlrun.api.schemas.AuthInfo())
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
        mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )

    # raw access key provided - secret should be created env should be set (to reference it)
    access_key = "some-access-key"
    username = "some-username"
    _, _, _, original_function_dict = _generate_original_function(
        access_key=access_key,
        kind=mlrun.runtimes.RuntimeKinds.job,
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    ensure_function_has_auth_set(
        function, mlrun.api.schemas.AuthInfo(username=username)
    )
    secret_name = k8s_secrets_mock.get_auth_secret_name(username, access_key)
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
        mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )


def test_obfuscate_v3io_access_key_env_var(
    db: Session, client: TestClient, k8s_secrets_mock: tests.api.conftest.K8sSecretsMock
):
    logger.info("Obfuscate function without access key, nothing should be changed")
    _, _, _, original_function_dict = _generate_original_function()
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _obfuscate_v3io_access_key_env_var(function, mlrun.api.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info(
        "Obfuscate function with access key without username when iguazio auth on - explode"
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
        _obfuscate_v3io_access_key_env_var(function, mlrun.api.schemas.AuthInfo())

    logger.info(
        "Obfuscate function with access key without username when iguazio auth off - skip"
    )
    _, _, _, original_function_dict = _generate_original_function(
        v3io_access_key=v3io_access_key
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required = (
        unittest.mock.Mock(return_value=False)
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _obfuscate_v3io_access_key_env_var(function, mlrun.api.schemas.AuthInfo())
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info(
        "Happy flow - obfuscate function with access key with username from env var - secret should be "
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
    _obfuscate_v3io_access_key_env_var(function, mlrun.api.schemas.AuthInfo())
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
    secret_name = k8s_secrets_mock.get_auth_secret_name(username, v3io_access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, v3io_access_key)
    _assert_env_var_from_secret(
        function,
        "V3IO_ACCESS_KEY",
        secret_name,
        mlrun.api.schemas.AuthSecretData.get_field_secret_key("access_key"),
    )

    logger.info(
        "obfuscate same function again, access key is already a reference - nothing should change"
    )
    original_function = mlrun.new_function(runtime=function)
    _obfuscate_v3io_access_key_env_var(function, mlrun.api.schemas.AuthInfo())
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
        "obfuscate same function again, access key is already a reference, but this time a dict - nothing "
        "should change"
    )
    function.spec.env.append(function.spec.env.pop().to_dict())
    original_function = mlrun.new_function(runtime=function)
    _obfuscate_v3io_access_key_env_var(
        function, mlrun.api.schemas.AuthInfo(username=username)
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
def test_obfuscate_v3io_volume_credentials(
    db: Session,
    client: TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    use_structs: bool,
):
    username = "volume-username"
    access_key = "volume-access-key"
    v3io_volume = mlrun.platforms.iguazio.v3io_to_vol(
        "some-v3io-volume-name", "", access_key
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

    logger.info("Obfuscate function without v3io volume, nothing should be changed")
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[regular_volume], volume_mounts=[regular_volume_mount]
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _obfuscate_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info("Obfuscate several edge cases, nothing should be changed")
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
    _obfuscate_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
        )
        == {}
    )

    logger.info(
        "Happy flow, username resolved from volume mount, obfuscation should be done, secret should be "
        "created, volume should reference it"
    )
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[v3io_volume], volume_mounts=[v3io_volume_mount]
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _obfuscate_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
            exclude_paths=["root['spec']['volumes'][0]['flexVolume']"],
        )
        == {}
    )
    secret_name = k8s_secrets_mock.get_auth_secret_name(username, access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    assert "accessKey" not in function.spec.volumes[0]["flexVolume"]["options"]
    assert function.spec.volumes[0]["flexVolume"]["secretRef"]["name"] == secret_name

    logger.info(
        "Happy flow, username resolved from env var, obfuscation should be done, secret should be "
        "created, volume should reference it"
    )
    k8s_secrets_mock.reset_mock()
    _, _, _, original_function_dict = _generate_original_function(
        volumes=[v3io_volume], v3io_username=username
    )
    original_function = mlrun.new_function(runtime=original_function_dict)
    function = mlrun.new_function(runtime=original_function_dict)
    _obfuscate_v3io_volume_credentials(function)
    assert (
        DeepDiff(
            original_function.to_dict(),
            function.to_dict(),
            ignore_order=True,
            exclude_paths=["root['spec']['volumes'][0]['flexVolume']"],
        )
        == {}
    )
    secret_name = k8s_secrets_mock.get_auth_secret_name(username, access_key)
    k8s_secrets_mock.assert_auth_secret(secret_name, username, access_key)
    assert "accessKey" not in function.spec.volumes[0]["flexVolume"]["options"]
    assert function.spec.volumes[0]["flexVolume"]["secretRef"]["name"] == secret_name


def test_generate_function_and_task_from_submit_run_body_imported_function_project_assignment(
    db: Session, client: TestClient, monkeypatch
):
    task_name = "task_name"
    task_project = "task-project"
    _mock_import_function(monkeypatch)
    submit_job_body = {
        "task": {
            "spec": {"function": "hub://gen_class_data"},
            "metadata": {"name": task_name, "project": task_project},
        },
        "function": {"spec": {"resources": {"limits": {}, "requests": {}}}},
    }
    parsed_function_object, task = _generate_function_and_task_from_submit_run_body(
        db, mlrun.api.schemas.AuthInfo(), submit_job_body
    )
    assert parsed_function_object.metadata.project == task_project


def test_get_obj_path(db: Session, client: TestClient):
    cases = [
        {"path": "/local/path", "expected_path": "/local/path"},
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
            "expected_path": "/home/jovyan/data/my/path",
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
            "real_path": "/root",
            "expected_path": "/root/my/path",
        },
    ]
    for case in cases:
        old_real_path = mlrun.mlconf.httpdb.real_path
        old_data_volume = mlrun.mlconf.httpdb.data_volume
        if case.get("real_path"):
            mlrun.mlconf.httpdb.real_path = case["real_path"]
        if case.get("data_volume"):
            mlrun.mlconf.httpdb.data_volume = case["data_volume"]
        result_path = mlrun.api.api.utils.get_obj_path(
            case.get("schema"), case.get("path"), case.get("user")
        )
        assert result_path == case["expected_path"]
        if case.get("real_path"):
            mlrun.mlconf.httpdb.real_path = old_real_path
        if case.get("data_volume"):
            mlrun.mlconf.httpdb.data_volume = old_data_volume


def _mock_import_function(monkeypatch):
    def _mock_import_function_to_dict(*args, **kwargs):
        _, _, _, original_function = _generate_original_function()
        return original_function

    monkeypatch.setattr(
        mlrun.run, "import_function_to_dict", _mock_import_function_to_dict
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
    function_name = "function_name"
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
