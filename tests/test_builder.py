# Copyright 2018 Iguazio
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
import re
import unittest.mock

import deepdiff
import pytest

import mlrun
import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s
import mlrun.builder
import mlrun.k8s_utils
import mlrun.utils.version
from mlrun.config import config


def test_build_runtime_use_base_image_when_no_build():
    fn = mlrun.new_function("some-function", "some-project", "some-tag", kind="job")
    base_image = "mlrun/ml-models"
    fn.build_config(base_image=base_image)
    assert fn.spec.image == ""
    ready = mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        fn,
    )
    assert ready is True
    assert fn.spec.image == base_image


def test_build_runtime_use_image_when_no_build():
    image = "mlrun/ml-models"
    fn = mlrun.new_function(
        "some-function", "some-project", "some-tag", image=image, kind="job"
    )
    assert fn.spec.image == image
    ready = mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        fn,
        with_mlrun=False,
    )
    assert ready is True
    assert fn.spec.image == image


def test_build_config_with_multiple_commands():
    image = "mlrun/ml-models"
    fn = mlrun.new_function(
        "some-function", "some-project", "some-tag", image=image, kind="job"
    )
    fn.build_config(commands=["pip install pandas", "pip install numpy"])
    assert len(fn.spec.build.commands) == 2

    fn.build_config(commands=["pip install pandas"])
    assert len(fn.spec.build.commands) == 2


def test_build_config_preserve_order():
    function = mlrun.new_function("some-function", kind="job")
    # run a lot of times as order change
    commands = []
    for index in range(10):
        commands.append(str(index))
    # when using un-stable (doesn't preserve order) methods to make a list unique (like list(set(x))) it's random
    # whether the order will be preserved, therefore run in a loop
    for _ in range(100):
        function.spec.build.commands = []
        function.build_config(commands=commands)
        assert function.spec.build.commands == commands


@pytest.mark.parametrize(
    "pull_mode,push_mode,secret,flags_expected",
    [
        ("auto", "auto", "", True),
        ("auto", "auto", "some-secret-name", False),
        ("enabled", "enabled", "some-secret-name", True),
        ("enabled", "enabled", "", True),
        ("disabled", "disabled", "some-secret-name", False),
        ("disabled", "disabled", "", False),
    ],
)
def test_build_runtime_insecure_registries(
    monkeypatch, pull_mode, push_mode, secret, flags_expected
):
    _patch_k8s_helper(monkeypatch)
    mlrun.mlconf.httpdb.builder.docker_registry = "registry.hub.docker.com/username"
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )

    insecure_flags = {"--insecure", "--insecure-pull"}
    mlrun.mlconf.httpdb.builder.insecure_pull_registry_mode = pull_mode
    mlrun.mlconf.httpdb.builder.insecure_push_registry_mode = push_mode
    mlrun.mlconf.httpdb.builder.docker_registry_secret = secret
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    assert (
        insecure_flags.issubset(
            set(
                mlrun.builder.get_k8s_helper()
                .create_pod.call_args[0][0]
                .pod.spec.containers[0]
                .args
            )
        )
        == flags_expected
    )


def test_build_runtime_target_image(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    registry = "registry.hub.docker.com/username"
    mlrun.mlconf.httpdb.builder.docker_registry = registry
    mlrun.mlconf.httpdb.builder.function_target_image_name_prefix_template = (
        "my-cool-prefix-{project}-{name}"
    )
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )
    image_name_prefix = (
        mlrun.mlconf.httpdb.builder.function_target_image_name_prefix_template.format(
            project=function.metadata.project, name=function.metadata.name
        )
    )

    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )

    # assert the default target image
    target_image = _get_target_image_from_create_pod_mock()
    assert target_image == f"{registry}/{image_name_prefix}:{function.metadata.tag}"

    # assert we can override the target image as long as we stick to the prefix
    function.spec.build.image = (
        f"{registry}/{image_name_prefix}-some-addition:{function.metadata.tag}"
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    target_image = _get_target_image_from_create_pod_mock()
    assert target_image == function.spec.build.image

    # assert the same with the registry enrich prefix
    # assert we can override the target image as long as we stick to the prefix
    function.spec.build.image = (
        f"{mlrun.builder.IMAGE_NAME_ENRICH_REGISTRY_PREFIX}username"
        f"/{image_name_prefix}-some-addition:{function.metadata.tag}"
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    target_image = _get_target_image_from_create_pod_mock()
    assert (
        target_image
        == f"{registry}/{image_name_prefix}-some-addition:{function.metadata.tag}"
    )

    # assert it raises if we don't stick to the prefix
    for invalid_image in [
        f"{mlrun.builder.IMAGE_NAME_ENRICH_REGISTRY_PREFIX}username/without-prefix:{function.metadata.tag}",
        f"{registry}/without-prefix:{function.metadata.tag}",
    ]:
        function.spec.build.image = invalid_image
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            mlrun.builder.build_runtime(
                mlrun.api.schemas.AuthInfo(),
                function,
            )

    # assert if we can not-stick to the regex if it's a different registry
    function.spec.build.image = (
        f"registry.hub.docker.com/some-other-username/image-not-by-prefix"
        f":{function.metadata.tag}"
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    target_image = _get_target_image_from_create_pod_mock()
    assert target_image == function.spec.build.image


def test_build_runtime_use_default_node_selector(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    mlrun.mlconf.httpdb.builder.docker_registry = "registry.hub.docker.com/username"
    node_selector = {
        "label-1": "val1",
        "label-2": "val2",
    }
    mlrun.mlconf.default_function_node_selector = base64.b64encode(
        json.dumps(node_selector).encode("utf-8")
    )
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().node_selector, node_selector, ignore_order=True
        )
        == {}
    )


def test_function_build_with_attributes_from_spec(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    mlrun.mlconf.httpdb.builder.docker_registry = "registry.hub.docker.com/username"
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )
    node_selector = {
        "label-1": "val1",
        "label-2": "val2",
    }
    node_name = "node_test"
    priority_class_name = "test-priority"

    function.spec.node_name = node_name
    function.spec.node_selector = node_selector
    function.spec.priority_class_name = priority_class_name
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().node_name, node_name, ignore_order=True
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().node_selector, node_selector, ignore_order=True
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().priority_class_name,
            priority_class_name,
            ignore_order=True,
        )
        == {}
    )


def test_function_build_with_default_requests(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    mlrun.mlconf.httpdb.builder.docker_registry = "registry.hub.docker.com/username"
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    expected_resources = {"requests": {}}
    # assert that both limits requirements and gpu requests are not defined
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().containers[0].resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )
    mlrun.mlconf.default_function_pod_resources.requests = {
        "cpu": "25m",
        "memory": "1m",
        "gpu": None,
    }
    expected_resources = {"requests": {"cpu": "25m", "memory": "1m"}}

    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().containers[0].resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )

    mlrun.mlconf.default_function_pod_resources = {
        "requests": {
            "cpu": "25m",
            "memory": "1m",
            "gpu": 2,
        },
        "limits": {
            "cpu": "1",
            "memory": "1G",
            "gpu": 2,
        },
    }
    expected_resources = {"requests": {"cpu": "25m", "memory": "1m"}}

    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    assert (
        deepdiff.DeepDiff(
            _create_pod_mock_pod_spec().containers[0].resources,
            expected_resources,
            ignore_order=True,
        )
        == {}
    )


def test_resolve_mlrun_install_command():
    pip_command = "python -m pip install"
    cases = [
        {
            "test_description": "when mlrun_version_specifier configured, expected to install mlrun_version_specifier",
            "mlrun_version_specifier": "mlrun[complete] @ git+https://github.com/mlrun/mlrun@v0.10.0",
            "client_version": "0.9.3",
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f"{pip_command} "
            f'"mlrun[complete] @ git+https://github.com/mlrun/mlrun@v0.10.0"',
        },
        {
            "test_description": "when mlrun_version_specifier is not configured and the server_mlrun_version_specifier"
            " is setup, expected to install the server_mlrun_version_specifier even if"
            " the client_version configured",
            "mlrun_version_specifier": None,
            "client_version": "0.9.3",
            "server_mlrun_version_specifier": "mlrun[complete]==0.10.0-server-version",
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete]==0.10.0-server-version"',
        },
        {
            "test_description": "when client_version is specified and stable and mlrun_version_specifier and"
            " server_mlrun_version_specifier are not configured,"
            " expected to install stable client version",
            "mlrun_version_specifier": None,
            "client_version": "0.9.3",
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete]==0.9.3"',
        },
        {
            "test_description": "when client_version is specified and unstable and mlrun_version_specifier and"
            " server_mlrun_version_specifier are not configured,"
            " expected to install mlrun/mlrun@development",
            "mlrun_version_specifier": None,
            "client_version": "unstable",
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete] @ git+'
            f'https://github.com/mlrun/mlrun@development"',
        },
        {
            "test_description": "when only the config.version is configured and unstable,"
            " expected to install mlrun/mlrun@development",
            "mlrun_version_specifier": None,
            "client_version": None,
            "server_mlrun_version_specifier": None,
            "version": "unstable",
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete] @ git+'
            f'https://github.com/mlrun/mlrun@development"',
        },
        {
            "test_description": "when only the config.version is configured and stable,"
            " expected to install config.version",
            "mlrun_version_specifier": None,
            "client_version": None,
            "server_mlrun_version_specifier": None,
            "version": "0.9.2",
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete]==0.9.2"',
        },
    ]
    for case in cases:
        config.httpdb.builder.mlrun_version_specifier = case.get(
            "server_mlrun_version_specifier"
        )
        config.package_path = case.get("package_path", "mlrun")
        if case.get("version") is not None:
            mlrun.utils.version.Version().get = unittest.mock.Mock(
                return_value={"version": case["version"]}
            )

        mlrun_version_specifier = case.get("mlrun_version_specifier")
        client_version = case.get("client_version")
        expected_result = case.get("expected_mlrun_install_command")

        result = mlrun.builder.resolve_mlrun_install_command(
            mlrun_version_specifier, client_version
        )
        assert (
            result == expected_result
        ), f"Test supposed to pass {case.get('test_description')}"


def test_build_runtime_ecr_with_ec2_iam_policy(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    mlrun.mlconf.httpdb.builder.docker_registry = (
        "aws_account_id.dkr.ecr.region.amazonaws.com"
    )
    project = mlrun.new_project("some-project")
    project.set_secrets(
        secrets={
            "AWS_ACCESS_KEY_ID": "test-a",
            "AWS_SECRET_ACCESS_KEY": "test-b",
        }
    )
    function = project.set_function(
        "hub://describe",
        name="some-function",
        kind="job",
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    pod_spec = _create_pod_mock_pod_spec()
    assert {"name": "AWS_SDK_LOAD_CONFIG", "value": "true", "value_from": None} in [
        env.to_dict() for env in pod_spec.containers[0].env
    ]

    # ensure both envvars are set without values so they wont interfere with the iam policy
    for env_name in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        assert {"name": env_name, "value": "", "value_from": None} in [
            env.to_dict() for env in pod_spec.containers[0].env
        ]

    # 1 for the AWS_SDK_LOAD_CONFIG=true
    # 2 for the AWS_ACCESS_KEY_ID="" and AWS_SECRET_ACCESS_KEY=""
    # 1 for the project secret
    # == 4
    assert len(pod_spec.containers[0].env) == 4, "expected 4 env items"

    assert len(pod_spec.init_containers) == 2
    for init_container in pod_spec.init_containers:
        if init_container.name == "create-repos":
            assert (
                "aws ecr create-repository --region region --repository-name mlrun/func-some-project-some-function"
                in init_container.args[1]
            )
            break
    else:
        pytest.fail("no create-repos init container")


def test_build_runtime_resolve_ecr_registry(monkeypatch):
    registry = "aws_account_id.dkr.ecr.us-east-2.amazonaws.com"
    for case in [
        {
            "name": "sanity",
            "repo": "some-repo",
            "tag": "latest",
        },
        {
            "name": "nested repo",
            "repo": "mlrun/some-repo",
            "tag": "1.2.3",
        },
        {
            "name": "no tag",
            "repo": "some-repo",
            "tag": None,
        },
    ]:
        _patch_k8s_helper(monkeypatch)
        mlrun.mlconf.httpdb.builder.docker_registry = ""
        function = mlrun.new_function(
            "some-function",
            "some-project",
            kind="job",
        )
        image = f"{registry}/{case.get('repo')}"
        if case.get("tag"):
            image += f":{case.get('tag')}"
        function.spec.build.image = image
        mlrun.builder.build_runtime(
            mlrun.api.schemas.AuthInfo(),
            function,
        )
        pod_spec = _create_pod_mock_pod_spec()
        for init_container in pod_spec.init_containers:
            if init_container.name == "create-repos":
                assert (
                    f"aws ecr create-repository --region us-east-2 --repository-name {case.get('repo')}"
                    in init_container.args[1]
                ), f"test case: {case.get('name')}"
                break
        else:
            pytest.fail(
                f"no create-repos init container, test case: {case.get('name')}"
            )


def test_build_runtime_ecr_with_aws_secret(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    mlrun.mlconf.httpdb.builder.docker_registry = (
        "aws_account_id.dkr.ecr.region.amazonaws.com"
    )
    mlrun.mlconf.httpdb.builder.docker_registry_secret = "aws-secret"
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    pod_spec = _create_pod_mock_pod_spec()
    assert "aws-secret" in [
        volume.secret.to_dict()["secret_name"]
        for volume in pod_spec.volumes
        if volume.secret
    ]
    aws_mount = {
        "mount_path": "/tmp",
        "mount_propagation": None,
        "name": "aws-secret",
        "read_only": None,
        "sub_path": None,
        "sub_path_expr": None,
    }
    assert aws_mount in [
        volume_mount.to_dict() for volume_mount in pod_spec.containers[0].volume_mounts
    ]

    aws_creds_location_env = {
        "name": "AWS_SHARED_CREDENTIALS_FILE",
        "value": "/tmp/credentials",
        "value_from": None,
    }
    assert aws_creds_location_env in [
        env.to_dict() for env in pod_spec.containers[0].env
    ]
    for init_container in pod_spec.init_containers:
        if init_container.name == "create-repos":
            assert aws_mount in [
                volume_mount.to_dict() for volume_mount in init_container.volume_mounts
            ]
            assert aws_creds_location_env in [
                env.to_dict() for env in init_container.env
            ]
            break
    else:
        pytest.fail("no create-repos init container")


def test_build_runtime_ecr_with_repository(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    repo_name = "my-repo"
    mlrun.mlconf.httpdb.builder.docker_registry = (
        f"aws_account_id.dkr.ecr.us-east-2.amazonaws.com/{repo_name}"
    )
    mlrun.mlconf.httpdb.builder.docker_registry_secret = "aws-secret"
    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
        requirements=["some-package"],
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    pod_spec = _create_pod_mock_pod_spec()

    for init_container in pod_spec.init_containers:
        if init_container.name == "create-repos":
            assert (
                f"aws ecr create-repository --region us-east-2 --repository-name "
                f"{repo_name}/func-some-project-some-function" in init_container.args[1]
            )
            break
    else:
        pytest.fail("no create-repos init container")


@pytest.mark.parametrize(
    "image_target,registry,default_repository,expected_dest",
    [
        (
            "test-image",
            None,
            None,
            "test-image",
        ),
        (
            "test-image",
            "test-registry",
            None,
            "test-registry/test-image",
        ),
        (
            "test-image",
            "test-registry/test-repository",
            None,
            "test-registry/test-repository/test-image",
        ),
        (
            ".test-image",
            None,
            "default.docker.registry/default-repository",
            "default.docker.registry/default-repository/test-image",
        ),
        (
            ".default-repository/test-image",
            None,
            "default.docker.registry/default-repository",
            "default.docker.registry/default-repository/test-image",
        ),
        (
            ".test-image",
            None,
            "default.docker.registry",
            "default.docker.registry/test-image",
        ),
    ],
)
def test_resolve_image_dest(image_target, registry, default_repository, expected_dest):
    docker_registry_secret = "default-docker-registry-secret"
    config.httpdb.builder.docker_registry = default_repository
    config.httpdb.builder.docker_registry_secret = docker_registry_secret

    image_target, _ = mlrun.builder._resolve_image_target_and_registry_secret(
        image_target, registry
    )
    assert image_target == expected_dest


@pytest.mark.parametrize(
    "image_target,registry,secret_name,default_secret_name,expected_secret_name",
    [
        (
            "test-image",
            None,
            None,
            "default-secret-name",
            None,
        ),
        (
            "test-image",
            None,
            "test-secret-name",
            "default-secret-name",
            "test-secret-name",
        ),
        (
            "test-image",
            "test-registry",
            None,
            "default-secret-name",
            None,
        ),
        (
            "test-image",
            "test-registry",
            "test-secret-name",
            "default-secret-name",
            "test-secret-name",
        ),
        (
            ".test-image",
            None,
            None,
            "default-secret-name",
            "default-secret-name",
        ),
        (
            ".test-image",
            None,
            "test-secret-name",
            "default-secret-name",
            "test-secret-name",
        ),
        (
            ".test-image",
            None,
            "test-secret-name",
            None,
            "test-secret-name",
        ),
        (
            ".test-image",
            None,
            None,
            None,
            None,
        ),
    ],
)
def test_resolve_registry_secret(
    image_target, registry, secret_name, default_secret_name, expected_secret_name
):
    docker_registry = "default.docker.registry/default-repository"
    config.httpdb.builder.docker_registry = docker_registry
    config.httpdb.builder.docker_registry_secret = default_secret_name

    _, secret_name = mlrun.builder._resolve_image_target_and_registry_secret(
        image_target, registry, secret_name
    )
    assert secret_name == expected_secret_name


def test_kaniko_pod_spec_default_service_account_enrichment(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    docker_registry = "default.docker.registry/default-repository"
    config.httpdb.builder.docker_registry = docker_registry

    service_account = "my-service-account"
    _mock_default_service_account(monkeypatch, service_account)

    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    pod_spec = _create_pod_mock_pod_spec()
    assert pod_spec.service_account == service_account


def test_kaniko_pod_spec_user_service_account_enrichment(monkeypatch):
    _patch_k8s_helper(monkeypatch)
    docker_registry = "default.docker.registry/default-repository"
    config.httpdb.builder.docker_registry = docker_registry

    _mock_default_service_account(monkeypatch, "my-default-service-account")

    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
    )
    service_account = "my-actual-sa"
    function.spec.service_account = service_account
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    pod_spec = _create_pod_mock_pod_spec()
    assert pod_spec.service_account == service_account


@pytest.mark.parametrize(
    "clone_target_dir,expected_workdir",
    [
        (None, r"WORKDIR .*\/tmp.*\/mlrun"),
        ("", r"WORKDIR .*\/tmp.*\/mlrun"),
        ("./path/to/code", r"WORKDIR .*\/tmp.*\/mlrun\/path\/to\/code"),
        ("rel_path", r"WORKDIR .*\/tmp.*\/mlrun\/rel_path"),
        ("/some/workdir", r"WORKDIR \/some\/workdir"),
    ],
)
def test_builder_workdir(monkeypatch, clone_target_dir, expected_workdir):
    _patch_k8s_helper(monkeypatch)
    mlrun.builder.make_kaniko_pod = unittest.mock.MagicMock()
    docker_registry = "default.docker.registry/default-repository"
    config.httpdb.builder.docker_registry = docker_registry

    function = mlrun.new_function(
        "some-function",
        "some-project",
        "some-tag",
        image="mlrun/mlrun",
        kind="job",
    )
    if clone_target_dir is not None:
        function.spec.clone_target_dir = clone_target_dir
    function.spec.build.source = "some-source.tgz"
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        function,
    )
    dockerfile = mlrun.builder.make_kaniko_pod.call_args[1]["dockertext"]
    dockerfile_lines = dockerfile.splitlines()
    expected_workdir_re = re.compile(expected_workdir)
    assert expected_workdir_re.match(dockerfile_lines[1])


def _get_target_image_from_create_pod_mock():
    return _create_pod_mock_pod_spec().containers[0].args[5]


def _create_pod_mock_pod_spec():
    return mlrun.builder.get_k8s_helper().create_pod.call_args[0][0].pod.spec


def _patch_k8s_helper(monkeypatch):
    get_k8s_helper_mock = unittest.mock.Mock()
    get_k8s_helper_mock.create_pod = unittest.mock.Mock(
        side_effect=lambda pod: (pod, "some-namespace")
    )
    get_k8s_helper_mock.get_project_secret_name = unittest.mock.Mock(
        side_effect=lambda name: "name"
    )
    get_k8s_helper_mock.get_project_secret_keys = unittest.mock.Mock(
        side_effect=lambda project, filter_internal: ["KEY"]
    )
    get_k8s_helper_mock.get_project_secret_data = unittest.mock.Mock(
        side_effect=lambda project, keys: {"KEY": "val"}
    )
    monkeypatch.setattr(
        mlrun.builder, "get_k8s_helper", lambda *args, **kwargs: get_k8s_helper_mock
    )
    monkeypatch.setattr(
        mlrun.k8s_utils, "get_k8s_helper", lambda *args, **kwargs: get_k8s_helper_mock
    )
    monkeypatch.setattr(
        mlrun.api.utils.singletons.k8s,
        "get_k8s",
        lambda *args, **kwargs: get_k8s_helper_mock,
    )


def _mock_default_service_account(monkeypatch, service_account):
    resolve_project_default_service_account_mock = unittest.mock.MagicMock()
    resolve_project_default_service_account_mock.return_value = (
        [],
        service_account,
    )
    monkeypatch.setattr(
        mlrun.api.api.utils,
        "resolve_project_default_service_account",
        resolve_project_default_service_account_mock,
    )
