import unittest.mock

import pytest

import mlrun
import mlrun.api.schemas
import mlrun.builder
import mlrun.k8s_utils
import mlrun.utils.version
from mlrun.config import config


def test_build_runtime_use_base_image_when_no_build():
    fn = mlrun.new_function("some-function", "some-project", "some-tag", kind="job")
    base_image = "mlrun/ml-models"
    fn.build_config(base_image=base_image)
    assert fn.spec.image == ""
    ready = mlrun.builder.build_runtime(mlrun.api.schemas.AuthInfo(), fn,)
    assert ready is True
    assert fn.spec.image == base_image


def test_build_runtime_use_image_when_no_build():
    image = "mlrun/ml-models"
    fn = mlrun.new_function(
        "some-function", "some-project", "some-tag", image=image, kind="job"
    )
    assert fn.spec.image == image
    ready = mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(), fn, with_mlrun=False,
    )
    assert ready is True
    assert fn.spec.image == image


def test_build_runtime_insecure_registries(monkeypatch):
    get_k8s_helper_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.builder, "get_k8s_helper", lambda *args, **kwargs: get_k8s_helper_mock
    )
    mlrun.builder.get_k8s_helper().create_pod = unittest.mock.Mock(
        side_effect=lambda pod: (pod, "some-namespace")
    )
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
    for case in [
        {
            "pull_mode": "auto",
            "push_mode": "auto",
            "secret": "",
            "flags_expected": True,
        },
        {
            "pull_mode": "auto",
            "push_mode": "auto",
            "secret": "some-secret-name",
            "flags_expected": False,
        },
        {
            "pull_mode": "enabled",
            "push_mode": "enabled",
            "secret": "some-secret-name",
            "flags_expected": True,
        },
        {
            "pull_mode": "enabled",
            "push_mode": "enabled",
            "secret": "",
            "flags_expected": True,
        },
        {
            "pull_mode": "disabled",
            "push_mode": "disabled",
            "secret": "some-secret-name",
            "flags_expected": False,
        },
        {
            "pull_mode": "disabled",
            "push_mode": "disabled",
            "secret": "",
            "flags_expected": False,
        },
    ]:
        mlrun.mlconf.httpdb.builder.insecure_pull_registry_mode = case["pull_mode"]
        mlrun.mlconf.httpdb.builder.insecure_push_registry_mode = case["push_mode"]
        mlrun.mlconf.httpdb.builder.docker_registry_secret = case["secret"]
        mlrun.builder.build_runtime(
            mlrun.api.schemas.AuthInfo(), function,
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
            == case["flags_expected"]
        )


def test_build_runtime_target_image(monkeypatch):
    get_k8s_helper_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.builder, "get_k8s_helper", lambda *args, **kwargs: get_k8s_helper_mock
    )
    mlrun.builder.get_k8s_helper().create_pod = unittest.mock.Mock(
        side_effect=lambda pod: (pod, "some-namespace")
    )
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
    image_name_prefix = mlrun.mlconf.httpdb.builder.function_target_image_name_prefix_template.format(
        project=function.metadata.project, name=function.metadata.name
    )

    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(), function,
    )

    # assert the default target image
    target_image = (
        mlrun.builder.get_k8s_helper()
        .create_pod.call_args[0][0]
        .pod.spec.containers[0]
        .args[5]
    )
    assert target_image == f"{registry}/{image_name_prefix}:{function.metadata.tag}"

    # assert we can override the target image as long as we stick to the prefix
    function.spec.build.image = (
        f"{registry}/{image_name_prefix}-some-addition:{function.metadata.tag}"
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(), function,
    )
    target_image = (
        mlrun.builder.get_k8s_helper()
        .create_pod.call_args[0][0]
        .pod.spec.containers[0]
        .args[5]
    )
    assert target_image == function.spec.build.image

    # assert the same with the registry enrich prefix
    # assert we can override the target image as long as we stick to the prefix
    function.spec.build.image = (
        f"{mlrun.builder.IMAGE_NAME_ENRICH_REGISTRY_PREFIX}username"
        f"/{image_name_prefix}-some-addition:{function.metadata.tag}"
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(), function,
    )
    target_image = (
        mlrun.builder.get_k8s_helper()
        .create_pod.call_args[0][0]
        .pod.spec.containers[0]
        .args[5]
    )
    assert (
        target_image
        == f"{registry}/{image_name_prefix}-some-addition:{function.metadata.tag}"
    )

    # assert it raises if we don't stick to the prefix
    for invalid_image in [
        f"{mlrun.builder.IMAGE_NAME_ENRICH_REGISTRY_PREFIX}username/without-prefix:{function.metadata.tag}"
        f"{registry}/without-prefix:{function.metadata.tag}"
    ]:
        function.spec.build.image = invalid_image
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            mlrun.builder.build_runtime(
                mlrun.api.schemas.AuthInfo(), function,
            )

    # assert if we can not-stick to the regex if it's a different registry
    function.spec.build.image = (
        f"registry.hub.docker.com/some-other-username/image-not-by-prefix"
        f":{function.metadata.tag}"
    )
    mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(), function,
    )
    target_image = (
        mlrun.builder.get_k8s_helper()
        .create_pod.call_args[0][0]
        .pod.spec.containers[0]
        .args[5]
    )
    assert target_image == function.spec.build.image


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
