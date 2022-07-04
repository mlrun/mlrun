import base64
import json
import unittest.mock

import deepdiff
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


def test_build_runtime_insecure_registries(monkeypatch):
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
            == case["flags_expected"]
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


def _get_target_image_from_create_pod_mock():
    return _create_pod_mock_pod_spec().containers[0].args[5]


def _create_pod_mock_pod_spec():
    return mlrun.builder.get_k8s_helper().create_pod.call_args[0][0].pod.spec


def _patch_k8s_helper(monkeypatch):
    get_k8s_helper_mock = unittest.mock.Mock()
    monkeypatch.setattr(
        mlrun.builder, "get_k8s_helper", lambda *args, **kwargs: get_k8s_helper_mock
    )
    mlrun.builder.get_k8s_helper().create_pod = unittest.mock.Mock(
        side_effect=lambda pod: (pod, "some-namespace")
    )
    mlrun.builder.get_k8s_helper().get_project_secret_name = unittest.mock.Mock(
        side_effect=lambda name: "name"
    )
    mlrun.builder.get_k8s_helper().get_project_secret_keys = unittest.mock.Mock(
        side_effect=lambda project, filter_internal: ["KEY"]
    )
    mlrun.builder.get_k8s_helper().get_project_secret_data = unittest.mock.Mock(
        side_effect=lambda project, keys: {"KEY": "val"}
    )
