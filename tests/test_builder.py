import mlrun
import mlrun.api.schemas
import mlrun.builder
from mlrun.config import config


def test_build_runtime_use_base_image_when_no_build():
    fn = mlrun.new_function("some-function", "some-project", "some-tag", kind="job")
    base_image = "mlrun/ml-models"
    fn.build_config(base_image=base_image)
    assert fn.spec.image == ""
    ready = mlrun.builder.build_runtime(
        mlrun.api.schemas.AuthInfo(),
        fn,
        with_mlrun=False,
        mlrun_version_specifier=None,
        skip_deployed=False,
        builder_env=None,
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
        mlrun_version_specifier=None,
        skip_deployed=False,
        builder_env=None,
    )
    assert ready is True
    assert fn.spec.image == image


def test_resolve_mlrun_install_command():
    pip_command = "python -m pip install"
    cases = [
        {
            "test_description": "when no version is specified, expected to install development branch",
            "mlrun_version_specifier": None,
            "client_version": None,
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f"{pip_command} "
            f'"mlrun[complete] @ git+https://github.com/mlrun/mlrun@development"',
        },
        {
            "test_description": "when only client_version is specified and stable, "
            "expected to install stable client version",
            "mlrun_version_specifier": None,
            "client_version": "0.9.3",
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete]==0.9.3"',
        },
        {
            "test_description": "when both mlrun_version_specifier and client_version configured,"
            " expected to install mlrun_version_specifier",
            "mlrun_version_specifier": "mlrun[complete] @ git+https://github.com/mlrun/mlrun@v0.10.0",
            "client_version": "0.9.3",
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f"{pip_command} "
            f'"mlrun[complete] @ git+https://github.com/mlrun/mlrun@v0.10.0"',
        },
        {
            "test_description": "when client_version and mlrun_version_specifier is setup in the builder configuration"
            " are specified, expected to install the builder configuration",
            "mlrun_version_specifier": None,
            "client_version": "0.9.3",
            "server_mlrun_version_specifier": "mlrun[complete]==0.10.0-server-version",
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete]==0.10.0-server-version"',
        },
        {
            "test_description": "when only client_version is specified and is unstable, "
            "expected to install mlrun/mlrun@development",
            "mlrun_version_specifier": None,
            "client_version": "unstable",
            "server_mlrun_version_specifier": None,
            "expected_mlrun_install_command": f'{pip_command} "mlrun[complete] @ git+'
            f'https://github.com/mlrun/mlru2n@development"',
        },
    ]
    for case in cases:
        config.httpdb.builder.mlrun_version_specifier = case.get(
            "server_mlrun_version_specifier"
        )
        config.package_path = case.get("package_path", "mlrun")
        config.version = case.get("version", "unstable")

        mlrun_version_specifier = case.get("mlrun_version_specifier")
        client_version = case.get("client_version")
        expected_result = case.get("expected_mlrun_install_command")

        result = mlrun.builder.resolve_mlrun_install_command(
            mlrun_version_specifier, client_version
        )
        assert (
            result == expected_result
        ), f"Test supposed to pass {case.get('test_description')}"
