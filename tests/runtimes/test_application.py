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

import base64
import pathlib
import unittest.mock

import pytest

import mlrun
import mlrun.common.constants
import mlrun.common.schemas
import mlrun.runtimes
import mlrun.utils
from mlrun.common.runtimes.constants import ProbeTimeConfig, ProbeType

assets_path = pathlib.Path(__file__).absolute().parent / "assets"


@pytest.fixture
def igz_version_mock():
    """Application runtime uses access key api gateway which requires igz version >= 3.5.5,
    so we need to mock the igz version to be 3.6.0 to pass the validation in the tests."""
    original_igz_version = mlrun.mlconf.igz_version
    mlrun.mlconf.igz_version = "3.6.0"
    yield
    mlrun.mlconf.igz_version = original_igz_version


def test_ensure_reverse_proxy_configurations():
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )
    mlrun.runtimes.ApplicationRuntime._ensure_reverse_proxy_configurations(fn)
    assert fn.kind == mlrun.runtimes.RuntimeKinds.application
    assert fn.spec.image == "mlrun/mlrun"
    assert fn.metadata.name == "application-test"
    assert fn.spec.min_replicas == 1
    assert fn.spec.max_replicas == 1
    _assert_function_code(fn)
    _assert_function_handler(fn)


def test_ensure_basic_credentials_configuration():
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fn.create_api_gateway(
            name="api-gateway",
            authentication_mode=mlrun.common.schemas.APIGatewayAuthenticationMode.basic,
        )


def test_create_application_runtime_with_command(rundb_mock, igz_version_mock):
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun", command="echo"
    )
    fn.deploy()
    assert fn.spec.config["spec.sidecars"][0]["command"] == ["echo"]
    assert fn.kind == mlrun.runtimes.RuntimeKinds.application
    assert fn.status.application_image == "mlrun/mlrun"
    assert fn.metadata.name == "application-test"
    _assert_function_code(fn)
    _assert_function_handler(fn)


def test_create_application_runtime_many_ports(rundb_mock, igz_version_mock):
    # deploy with default value
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun", command="echo"
    )
    # both should be the same
    assert fn.spec.internal_application_port == 8050
    assert fn.spec.application_ports == [8050]

    # should replace both application_ports and internal application port
    fn.with_sidecar("echo", command="echo", ports=[80, 22])
    fn.deploy()
    assert fn.spec.application_ports == [80, 22]

    # should reset internal application port and reorder application ports
    fn.spec.internal_application_port = 22
    assert fn.spec.application_ports == [22, 80]
    assert fn.spec.internal_application_port == 22


def test_create_application_runtime_multiple_ports_different_nuclio_versions(
    rundb_mock,
):
    # Test multiple ports with different nuclio versions
    for nuclio_version in ["1.14.13", "1.14.14", "1.14.15"]:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(mlrun.mlconf, "nuclio_version", nuclio_version)
            fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
                "application-test",
                kind="application",
                image="mlrun/mlrun",
                command="echo",
            )
            if nuclio_version >= "1.14.14":
                fn.with_sidecar("echo", command="echo", ports=[80, 22])
                fn.deploy()
                assert fn.spec.application_ports == [80, 22]
            else:
                with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
                    fn.with_sidecar("echo", command="echo", ports=[80, 22])
                with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
                    fn.spec.application_ports = [22, 80]


def test_application_runtime_update_port(rundb_mock, igz_version_mock):
    # deploy with default value
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun", command="echo"
    )
    # both should be the same
    assert fn.spec.internal_application_port == 8050
    assert fn.spec.application_ports == [8050]

    # simulate the case where the user updates the application port (without setting application ports)

    fn.set_internal_application_port(5000)

    assert fn.spec.internal_application_port == 5000
    assert fn.spec.application_ports == [5000]


def test_deploy_application_runtime(rundb_mock, igz_version_mock):
    image = "my/web-app:latest"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image=image
    )
    fn.deploy()
    _assert_application_post_deploy_spec(fn, image)


def test_consecutive_deploy_application_runtime(rundb_mock, igz_version_mock):
    image = "my/web-app:latest"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image=image
    )
    fn.deploy()
    _assert_application_post_deploy_spec(fn, image)

    fn.deploy()
    _assert_application_post_deploy_spec(fn, image)

    # Change the image and deploy again
    image = "another/web-app:latest"
    fn.spec.image = image
    fn.deploy()

    # Ensure the image is updated
    _assert_application_post_deploy_spec(fn, image)


@pytest.mark.parametrize(
    "sidecars, expected_error_message",
    [
        ([], "Application spec must include a sidecar configuration"),
        ([{}], "Application sidecar spec must include an image"),
        (
            [{"image": "my/web-app:latest"}],
            "Application sidecar spec must include at least one port",
        ),
        (
            [{"image": "my/web-app:latest", "ports": [{}]}],
            "Application sidecar port spec must include a containerPort",
        ),
        (
            [{"image": "my/web-app:latest", "ports": [{"containerPort": 8050}]}],
            "Application sidecar port spec must include a name",
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8050, "name": "sidecar-port"}],
                    "args": ["--help"],
                }
            ],
            "Application sidecar spec must include a command if args are provided",
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8050, "name": "sidecar-port"}],
                }
            ],
            None,
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8050, "name": "sidecar-port"}],
                    "command": ["echo"],
                }
            ],
            None,
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8050, "name": "sidecar-port"}],
                    "command": ["echo"],
                    "args": ["--help"],
                }
            ],
            None,
        ),
    ],
)
def test_pre_deploy_validation(sidecars, expected_error_message):
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="my/web-app:latest"
    )
    fn.spec.config["spec.sidecars"] = sidecars
    if expected_error_message:
        with pytest.raises(mlrun.errors.MLRunBadRequestError) as exc:
            fn.pre_deploy_validation()
        assert expected_error_message in str(exc.value)
    else:
        fn.pre_deploy_validation()


def test_image_enriched_on_build_application_image(remote_builder_mock):
    project = "test-project"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        project=project,
    )
    fn._build_application_image()
    assert fn.spec.image == f".mlrun/func-{project}-application-test:latest"
    assert fn.status.state == mlrun.common.schemas.FunctionState.ready


def test_application_image_build(remote_builder_mock, igz_version_mock):
    project = "test-project"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        requirements=["mock"],
        project=project,
    )
    assert fn.requires_build()
    fn.deploy()
    _assert_application_post_deploy_spec(
        fn, f".mlrun/func-{project}-application-test:latest"
    )


@pytest.mark.parametrize(
    "source,commands,requirements,expected",
    [
        # No source, no commands, no requirements - no build needed
        (None, None, None, False),
        # store:// URI alone doesn't require build (init container handles it)
        ("store://artifacts/project/my-source", None, None, False),
        # store:// URI with commands requires build
        ("store://artifacts/project/my-source", ["pip install foo"], None, True),
        # store:// URI with requirements requires build
        ("store://artifacts/project/my-source", None, ["pandas"], True),
        # Remote source without load_source_on_run requires build
        ("https://github.com/repo.git", None, None, True),
        # Commands alone require build
        (None, ["pip install foo"], None, True),
        # Requirements alone require build
        (None, None, ["pandas"], True),
    ],
)
def test_application_requires_build(source, commands, requirements, expected):
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
    )
    if source:
        fn.spec.build.source = source
    if commands:
        fn.spec.build.commands = commands
    if requirements:
        fn.spec.build.requirements = requirements

    assert fn.requires_build() == expected


def test_application_default_api_gateway(rundb_mock, igz_version_mock):
    function_name = "application-test"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        function_name,
        kind="application",
        image="mlrun/mlrun",
    )
    fn.deploy()
    api_gateway = fn.status.api_gateway
    assert api_gateway is not None
    assert api_gateway.name == function_name
    assert len(api_gateway.spec.functions) == 1
    assert function_name in api_gateway.spec.functions[0]


def test_application_disable_default_api_gateway(rundb_mock, igz_version_mock):
    function_name = "application-test"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        function_name,
        kind="application",
        image="mlrun/mlrun",
    )
    fn.deploy(create_default_api_gateway=False)
    assert fn.status.api_gateway is None

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=f"Non-default API gateway cannot use the default gateway name, name='{fn.metadata.name}'.",
    ):
        fn.create_api_gateway(name=fn.resolve_default_api_gateway_name())

    url = fn.create_api_gateway(
        "my-gateway",
        authentication_mode=mlrun.common.schemas.APIGatewayAuthenticationMode.basic,
        authentication_creds=("username", "password"),
    )

    assert url == f"https://{fn.status.external_invocation_urls[0]}"


def test_application_api_gateway_ssl_redirect(rundb_mock, igz_version_mock):
    function: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
    )
    # ssl redirect is enabled by default when running in iguazio
    function.deploy()

    ssl_redirect_annotation = "nginx.ingress.kubernetes.io/force-ssl-redirect"
    api_gateway = function.status.api_gateway
    assert api_gateway is not None
    assert ssl_redirect_annotation in api_gateway.metadata.annotations
    assert api_gateway.metadata.annotations[ssl_redirect_annotation] == "true"


@pytest.mark.parametrize("gateway_timeout", [50, None, 0])
def test_application_api_gateway_timeout_annotations(rundb_mock, gateway_timeout):
    function: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
    )

    function.deploy(create_default_api_gateway=False)
    function.create_api_gateway(
        name="my-gateway", gateway_timeout=gateway_timeout, set_as_default=True
    )

    annotations = [
        "nginx.ingress.kubernetes.io/proxy-connect-timeout",
        "nginx.ingress.kubernetes.io/proxy-read-timeout",
        "nginx.ingress.kubernetes.io/proxy-send-timeout",
    ]
    api_gateway = function.status.api_gateway
    assert api_gateway is not None
    for annotation in annotations:
        if gateway_timeout:
            annotation_value = api_gateway.metadata.annotations.get(annotation)
            assert annotation_value == str(gateway_timeout)
            assert int(annotation_value) == gateway_timeout
            assert annotation not in function.metadata.annotations
        else:
            assert annotation not in api_gateway.metadata.annotations
            assert annotation not in function.metadata.annotations


def test_application_runtime_resources(rundb_mock, igz_version_mock):
    image = "my/web-app:latest"
    app_name = "application-test"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        app_name,
        kind="application",
        image=image,
    )
    cpu_requests = "0.7"
    memory_requests = "1.2Gi"
    cpu_limits = "2"
    memory_limits = "4Gi"
    fn.with_requests(cpu=cpu_requests, mem=memory_requests)
    fn.with_limits(cpu=cpu_limits, mem=memory_limits)

    fn.deploy()

    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": f"{app_name}-sidecar",
            "ports": [
                {
                    "containerPort": 8050,
                    "name": "application-t-0",
                    "protocol": "TCP",
                }
            ],
            "resources": {
                "requests": {"cpu": cpu_requests, "memory": memory_requests},
                "limits": {"cpu": cpu_limits, "memory": memory_limits},
            },
        }
    ]

    # assert the resources for the function itself remain the defaults
    assert fn.spec.resources == {}


def test_deploy_reverse_proxy_image(rundb_mock, igz_version_mock):
    mlrun.get_or_create_project("test-deploy-reverse-proxy", allow_cross_project=True)
    mlrun.runtimes.ApplicationRuntime.deploy_reverse_proxy_image()
    assert mlrun.runtimes.ApplicationRuntime.reverse_proxy_image


def _assert_function_code(fn, file_path=None):
    file_path = (
        file_path or mlrun.runtimes.ApplicationRuntime.get_filename_and_handler()[0]
    )
    expected_code = pathlib.Path(file_path).read_text()
    expected_code_encoded = base64.b64encode(expected_code.encode("utf-8")).decode(
        "utf-8"
    )
    assert fn.spec.build.functionSourceCode == expected_code_encoded


def _assert_function_handler(fn):
    (
        filepath,
        expected_handler,
    ) = mlrun.runtimes.ApplicationRuntime.get_filename_and_handler()
    expected_filename = pathlib.Path(filepath).name
    expected_module = mlrun.utils.normalize_name(expected_filename.split(".")[0])
    # '-nuclio' suffix is added by nuclio-jupyter
    expected_function_handler = f"{expected_module}-nuclio:{expected_handler}"
    assert fn.spec.function_handler == expected_function_handler


def _assert_application_post_deploy_spec(fn, image):
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": "application-test-sidecar",
            "ports": [
                {
                    "containerPort": 8050,
                    "name": "application-t-0",
                    "protocol": "TCP",
                }
            ],
        }
    ]
    assert fn.get_env("SIDECAR_PORT") == "8050"
    assert fn.status.application_image == image
    assert not fn.spec.image


def test_set_probe_readiness():
    """Test setting a readiness probe with HTTP configuration"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="readiness",
        http_path="/api/healthz",
        initial_delay_seconds=10,
        period_seconds=20,
        timeout_seconds=30,
        failure_threshold=40,
    )

    sidecar = fn._get_sidecar()
    assert sidecar is not None
    assert ProbeType.READINESS.key in sidecar
    probe = sidecar[ProbeType.READINESS.key]
    assert probe["httpGet"]["path"] == "/api/healthz"
    assert probe["httpGet"]["scheme"] == "HTTP"
    assert probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 10
    assert probe[ProbeTimeConfig.PERIOD_SECONDS.value] == 20
    assert probe[ProbeTimeConfig.TIMEOUT_SECONDS.value] == 30
    assert probe[ProbeTimeConfig.FAILURE_THRESHOLD.value] == 40


def test_set_probe_liveness_with_port():
    """Test setting a liveness probe with explicit port"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="liveness",
        http_path="/health",
        http_port=8080,
        http_scheme="HTTPS",
        initial_delay_seconds=15,
        period_seconds=10,
        failure_threshold=3,
        timeout_seconds=5,
    )

    sidecar = fn._get_sidecar()
    assert sidecar is not None
    assert ProbeType.LIVENESS.key in sidecar
    probe = sidecar[ProbeType.LIVENESS.key]
    assert probe["httpGet"]["path"] == "/health"
    assert probe["httpGet"]["port"] == 8080
    assert probe["httpGet"]["scheme"] == "HTTPS"
    assert probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 15
    assert probe[ProbeTimeConfig.PERIOD_SECONDS.value] == 10
    assert probe[ProbeTimeConfig.FAILURE_THRESHOLD.value] == 3
    assert probe[ProbeTimeConfig.TIMEOUT_SECONDS.value] == 5


def test_set_probe_with_config_override():
    """Test that explicit parameters override config dict values"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="startup",
        initial_delay_seconds=15,
        config={
            "tcpSocket": {"port": 8080},
            "initialDelaySeconds": 20,
            "periodSeconds": 30,
        },
    )

    sidecar = fn._get_sidecar()
    assert sidecar is not None
    assert ProbeType.STARTUP.key in sidecar
    probe = sidecar[ProbeType.STARTUP.key]
    assert probe["tcpSocket"]["port"] == 8080
    assert probe[ProbeTimeConfig.PERIOD_SECONDS.value] == 30
    assert probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 15


def test_set_probe_replace_existing():
    """Test that calling set_probe again replaces the existing configuration"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="readiness",
        http_path="/old/path",
        initial_delay_seconds=10,
        failure_threshold=5,
    )

    fn.set_probe(
        type="readiness",
        http_path="/new/path",
        initial_delay_seconds=20,
    )

    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    probe = sidecar[ProbeType.READINESS.key]
    assert probe["httpGet"]["path"] == "/new/path"
    assert probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 20
    assert ProbeTimeConfig.FAILURE_THRESHOLD.value not in probe


def test_set_probe_invalid_type():
    """Test that invalid probe type raises ValueError"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    with pytest.raises(ValueError, match="Invalid probe type"):
        fn.set_probe(type="invalid_type")

    with pytest.raises(ValueError, match="Invalid probe type"):
        fn.set_probe(type=None)

    with pytest.raises(ValueError, match="Invalid probe type"):
        fn.set_probe(type="")


def test_set_probe_empty_value():
    """Test that empty values set raise an error"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    with pytest.raises(
        ValueError,
        match="Empty probe configuration: at least one parameter must be set",
    ):
        fn.set_probe(type="readiness")


def test_set_probe_string_type():
    """Test that string probe type is accepted and converted"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="readiness",
        http_path="/health",
    )

    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar


def test_set_probe_no_http_path():
    """Test setting probe without HTTP path (only timing parameters)"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="readiness",
        initial_delay_seconds=10,
        period_seconds=5,
    )

    sidecar = fn._get_sidecar()
    probe = sidecar[ProbeType.READINESS.key]
    assert "httpGet" not in probe
    assert probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 10
    assert probe[ProbeTimeConfig.PERIOD_SECONDS.value] == 5


def test_set_probe_multiple_probes():
    """Test setting multiple different probe types"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.set_probe(
        type="readiness",
        http_path="/readiness",
        initial_delay_seconds=10,
        period_seconds=5,
    )
    fn.set_probe(
        type="liveness",
        http_path="/liveness",
        initial_delay_seconds=15,
        period_seconds=10,
        timeout_seconds=3,
    )

    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    readiness_probe = sidecar[ProbeType.READINESS.key]
    assert readiness_probe["httpGet"]["path"] == "/readiness"
    assert readiness_probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 10
    assert readiness_probe[ProbeTimeConfig.PERIOD_SECONDS.value] == 5

    assert ProbeType.LIVENESS.key in sidecar
    liveness_probe = sidecar[ProbeType.LIVENESS.key]
    assert liveness_probe["httpGet"]["path"] == "/liveness"
    assert liveness_probe[ProbeTimeConfig.INITIAL_DELAY_SECONDS.value] == 15
    assert liveness_probe[ProbeTimeConfig.PERIOD_SECONDS.value] == 10
    assert liveness_probe[ProbeTimeConfig.TIMEOUT_SECONDS.value] == 3


def test_delete_probe():
    """Test deleting a probe configuration"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "scheme": "HTTP",
                                }
                            },
                        }
                    ]
                }
            }
        },
    )

    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    fn.delete_probe(type="readiness")
    assert ProbeType.READINESS.key not in sidecar


def test_delete_probe_nonexistent():
    """Test deleting a probe that doesn't exist (should not raise error)"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    fn.delete_probe(type="liveness")
    sidecar = fn._get_sidecar()
    assert sidecar is None

    fn.spec.config["spec.sidecars"] = [
        {
            "name": "application-test-sidecar",
            "readinessProbe": {
                "httpGet": {
                    "path": "/readiness",
                    "scheme": "HTTP",
                }
            },
        }
    ]
    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    fn.delete_probe(type="readiness")
    assert ProbeType.READINESS.key not in sidecar


def test_delete_probe_invalid_type():
    """Test that invalid probe type raises ValueError"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    with pytest.raises(ValueError, match="Invalid probe type"):
        fn.delete_probe(type="invalid_type")


def test_delete_probe_multiple_probes():
    """Test deleting one probe while others remain"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/readiness",
                                    "scheme": "HTTP",
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/liveness",
                                    "scheme": "HTTP",
                                }
                            },
                            "startupProbe": {
                                "httpGet": {
                                    "path": "/startup",
                                    "scheme": "HTTP",
                                }
                            },
                        }
                    ]
                }
            }
        },
    )

    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    assert ProbeType.LIVENESS.key in sidecar
    assert ProbeType.STARTUP.key in sidecar
    fn.delete_probe(type="readiness")
    assert ProbeType.READINESS.key not in sidecar
    assert ProbeType.LIVENESS.key in sidecar
    assert ProbeType.STARTUP.key in sidecar


def test_enrich_sidecar_probe_ports_without_port():
    """Test enriching HTTP probe without port when internal_application_port is set"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "internal_application_port": 8080,
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "scheme": "HTTP",
                                }
                            },
                        }
                    ]
                },
            }
        },
    )

    fn._enrich_sidecar_probe_ports()

    sidecar = fn._get_sidecar()
    assert sidecar is not None
    assert ProbeType.READINESS.key in sidecar
    probe = sidecar[ProbeType.READINESS.key]
    assert probe["httpGet"]["port"] == 8080
    assert probe["httpGet"]["path"] == "/health"


def test_enrich_sidecar_probe_ports_with_existing_port():
    """Test that probes with existing ports are not enriched"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "internal_application_port": 8080,
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 9090,
                                    "scheme": "HTTP",
                                }
                            },
                        }
                    ]
                },
            }
        },
    )

    fn._enrich_sidecar_probe_ports()
    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    probe = sidecar[ProbeType.READINESS.key]
    assert probe["httpGet"]["port"] == 9090


def test_enrich_sidecar_probe_ports_multiple_probes():
    """Test enriching multiple probes, some with ports, some without"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "internal_application_port": 8080,
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/readiness",
                                    "scheme": "HTTP",
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/liveness",
                                    "port": 9090,
                                    "scheme": "HTTP",
                                }
                            },
                            "startupProbe": {
                                "httpGet": {
                                    "path": "/startup",
                                    "scheme": "HTTP",
                                }
                            },
                        }
                    ]
                },
            }
        },
    )

    fn._enrich_sidecar_probe_ports()

    sidecar = fn._get_sidecar()
    assert ProbeType.READINESS.key in sidecar
    readiness_probe = sidecar[ProbeType.READINESS.key]
    assert readiness_probe["httpGet"]["port"] == 8080
    assert ProbeType.LIVENESS.key in sidecar
    liveness_probe = sidecar[ProbeType.LIVENESS.key]
    assert liveness_probe["httpGet"]["port"] == 9090
    assert ProbeType.STARTUP.key in sidecar
    startup_probe = sidecar[ProbeType.STARTUP.key]
    assert startup_probe["httpGet"]["port"] == 8080


def test_enrich_sidecar_probe_ports_no_internal_port_error():
    """Test that error is raised when internal_application_port is not set and probe needs enrichment"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "scheme": "HTTP",
                                }
                            },
                        }
                    ]
                }
            }
        },
    )

    del fn.spec._internal_application_port
    with pytest.raises(AttributeError):
        fn._enrich_sidecar_probe_ports()


def test_enrich_sidecar_probe_ports_no_sidecar():
    """Test that method returns early when there's no sidecar"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "internal_application_port": 8080,
            }
        },
    )

    fn._enrich_sidecar_probe_ports()
    sidecar = fn._get_sidecar()
    assert sidecar is None


def test_enrich_sidecar_probe_ports_no_probes():
    """Test that method handles sidecar with no probes"""
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test",
        kind="application",
        image="mlrun/mlrun",
        runtime={
            "spec": {
                "internal_application_port": 8080,
                "config": {
                    "spec.sidecars": [
                        {
                            "name": "application-test-sidecar",
                        }
                    ]
                },
            }
        },
    )

    fn._enrich_sidecar_probe_ports()
    sidecar = fn._get_sidecar()
    assert sidecar is not None
    assert ProbeType.READINESS.key not in sidecar
    assert ProbeType.LIVENESS.key not in sidecar
    assert ProbeType.STARTUP.key not in sidecar


@pytest.mark.parametrize(
    "source,setup_file,expected",
    [
        ("local_file.py", True, True),
        ("directory_path", False, False),
        ("", False, False),
        ("store://artifacts/project/file", False, False),
        ("https://example.com/file.py", False, False),
        ("git://github.com/repo.git", False, False),
        ("/non/existent/path.py", False, False),
    ],
)
def test_is_single_local_file(tmp_path, source, setup_file, expected):
    # Test _is_single_local_file identifies local files vs remote/invalid sources.
    func_name = "application-test"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        func_name,
        kind="application",
        image="mlrun/mlrun",
    )

    if setup_file:
        # Create a temporary file for local file case
        file_path = tmp_path / source
        file_path.write_text("def handler(): pass")
        test_source = str(file_path)
    elif source == "directory_path":
        # Use tmp_path as directory
        test_source = str(tmp_path)
    else:
        test_source = source

    assert fn._is_single_local_file(test_source) is expected


def test_upload_source_as_artifact(tmp_path):
    # Test that local single file is uploaded as artifact
    func_name = "application-test"
    # Create a temporary source file
    source_file = tmp_path / "handler.py"
    source_file.write_text("def handler(): pass")

    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        func_name,
        kind="application",
        image="mlrun/mlrun",
        project="test-project",
    )
    fn.spec.build.source = str(source_file)

    mock_artifact = unittest.mock.MagicMock()
    mock_artifact.uri = "store://artifacts/test-project/application-test-source"

    mock_project = unittest.mock.MagicMock()
    mock_project.log_artifact.return_value = mock_artifact

    with unittest.mock.patch(
        "mlrun.get_or_create_project", return_value=mock_project
    ) as mock_get_project:
        original_path, artifact_uri = fn._upload_source_as_artifact()

    # Verify project was retrieved
    mock_get_project.assert_called_once_with("test-project")

    # Verify artifact was logged with correct parameters
    mock_project.log_artifact.assert_called_once_with(
        item="application-test-source",
        local_path=str(source_file),
        artifact_path=mlrun.common.constants.MLRUN_INTERNAL_ARTIFACT_PATH,
        upload=True,
        labels={
            mlrun.common.constants.MLRunInternalLabels.function_name: func_name,
            mlrun.common.constants.MLRunInternalLabels.system_generated: "true",
        },
    )

    # Verify source was swapped to artifact URI and original path is returned for restore
    assert (
        fn.spec.build.source == "store://artifacts/test-project/application-test-source"
    )
    assert original_path == str(source_file)
    assert artifact_uri == "store://artifacts/test-project/application-test-source"


@pytest.mark.parametrize(
    "source",
    [
        "store://artifacts/test-project/existing-artifact",
        "https://github.com/org/repo.git",
        "s3://bucket/path/file.py",
        "",
    ],
)
def test_upload_source_as_artifact_skip_non_local(source):
    # Test that non-local sources (store URIs, remote URLs) and empty path are not uploaded
    func_name = "application-test"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        func_name,
        kind="application",
        image="mlrun/mlrun",
        project="test-project",
    )
    fn.spec.build.source = source

    with unittest.mock.patch("mlrun.get_or_create_project") as mock_get_project:
        fn._upload_source_as_artifact()

    # Verify project was not called (upload skipped)
    mock_get_project.assert_not_called()

    # Verify source remains unchanged
    assert fn.spec.build.source == source


def test_upload_source_as_artifact_no_project_error():
    # Test that missing project raises an error
    func_name = "application-test"
    # Create a temporary source file
    with unittest.mock.patch("os.path.isfile", return_value=True):
        fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
            func_name,
            kind="application",
            image="mlrun/mlrun",
        )
        fn.metadata.project = None
        fn.spec.build.source = "/path/to/handler.py"

        with pytest.raises(
            mlrun.errors.MLRunMissingProjectError,
            match="Project is required to upload source as artifact",
        ):
            fn._upload_source_as_artifact()


def test_set_function_single_file_application(tmp_path):
    # Test that set_function with single .py file works for application runtime
    source_file = tmp_path / "handler.py"
    source_file.write_text("def handler(): pass")

    project = mlrun.get_or_create_project("test-proj", allow_cross_project=True)
    fn = project.set_function(
        str(source_file),
        name="my-app",
        kind="application",
        image="mlrun/mlrun",
    )

    assert fn.kind == "application"
    assert fn.metadata.project == "test-proj"
    assert fn.spec.build.source == str(source_file)


@pytest.mark.parametrize(
    "target_dir,should_succeed,error_match",
    [
        ("/my/custom/path", True, None),
        ("/another/path", True, None),
        ("", False, "target_dir is required"),
        ("relative/path", False, "must be an absolute path"),
    ],
)
def test_set_source_target(target_dir, should_succeed, error_match):
    """
    Test set_source_target with various valid and invalid inputs.

    Valid absolute paths should set spec.build.source_code_target_dir.
    Empty or relative paths should raise MLRunInvalidArgumentError.
    """
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.new_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )

    if should_succeed:
        fn.set_source_target(target_dir)
        assert fn.spec.build.source_code_target_dir == target_dir
    else:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match=error_match):
            fn.set_source_target(target_dir)
