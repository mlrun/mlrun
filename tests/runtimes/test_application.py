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
import pathlib

import pytest

import mlrun
import mlrun.common.schemas


def test_create_application_runtime():
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )
    assert fn.kind == mlrun.runtimes.RuntimeKinds.application
    assert fn.spec.image == "mlrun/mlrun"
    assert fn.metadata.name == "application-test"
    _assert_function_code(fn)
    _assert_function_handler(fn)


def test_create_application_runtime_with_command(rundb_mock):
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


def test_deploy_application_runtime(rundb_mock):
    image = "my/web-app:latest"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test", kind="application", image=image
    )
    fn.deploy()
    _assert_application_post_deploy_spec(fn, image)


def test_consecutive_deploy_application_runtime(rundb_mock):
    image = "my/web-app:latest"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
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
            [{"image": "my/web-app:latest", "ports": [{"containerPort": 8080}]}],
            "Application sidecar port spec must include a name",
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8080, "name": "http"}],
                    "args": ["--help"],
                }
            ],
            "Application sidecar spec must include a command if args are provided",
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8080, "name": "http"}],
                }
            ],
            None,
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8080, "name": "http"}],
                    "command": ["echo"],
                }
            ],
            None,
        ),
        (
            [
                {
                    "image": "my/web-app:latest",
                    "ports": [{"containerPort": 8080, "name": "http"}],
                    "command": ["echo"],
                    "args": ["--help"],
                }
            ],
            None,
        ),
    ],
)
def test_pre_deploy_validation(sidecars, expected_error_message):
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
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
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test",
        kind="application",
    )
    fn._build_application_image()
    assert fn.spec.image == ".mlrun/func-default-application-test:latest"
    assert fn.status.state == mlrun.common.schemas.FunctionState.ready


def test_application_image_build(remote_builder_mock):
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test",
        kind="application",
        requirements=["mock"],
    )
    assert fn.requires_build()
    fn.deploy()
    _assert_application_post_deploy_spec(
        fn, ".mlrun/func-default-application-test:latest"
    )


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
    filepath, expected_handler = (
        mlrun.runtimes.ApplicationRuntime.get_filename_and_handler()
    )
    expected_filename = pathlib.Path(filepath).name
    expected_function_handler = f"{expected_filename.split('.')[0]}:{expected_handler}"
    assert fn.spec.function_handler == expected_function_handler


def _assert_application_post_deploy_spec(fn, image):
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": "application-test-sidecar",
            "ports": [{"containerPort": 8080, "name": "http", "protocol": "TCP"}],
        }
    ]
    assert fn.get_env("SIDECAR_PORT") == "8080"
    assert fn.status.application_image == image
    assert not fn.spec.image
