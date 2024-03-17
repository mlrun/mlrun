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
import pathlib

import mlrun


def test_create_application_runtime():
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )
    assert fn.kind == mlrun.runtimes.RuntimeKinds.application
    assert fn.spec.image == "mlrun/mlrun"
    assert fn.metadata.name == "application-test"
    # base64 prefix of the reverse proxy code
    assert (
        "Ly8gQ29weXJpZ2h0IDIwMjQgSWd1YXppbwovLwovLyBMaWN"
        in fn.spec.build.functionSourceCode
    )

    filepath, expected_handler = (
        mlrun.runtimes.ApplicationRuntime.get_filename_and_handler()
    )
    expected_filename = pathlib.Path(filepath).name
    expected_function_handler = f"{expected_filename.split('.')[0]}:{expected_handler}"
    assert fn.spec.function_handler == expected_function_handler


def test_deploy_application_runtime(rundb_mock):
    image = "my/web-app:latest"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test", kind="application", image=image
    )
    fn.deploy()
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


def test_consecutive_deploy_application_runtime(rundb_mock):
    image = "my/web-app:latest"
    fn: mlrun.runtimes.ApplicationRuntime = mlrun.code_to_function(
        "application-test", kind="application", image=image
    )
    fn.deploy()
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": "application-test-sidecar",
            "ports": [{"containerPort": 8080, "name": "http", "protocol": "TCP"}],
        }
    ]
    assert fn.status.application_image == image
    assert not fn.spec.image

    fn.deploy()
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": "application-test-sidecar",
            "ports": [{"containerPort": 8080, "name": "http", "protocol": "TCP"}],
        }
    ]
    assert fn.status.application_image == image
    assert not fn.spec.image

    # Change the image and deploy again
    image = "another/web-app:latest"
    fn.spec.image = image
    fn.deploy()

    # Ensure the image is updated
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": "application-test-sidecar",
            "ports": [{"containerPort": 8080, "name": "http", "protocol": "TCP"}],
        }
    ]
    assert fn.status.application_image == image
    assert not fn.spec.image
