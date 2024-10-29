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
import sys
from contextlib import nullcontext as does_not_raise

import pytest
from deepdiff import DeepDiff

import mlrun
import mlrun.errors
from mlrun import code_to_function
from mlrun.utils.helpers import resolve_git_reference_from_source
from tests.runtimes.test_base import TestAutoMount


def test_generate_nuclio_volumes():
    volume_1_name = "volume-name"
    volume_1 = {
        "name": volume_1_name,
        "flexVolume": {
            "driver": "v3io/fuse",
            "options": {
                "container": "users",
                "accessKey": "4dbc1521-f6f2-4b28-aeac-29073413b9ae",
                "subPath": "/pipelines/.mlrun",
            },
        },
    }
    volume_2_name = "second-volume-name"
    volume_2 = {
        "name": volume_2_name,
        "secret": {"secretName": "secret-name"},
    }
    volume_1_volume_mount_1 = {
        "name": volume_1_name,
        "mountPath": "/v3io/volume/mount/path",
    }
    volume_1_volume_mount_2 = {
        "name": volume_1_name,
        "mountPath": "/v3io/volume/mount/2/path",
    }
    volume_2_volume_mount_1 = {
        "name": volume_2_name,
        "mountPath": "/secret/second/volume/mount/path",
    }
    runtime = {
        "kind": "nuclio",
        "metadata": {"name": "some-function", "project": "default"},
        "spec": {
            "volumes": [volume_1, volume_2],
            "volume_mounts": [
                volume_1_volume_mount_1,
                volume_1_volume_mount_2,
                volume_2_volume_mount_1,
            ],
        },
    }
    expected_nuclio_volumes = [
        {"volume": volume_1, "volumeMount": volume_1_volume_mount_1},
        {"volume": volume_1, "volumeMount": volume_1_volume_mount_2},
        {"volume": volume_2, "volumeMount": volume_2_volume_mount_1},
    ]
    function = mlrun.new_function(runtime=runtime)
    nuclio_volumes = function.spec.generate_nuclio_volumes()
    assert (
        DeepDiff(
            expected_nuclio_volumes,
            nuclio_volumes,
            ignore_order=True,
        )
        == {}
    )


class TestAutoMountNuclio(TestAutoMount):
    def setup_method(self, method):
        super().setup_method(method)
        self.assets_path = (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )
        self.code_filename = str(self.assets_path / "sample_function.py")
        self.code_handler = "test_func"

    def _generate_runtime(self, disable_auto_mount=False):
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind="nuclio",
            image=self.image_name,
            description="test function",
        )
        runtime.spec.disable_auto_mount = disable_auto_mount
        return runtime

    def _execute_run(self, runtime):
        runtime.deploy(project=self.project)


def test_http_trigger():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.with_http(
        workers=2,
        host="x",
        worker_timeout=5,
        extra_attributes={"yy": "123"},
    )

    trigger = function.spec.config["spec.triggers.http"]
    print(trigger)
    assert trigger["maxWorkers"] == 2
    assert trigger["attributes"]["ingresses"] == {"0": {"host": "x", "paths": ["/"]}}
    assert trigger["attributes"]["yy"] == "123"
    assert trigger["workerAvailabilityTimeoutMilliseconds"] == 5000
    assert (
        trigger["annotations"]["nginx.ingress.kubernetes.io/proxy-connect-timeout"]
        == "65"
    )


def test_v3io_stream_trigger():
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.add_v3io_stream_trigger(
        "v3io:///projects/x/y",
        name="mystream",
        extra_attributes={"yy": "123"},
        ack_window_size=10,
    )
    trigger = function.spec.config["spec.triggers.mystream"]
    assert trigger["attributes"]["containerName"] == "projects"
    assert trigger["attributes"]["streamPath"] == "x/y"
    assert trigger["password"] == mlrun.model.Credentials.generate_access_key
    assert trigger["attributes"]["yy"] == "123"
    assert trigger["attributes"]["ackWindowSize"] == 10


@pytest.mark.parametrize(
    "consumer_group,expected",
    [
        ("my_group", does_not_raise()),
        ("_mygroup", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
    ],
)
def test_v3io_stream_trigger_validate_consumer_group(consumer_group, expected):
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    with expected:
        function.add_v3io_stream_trigger(
            "v3io:///projects/x/y",
            name="mystream",
            group=consumer_group,
        )
        trigger = function.spec.config["spec.triggers.mystream"]
        assert trigger["attributes"]["consumerGroup"] == consumer_group


def test_resolve_git_reference_from_source():
    cases = [
        # source, (repo, refs, branch)
        ("repo", ("repo", "", "")),
        ("repo#br", ("repo", "", "br")),
        ("repo#refs/heads/main", ("repo", "refs/heads/main", "")),
        ("repo#refs/heads/main#commit", ("repo", "refs/heads/main#commit", "")),
    ]
    for source, expected in cases:
        assert expected == resolve_git_reference_from_source(source)


@pytest.mark.parametrize("function_kind", ["serving", "remote"])
def test_update_credentials_from_remote_build(function_kind):
    secret_name = "secret-name"
    remote_data = {
        "metadata": {"credentials": {"access_key": secret_name}},
        "spec": {
            "env": [
                {"name": "V3IO_ACCESS_KEY", "value": secret_name},
                {"name": "MLRUN_AUTH_SESSION", "value": secret_name},
            ],
        },
    }

    function = mlrun.new_function("tst", kind=function_kind)
    function.metadata.credentials.access_key = "access_key"
    function.spec.env = [
        {"name": "V3IO_ACCESS_KEY", "value": "access_key"},
        {"name": "MLRUN_AUTH_SESSION", "value": "access_key"},
    ]
    function._update_credentials_from_remote_build(remote_data)

    assert function.metadata.credentials.access_key == secret_name
    assert function.spec.env == remote_data["spec"]["env"]


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("valid_tag", does_not_raise()),
        ("invalid%$tag", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("too-long-tag" * 10, pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
    ],
)
def test_invalid_tags(tag, expected, rundb_mock):
    function = mlrun.new_function("test", kind="nuclio", tag=tag)
    with expected:
        function.pre_deploy_validation()


@pytest.mark.parametrize(
    "command, args, expected_sidecars",
    (
        [
            None,
            ["a", "b"],
            [
                {
                    "name": "tst-sidecar",
                    "ports": [
                        {
                            "containerPort": None,
                            "name": "tst-sidecar-0",
                            "protocol": "TCP",
                        }
                    ],
                }
            ],
        ],
        [
            "abc",
            ["a", "b"],
            [
                {
                    "args": ["a", "b"],
                    "command": ["abc"],
                    "name": "tst-sidecar",
                    "ports": [
                        {
                            "containerPort": None,
                            "name": "tst-sidecar-0",
                            "protocol": "TCP",
                        }
                    ],
                }
            ],
        ],
    ),
)
def test_with_sidecar(command: str, args: list, expected_sidecars: list):
    function: mlrun.runtimes.RemoteRuntime = mlrun.new_function("tst", kind="nuclio")
    function.with_sidecar(
        command=command,
        args=args,
    )

    assert function.spec.config["spec.sidecars"] == expected_sidecars
