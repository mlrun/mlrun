import pathlib
import sys

from deepdiff import DeepDiff

import mlrun
from mlrun import code_to_function
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
    assert DeepDiff(expected_nuclio_volumes, nuclio_volumes, ignore_order=True,) == {}


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
        workers=2, host="x", worker_timeout=5, extra_attributes={"yy": "123"},
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
        access_key="x",
    )

    print(function.spec.config)
    trigger = function.spec.config["spec.triggers.mystream"]
    assert trigger["attributes"]["containerName"] == "projects"
    assert trigger["attributes"]["streamPath"] == "x/y"
    assert trigger["password"] == "x"
    assert trigger["attributes"]["yy"] == "123"
