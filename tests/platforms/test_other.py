import deepdiff

import mlrun
import mlrun.errors


def test_mount_configmap():
    expected_volume = {"configMap": {"name": "my-config-map"}, "name": "my-volume"}
    expected_volume_mount = {"mountPath": "/myConfMapPath", "name": "my-volume"}

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(
        mlrun.platforms.mount_configmap(
            configmap_name="my-config-map",
            mount_path="/myConfMapPath",
            volume_name="my-volume",
        )
    )

    assert (
        deepdiff.DeepDiff([expected_volume], function.spec.volumes, ignore_order=True,)
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            [expected_volume_mount], function.spec.volume_mounts, ignore_order=True,
        )
        == {}
    )


def test_mount_hostpath():
    expected_volume = {"hostPath": {"path": "/tmp", "type": ""}, "name": "my-volume"}
    expected_volume_mount = {"mountPath": "/myHostPath", "name": "my-volume"}

    function = mlrun.new_function(
        "function-name", "function-project", kind=mlrun.runtimes.RuntimeKinds.job
    )
    function.apply(
        mlrun.platforms.mount_hostpath(
            host_path="/tmp", mount_path="/myHostPath", volume_name="my-volume"
        )
    )

    assert (
        deepdiff.DeepDiff([expected_volume], function.spec.volumes, ignore_order=True,)
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            [expected_volume_mount], function.spec.volume_mounts, ignore_order=True,
        )
        == {}
    )
