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
# this file is based on the code from kubeflow pipelines git
import os

from .iguazio import mount_v3io


def mount_pvc(pvc_name, volume_name="pipeline", volume_mount_path="/mnt/pipeline"):
    """
    Modifier function to apply to a Container Op to simplify volume, volume mount addition and
    enable better reuse of volumes, volume claims across container ops.

    Usage::

        train = train_op(...)
        train.apply(mount_pvc('claim-name', 'pipeline', '/mnt/pipeline'))
    """

    def _mount_pvc(task):
        from kubernetes import client as k8s_client

        local_pvc = k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)
        return task.add_volume(
            k8s_client.V1Volume(name=volume_name, persistent_volume_claim=local_pvc)
        ).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path=volume_mount_path, name=volume_name)
        )

    return _mount_pvc


def auto_mount(pvc_name="", volume_mount_path="", volume_name=None):
    """choose the mount based on env variables and params

    volume will be selected by the following order:
    - k8s PVC volume when both pvc_name and volume_mount_path are set
    - iguazio v3io volume when V3IO_ACCESS_KEY and V3IO_USERNAME env vars are set
    - k8s PVC volume when env var is set: MLRUN_PVC_MOUNT=<pvc-name>:<mount-path>
    """
    if pvc_name and volume_mount_path:
        return mount_pvc(
            pvc_name=pvc_name,
            volume_mount_path=volume_mount_path,
            volume_name=volume_name or "pvc",
        )
    if "MLRUN_PVC_MOUNT" in os.environ:
        mount = os.environ.get("MLRUN_PVC_MOUNT")
        items = mount.split(":")
        if len(items) != 2:
            raise ValueError("MLRUN_PVC_MOUNT should include <pvc-name>:<mount-path>")
        return mount_pvc(
            pvc_name=items[0],
            volume_mount_path=items[1],
            volume_name=volume_name or "pvc",
        )
    if "V3IO_ACCESS_KEY" in os.environ:
        return mount_v3io(name=volume_name or "v3io")
    raise ValueError("failed to auto mount, need to set env vars")


def mount_secret(secret_name, mount_path, volume_name="secret", items=None):
    """Modifier function to mount kubernetes secret as files(s)

    :param secret_name:  k8s secret name
    :param mount_path:   path to mount inside the container
    :param volume_name:  unique volume name
    :param items:        If unspecified, each key-value pair in the Data field
                         of the referenced Secret will be projected into the
                         volume as a file whose name is the key and content is
                         the value.
                         If specified, the listed keys will be projected into
                         the specified paths, and unlisted keys will not be
                         present.
    """

    def _mount_secret(task):
        from kubernetes import client as k8s_client

        vol = k8s_client.V1SecretVolumeSource(secret_name=secret_name, items=items)
        return task.add_volume(
            k8s_client.V1Volume(name=volume_name, secret=vol)
        ).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path=mount_path, name=volume_name)
        )

    return _mount_secret
