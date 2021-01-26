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

import uuid
from copy import deepcopy

from kubernetes import client
from kfp.dsl import ContainerOp

from .utils import (
    apply_kfp,
    set_named_item,
    get_item_name,
    get_resource_labels,
    generate_resources,
)
from ..utils import normalize_name, update_in
from .base import BaseRuntime, FunctionSpec


class KubeResourceSpec(FunctionSpec):
    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        volumes=None,
        volume_mounts=None,
        env=None,
        resources=None,
        default_handler=None,
        entry_points=None,
        description=None,
        workdir=None,
        replicas=None,
        image_pull_policy=None,
        service_account=None,
        build=None,
        image_pull_secret=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            build=build,
            entry_points=entry_points,
            description=description,
            workdir=workdir,
            default_handler=default_handler,
        )
        self._volumes = {}
        self._volume_mounts = {}
        self.volumes = volumes or []
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self.resources = resources or {}
        self.replicas = replicas
        self.image_pull_policy = image_pull_policy
        self.service_account = service_account
        self.image_pull_secret = image_pull_secret

    @property
    def volumes(self) -> list:
        return list(self._volumes.values())

    @volumes.setter
    def volumes(self, volumes):
        self._volumes = {}
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

    @property
    def volume_mounts(self) -> list:
        return list(self._volume_mounts.values())

    @volume_mounts.setter
    def volume_mounts(self, volume_mounts):
        self._volume_mounts = {}
        if volume_mounts:
            for volume_mount in volume_mounts:
                self._set_volume_mount(volume_mount)

    def update_vols_and_mounts(self, volumes, volume_mounts):
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

        if volume_mounts:
            for volume_mount in volume_mounts:
                self._set_volume_mount(volume_mount)

    def _set_volume_mount(self, volume_mount):
        # calculate volume mount hash
        volume_name = get_item_name(volume_mount, "name")
        volume_sub_path = get_item_name(volume_mount, "subPath")
        volume_mount_path = get_item_name(volume_mount, "mountPath")
        volume_mount_key = hash(f"{volume_name}-{volume_sub_path}-{volume_mount_path}")
        self._volume_mounts[volume_mount_key] = volume_mount


class KubeResource(BaseRuntime):
    kind = "job"
    _is_nested = True

    def __init__(self, spec=None, metadata=None):
        super().__init__(metadata, spec)
        self._cop = ContainerOp("name", "image")
        self.verbose = False

    @property
    def spec(self) -> KubeResourceSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", KubeResourceSpec)

    def to_dict(self, fields=None, exclude=None, strip=False):
        struct = super().to_dict(fields, exclude, strip=strip)
        api = client.ApiClient()
        struct = api.sanitize_for_serialization(struct)
        if strip:
            spec = struct["spec"]
            for attr in ["volumes", "volume_mounts"]:
                if attr in spec:
                    del spec[attr]
            if "env" in spec and spec["env"]:
                for ev in spec["env"]:
                    if ev["name"].startswith("V3IO_"):
                        ev["value"] = ""
        return struct

    def apply(self, modify):
        return apply_kfp(modify, self._cop, self)

    def set_env_from_secret(self, name, secret=None, secret_key=None):
        """set pod environment var from secret"""
        secret_key = secret_key or name
        value_from = client.V1EnvVarSource(
            secret_key_ref=client.V1SecretKeySelector(name=secret, key=secret_key)
        )
        return self._set_env(name, value_from=value_from)

    def set_env(self, name, value):
        """set pod environment var from value"""
        return self._set_env(name, value=str(value))

    def _set_env(self, name, value=None, value_from=None):
        new_var = client.V1EnvVar(name=name, value=value, value_from=value_from)
        i = 0
        for v in self.spec.env:
            if get_item_name(v) == name:
                self.spec.env[i] = new_var
                return self
            i += 1
        self.spec.env.append(new_var)
        return self

    def set_envs(self, env_vars):
        """set pod environment var key/value dict"""
        for name, value in env_vars.items():
            self.set_env(name, value)
        return self

    def gpus(self, gpus, gpu_type="nvidia.com/gpu"):
        update_in(self.spec.resources, ["limits", gpu_type], gpus)

    def with_limits(self, mem=None, cpu=None, gpus=None, gpu_type="nvidia.com/gpu"):
        """set pod cpu/memory/gpu limits"""
        update_in(
            self.spec.resources,
            "limits",
            generate_resources(mem=mem, cpu=cpu, gpus=gpus, gpu_type=gpu_type),
        )

    def with_requests(self, mem=None, cpu=None):
        """set requested (desired) pod cpu/memory/gpu resources"""
        update_in(self.spec.resources, "requests", generate_resources(mem=mem, cpu=cpu))

    def _get_meta(self, runobj, unique=False):
        namespace = self._get_k8s().resolve_namespace()

        labels = get_resource_labels(self, runobj, runobj.spec.scrape_metrics)
        new_meta = client.V1ObjectMeta(namespace=namespace, labels=labels)

        name = runobj.metadata.name or "mlrun"
        norm_name = "{}-".format(normalize_name(name))
        if unique:
            norm_name += uuid.uuid4().hex[:8]
            new_meta.name = norm_name
            runobj.set_label("mlrun/job", norm_name)
        else:
            new_meta.generate_name = norm_name
        return new_meta

    def copy(self):
        self._cop = None
        fn = deepcopy(self)
        self._cop = ContainerOp("name", "image")
        fn._cop = ContainerOp("name", "image")
        return fn
