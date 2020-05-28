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

from .utils import apply_kfp, set_named_item, get_item_name
from ..utils import normalize_name, update_in
from .base import BaseRuntime, FunctionSpec


class KubeResourceSpec(FunctionSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 volumes=None, volume_mounts=None, env=None, resources=None,
                 default_handler=None, entry_points=None, description=None,
                 workdir=None, replicas=None, image_pull_policy=None,
                 service_account=None, build=None, image_pull_secret=None):
        super().__init__(command=command, args=args, image=image, mode=mode,
                         build=build, entry_points=entry_points,
                         description=description,
                         workdir=workdir,
                         default_handler=default_handler)
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
            for vol in volume_mounts:
                set_named_item(self._volume_mounts, vol)

    def update_vols_and_mounts(self, volumes, volume_mounts):
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

        if volume_mounts:
            for vol in volume_mounts:
                set_named_item(self._volume_mounts, vol)


class KubeResource(BaseRuntime):
    kind = 'job'
    _is_nested = True

    def __init__(self, spec=None, metadata=None):
        super().__init__(metadata, spec)
        self._cop = ContainerOp('name', 'image')
        self.verbose = False

    @property
    def spec(self) -> KubeResourceSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', KubeResourceSpec)

    def to_dict(self, fields=None, exclude=None, strip=False):
        struct = super().to_dict(fields, exclude, strip=strip)
        api = client.ApiClient()
        struct = api.sanitize_for_serialization(struct)
        if strip:
            spec = struct['spec']
            for attr in ['volumes', 'volume_mounts']:
                if attr in spec:
                    del spec[attr]
            if 'env' in spec and spec['env']:
                for ev in spec['env']:
                    if ev['name'].startswith('V3IO_'):
                        ev['value'] = ''
        return struct

    def apply(self, modify):
        return apply_kfp(modify, self._cop, self)

    def set_env(self, name, value):
        i = 0
        new_var = client.V1EnvVar(name=name, value=value)
        for v in self.spec.env:
            if get_item_name(v) == name:
                self.spec.env[i] = new_var
                return self
            i += 1
        self.spec.env.append(new_var)
        return self

    def set_envs(self, env_vars):
        for name, value in env_vars.items():
            self.set_env(name, value)
        return self

    def gpus(self, gpus, gpu_type='nvidia.com/gpu'):
        update_in(self.spec.resources, ['limits', gpu_type], gpus)

    def with_limits(self, mem=None, cpu=None, gpus=None, gpu_type='nvidia.com/gpu'):
        limits = {}
        if gpus:
            limits[gpu_type] = gpus
        if mem:
            limits['memory'] = mem
        if cpu:
            limits['cpu'] = cpu
        update_in(self.spec.resources, 'limits', limits)

    def with_requests(self, mem=None, cpu=None):
        requests = {}
        if mem:
            requests['memory'] = mem
        if cpu:
            requests['cpu'] = cpu
        update_in(self.spec.resources, 'requests', requests)

    def _get_meta(self, runobj, unique=False):
        namespace = self._get_k8s().resolve_namespace()
        uid = runobj.metadata.uid
        labels = {'mlrun/class': self.kind, 'mlrun/uid': uid}
        new_meta = client.V1ObjectMeta(namespace=namespace,
                                       labels=labels)

        name = runobj.metadata.name or 'mlrun'
        norm_name = '{}-'.format(normalize_name(name))
        if unique:
            norm_name += uuid.uuid4().hex[:8]
            new_meta.name = norm_name
            runobj.set_label('mlrun/job', norm_name)
        else:
            new_meta.generate_name = norm_name
        return new_meta

    def copy(self):
        self._cop = None
        fn = deepcopy(self)
        self._cop = ContainerOp('name', 'image')
        fn._cop = ContainerOp('name', 'image')
        return fn