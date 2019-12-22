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
import json
import uuid
from kubernetes import client

from .utils import AsyncLogWriter, apply_kfp, set_named_item
from ..model import RunObject
from ..utils import normalize_name, update_in, logger
from .base import RunError, FunctionSpec
from .container import ContainerRuntime


class KubejobSpec(FunctionSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 volumes=None, volume_mounts=None, env=None, resources=None,
                 entry_points=None, description=None, replicas=None,
                 image_pull_policy=None, service_account=None, build=None):
        super().__init__(command=command, args=args, image=image,
                         mode=mode, build=build,
                         entry_points=entry_points, description=description)
        self._volumes = {}
        self._volume_mounts = {}
        self.volumes = volumes or []
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self.resources = resources or {}
        self.replicas = replicas
        self.image_pull_policy = image_pull_policy
        self.service_account = service_account

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


class KubejobRuntime(ContainerRuntime):
    kind = 'job'
    _is_nested = True

    def __init__(self, spec=None, metadata=None):
        try:
            from kfp.dsl import ContainerOp
        except (ImportError, ModuleNotFoundError) as e:
            print('KubeFlow pipelines sdk is not installed, use "pip install kfp"')
            raise e

        super().__init__(metadata, spec)
        self._cop = ContainerOp('name', 'image')

    @property
    def spec(self) -> KubejobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', KubejobSpec)

    def to_dict(self, fields=None, exclude=None):
        struct = super().to_dict(fields, exclude)
        api = client.ApiClient()
        return api.sanitize_for_serialization(struct)

    def apply(self, modify):
        return apply_kfp(modify, self._cop, self)

    def set_env(self, name, value):
        self.spec.env.append(client.V1EnvVar(name=name, value=value))
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

    # TODO: Verify if gpus are needed here too
    def with_requests(self, mem=None, cpu=None):
        requests = {}
        if mem:
            requests['memory'] = mem
        if cpu:
            requests['cpu'] = cpu
        update_in(self.spec.resources, 'requests', requests)

    def _run(self, runobj: RunObject, execution):

        with_mlrun = (not self.spec.mode) or (self.spec.mode != 'pass')
        command, args, extra_env = self._get_cmd_args(runobj, with_mlrun)
        extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]

        if runobj.metadata.iteration:
            self.store_run(runobj)
        k8s = self._get_k8s()
        new_meta = self._get_meta(runobj)

        pod_spec = func_to_pod(self._image_path(), self, extra_env, command, args)
        pod = client.V1Pod(metadata=new_meta, spec=pod_spec)
        try:
            pod_name, namespace =  k8s.create_pod(pod)
        except client.rest.ApiException as e:
            raise RunError(str(e))

        if pod_name and (self.interactive or self.kfp):
            writer = AsyncLogWriter(self._db_conn, runobj)
            status = k8s.watch(pod_name, namespace, writer=writer)

            if status in ['failed', 'error']:
                raise RunError(f'pod exited with {status}, check logs')
        else:
            txt = 'Job is running in the background, pod: {}'.format(pod_name)
            logger.info(txt)
            runobj.status.status_text = txt

        return None

    def _get_meta(self, runobj, unique=False):
        namespace = self._get_k8s().ns()
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


def func_to_pod(image, runtime, extra_env, command, args):

    container = client.V1Container(name='base',
                                   image=image,
                                   env=extra_env + runtime.spec.env,
                                   command=[command],
                                   args=args,
                                   image_pull_policy=runtime.spec.image_pull_policy,
                                   volume_mounts=runtime.spec.volume_mounts,
                                   resources=runtime.spec.resources)

    pod_spec = client.V1PodSpec(containers=[container],
                                restart_policy='Never',
                                volumes=runtime.spec.volumes,
                                service_account=runtime.spec.service_account)

    return pod_spec
