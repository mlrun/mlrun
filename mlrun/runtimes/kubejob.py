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
from kubernetes import client

from ..model import RunObject
from ..utils import normalize_name, update_in
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
        self.volumes = volumes or []
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self.resources = resources or {}
        self.replicas = replicas
        self.image_pull_policy = image_pull_policy
        self.service_account = service_account


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

    def apply(self, modify):
        modify(self._cop)
        self._merge()
        return self

    def _merge(self):
        api = client.ApiClient()
        for k, v in self._cop.pod_labels.items():
            self.metadata.labels[k] = v
        for k, v in self._cop.pod_annotations.items():
            self.metadata.annotations[k] = v
        if self._cop.container.env:
            [self.spec.env.append(e) for e in self._cop.container.env]
            self._cop.container.env.clear()
        if self._cop.volumes:
            [self.spec.volumes.append(v) for v in
             api.sanitize_for_serialization(self._cop.volumes)]
            self._cop.volumes.clear()
        if self._cop.container.volume_mounts:
            [self.spec.volume_mounts.append(v) for v in
             api.sanitize_for_serialization(self._cop.container.volume_mounts)]
            self._cop.container.volume_mounts.clear()

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

        with_mlrun = self.spec.mode != 'pass'
        command, args, extra_env = self._get_cmd_args(runobj, with_mlrun)
        extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]

        if not self._is_built:
            ready = self._build_image(True, with_mlrun, execution)
            if not ready:
                raise RunError("can't run task, image is not built/ready")

        k8s = self._get_k8s()
        execution.set_state('submit')
        new_meta = self._get_meta(runobj)

        pod_spec = func_to_pod(self._image_path(), self, extra_env, command, args)
        pod = client.V1Pod(metadata=new_meta, spec=pod_spec)
        try:
            pod_name, namespace =  k8s.create_pod(pod)
        except client.rest.ApiException as e:
            raise RunError(str(e))

        status = 'unknown'
        if pod_name:
            status = k8s.watch(pod_name, namespace)

        if self._db_conn and pod_name:
            project = runobj.metadata.project or ''
            self._db_conn.store_log(new_meta.uid, project,
                                   k8s.logs(pod_name, namespace))
        if status in ['failed', 'error']:
            raise RunError(f'pod exited with {status}, check logs')

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
