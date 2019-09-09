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
from base64 import b64decode, b64encode
from kubernetes import client
from os import environ

from ..model import RunObject, ImageBuilder, BaseMetadata
from ..utils import get_in, logger, normalize_name, update_in
from ..k8s_utils import k8s_helper
from .base import RunRuntime, RunError, FunctionSpec
from ..builder import build_image


def make_nuclio_job(filename='', handler='', image=None, secret=None, kind=None):
    from nuclio import build_file
    name, spec, code = build_file(filename, handler=handler)
    r = KubejobRuntime()
    r.kind = kind or 'job'
    h = get_in(spec, 'spec.handler', '').split(':')
    r.handler = h[0] if len(h) <= 1 else h[1]
    r.metadata = get_in(spec, 'spec.metadata')
    r.build.base_image = get_in(spec, 'spec.build.baseImage')
    r.build.commands = get_in(spec, 'spec.build.commands')
    r.build.inline_code = get_in(spec, 'spec.build.functionSourceCode')
    r.build.image = get_in(spec, 'spec.build.image', image)
    r.build.secret = get_in(spec, 'spec.build.secret', secret)
    r.spec.env = get_in(spec, 'spec.env')
    for vol in get_in(spec, 'spec.volumes', []):
        r.spec.volumes.append(vol.get('volume'))
        r.spec.volume_mounts.append(vol.get('volumeMount'))
    return r


class KubejobSpec(FunctionSpec):
    def __init__(self, command=None, args=None, image=None, rundb=None, mode=None, workers=None,
                 volumes=None, volume_mounts=None, env=None, resources=None,
                 replicas=None, image_pull_policy=None, service_account=None):
        super().__init__(command=command, args=args, image=image, rundb=rundb, mode=mode, workers=workers)
        self.volumes = volumes or []
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self.resources = resources or {}
        self.replicas = replicas
        self.image_pull_policy = image_pull_policy
        self.service_account = service_account


class KubejobRuntime(RunRuntime):
    kind = 'job'

    def __init__(self, spec=None, metadata=None, build=None):
        try:
            from kfp.dsl import ContainerOp
        except ImportError as e:
            print('KubeFlow pipelines sdk is not installed, use "pip install kfp"')
            raise e

        super().__init__(spec, metadata)
        self._build = None
        self.build = build
        self._cop = ContainerOp('name', 'image')
        self._k8s = None
        self._is_built = False

    @property
    def spec(self) -> KubejobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', KubejobSpec)

    def _get_k8s(self):
        if not self._k8s:
            self._k8s = k8s_helper()
        return self._k8s

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    def apply(self, modify):
        modify(self._cop)
        self._merge()
        return self

    def _merge(self):
        for k, v in self._cop.pod_labels.items():
            self.metadata.labels[k] = v
        for k, v in self._cop.pod_annotations.items():
            self.metadata.annotations[k] = v
        if self._cop.container.env:
            [self.spec.env.append(e) for e in self._cop.container.env]
            self._cop.container.env.clear()
        if self._cop.volumes:
            [self.spec.volumes.append(v) for v in self._cop.volumes]
            self._cop.volumes.clear()
        if self._cop.container.volume_mounts:
            [self.spec.volume_mounts.append(v) for v in self._cop.container.volume_mounts]
            self._cop.container.volume_mounts.clear()

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, 'build', ImageBuilder)

    def set_env(self, name, value):
        self.spec.env.append(client.V1EnvVar(name=name, value=value))
        return self

    def with_limits(self, mem=None, cpu=None, gpus=None, gpu_type='nvidia.com/gpu'):
        limits = {}
        if gpus:
            limits[gpu_type] = gpus
        if mem:
            limits['memory'] = mem
        if cpu:
            limits['cpu'] = cpu
        update_in(self.spec.resources, 'limits', limits)

    def with_code(self, from_file='', body=None):
        if (not body and not from_file) or (from_file and from_file.endswith('.ipynb')):
            from nuclio import build_file
            name, spec, code = build_file(from_file)
            self.build.inline_code = get_in(spec, 'spec.build.functionSourceCode')
            return self

        if from_file:
            with open(from_file) as fp:
                body = fp.read()
        self.build.inline_code = b64encode(body.encode('utf-8')).decode('utf-8')
        return self

    def build_image(self, image, base_image=None, commands: list = None,
                    secret=None, with_mlrun=True, watch=True):
        self.build.image = image
        self.spec.image = ''
        if commands and isinstance(commands, list):
            self.build.commands = self.build.commands or []
            self.build.commands += commands
        if secret:
            self.build.secret = secret
        if base_image:
            self.build.base_image = base_image
        ready = self._build_image(watch, with_mlrun)
        return self

    def _build_image(self, watch=False, with_mlrun=True, execution=None):
        pod = self.build.build_pod
        if not self._is_built and pod:
            k8s = self._get_k8s()
            status = k8s.get_pod_status(pod)
            if status == 'succeeded':
                self.build.build_pod = None
                self._is_built = True
                logger.info('build completed successfully')
                return True
            if status in ['failed', 'error']:
                raise RunError(' build {}, watch the build pod logs: {}'.format(status, pod))
            logger.info('builder status is: {}, wait for it to complete'.format(status))
            return False

        if not self.build.commands and self.spec.mode == 'pass' and not self.build.source:
            if not self.spec.image and not self.build.base_image:
                raise RunError('image or base_image must be specified')
            self.spec.image = self.spec.image or self.build.base_image
        if self.spec.image:
            self._is_built = True
            return True

        if execution:
            execution.set_state('build')
        ready = _build(self, with_mlrun, watch)
        self._is_built = ready

    def _run(self, runobj: RunObject, execution):

        extra_env = [{'name': 'MLRUN_EXEC_CONFIG', 'value': runobj.to_json()}]
        if self.spec.rundb:
            extra_env.append({'name': 'MLRUN_META_DBPATH', 'value': self.spec.rundb})

        args = self.spec.args
        command = self.spec.command
        if self.build.inline_code:
            extra_env.append({'name': 'MLRUN_EXEC_CODE', 'value': self.build.inline_code})
            if self.spec.mode != 'pass':
                command = 'mlrun'
                args = ['run', '--from-env', 'main.py']
                if runobj.spec.handler:
                    args += ['--handler', runobj.spec.handler]

        if not self._is_built:
            ready = self._build_image(True, self.spec.mode != 'pass', execution)
            if not ready:
                raise RunError("can't run task, image is not built/ready")

        k8s = self._get_k8s()
        execution.set_state('submit')
        new_meta = self._get_meta(runobj)

        pod_spec = func_to_pod(image_path(self.spec.image), self, extra_env, command, args)
        pod = client.V1Pod(metadata=new_meta, spec=pod_spec)
        try:
            pod_name, namespace =  k8s.create_pod(pod)
        except client.rest.ApiException as e:
            print(str(e))
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


def image_path(image):
    if not image.startswith('.'):
        return image
    if 'DEFAULT_DOCKER_REGISTRY' in environ:
        return '{}/{}'.format(environ.get('DEFAULT_DOCKER_REGISTRY'), image[1:])
    if 'IGZ_NAMESPACE_DOMAIN' in environ:
        return 'docker-registry.{}:80/{}'.format(environ.get('IGZ_NAMESPACE_DOMAIN'), image[1:])
    raise RunError('local container registry is not defined')


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


def _build(runtime, with_mlrun, interactive=False):
    build = runtime.build
    namespace = runtime.metadata.namespace
    inline = None
    if build.inline_code:
        inline = b64decode(build.inline_code).decode('utf-8')
    if not build.image:
        raise ValueError('build spec must have a taget image, set build.image = <target image>')
    logger.info(f'building image ({build.image})')
    status = build_image(build.image,
                         base_image=build.base_image or 'python:3.6-jessie',
                         commands=build.commands,
                         namespace=namespace,
                         #inline_code=inline,
                         source=build.source,
                         secret_name=build.secret,
                         interactive=interactive,
                         with_mlrun=with_mlrun)
    build.build_pod = None
    if status == 'skipped':
        runtime.spec.image = runtime.build.base_image
        return True

    if status.startswith('build:'):
        build.build_pod = status[6:]
        return False

    logger.info('build completed with {}'.format(status))
    if status in ['failed', 'error']:
        raise RunError(' build {}!'.format(status))

    local = '' if build.secret else '.'
    runtime.spec.image = local + build.image
    return True
