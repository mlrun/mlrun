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

from base64 import b64decode, b64encode
from kubernetes import client
from os import environ

from ..model import RunObject, ImageBuilder, RunRuntime
from ..utils import get_in, logger, normalize_name
from ..k8s_utils import k8s_helper
from .base import MLRuntime, RunError
from ..builder import build_image


def make_nuclio_job(filename='', handler='', image=None, secret=None, kind=None):
    from nuclio import build_file
    name, spec, code = build_file(filename, handler=handler)
    r = K8sRuntime()
    r.kind = kind or 'job'
    h = get_in(spec, 'spec.handler', '').split(':')
    r.handler = h[0] if len(h) <= 1 else h[1]
    r.metadata = get_in(spec, 'spec.metadata')
    r.build.base_image = get_in(spec, 'spec.build.baseImage')
    r.build.commands = get_in(spec, 'spec.build.commands')
    r.build.inline_code = get_in(spec, 'spec.build.functionSourceCode')
    r.build.image = get_in(spec, 'spec.build.image', image)
    r.build.secret = get_in(spec, 'spec.build.secret', secret)
    r.env = get_in(spec, 'spec.env')
    for vol in get_in(spec, 'spec.volumes', []):
        r.volumes.append(vol.get('volume'))
        r.volume_mounts.append(vol.get('volumeMount'))
    return r


class K8sRuntime(RunRuntime):
    def __init__(self, kind=None, command=None, args=None, image=None, handler=None,
                 metadata=None, build=None, volumes=None, volume_mounts=None,
                 env=None, resources=None, image_pull_policy=None,
                 service_account=None):
        try:
            from kfp.dsl import ContainerOp
        except ImportError as e:
            print('KubeFlow pipelines sdk is not installed, use "pip install kfp"')
            raise e

        super().__init__(kind, command, args, image, handler, metadata, None)
        self._build = None
        self.build = build
        self.volumes = volumes or []
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self.resources = resources
        self.image_pull_policy = image_pull_policy
        self.service_account = service_account
        self._cop = ContainerOp('name', 'image')

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
            [self.env.append(e) for e in self._cop.container.env]
            self._cop.container.env.clear()
        if self._cop.volumes:
            [self.volumes.append(v) for v in self._cop.volumes]
            self._cop.volumes.clear()
        if self._cop.container.volume_mounts:
            [self.volume_mounts.append(v) for v in self._cop.container.volume_mounts]
            self._cop.container.volume_mounts.clear()

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, 'build', ImageBuilder)

    def set_env(self, name, value):
        self.env.append(client.V1EnvVar(name=name, value=value))
        return self

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

    def build_image(self, image, with_mlrun=True):
        self.build.image = image
        _build(self, with_mlrun)
        return self


class KubejobRuntime(MLRuntime):
    kind = 'job'

    def set_runtime(self, runtime: K8sRuntime):
        self.runtime = K8sRuntime.from_dict(runtime)

    def _run(self, runobj: RunObject):

        runtime = self.runtime
        meta = runtime.metadata or {}
        namespace = meta.namespace or 'default-tenant'

        extra_env = [{'name': 'MLRUN_EXEC_CONFIG', 'value': runobj.to_json()}]
        if self.rundb:
            extra_env.append({'name': 'MLRUN_META_DBPATH', 'value': self.rundb})

        source_code = runtime.build.inline_code
        if source_code:
            extra_env.append({'name': 'MLRUN_EXEC_CODE', 'value': source_code})
            if self.mode != 'pass':
                runtime.command = 'mlrun'
                runtime.args = ['run', '--from-env', 'main.py']
                if runtime.handler:
                    runtime.args += ['--handler', runtime.handler]
            if not runtime.build.commands and self.mode == 'pass':
                runtime.image = runtime.image or runtime.build.base_image

        if not runtime.image and (runtime.build.source or runtime.build.commands or self.mode != 'pass'):
            _build(runtime, namespace, self.mode != 'pass')
        if not runtime.image:
            raise RunError('job submitted without image, set runtime.image')

        uid = runobj.metadata.uid
        name = runobj.metadata.name or 'mlrun'
        runtime.set_label('mlrun/class', self.kind)
        runtime.set_label('mlrun/uid', uid)
        norm_name = '{}-'.format(normalize_name(name))
        new_meta = client.V1ObjectMeta(generate_name=norm_name,
                                       namespace=namespace,
                                       labels=meta.labels,
                                       annotations=meta.annotations)

        k8s = k8s_helper()
        pod_name, namespace = self._submit(k8s, runtime, new_meta, extra_env)
        status = 'unknown'
        if pod_name:
            status = k8s.watch(pod_name, namespace)

        if self.db_conn and pod_name:
            project = runobj.metadata.project or ''
            self.db_conn.store_log(uid, project,
                                   k8s.logs(pod_name, namespace))
        if status in ['failed', 'error']:
            raise RunError(f'pod exited with {status}, check logs')

        return None


    @staticmethod
    def _submit(k8s, runtime, metadata, extra_env):
        pod_spec = func_to_pod(image_path(runtime.image), runtime, extra_env)
        pod = client.V1Pod(metadata=metadata, spec=pod_spec)
        try:
            return k8s.create_pod(pod)
        except client.rest.ApiException as e:
            print(str(e))
            raise RunError(str(e))


def image_path(image):
    if 'DEFAULT_DOCKER_REGISTRY' in environ:
        return '{}/{}'.format(environ.get('DEFAULT_DOCKER_REGISTRY'), image)
    return image


def func_to_pod(image, runtime, extra_env=[]):

    container = client.V1Container(name='base',
                                   image=image,
                                   env=extra_env + runtime.env,
                                   command=[runtime.command],
                                   args=runtime.args,
                                   image_pull_policy=runtime.image_pull_policy,
                                   volume_mounts=runtime.volume_mounts,
                                   resources=runtime.resources)

    pod_spec = client.V1PodSpec(containers=[container],
                                restart_policy='Never',
                                volumes=runtime.volumes,
                                service_account=runtime.service_account)

    return pod_spec


def _build(runtime, with_mlrun):
    build = runtime.build
    namespace = runtime.metadata.namespace
    inline = None
    if build.inline_code:
        inline = b64decode(build.inline_code).decode('utf-8')
    if not build.image:
        raise ValueError('build spec must have image, set runtime.build.image = <target image>')
    logger.info(f'building image ({build.image})')
    status = build_image(build.image,
                         base_image=build.base_image or 'python:3.6-jessie',
                         commands=build.commands,
                         namespace=namespace,
                         #inline_code=inline,
                         source=build.source,
                         secret_name=build.secret,
                         with_mlrun=with_mlrun)
    if status == 'skipped':
        runtime.image = runtime.build.base_image
        return

    logger.info('build completed with {}'.format(status))
    if status in ['failed', 'error']:
        raise RunError(' build {}!'.format(status))

    runtime.image = build.image
