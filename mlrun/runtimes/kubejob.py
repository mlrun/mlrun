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

from base64 import b64decode
from kubernetes import client
from os import environ

from ..model import RunObject, K8sRuntime
from ..utils import get_in, logger
from ..k8s_utils import k8s_helper
from .base import MLRuntime, RunError
from ..builder import build_image


def nuclio_to_k8s(kind, spec, image=''):
    r = K8sRuntime()
    r.metadata = get_in(spec, 'spec.metadata')
    r.build.base_image = get_in(spec, 'spec.build.baseImage')
    r.build.commands = get_in(spec, 'spec.build.commands')
    r.build.inline_code = get_in(spec, 'spec.build.functionSourceCode')
    r.build.image = get_in(spec, 'spec.build.image', image)
    r.env = get_in(spec, 'spec.env')
    for vol in get_in(spec, 'spec.volumes', []):
        r.volumes.append(vol.get('volume'))
        r.volume_mounts.append(vol.get('volumeMount'))
    return r


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
        if runtime.image and source_code:
            extra_env.append({'name': 'MLRUN_EXEC_CODE', 'value': source_code})
            runtime.command = 'mlrun'
            runtime.args = ['run', '--from-env', 'main.py']

        if not runtime.image and (runtime.build.source or runtime.build.inline_code):
            self._build(runtime, namespace, self.mode != 'noctx')
        if not runtime.image:
            raise RunError('job submitted without image, set runtime.image')

        uid = runobj.metadata.uid
        name = runobj.metadata.name or 'mlrun'
        runtime.set_label('mlrun/class', self.kind)
        runtime.set_label('mlrun/uid', uid)
        new_meta = client.V1ObjectMeta(generate_name=f'{name}-',
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
    def _build(runtime, namespace, with_mlrun):
        build = runtime.build
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
                             inline_code=inline,
                             source=build.source,
                             secret_name=build.secret,
                             with_mlrun=with_mlrun)
        logger.info(f'build completed with {status}')
        if status in ['failed', 'error']:
            raise RunError(f' build {status}!')

        runtime.image = build.image
        runtime.command = 'python'
        runtime.args = ['main.py']


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
