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

from ..model import RunObject, K8sRuntime
from ..utils import get_in, logger
from ..k8s_utils import k8s_helper
from .base import MLRuntime, RunError
from ..builder import build_image


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

        source_code = get_in(runtime.build, 'functionSourceCode')
        if runtime.image and source_code:
            extra_env.append({'name': 'MLRUN_EXEC_CODE', 'value': source_code})
            runtime.command = 'mlrun'
            runtime.args = ['run', '--from-env', 'main.py']

        if not runtime.image:
            self._build(runtime, namespace, self.mode != 'noctx')

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
        build_spec = get_in(runtime.spec, 'build')
        if build_spec:
            image = _build(build_spec, namespace, with_mlrun)
            runtime.image = image
            runtime.command = 'python'
            runtime.args = ['main.py']

    @staticmethod
    def _submit(k8s, runtime, metadata, extra_env):
        pod_spec = func_to_pod(runtime, extra_env)
        pod = client.V1Pod(metadata=metadata, spec=pod_spec)
        try:
            return k8s.create_pod(pod)
        except client.rest.ApiException as e:
            print(str(e))
            raise RunError(str(e))


def func_to_pod(runtime, extra_env=[]):

    container = client.V1Container(name='base',
                                   image=runtime.image,
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


def _build(build, namespace, with_mlrun=True):
    inline = get_in(build, 'functionSourceCode')
    if not inline:
        raise ValueError('build spec must have functionSourceCode and baseImage')
    inline = b64decode(inline).decode('utf-8')
    base_image = get_in(build, 'baseImage', 'python:3.6-jessie')
    commands = get_in(build, 'commands')
    image = get_in(build, 'image')
    if not image:
        raise ValueError('build spec must have image, use %nuclio config spec.build.image = <target image>')
    logger.info(f'building image ({image})')
    status = build_image(image,
                         base_image=base_image,
                         commands=commands,
                         namespace=namespace,
                         inline_code=inline,
                         with_mlrun=with_mlrun)
    logger.info(f'build completed with {status}')
    if status in ['failed', 'error']:
        raise RunError(f' build {status}!')
    return image

