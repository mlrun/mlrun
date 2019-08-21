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
from base64 import b64decode

from ..utils import get_in, update_in, logger
from ..k8s_utils import k8s_helper, BasePod
from .base import MLRuntime, RunError
from builder import build_image

from kubernetes import client


class KubejobRuntime(MLRuntime):
    kind = 'job'

    def _run(self, struct):

        runtime = get_in(struct, 'spec.runtime')
        func_spec = get_in(runtime, 'spec')
        meta = get_in(runtime, 'metadata', {})
        if not func_spec:
            raise ValueError('job runtime must have a spec')
        namespace = get_in(meta, 'namespace', 'default-tenant')

        build = get_in(func_spec, 'build')
        if build:
            inline = get_in(build, 'functionSourceCode')
            if not inline:
                raise ValueError('build spec must have functionSourceCode and baseImage')
            inline = b64decode(inline).decode('utf-8')
            base_image = get_in(build, 'baseImage')
            commands = get_in(build, 'commands')
            image = get_in(build, 'image')
            logger.info(f'building image ({image})')
            status = build_image(image,
                                 base_image=base_image,
                                 commands=commands,
                                 namespace=namespace,
                                 inline_code=inline,
                                 with_mlrun=self.mode != 'noctx')
            logger.info(f'build completed with {status}')
            if status in ['failed', 'error']:
                raise RunError(f' build {status}!')
            update_in(func_spec, 'image', image)

        extra_env = [{'name': 'MLRUN_EXEC_CONFIG', 'value': json.dumps(struct)}]
        if self.rundb:
            extra_env.append({'name': 'MLRUN_META_DBPATH', 'value': self.rundb})

        cmd = [self.command]
        pod_spec = func_to_pod(func_spec, cmd, self.args, extra_env)

        uid = get_in(struct, 'metadata.uid')
        name = get_in(struct, 'metadata.name', 'mlrun')
        labels = get_in(meta, 'labels', {})
        labels['mlrun/class'] = 'job'
        labels['mlrun/uid'] = uid
        new_meta = client.V1ObjectMeta(generate_name=f'{name}-',
                                       namespace=namespace,
                                       labels=labels,
                                       annotations=get_in(meta, 'annotations'))

        k8s = k8s_helper()
        pod_name, namespace = self._submit(k8s, pod_spec, new_meta)
        status = k8s.watch(pod_name, namespace)

        if self.db_conn:
            uid = get_in(struct, 'metadata.uid')
            project = get_in(struct, 'metadata.project', '')
            self.db_conn.store_log(uid, project,
                                   k8s.logs(pod_name, namespace))
        if status in ['failed', 'error']:
            raise RunError(f'pod exited with {status}, check logs')

        return None

    @staticmethod
    def _submit(k8s, pod_spec, metadata):
        pod = client.V1Pod(metadata=metadata, spec=pod_spec)
        try:
            return k8s.create_pod(pod)
        except client.rest.ApiException as e:
            print(str(e))
            raise RunError(str(e))


def func_to_pod(spec, command=None, args=None, extra_env=None):

    volumes = get_in(spec, 'volumes')
    mounts = []
    vols = []
    if volumes:
        for vol in volumes:
            mounts.append(vol['volumeMount'])
            vols.append(vol['volume'])

    env = get_in(spec, 'env', [])
    if extra_env:
        env = extra_env + env

    container = client.V1Container(name='base',
                                   image=get_in(spec, 'image'),
                                   env=env,
                                   command=command,
                                   args=args,
                                   image_pull_policy=get_in(spec, 'imagePullPolicy'),
                                   volume_mounts=mounts,
                                   resources=get_in(spec, 'resources'))

    pod_spec = client.V1PodSpec(containers=[container],
                                restart_policy='Never',
                                volumes=vols,
                                service_account=get_in(spec, 'serviceAccount'))

    return pod_spec

