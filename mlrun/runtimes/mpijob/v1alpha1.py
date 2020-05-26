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
from copy import deepcopy
import typing

from mlrun.execution import MLClientCtx
from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime
from mlrun.model import RunObject
from mlrun.utils import update_in, get_in

from kubernetes import client


class MpiRuntimeV1Alpha1(AbstractMPIJobRuntime):
    _mpijob_template = {
        'apiVersion': 'kubeflow.org/v1alpha1',
        'kind': 'MPIJob',
        'metadata': {
            'name': '',
            'namespace': 'default-tenant'
        },
        'spec': {
            'replicas': 1,
            'template': {
                'metadata': {},
                'spec': {
                    'containers': [{
                        'image': 'mlrun/mpijob',
                        'name': 'base',
                        'command': [],
                        'env': [],
                        'volumeMounts': [],
                        'securityContext': {
                            'capabilities': {'add': ['IPC_LOCK']}},
                        'resources': {
                            'limits': {}}}],
                    'volumes': []
                }}}}

    def _update_container(self, struct, key, value):
        struct['spec']['template']['spec']['containers'][0][key] = value

    def _get_crd_info(self) -> typing.Tuple[str, str, str]:
        mpi_group = 'kubeflow.org'
        mpi_version = 'v1alpha1'
        mpi_plural = 'mpijobs'

        return mpi_group, mpi_version, mpi_plural

    def _generate_mpi_job(self, runobj: RunObject, execution: MLClientCtx, meta: client.V1ObjectMeta) -> typing.Dict:
        job = deepcopy(self._mpijob_template)

        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name
        update_in(job, 'metadata', meta.to_dict())
        update_in(job, 'spec.template.metadata.labels', pod_labels)
        update_in(job, 'spec.replicas', self.spec.replicas or 1)
        if self.spec.image:
            self._update_container(job, 'image', self.full_image_path())
        update_in(job, 'spec.template.spec.volumes', self.spec.volumes)
        self._update_container(job, 'volumeMounts', self.spec.volume_mounts)

        extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
        if self.spec.rundb:
            extra_env['MLRUN_DBPATH'] = self.spec.rundb
        extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]
        self._update_container(job, 'env', extra_env + self.spec.env)
        if self.spec.image_pull_policy:
            self._update_container(
                job, 'imagePullPolicy', self.spec.image_pull_policy)
        if self.spec.resources:
            self._update_container(job, 'resources', self.spec.resources)
        if self.spec.workdir:
            self._update_container(job, 'workingDir', self.spec.workdir)

        if self.spec.image_pull_secret:
            update_in(job, 'spec.template.spec.imagePullSecrets',
                      [{'name': self.spec.image_pull_secret}])

        if self.spec.command:
            self._update_container(
                job, 'command',
                ['mpirun', 'python', self.spec.command] + self.spec.args)

        return job

    def _get_job_launcher_status(self, resp: typing.List) -> str:
        return get_in(resp, 'status.launcherStatus')

    def _generate_pods_selector(self, name: str, launcher: bool) -> str:
        selector = 'mlrun/class=mpijob'
        if name:
            selector += ',mpi_job_name={}'.format(name)
        if launcher:
            selector += ',mpi_role_type=launcher'

        return selector
