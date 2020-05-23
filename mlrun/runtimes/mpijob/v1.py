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
import shlex
from copy import deepcopy

from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime
from mlrun.model import RunObject
from mlrun.utils import update_in, get_in

from kubernetes import client

_mpijob_pod_template = {
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
    },
    'metadata': {}
}


def _generate_mpi_job(launcher_pod_template, worker_pod_template):
    return {
     'apiVersion': 'kubeflow.org/v1',
     'kind': 'MPIJob',
     'metadata': {
         'name': '',
         'namespace': 'default-tenant'
     },
     'spec': {
         'mpiReplicaSpecs': {
             'Launcher': {
                 'template': launcher_pod_template
             }, 'Worker': {
                 'replicas': 1,
                 'template': worker_pod_template
             }
         }
     }}


def _update_container(struct, key, value):
    struct['spec']['containers'][0][key] = value


class MpiRuntimeV1(AbstractMPIJobRuntime):

    def _get_crd_info(self) -> tuple:
        mpi_group = 'kubeflow.org'
        mpi_version = 'v1'
        mpi_plural = 'mpijobs'

        return mpi_group, mpi_version, mpi_plural

    def _generate_mpi_job(self, runobj: RunObject, meta: client.V1ObjectMeta) -> dict:
        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name

        # Populate mpijob object

        # start by populating pod templates
        launcher_pod_template = deepcopy(_mpijob_pod_template)
        worker_pod_template = deepcopy(_mpijob_pod_template)

        # configuration for both launcher and workers
        for pod_template in [launcher_pod_template, worker_pod_template]:
            if self.spec.image:
                _update_container(pod_template, 'image', self.full_image_path())
            _update_container(pod_template, 'volumeMounts', self.spec.volume_mounts)
            extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
            if self.spec.rundb:
                extra_env['MLRUN_DBPATH'] = self.spec.rundb
            extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]
            _update_container(pod_template, 'env', extra_env + self.spec.env)
            if self.spec.image_pull_policy:
                _update_container(
                    pod_template, 'imagePullPolicy', self.spec.image_pull_policy)
            if self.spec.workdir:
                _update_container(pod_template, 'workingDir', self.spec.workdir)
            if self.spec.image_pull_secret:
                update_in(pod_template, 'spec.imagePullSecrets',
                          [{'name': self.spec.image_pull_secret}])
            update_in(pod_template, 'metadata.labels', pod_labels)
            update_in(pod_template, 'spec.volumes', self.spec.volumes)

        # configuration for workers only
        # update resources only for workers because the launcher doesn't require
        # special resources (like GPUs, Memory, etc..)
        if self.spec.resources:
            _update_container(worker_pod_template, 'resources', self.spec.resources)

        # configuration for launcher only
        quoted_args = []
        for arg in self.spec.args:
            quoted_args.append(shlex.quote(arg))
        if self.spec.command:
            _update_container(
                launcher_pod_template, 'command',
                ['mpirun', 'python', shlex.quote(self.spec.command)] + quoted_args)

        # generate mpi job using both pod templates
        job = _generate_mpi_job(launcher_pod_template, worker_pod_template)

        # update the replicas only for workers
        update_in(job, 'spec.mpiReplicaSpecs.Worker.replicas', self.spec.replicas or 1)

        update_in(job, 'metadata', meta.to_dict())

        return job

    def _get_job_launcher_status(self, resp: list) -> str:
        launcher_status = get_in(resp, 'status.replicaStatuses.Launcher')
        if launcher_status is None:
            return ''

        for status in ['active', 'failed', 'succeeded']:
            if launcher_status.get(status, 0) == 1:
                return status

        return ''

    def _pretty_print_jobs(self, items: list):
        print('{:10} {:20} {:21} {}'.format(
            'status', 'name', 'start', 'end'))
        for i in items:
            print('{:10} {:20} {:21} {}'.format(
                self._get_job_launcher_status(),
                get_in(i, 'metadata.name', ''),
                get_in(i, 'status.startTime', ''),
                get_in(i, 'status.completionTime', ''),
            ))

    def _generate_pods_selector(self, name: str, launcher: bool) -> str:
        selector = 'mpi-job-name={}'.format(name)
        if launcher:
            selector += ',mpi-job-role=launcher'

        return selector
