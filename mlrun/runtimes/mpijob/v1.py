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
import typing

from mlrun.runtimes.mpijob.abstract import AbstractMPIJobRuntime
from mlrun.model import RunObject
from mlrun.utils import update_in, get_in

from kubernetes import client


class MpiRuntimeV1(AbstractMPIJobRuntime):
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

    def _generate_mpi_job_template(self, launcher_pod_template, worker_pod_template):
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

    def _update_container(self, struct, key, value):
        struct['spec']['containers'][0][key] = value

    def _get_crd_info(self) -> typing.Tuple[str, str, str]:
        mpi_group = 'kubeflow.org'
        mpi_version = 'v1'
        mpi_plural = 'mpijobs'

        return mpi_group, mpi_version, mpi_plural

    def _enrich_launcher_configurations(self, launcher_pod_template):
        quoted_args = []
        for arg in self.spec.args:
            quoted_args.append(shlex.quote(arg))
        if self.spec.command:
            self._update_container(
                launcher_pod_template, 'command',
                ['mpirun', 'python', shlex.quote(self.spec.command)] + quoted_args)

    def _enrich_worker_configurations(self, worker_pod_template):
        if self.spec.resources:
            self._update_container(worker_pod_template, 'resources', self.spec.resources)

    def _generate_mpi_job(self, runobj: RunObject, meta: client.V1ObjectMeta) -> dict:
        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name

        # Populate mpijob object

        # start by populating pod templates
        launcher_pod_template = deepcopy(self._mpijob_pod_template)
        worker_pod_template = deepcopy(self._mpijob_pod_template)

        # configuration for both launcher and workers
        for pod_template in [launcher_pod_template, worker_pod_template]:
            if self.spec.image:
                self._update_container(pod_template, 'image', self.full_image_path())
            self._update_container(pod_template, 'volumeMounts', self.spec.volume_mounts)
            extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
            if self.spec.rundb:
                extra_env['MLRUN_DBPATH'] = self.spec.rundb
            extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]
            self._update_container(pod_template, 'env', extra_env + self.spec.env)
            if self.spec.image_pull_policy:
                self._update_container(
                    pod_template, 'imagePullPolicy', self.spec.image_pull_policy)
            if self.spec.workdir:
                self._update_container(pod_template, 'workingDir', self.spec.workdir)
            if self.spec.image_pull_secret:
                update_in(pod_template, 'spec.imagePullSecrets',
                          [{'name': self.spec.image_pull_secret}])
            update_in(pod_template, 'metadata.labels', pod_labels)
            update_in(pod_template, 'spec.volumes', self.spec.volumes)

        # configuration for workers only
        # update resources only for workers because the launcher doesn't require
        # special resources (like GPUs, Memory, etc..)
        self._enrich_worker_configurations(worker_pod_template)

        # configuration for launcher only
        self._enrich_launcher_configurations(launcher_pod_template)

        # generate mpi job using both pod templates
        job = self._generate_mpi_job_template(launcher_pod_template, worker_pod_template)

        # update the replicas only for workers
        update_in(job, 'spec.mpiReplicaSpecs.Worker.replicas', self.spec.replicas or 1)

        update_in(job, 'metadata', meta.to_dict())

        return job

    def _get_job_launcher_status(self, resp: typing.List) -> str:
        launcher_status = get_in(resp, 'status.replicaStatuses.Launcher')
        if launcher_status is None:
            return ''

        for status in ['active', 'failed', 'succeeded']:
            if launcher_status.get(status, 0) == 1:
                return status

        return ''

    def _generate_pods_selector(self, name: str, launcher: bool) -> str:
        selector = 'mpi-job-name={}'.format(name)
        if launcher:
            selector += ',mpi-job-role=launcher'

        return selector
