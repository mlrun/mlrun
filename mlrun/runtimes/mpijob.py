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
import time
from copy import deepcopy

from .utils import AsyncLogWriter, RunError
from ..model import RunObject
from .kubejob import KubejobRuntime
from ..utils import update_in, logger, get_in
from ..execution import MLClientCtx

from kubernetes import client


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
                 'image': 'zilbermanor/horovod_cpu:0.2',
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

mpi_group = 'kubeflow.org'
mpi_version = 'v1alpha1'
mpi_plural = 'mpijobs'


def _update_container(struct, key, value):
    struct['spec']['template']['spec']['containers'][0][key] = value


class MpiRuntime(KubejobRuntime):
    kind = 'mpijob'
    _is_nested = False

    def _run(self, runobj: RunObject, execution: MLClientCtx):

        if runobj.metadata.iteration:
            self.store_run(runobj)
        job = deepcopy(_mpijob_template)
        meta = self._get_meta(runobj, True)

        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name
        update_in(job, 'metadata', meta.to_dict())
        update_in(job, 'spec.template.metadata.labels', pod_labels)
        update_in(job, 'spec.replicas', self.spec.replicas or 1)
        if self.spec.image:
            _update_container(job, 'image', self._image_path())
        update_in(job, 'spec.template.spec.volumes', self.spec.volumes)
        _update_container(job, 'volumeMounts', self.spec.volume_mounts)

        extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
        if self.spec.rundb:
            extra_env['MLRUN_DBPATH'] = self.spec.rundb
        extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]
        _update_container(job, 'env', extra_env + self.spec.env)
        if self.spec.image_pull_policy:
            _update_container(
                job, 'imagePullPolicy', self.spec.image_pull_policy)
        if self.spec.resources:
            _update_container(job, 'resources', self.spec.resources)

        if self.spec.command:
            _update_container(
                job, 'command',
                ['mpirun', 'python', self.spec.command] + self.spec.args)

        resp = self._submit_mpijob(job, meta.namespace)
        state = None
        for _ in range(60):
            resp = self.get_job(meta.name, meta.namespace)
            state = get_in(resp, 'status.launcherStatus')
            if resp and state:
                break
            time.sleep(1)

        if resp:
            logger.info('MpiJob {} state={}'.format(
                meta.name, state or 'unknown'))
            if state:
                state = state.lower()
                launcher, status = self._get_launcher(meta.name,
                                                      meta.namespace)
                execution.set_hostname(launcher)
                execution.set_state('running' if state == 'active' else state)
                if self.kfp:
                    writer = AsyncLogWriter(self._db_conn, runobj)
                    status = self._get_k8s().watch(
                        launcher, meta.namespace, writer=writer)
                    logger.info(
                        'MpiJob {} finished with state {}'.format(
                            meta.name, status))
                    if status == 'succeeded':
                        execution.set_state('completed')
                    else:
                        execution.set_state('error', 'MpiJob {} finished with state {}'.format(meta.name, status))
                else:
                    txt = 'MpiJob {} launcher pod {} state {}'.format(
                        meta.name, launcher, state)
                    logger.info(txt)
                    runobj.status.status_text = txt
            else:
                txt = 'MpiJob status unknown or failed, check pods: {}'.format(
                    self.get_pods(meta.name, meta.namespace))
                logger.warning(txt)
                runobj.status.status_text = txt

        return None

    def _submit_mpijob(self, job, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                mpi_group, mpi_version, namespace=namespace,
                plural=mpi_plural, body=job)
            name = get_in(resp, 'metadata.name', 'unknown')
            logger.info('MpiJob {} created'.format(name))
            return resp
        except client.rest.ApiException as e:
            logger.error("Exception when creating MPIJob: %s" % e)
            raise RunError("Exception when creating MPIJob: %s" % e)

    def delete_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            # delete the mpi job\
            body = client.V1DeleteOptions()
            resp = k8s.crdapi.delete_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name, body)
            logger.info('del status: {}'.format(
                get_in(resp, 'status', 'unknown')))
        except client.rest.ApiException as e:
            print("Exception when deleting MPIJob: %s" % e)

    def list_jobs(self, namespace=None, selector='', show=True):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.list_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural,
                watch=False, label_selector=selector)
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)

        items = []
        if resp:
            items = resp.get('items', [])
            if show and items:
                print('{:10} {:20} {:21} {}'.format(
                    'status', 'name', 'start', 'end'))
                for i in items:
                    print('{:10} {:20} {:21} {}'.format(
                        get_in(i, 'status.launcherStatus', ''),
                        get_in(i, 'metadata.name', ''),
                        get_in(i, 'status.startTime', ''),
                        get_in(i, 'status.completionTime', ''),
                    ))
        return items

    def get_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.get_namespaced_custom_object(
                mpi_group, mpi_version, namespace, mpi_plural, name)
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)
        return resp

    def get_pods(self, name=None, namespace=None, launcher=False):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        selector = 'mlrun/class=mpijob'
        if name:
            selector += ',mpi_job_name={}'.format(name)
        if launcher:
            selector += ',mpi_role_type=launcher'
        pods = k8s.list_pods(selector=selector, namespace=namespace)
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def watch(self, name, namespace=None):
        pods = self.get_pods(name, namespace, launcher=True)
        if not pods:
            logger.error('no pod matches that job name')
            return
        k8s = self._get_k8s()
        pod, status = self._get_launcher(name, namespace)
        logger.info('watching pod {}, status = {}'.format(pod, status))
        k8s.watch(pod, namespace)

    def _get_launcher(self, name, namespace=None):
        pods = self.get_pods(name, namespace, launcher=True)
        if not pods:
            logger.error('no pod matches that job name')
            return
        # TODO: Why was this here?
        # k8s = self._get_k8s()
        return list(pods.items())[0]
