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
from copy import deepcopy
from os import environ
from pprint import pprint

from ..model import RunObject
from .kubejob import KubejobRuntime
from ..utils import dict_to_yaml, update_in, logger, get_in
from ..k8s_utils import k8s_helper
import importlib

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


def _update_container(struct, key, value):
    struct['spec']['template']['spec']['containers'][0][key] = value


class MpiRuntime(KubejobRuntime):
    kind = 'mpijob'

    def _run(self, runobj: RunObject, execution):

        job = deepcopy(_mpijob_template)
        meta = self._get_meta(runobj, True)

        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name
        update_in(job, 'metadata', meta.to_dict())
        update_in(job, 'spec.template.metadata.labels', pod_labels)
        #update_in(job, 'spec.template.metadata.namespace', meta.namespace)
        update_in(job, 'spec.replicas', self.replicas or 1)
        if self.image:
            _update_container(job, 'image', self.image)
        update_in(job, 'spec.template.spec.volumes', self.volumes)
        _update_container(job, 'volumeMounts', self.volume_mounts)
        if self.command:
            _update_container(job, 'command',
                              ['mpirun', 'python', self.command] + self.args)

        self._submit_mpijob(job, meta.namespace)

        return None

    def _submit_mpijob(self, job, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                MpiJob.group, MpiJob.version, namespace=namespace,
                plural='mpijobs', body=job)
            name = get_in(resp, 'metadata.name', 'unknown')
            logger.info('MpiJob {} created'.format(name))
            logger.info('use runner.watch({}) to see logs'.format(name))
            return resp
        except client.rest.ApiException as e:
            logger.error("Exception when creating MPIJob: %s" % e)

    def delete_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            # delete the mpi job\
            body = client.V1DeleteOptions()
            resp = k8s.crdapi.delete_namespaced_custom_object(
                MpiJob.group, MpiJob.version, namespace, MpiJob.plural, name, body)
            logger.info('del status: {}'.format(get_in(resp, 'status', 'unknown')))
        except client.rest.ApiException as e:
            print("Exception when deleting MPIJob: %s" % e)

    def list_jobs(self, namespace=None, selector='', show=True):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.list_namespaced_custom_object(
                MpiJob.group, MpiJob.version, namespace, MpiJob.plural,
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
                MpiJob.group, MpiJob.version, namespace, MpiJob.plural, name)
        except client.rest.ApiException as e:
            print("Exception when reading MPIJob: %s" % e)
        return resp

    def get_pods(self, name=None, namespace=None, lancher=False):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        selector = 'mlrun/class=mpijob'
        if name:
            selector += ',mpi_job_name={}'.format(name)
        if lancher:
            selector += ',mpi_role_type=launcher'
        pods = k8s.list_pods(selector=selector, namespace=namespace)
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def watch(self, name, namespace=None):
        pods = self.get_pods(name, namespace, lancher=True)
        if not pods:
            logger.error('no pod matches that job name')
            return
        k8s = self._get_k8s()
        pod, status = list(pods.items())[0]
        logger.info('watching pod {}, status = {}'.format(pod, status))
        k8s.watch(pod, namespace)


class MpiJob:
    """
    A wrapper over Kubernetes MPIJob (Horovod).

    Example:

       from mpijob import MpiJob

       job = MpiJob('myname', 'img', ['a','b'])
       job.volume()   # add v3io volume
       print(job.to_yaml())
       job.submit()

    """
    group = 'kubeflow.org'
    version = 'v1alpha1'
    plural = 'mpijobs'

    def __init__(self, name, image=None, command=None,
                 replicas=0, namespace='default-tenant', struct=None):
        from kubernetes import config

        self.api_instance = None
        self.name = name
        self.namespace = namespace
        if struct:
            self._struct = struct
        else:
            self._struct = deepcopy(_mpijob_template)
            self._struct['metadata'] = {'name': name, 'namespace': namespace}
        self._update_container('name', name)
        if image:
            self._update_container('image', image)
        if command:
            self._update_container('command', ['mpirun', 'python'] + command)
        if replicas:
            self._struct['spec']['replicas'] = replicas
        config.load_incluster_config()
        self.api_instance = client.CustomObjectsApi()



    def gpus(self, num, gpu_type='nvidia.com/gpu'):
        self._update_container('resources', {'limits' : {gpu_type: num}})
        return self

