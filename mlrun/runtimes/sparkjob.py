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

from pprint import pprint
import yaml
import time
from copy import deepcopy
from .kubejob import KubejobRuntime, KubejobSpec
from ..model import ModelObj
from ..utils import logger, get_in
from ..model import RunObject
from ..utils import dict_to_yaml, update_in, logger, get_in
from ..platforms.iguazio import mount_v3io, mount_v3iod, mount_spark_conf

from kubernetes import client

igz_deps = {'jars': ['/igz/java/libs/v3io-hcfs_2.11-{0}.jar',
                     '/igz/java/libs/v3io-spark2-streaming_2.11-{0}.jar',
                     '/igz/java/libs/v3io-spark2-object-dataframe_2.11-{0}.jar',
                     '/igz/java/libs/scala-library-2.11.12.jar'],
            'files': ['/igz/java/libs/v3io-py-{0}.zip']}

_sparkjob_template = {
 'apiVersion': 'sparkoperator.k8s.io/v1beta1',
 'kind': 'SparkApplication',
 'metadata': {
     'name': '',
     'namespace': 'default-tenant'
 },
 'spec': {
     'type': 'Python',
     'pythonVersion': '2',
     'mode': 'cluster',
     'image': '',
     'imagePullPolicy': 'Always',
     'mainApplicationFile': '',
     'sparkVersion': '2.4.0',
     'restartPolicy': {
         'type': 'OnFailure',
         'onFailureRetries': 3,
         'onFailureRetryInterval': 10,
         'onSubmissionFailureRetries': 5,
         'onSubmissionFailureRetryInterval': 20,
     },
     'deps': {},
     'volumes': [],
     'driver': {
         'cores': 0.1,
         'coreLimit': '200m',
         'memory': '512m',
         'labels': {},
         'serviceAccount': 'spark-operator-spark',
         'volumeMounts': [],
         'env': [],
     },
     'executor': {
         'cores': 0,
         'instances': 0,
         'memory': '',
         'labels': {},
         'volumeMounts': [],
         'env': [],
     },
 },
}

class SparkJobSpec(KubejobSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 volumes=None, volume_mounts=None, env=None, resources=None, replicas=None,
                 image_pull_policy=None, service_account=None, driver_resources=None,
                 type=None, python_version=None, spark_version=None, restart_policy=None, deps=None):
        super().__init__(command=command,
                         args=args,
                         image=image,
                         mode=mode,
                         volumes=volumes,
                         volume_mounts=volume_mounts,
                         env=env,
                         resources=resources,
                         replicas=replicas,
                         image_pull_policy=image_pull_policy,
                         service_account=service_account)
        self.driver_resources = driver_resources
        self.type = type
        self.python_version = python_version
        self.spark_version = spark_version
        self.restart_policy = restart_policy
        self.deps = deps

class SparkRuntime(KubejobRuntime):
    group = 'sparkoperator.k8s.io'
    version = 'v1beta1'
    apiVersion = group + '/' + version
    kind = 'SparkApplication'
    plural = 'sparkapplications'

    def _run(self, runobj: RunObject, execution):
        job = deepcopy(_sparkjob_template)
        meta = self._get_meta(runobj, True)

        pod_labels = deepcopy(meta.labels)
        pod_labels['mlrun/job'] = meta.name
        update_in(job, 'metadata', meta.to_dict())
        update_in(job, 'spec.driver.labels', pod_labels)
        update_in(job, 'spec.executor.labels', pod_labels)
        update_in(job, 'spec.executor.instances', self.spec.replicas or 1)
        if self.spec.image:
            update_in(job, 'spec.image', self.spec.image)
        update_in(job, 'spec.volumes', self.spec.volumes)
        update_in(job, 'spec.driver.env', self.spec.env)
        update_in(job, 'spec.executor.env', self.spec.env)
        update_in(job, 'spec.driver.volumeMounts', self.spec.volume_mounts)
        update_in(job, 'spec.executor.volumeMounts', self.spec.volume_mounts)
        update_in(job, 'spec.deps', self.spec.deps)
        if 'requests' in self.spec.resources:
            if 'cpu' in self.spec.resources['requests']:
                update_in(job, 'spec.executor.cores', self.spec.resources['requests']['cpu'])
        if 'limits' in self.spec.resources:
            if 'cpu' in self.spec.resources['limits']:
                update_in(job, 'spec.executor.coreLimit', self.spec.resources['limits']['cpu'])
            if 'memory' in self.spec.resources['limits']:
                update_in(job, 'spec.executor.memory', self.spec.resources['limits']['memory'])
        if self.spec.command:
            update_in(job, 'spec.mainApplicationFile', self.spec.command)
        update_in(job, 'spec.args', self.spec.args)
        self._submit_job(job, meta.namespace)

    def _submit_job(self, job, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.ns(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                SparkRuntime.group, SparkRuntime.version, namespace=namespace,
                plural=SparkRuntime.plural, body=job)
            name = get_in(resp, 'metadata.name', 'unknown')
            logger.info('SparkJob {} created'.format(name))
            return resp
        except client.rest.ApiException as e:
            logger.error("Exception when creating SparkJob: %s" % e)

    def _update_igz_jars(self, igz_version, deps=igz_deps):
        if not self.spec.deps:
            self.spec.deps = {}
        if 'jars' in deps:
            if 'jars' not in self.spec.deps:
                self.spec.deps['jars'] = []
            self.spec.deps['jars'] += [x.format(igz_version) for x in deps['jars']]
        if 'files' in deps:
            if 'files' not in self.spec.deps:
                self.spec.deps['files'] = []
            self.spec.deps['files'] += [x.format(igz_version) for x in deps['files']]

    def with_igz_spark(self, igz_version):
        self._update_igz_jars(igz_version=igz_version)
        self.apply(mount_v3io(name='v3io-fuse', remote='/', mount_path='/v3io'))
        self.apply(mount_v3iod())
        self.apply(mount_spark_conf())

    @property
    def spec(self) -> SparkJobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', SparkJobSpec)
