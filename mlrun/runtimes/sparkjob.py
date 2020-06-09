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
from typing import Tuple

from kubernetes import client
from kubernetes.client.rest import ApiException

from mlrun.runtimes.base import BaseRuntimeHandler
from .base import RunError
from .kubejob import KubejobRuntime
from .pod import KubeResourceSpec
from ..execution import MLClientCtx
from ..model import RunObject
from ..platforms.iguazio import mount_v3io, mount_v3iod
from ..utils import update_in, logger, get_in

igz_deps = {'jars': ['/igz/java/libs/v3io-hcfs_2.11-{0}.jar',
                     '/igz/java/libs/v3io-spark2-streaming_2.11-{0}.jar',
                     '/igz/java/libs/v3io-spark2-object-dataframe_2.11-{0}.jar',
                     '/igz/java/libs/scala-library-2.11.12.jar'],
            'files': ['/igz/java/libs/v3io-py-{0}.zip']}

_sparkjob_template = {
 'apiVersion': 'sparkoperator.k8s.io/v1beta2',
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
     'serviceAccount': 'spark-operator-spark',
     'driver': {
         'cores': 1,
         'coreLimit': '1200m',
         'memory': '512m',
         'labels': {},
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


class SparkJobSpec(KubeResourceSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 volumes=None, volume_mounts=None, env=None, resources=None,
                 replicas=None, image_pull_policy=None, service_account=None,
                 image_pull_secret=None, driver_resources=None, type=None,
                 python_version=None, spark_version=None, restart_policy=None,
                 deps=None):

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
                         service_account=service_account,
                         image_pull_secret=image_pull_secret)

        self.driver_resources = driver_resources
        self.type = type
        self.python_version = python_version
        self.spark_version = spark_version
        self.restart_policy = restart_policy
        self.deps = deps


class SparkRuntime(KubejobRuntime):
    group = 'sparkoperator.k8s.io'
    version = 'v1beta2'
    apiVersion = group + '/' + version
    kind = 'spark'
    plural = 'sparkapplications'

    def _run(self, runobj: RunObject, execution: MLClientCtx):
        if runobj.metadata.iteration:
            self.store_run(runobj)
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

        extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
        # if self.spec.rundb:
        #     extra_env['MLRUN_DBPATH'] = self.spec.rundb
        extra_env = [{'name': k, 'value': v} for k, v in extra_env.items()]

        update_in(job, 'spec.driver.env', extra_env + self.spec.env)
        update_in(job, 'spec.executor.env', extra_env + self.spec.env)
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
        update_in(job, 'spec.arguments', self.spec.args)
        resp = self._submit_job(job, meta.namespace)
        name = get_in(resp, 'metadata.name', 'unknown')

        state = get_in(resp, 'status.applicationState.state','SUBMITTED')
        logger.info('SparkJob {} state={}'.format(meta.name, 'STARTING'))
        while state not in ["RUNNING", "COMPLETED", "FAILED"]:
            resp = self.get_job(meta.name, meta.namespace)
            state = get_in(resp, 'status.applicationState.state')
            time.sleep(1)

        if state == "FAILED":
            logger.error('SparkJob {} state={}'.format(meta.name, state or 'unknown'))
            execution.set_state('error', 'SparkJob {} finished with state {}'.format(meta.name, state or 'unknown'))

        if resp:
            logger.info('SparkJob {} state={}'.format(meta.name, state or 'unknown'))
            if state:
                driver, status = self._get_driver(meta.name, meta.namespace)
                execution.set_hostname(driver)
                execution.set_state(state.lower())
                if self.kfp:
                    status = self._get_k8s().watch(driver, meta.namespace)
                    logger.info('SparkJob {} finished with state {}'.format(meta.name, status))
                    if status == 'succeeded':
                        execution.set_state('completed')
                    else:
                        execution.set_state('error', 'SparkJob {} finished with state {}'.format(meta.name, status))
                else:
                    logger.info('SparkJob {} driver pod {} state {}'.format(meta.name, driver, status))
                    logger.info('use .watch({}) to see logs'.format(meta.name))
            else:
                logger.error('SparkJob status unknown or failed, check pods: {}'.format(self.get_pods(meta.name, meta.namespace)))
                execution.set_state('error', 'SparkJob {} finished with unknown state'.format(meta.name))

        return None

    def _submit_job(self, job, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.create_namespaced_custom_object(
                SparkRuntime.group, SparkRuntime.version, namespace=namespace,
                plural=SparkRuntime.plural, body=job)
            name = get_in(resp, 'metadata.name', 'unknown')
            logger.info('SparkJob {} created'.format(name))
            return resp
        except client.rest.ApiException as e:
            crd = '{}/{}/{}'.format(SparkRuntime.group, SparkRuntime.version, SparkRuntime.plural)
            logger.error("Exception when creating SparkJob ({}): {}".format(crd, e))
            raise RunError("Exception when creating SparkJob: %s" % e)

    def get_job(self, name, namespace=None):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        try:
            resp = k8s.crdapi.get_namespaced_custom_object(
                SparkRuntime.group, SparkRuntime.version, namespace, SparkRuntime.plural, name)
        except client.rest.ApiException as e:
            print("Exception when reading SparkJob: %s" % e)
        return resp

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
        self.apply(mount_v3iod(namespace='default-tenant', v3io_config_configmap='spark-operator-v3io-config', v3io_auth_secret='spark-operator-v3io-auth'))

    def get_pods(self, name=None, namespace=None, driver=False):
        k8s = self._get_k8s()
        namespace = k8s.resolve_namespace(namespace)
        selector = 'mlrun/class=spark'
        if name:
            selector += ',sparkoperator.k8s.io/app-name={}'.format(name)
        if driver:
            selector += ',spark-role=driver'
        pods = k8s.list_pods(selector=selector, namespace=namespace)
        if pods:
            return {p.metadata.name: p.status.phase for p in pods}

    def _get_driver(self, name, namespace=None):
        pods = self.get_pods(name, namespace, driver=True)
        if not pods:
            logger.error('no pod matches that job name')
            return
        k8s = self._get_k8s()
        return list(pods.items())[0]

    @property
    def spec(self) -> SparkJobSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', SparkJobSpec)


class SparkRuntimeHandler(BaseRuntimeHandler):

    @staticmethod
    def _get_object_label_selector(object_id: str) -> str:
        return f'mlrun/uid={object_id}'

    @staticmethod
    def _get_default_label_selector() -> str:
        return 'mlrun/class=spark'

    @staticmethod
    def _get_crd_info() -> Tuple[str, str, str]:
        return SparkRuntime.group, SparkRuntime.version, SparkRuntime.plural

    @staticmethod
    def _is_crd_object_in_transient_state(crd_object) -> bool:
        # it is less likely that there will be new stable states, or the existing ones will change so better to resolve
        # whether it's a transient state by checking if it's not a stable state
        return crd_object.get('status', {}).get('applicationState', {}).get('state') not in ['COMPLETED', 'FAILED']
