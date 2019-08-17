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
from datetime import datetime
from sys import stdout

from kubernetes import config, client, watch
from kubernetes.client.rest import ApiException
from .platforms.iguazio import v3io_to_vol
from .utils import logger


class k8s_helper:
    def __init__(self, namespace=None, config_file=None):
        self.namespace = namespace
        self._init_k8s_config(config_file)
        self.v1api = client.CoreV1Api()

    def _ns(self, namespace):
        return namespace or self.namespace

    def _init_k8s_config(self, config_file):
        try:
            config.load_incluster_config()
            logger.info('using in-cluster config.')
        except:
            logger.info('cannot find in-cluster config, trying the local kubernetes config.')
            try:
                config.load_kube_config(config_file)
                logger.info('using local kubernetes config.')
            except:
                raise RuntimeError('cannot find local kubernetes config file')

    def list_pods(self, namespace=None, selector='', states=None):
        try:
            resp = self.v1api.list_namespaced_pod(self._ns(namespace), watch=False, label_selector=selector)
        except ApiException as e:
            logger.error('failed to list pods: {}'.format(e))
            raise e

        items = []
        for i in resp.items:
            if not states or i.status.phase in states:
                items.append(i)
        return items

    def clean_pods(self, namespace=None, selector='', states=None):
        if not selector and not states:
            raise ValueError('labels selector or states list must be specified')
        items = self.list_pods(namespace, selector, states)
        for item in items:
            self.del_pod(item.metadata.name, item.metadata.namespace)

    def create_pod(self, pod):
        if hasattr(pod, 'pod'):
            pod = pod.pod
        pod.metadata.namespace = self._ns(pod.metadata.namespace)
        try:
            resp = self.v1api.create_namespaced_pod(pod.metadata.namespace, pod)
        except ApiException as e:
            logger.error('failed to create pod: {}'.format(e))
            raise e

        logger.info(f'Pod {resp.metadata.name} created')
        return resp.metadata.name

    def del_pod(self, name, namespace=None):
        try:
            api_response = self.v1api.delete_namespaced_pod(name,
                                                            self._ns(namespace),
                                                            grace_period_seconds=0,
                                                            propagation_policy='Background')
            return api_response
        except ApiException as e:
            # ignore error if pod is already removed
            if e.status != 404:
                logger.error('failed to delete pod: {}'.format(e))
            raise e

    def get_pod(self, name, namespace=None):
        try:
            api_response = self.v1api.read_namespaced_pod(name=name, namespace=self._ns(namespace))
            return api_response
        except ApiException as e:
            if e.status != 404:
                logger.error('failed to get pod: {}'.format(e))
                raise e
            return None

    def logs(self, name, namespace=None):
        try:
            resp = self.v1api.read_namespaced_pod_log(
                name=name, namespace=self._ns(namespace))
        except ApiException as e:
            logger.error('failed to get pod logs: {}'.format(e))
            raise e

        return resp

    def run_job(self, pod, timeout=600):
        if hasattr(pod, 'pod'):
            pod = pod.pod
        namespace = self._ns(self._ns(pod.metadata.namespace))
        pod_name = self.create_pod(pod)
        if not pod_name:
            logger.error('failed to create pod')
            return 'error'

        start_time = datetime.now()
        while True:
            try:
                pod = self.get_pod(pod_name, namespace)
                if not pod:
                    return 'error'
                status = pod.status.phase.lower()
                if status in ['running', 'completed', 'succeeded']:
                    print('')
                    break
                if status == 'failed':
                    return 'failed'
                elapsed_time = (datetime.now() - start_time).seconds
                if elapsed_time > timeout:
                    return 'timeout'
                time.sleep(2)
                stdout.write('.')
                if status != 'pending':
                    logger.warning(f'pod state in loop is {status}')
            except ApiException as e:
                logger.error('failed waiting for pod: {}\n'.format(str(e)))
                self.del_pod(pod_name, namespace)
                return 'error'
        w = watch.Watch()
        for out in w.stream(self.v1api.read_namespaced_pod_log,
                            name=pod_name, namespace=namespace):
            print(out)
        pod_state = self.get_pod(pod_name, namespace).status.phase.lower()
        self.del_pod(pod_name, namespace)
        if pod_state == 'failed':
            logger.error('pod exited with error')
        return pod_state

    def create_cfgmap(self, name, data, namespace='', labels=None):
        body = client.V1ConfigMap()
        namespace = self._ns(namespace)
        body.data = data
        if name.endswith('*'):
            body.metadata = client.V1ObjectMeta(generate_name=name[:-1],
                                                namespace=namespace,
                                                labels=labels)
        else:
            body.metadata = client.V1ObjectMeta(name=name,
                                                namespace=namespace,
                                                labels=labels)

        try:
            resp = self.v1api.create_namespaced_config_map(namespace, body)
        except ApiException as e:
            logger.error('failed to create configmap: {}'.format(e))
            raise e

        logger.info(f'ConfigMap {resp.metadata.name} created')
        return resp.metadata.name

    def del_cfgmap(self, name, namespace=None):
        try:
            api_response = self.v1api.delete_namespaced_config_map(
                name,
                self._ns(namespace),
                grace_period_seconds=0,
                propagation_policy='Background')

            return api_response
        except ApiException as e:
            # ignore error if ConfigMap is already removed
            if e.status != 404:
                logger.error('failed to delete ConfigMap: {}'.format(e))
            raise e

    def list_cfgmap(self, namespace=None, selector=''):
        try:
            resp = self.v1api.list_namespaced_config_map(self._ns(namespace), watch=False, label_selector=selector)
        except ApiException as e:
            logger.error('failed to list ConfigMaps: {}'.format(e))
            raise e

        items = []
        for i in resp.items:
            items.append(i)
        return items


class BasePod:

    def __init__(self, task_name='', image=None, command=None, args=None, namespace=''):
        self.namespace = namespace
        self.name = ''
        self.task_name = task_name
        self.image = image
        self.command = command
        self.args = args
        self._volumes = []
        self._mounts = []
        self.env = []
        self.labels = {'task-name': task_name}
        self.annotations = None

    @property
    def pod(self):
        return self._get_spec()

    def add_volume(self, volume: client.V1Volume, mount_path):
        self._mounts.append(client.V1VolumeMount(name=volume.name, mount_path=mount_path))
        self._volumes.append(volume)

    def _get_spec(self):

        container = client.V1Container(name='base',
                                       image=self.image,
                                       env=self.env,
                                       command=self.command,
                                       args=self.args,
                                       volume_mounts=self._mounts,
                                       )

        pod_spec = client.V1PodSpec(containers=[container],
                                    restart_policy='Never',
                                    volumes=self._volumes)

        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(generate_name=f'{self.task_name}-',
                                         namespace=self.namespace,
                                         labels=self.labels,
                                         annotations=self.annotations),
            spec=pod_spec)
        return pod

    def mount_v3io(self, remote='~/', mount_path='/User', access_key='', user='', name='v3io'):
        self.add_volume(v3io_to_vol(name, remote, access_key, user),
                        mount_path=mount_path)

    def mount_cfgmap(self, name, path='/config'):
        self.add_volume(client.V1Volume(
            name=name,
            config_map=client.V1ConfigMapVolumeSource(name=name)),
            mount_path=path)


class KanikoPod(BasePod):
    def __init__(self, dockerfile='/User/Dockerfile',
                 context='/User', dest='yhaviv/ktests:latest',
                 secret_name='my-docker'):

        super().__init__('kaniko', 'gcr.io/kaniko-project/executor:latest',
                         args=["--dockerfile", dockerfile,
                               "--context", context,
                               "--destination", dest])

        self.add_volume(client.V1Volume(name='registry-creds',
                                        secret=client.V1SecretVolumeSource(
                                             secret_name=secret_name,
                                             items=[{'key': '.dockerconfigjson', 'path': '.docker/config.json'}],
                                        )),
                        mount_path='/root/')
