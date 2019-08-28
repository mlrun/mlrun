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
from os import environ
from kubernetes import client

from ..model import RunObject, K8sRuntime
from .base import MLRuntime
from ..lists import RunList

class DaskCluster(K8sRuntime):
    kind = 'dask'
    def __init__(self, kind=None, command=None, args=None, image=None,
                 metadata=None, build=None, volumes=None, volume_mounts=None,
                 env=None, resources=None, image_pull_policy=None,
                 service_account=None, extra_pip=None):
        args = args or ['dask-worker']
        super().__init__(kind, command, args, image, metadata, build, volumes,
                         volume_mounts, env, resources, image_pull_policy,
                         service_account)
        self._cluster = None
        self.extra_pip = extra_pip
        self.set_label('mlrun/class', self.kind)

    def to_pod(self):
        image = self.image or 'daskdev/dask:latest'
        env = self.env
        if self.extra_pip:
            env.append(self.extra_pip)
        container = client.V1Container(name='base',
                                       image=image,
                                       env=self.env,
                                       command=None,
                                       args=self.args,
                                       image_pull_policy=self.image_pull_policy,
                                       volume_mounts=self.volume_mounts,
                                       resources=self.resources)

        pod_spec = client.V1PodSpec(containers=[container],
                                    restart_policy='Never',
                                    volumes=self.volumes,
                                    service_account=self.service_account)

        meta = client.V1ObjectMeta(namespace=self.metadata.namespace or 'default-tenant',
                                   labels=self.metadata.labels,
                                   annotations=self.metadata.annotations)

        pod = client.V1Pod(metadata=meta, spec=pod_spec)
        return pod

    @property
    def initialized(self):
        return True if self._cluster else False

    @property
    def cluster(self):
        if not self._cluster:
            try:
                from dask_kubernetes import KubeCluster
            except ImportError as e:
                print('missing dask_kubernetes, please run "pip install dask_kubernetes"')
                raise e
            self._cluster = KubeCluster(self.to_pod())
        return self._cluster

    @property
    def client(self):
        import distributed
        return distributed.Client(self.cluster)


class DaskRuntime(MLRuntime):
    kind = 'dask'

    def _run(self, runobj: RunObject):
        self._force_handler()
        from dask import delayed
        if self.rundb:
            # todo: remote dask via k8s spec env
            environ['MLRUN_META_DBPATH'] = self.rundb

        task = delayed(self.handler)(runobj.to_dict())
        out = task.compute()
        if isinstance(out, dict):
            return out
        return json.loads(out)

    def _run_many(self, tasks):
        self._force_handler()
        from dask.distributed import Client, default_client, as_completed
        try:
            client = default_client()
        except ValueError:
            client = Client()  # todo: k8s client

        tasks = list(tasks)
        results = RunList()
        futures = client.map(self.handler, tasks)
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                if result:
                    result = self._post_run(result)
                    results.append(result)
                else:
                    print("Dask RESULT = None")

        return results
