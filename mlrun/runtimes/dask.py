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

from ..execution import MLClientCtx
from .local import get_func_arg
from ..model import RunObject
from .base import RunRuntime
from .kubejob import KubejobRuntime
from ..lists import RunList

class DaskCluster(KubejobRuntime):
    kind = 'dask'

    def __init__(self, kind=None, command=None, args=None, image=None,
                 metadata=None, build=None, volumes=None, volume_mounts=None,
                 env=None, resources=None, image_pull_policy=None,
                 service_account=None, rundb=None, extra_pip=None):
        args = args or ['dask-worker']
        super().__init__(kind, command, args, image, None, metadata, build, volumes,
                         volume_mounts, env, resources, image_pull_policy,
                         service_account, rundb)
        self._cluster = None
        self.extra_pip = extra_pip
        self.build.base_image = self.build.base_image or 'daskdev/dask:latest'
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

    def cluster(self, scale=0):
        if not self._cluster:
            try:
                from dask_kubernetes import KubeCluster
                from dask.distributed import Client
            except ImportError as e:
                print('missing dask_kubernetes, please run "pip install dask_kubernetes"')
                raise e
            self._cluster = KubeCluster(self.to_pod())
            if not scale:
                self._cluster.adapt()
            else:
                self._cluster.scale(scale)
            Client(self._cluster)
        return self._cluster

    @property
    def client(self):
        from dask.distributed import Client, default_client
        try:
            return default_client()
        except ValueError:
            if self._cluster:
                return Client(self._cluster)
            return Client()

    def close(self):
        from dask.distributed import Client, default_client, as_completed
        try:
            client = default_client()
            client.close()
        except ValueError:
            pass
        if self._cluster:
            self._cluster.close()

    def _run(self, runobj: RunObject, execution):
        handler = runobj.spec.handler
        self._force_handler(handler)
        from dask import delayed
        if self.rundb:
            # todo: remote dask via k8s spec env
            environ['MLRUN_META_DBPATH'] = self.rundb

        arg_list = get_func_arg(handler, runobj, execution)
        try:
            task = delayed(handler)(*arg_list)
            out = task.compute()
        except Exception as e:
            err = str(e)
            execution.set_state(error=err)

        if out:
            print('out:', out)
            execution.log_result('return', out)

        return None

    def _run_many(self, tasks, execution, runobj: RunObject):
        handler = runobj.spec.handler
        self._force_handler(handler)
        futures = []
        contexts = []
        tasks = list(tasks)
        for task in tasks:
            ctx = MLClientCtx.from_dict(task.to_dict(),
                                        self.rundb,
                                        autocommit=True)
            args = get_func_arg(handler, task, ctx)
            resp = self.client.submit(handler, *args)
            futures.append(resp)
            contexts.append(ctx)

        resps = self.client.gather(futures)
        results = RunList()
        for r, c, t in zip(resps, contexts, tasks):
            if r:
                c.log_result('return', r)
            # todo: handle task errors
            resp = self._post_run(task=t)
            results.append(resp)

        print(resps)
        return results
