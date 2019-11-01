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
from .kubejob import KubejobRuntime, KubejobSpec
from ..lists import RunList
from ..config import config

class DaskSpec(KubejobSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 volumes=None, volume_mounts=None, env=None, resources=None, build=None,
                 entry_points=None, description=None, replicas=None,
                 image_pull_policy=None, service_account=None, extra_pip=None):

        super().__init__(command=command, args=args, image=image,
                         mode=mode, volumes=volumes, volume_mounts=volume_mounts,
                         env=env, resources=resources, replicas=replicas, image_pull_policy=image_pull_policy,
                         service_account=service_account, build=build,
                         entry_points=entry_points, description=description)
        self.extra_pip = extra_pip
        self.args = args or ['dask-worker']


class DaskCluster(KubejobRuntime):
    kind = 'dask'
    _is_nested = False

    def __init__(self, spec=None,
                 metadata=None):
        super().__init__(spec, metadata)
        self._cluster = None
        self.spec.build.base_image = self.spec.build.base_image or 'daskdev/dask:latest'
        self.set_label('mlrun/class', self.kind)

    @property
    def spec(self) -> DaskSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', DaskSpec)

    def to_pod(self):
        image = self._image_path() or 'daskdev/dask:latest'
        env = self.spec.env
        namespace = self.metadata.namespace or config.namespace
        if self.spec.extra_pip:
            env.append(self.spec.extra_pip)
        container = client.V1Container(name='base',
                                       image=image,
                                       env=env,
                                       command=None,
                                       args=self.spec.args,
                                       image_pull_policy=self.spec.image_pull_policy,
                                       volume_mounts=self.spec.volume_mounts,
                                       resources=self.spec.resources)

        pod_spec = client.V1PodSpec(containers=[container],
                                    restart_policy='Never',
                                    volumes=self.spec.volumes,
                                    service_account=self.spec.service_account)

        meta = client.V1ObjectMeta(namespace=namespace,
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
        if self.spec.rundb:
            # todo: remote dask via k8s spec env
            environ['MLRUN_DBPATH'] = self.spec.rundb

        arg_list = get_func_arg(handler, runobj, execution)
        try:
            task = delayed(handler)(*arg_list)
            out = task.compute()
        except Exception as e:
            err = str(e)
            execution.set_state(error=err)

        if out:
            execution.log_result('return', out)

        return execution.to_dict()

    def _run_many(self, tasks, execution, runobj: RunObject):
        handler = runobj.spec.handler
        self._force_handler(handler)
        futures = []
        contexts = []
        tasks = list(tasks)
        for task in tasks:
            ctx = MLClientCtx.from_dict(task.to_dict(),
                                        self.spec.rundb,
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
