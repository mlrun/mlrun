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

from ..model import RunObject
from .base import MLRuntime
from ..lists import RunList

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
