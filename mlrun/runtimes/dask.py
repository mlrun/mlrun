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

from datetime import datetime
import json
from os import environ
from .base import MLRuntime, task_gen, results_to_iter_status

class DaskRuntime(MLRuntime):
    kind = 'dask'

    def run(self):
        self._force_handler()
        from dask import delayed
        if self.rundb:
            # todo: remote dask via k8s spec env
            environ['MLRUN_META_DBPATH'] = self.rundb

        task = delayed(self.handler)(self.struct)
        out = task.compute()
        if isinstance(out, dict):
            return out
        return json.loads(out)

    def run_many(self, hyperparams={}):
        start = datetime.now()
        self._force_handler()
        from dask.distributed import Client, default_client, as_completed
        try:
            client = default_client()
        except ValueError:
            client = Client()  # todo: k8s client

        base_struct = self.struct
        tasks = list(task_gen(base_struct, hyperparams))
        results = []
        futures = client.map(self.handler, tasks)
        for batch in as_completed(futures, with_results=True).batches():
            for future, result in batch:
                if result:
                    results.append(json.loads(result))
                else:
                    print("Dask RESULT = None")

        base_struct['status'] = {'start_time': str(start)}
        base_struct['spec']['hyperparams'] = hyperparams
        results_to_iter_status(base_struct, results)
        return base_struct
