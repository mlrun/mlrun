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
import requests

from .base import MLRuntime, task_gen, results_to_iter_status
from mlrun.secrets import SecretsStore


class RemoteRuntime(MLRuntime):
    kind = 'remote'

    def run(self):
        self._secrets = SecretsStore.from_dict(self.struct['spec'])
        self.struct['spec']['secret_sources'] = self._secrets.to_serial()
        log_level = self.struct['spec'].get('log_level', 'info')
        headers = {'x-nuclio-log-level': log_level}
        try:
            resp = requests.put(self.command, json=json.dumps(self.struct), headers=headers)
        except OSError as err:
            print('ERROR: %s', str(err))
            raise OSError('error: cannot run function at url {}'.format(self.command))

        if not resp.ok:
            print('bad resp!!\n', resp.text)
            return None

        logs = resp.headers.get('X-Nuclio-Logs')
        if logs:
            logs = json.loads(logs)
            for line in logs:
                print(line)

        return self.save_run(resp.json())


