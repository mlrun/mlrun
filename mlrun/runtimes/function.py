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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from aiohttp.client import ClientSession


class RemoteRuntime(MLRuntime):
    kind = 'remote'

    def _run(self, struct):
        secrets = SecretsStore.from_dict(struct['spec'])
        struct['spec']['secret_sources'] = secrets.to_serial()
        log_level = struct['spec'].get('log_level', 'info')
        headers = {'x-nuclio-log-level': log_level}
        try:
            resp = requests.put(self.command, json=json.dumps(struct), headers=headers)
        except OSError as err:
            print('ERROR: %s', str(err))
            raise OSError('error: cannot run function at url {}'.format(self.command))

        if not resp.ok:
            print('bad resp!!\n', resp.text)
            return None

        logs = resp.headers.get('X-Nuclio-Logs')
        if logs:
            print(parse_logs(logs))

        return resp.json()

    async def _run_many(self, tasks):
        secrets = SecretsStore.from_dict(self.struct['spec'])
        secrets = secrets.to_serial()
        log_level = self.struct['spec'].get('log_level', 'info')
        headers = {'x-nuclio-log-level': log_level}

        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(
            invoke_async(tasks, self.command, headers, secrets))

        loop.run_until_complete(future)
        return future.result()


def parse_logs(logs):
    logs = json.loads(logs)
    lines = ''
    for line in logs:
        extra = []
        for k, v in line.items():
            if k not in ['time', 'level', 'name', 'message']:
                extra.append('{}={}'.format(k, v))
        print(line)
        line['extra'] = ', '.join(extra)
        lines += '{time:<18} {level:<6} {name}  {message}  {extra}\n'.format(**line)


async def submit(session, url, body, headers=None):
    async with session.put(url, json=body, headers=headers) as response:
        text = await response.text()
        logs = response.headers.get('X-Nuclio-Logs')
        return response.status, text, logs


async def invoke_async(runs, url, headers, secrets):
    results = []
    tasks = []

    async with ClientSession() as session:
        for run in runs:
            run['spec']['secret_sources'] = secrets
            tasks.append(asyncio.ensure_future(
                submit(session, url, json.dumps(run), headers),
            ))

        for status, resp, logs in await asyncio.gather(*tasks):

            if status != 200:
                print("failed to access {} - {}".format(url, resp))
            else:
                results.append(json.loads(resp))

            if logs:
                print('----------\n', parse_logs(logs))

    return results

