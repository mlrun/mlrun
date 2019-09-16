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
from datetime import datetime
import asyncio
from aiohttp.client import ClientSession
import logging
from sys import stdout

from .base import BaseRuntime, RunError
from ..utils import logger
from ..lists import RunList
from ..model import RunObject

from nuclio_sdk import Context as _Context, Logger
from nuclio_sdk.logger import HumanReadableFormatter
from nuclio_sdk import Event
import nuclio


class RemoteRuntime(BaseRuntime):
    kind = 'remote'
    #kind = 'nuclio'

    def __init__(self, metadata=None, spec=None):
        super().__init__(metadata, spec)
        self._config = nuclio.ConfigSpec()
        self.verbose = False
        self.dashboard = ''
        self.kind = ''

    def set_env(self, name, value):
        self._config.set_env(name, value)
        return self

    def set_config(self, key, value):
        self._config.set_config(key, value)
        return self

    def add_volume(self, local, remote, kind='', name='fs',
                   key='', readonly=False):
        self._config.add_volume(local, remote, kind, name, key, readonly)
        return self

    def add_trigger(self, name, spec):
        self._config.add_trigger(name, spec)
        return self

    def with_http(self, workers=8, port=0,
                  host=None, paths=None, canary=None):
        self._config.with_http(workers, port, host, paths, canary)
        return self

    def with_v3io(self):
        self._config.with_v3io()
        return self

    def deploy(self, source='', project='', handler='',
                tag='', archive=False, files=[], output_dir='', kind=None):

        self.set_config('metadata.labels.mlrun/class', self.kind)
        project = project or self.metadata.project or 'mlrun'
        addr = nuclio.deploy_file(source, name=self.metadata.name, project=project,
                                  dashboard_url=self.dashboard, verbose=self.verbose,
                                  spec=self._config, tag=tag, handler=handler, kind=kind,
                                  archive=archive, files=files, output_dir=output_dir)
        self.spec.command = 'http://{}'.format(addr)
        self.kind = kind
        return self.spec.command

    def _run(self, runobj: RunObject, execution):
        if self._secrets:
            runobj.spec.secret_sources = self._secrets.to_serial()
        log_level = execution.log_level
        command = self.spec.command
        if runobj.spec.handler:
            command = '{}/{}'.format(command, runobj.spec.handler_name)
        headers = {'x-nuclio-log-level': log_level}
        try:
            resp = requests.put(command, json=runobj.to_dict(), headers=headers)
        except OSError as err:
            logger.error('error invoking function: {}'.format(err))
            raise OSError('error: cannot run function at url {}'.format(command))

        if not resp.ok:
            logger.error('bad function resp!!\n{}'.format(resp.text))
            raise RunError('bad function response')

        logs = resp.headers.get('X-Nuclio-Logs')
        if logs:
            print(parse_logs(logs))

        return resp.json()

    def _run_many(self, tasks, execution, runobj: RunObject):
        secrets = self._secrets.to_serial() if self._secrets else None
        log_level = execution.log_level
        headers = {'x-nuclio-log-level': log_level}

        command = self.spec.command
        if runobj.spec.handler:
            command = '{}/{}'.format(command, runobj.spec.handler_name)
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(
            self.invoke_async(tasks, command, headers, secrets))

        loop.run_until_complete(future)
        return future.result()

    async def invoke_async(self, runs, url, headers, secrets):
        results = RunList()
        tasks = []

        async with ClientSession() as session:
            for run in runs:
                self.store_run(run)
                run.spec.secret_sources = secrets or []
                tasks.append(asyncio.ensure_future(
                    submit(session, url, run.to_dict(), headers),
                ))

            for status, resp, logs in await asyncio.gather(*tasks):

                if status != 200:
                    logger.error("failed to access {} - {}".format(url, resp))
                else:
                    results.append(json.loads(resp))

                if logs:
                    parsed = parse_logs(logs)
                    if parsed:
                        print(parsed, '----------')

        return results


def parse_logs(logs):
    logs = json.loads(logs)
    lines = ''
    for line in logs:
        extra = []
        for k, v in line.items():
            if k not in ['time', 'level', 'name', 'message']:
                extra.append('{}={}'.format(k, v))
        line['extra'] = ', '.join(extra)
        line['time'] = datetime.fromtimestamp(float(line['time'])/1000).strftime('%Y-%m-%d %H:%M:%S.%f')
        lines += '{time}  {level:<6} {message}  {extra}\n'.format(**line)

    return lines


async def submit(session, url, body, headers=None):
    async with session.put(url, json=body, headers=headers) as response:
        text = await response.text()
        logs = response.headers.get('X-Nuclio-Logs', None)
        return response.status, text, logs


def fake_nuclio_context(body, headers=None):

    class FunctionContext(_Context):
        """Wrapper around nuclio_sdk.Context to make automatically create
        logger"""

        def __getattribute__(self, attr):
            value = object.__getattribute__(self, attr)
            if value is None and attr == 'logger':
                value = self.logger = Logger(level=logging.INFO)
                value.set_handler(
                    'mlrun', stdout, HumanReadableFormatter())
            return value

        def set_logger_level(self, verbose=False):
            if verbose:
                level = logging.DEBUG
            else:
                level = logging.INFO
            value = self.logger = Logger(level=level)
            value.set_handler('mlrun', stdout, HumanReadableFormatter())

    return FunctionContext(), Event(body=body, headers=headers)


