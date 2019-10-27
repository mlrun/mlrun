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
import requests
from datetime import datetime
import asyncio
from aiohttp.client import ClientSession
import logging
from sys import stdout
from kubernetes import client
from nuclio.deploy import deploy_config

from ..kfpops import deploy_op
from ..platforms.iguazio import v3io_to_vol
from .base import BaseRuntime, RunError, FunctionSpec
from ..utils import logger, update_in
from ..lists import RunList
from ..model import RunObject

from nuclio_sdk import Context as _Context, Logger
from nuclio_sdk.logger import HumanReadableFormatter
from nuclio_sdk import Event
import nuclio

serving_handler = 'handler'


def new_model_server(name, model_class: str, models: dict = None, filename='',
                     protocol='', image='', endpoint='', explainer=False,
                     workers=8, canary=None):
    f = RemoteRuntime()
    f.metadata.name = name
    if not image:
        bname, spec, code = nuclio.build_file(filename, handler=serving_handler, kind='serving')
        f.spec.base_spec = spec

    f.serving(models, model_class, protocol, image=image, endpoint=endpoint,
              explainer=explainer, workers=workers, canary=canary)
    return f


class NuclioSpec(FunctionSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 entry_points=None, description=None,
                 volumes=None, env=None, resources=None,
                 config=None, build_commands=None, base_spec=None,
                 source=None, image_pull_policy=None, function_kind=None,
                 service_account=None):
        super().__init__(command=command, args=args, image=image,
                         mode=mode, build=None,
                         entry_points=entry_points, description=description)

        self.base_spec = base_spec or ''
        self.function_kind = function_kind or 'mlrun'
        self.source = source or ''
        self.volumes = volumes or []
        self.build_commands = build_commands or []
        self.env = env or {}
        self.config = config or {}
        self.resources = resources or {}
        self.image_pull_policy = image_pull_policy
        self.service_account = service_account
        self.function_handler = ''


class RemoteRuntime(BaseRuntime):
    kind = 'remote'
    #kind = 'nuclio'

    def __init__(self, metadata=None, spec=None):
        super().__init__(metadata, spec)
        self.verbose = False

    @property
    def spec(self) -> NuclioSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', NuclioSpec)

    def set_env(self, name, value):
        self.spec.env[name] = value
        return self

    def set_config(self, key, value):
        self.spec.config[key] = value
        return self

    def add_volume(self, local, remote, name='fs',
                   access_key='', user=''):
        vol = v3io_to_vol(name, remote=remote, access_key=access_key, user=user)
        api = client.ApiClient()
        vol = api.sanitize_for_serialization(vol)
        self.spec.volumes.append({'volume': vol,
                                  'volumeMount': {'name': name, 'mountPath': local}})
        return self

    def add_trigger(self, name, spec):
        if hasattr(spec, 'to_dict'):
            spec = spec.to_dict()
        self.spec.config['spec.triggers.{}'.format(name)] = spec
        return self

    def with_v3io(self, local='', remote=''):
        for key in ['V3IO_FRAMESD', 'V3IO_USERNAME',
                    'V3IO_ACCESS_KEY', 'V3IO_API']:
            if key in environ:
                self.spec.env[key] = environ[key]
        if local and remote:
            self.add_volume(local, remote)
        return self

    def with_http(self, workers=8, port=0,
                  host=None, paths=None, canary=None):
        self.add_trigger('http', nuclio.HttpTrigger(
            workers, port=port, host=host, paths=paths, canary=canary))
        return self

    def add_model(self, key, model):
        if model.startswith('v3io://'):
            model = '/User/' + '/'.join(model.split('/')[5:])
        if '://' not in model:
            model = 'file://' + model
        if not model.endswith('/'):
            model = model[:model.rfind('/')]
        self.set_env('SERVING_MODEL_{}'.format(key), model)
        return self

    def serving(self, models: dict = None, model_class='', protocol='', image='',
                endpoint='', explainer=False, workers=8, canary=None):

        if models:
            for k, v in models.items():
                self.set_env('SERVING_MODEL_{}'.format(k), v)

        self.set_env('TRANSPORT_PROTOCOL', protocol or 'seldon')
        self.set_env('ENABLE_EXPLAINER', str(explainer))
        self.set_env('MODEL_CLASS', model_class)
        self.with_http(workers, host=endpoint, canary=canary)
        self.spec.function_kind = 'serving'

        if image:
            config = nuclio.config.new_config()
            update_in(config, 'spec.handler',
                      self.spec.function_handler or 'main:{}'.format(serving_handler))
            update_in(config, 'spec.image', image)
            update_in(config, 'spec.build.codeEntryType', 'image')
            self.spec.base_spec = config

        return self

    def deploy(self, source='', dashboard='', project='', tag='',
               kind=None):

        self.set_config('metadata.labels.mlrun/class', self.kind)
        spec = nuclio.ConfigSpec(env=self.spec.env, config=self.spec.config)
        spec.cmd = self.spec.build_commands
        kind = kind or self.spec.function_kind
        project = project or self.metadata.project or 'mlrun'
        source = source or self.spec.source
        handler = self.spec.function_handler

        if self.spec.base_spec:
            config = nuclio.config.extend_config(self.spec.base_spec, spec, tag,
                                                 self.spec.source)
            update_in(config, 'metadata.name', self.metadata.name)
            update_in(config, 'spec.volumes', self.spec.volumes)

            addr = nuclio.deploy.deploy_config(
                config, dashboard, name=self.metadata.name,
                project=project, tag=tag, verbose=self.verbose,
                create_new=True)
        else:

            name, config, code = nuclio.build_file(source, name=self.metadata.name,
                                            project=project,
                                            handler=handler,
                                            tag=tag, spec=spec,
                                            kind=kind, verbose=self.verbose)

            update_in(config, 'spec.volumes', self.spec.volumes)
            addr = deploy_config(config, dashboard_url=dashboard, name=name, project=project,
                                 tag=tag, verbose=self.verbose, create_new=True)

        self.spec.command = 'http://{}'.format(addr)
        return self.spec.command

    def deploy_step(self, source='', dashboard='', project='', models={}):
        name = 'deploy_{}'.format(self.metadata.name or 'function')
        return deploy_op(name, self, source=source, dashboard=dashboard,
                         project=project, models=models)

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


