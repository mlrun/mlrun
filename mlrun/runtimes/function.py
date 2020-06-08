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
from urllib.parse import urlparse

import requests
from datetime import datetime
import asyncio
from aiohttp.client import ClientSession
from nuclio.deploy import deploy_config
import nuclio

from .pod import KubeResourceSpec, KubeResource
from ..kfpops import deploy_op
from ..platforms.iguazio import mount_v3io, add_or_refresh_credentials, is_iguazio_endpoint
from .base import RunError, FunctionStatus
from .utils import log_std, set_named_item, get_item_name
from ..utils import logger, update_in, get_in, tag_image
from ..lists import RunList
from ..model import RunObject
from ..config import config as mlconf

serving_handler = 'handler'
default_max_replicas = 4


def new_model_server(name, model_class: str, models: dict = None, filename='',
                     protocol='', image='', endpoint='', explainer=False,
                     workers=8, canary=None, handler=None):
    f = RemoteRuntime()
    if not image:
        name, spec, code = nuclio.build_file(
            filename, name=name, handler=serving_handler, kind='serving')
        f.spec.base_spec = spec
    elif handler:
        f.spec.function_handler = handler

    f.metadata.name = name
    f.serving(models, model_class, protocol, image=image, endpoint=endpoint,
              explainer=explainer, workers=workers, canary=canary)
    return f


class NuclioSpec(KubeResourceSpec):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 entry_points=None, description=None, replicas=None,
                 min_replicas=None, max_replicas=None,
                 volumes=None, volume_mounts=None, env=None, resources=None,
                 config=None, base_spec=None, no_cache=None,
                 source=None, image_pull_policy=None, function_kind=None,
                 service_account=None):

        super().__init__(command=command,
                         args=args,
                         image=image,
                         mode=mode,
                         volumes=volumes,
                         volume_mounts=volume_mounts,
                         env=env,
                         resources=resources,
                         replicas=replicas,
                         image_pull_policy=image_pull_policy,
                         service_account=service_account)

        super().__init__(command=command, args=args, image=image,
                         mode=mode, build=None,
                         entry_points=entry_points, description=description)

        self.base_spec = base_spec or ''
        self.function_kind = function_kind
        self.source = source or ''
        self.config = config or {}
        self.function_handler = ''
        self.no_cache = no_cache
        self.replicas = replicas
        self.min_replicas = min_replicas or 0
        self.max_replicas = max_replicas or default_max_replicas

    @property
    def volumes(self) -> list:
        return list(self._volumes.values())

    @volumes.setter
    def volumes(self, volumes):
        self._volumes = {}
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

    @property
    def volume_mounts(self) -> list:
        return list(self._volume_mounts.values())

    @volume_mounts.setter
    def volume_mounts(self, volume_mounts):
        self._volume_mounts = {}
        if volume_mounts:
            for vol in volume_mounts:
                set_named_item(self._volume_mounts, vol)

    def update_vols_and_mounts(self, volumes, volume_mounts):
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

        if volume_mounts:
            for vol in volume_mounts:
                set_named_item(self._volume_mounts, vol)

    def to_nuclio_vol(self):
        vols = []
        for name, vol in self._volumes.items():
            if name not in self._volume_mounts:
                raise ValueError('found volume without a volume mount ({})'.format(name))
            vols.append({
                'volume': vol,
                'volumeMount': self._volume_mounts[name],
            })
        return vols


class NuclioStatus(FunctionStatus):
    def __init__(self, state=None, nuclio_name=None, address=None):
        super().__init__(state)

        self.nuclio_name = nuclio_name
        self.address = address


class RemoteRuntime(KubeResource):
    kind = 'remote'

    @property
    def spec(self) -> NuclioSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', NuclioSpec)

    @property
    def status(self) -> NuclioStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, 'status', NuclioStatus)

    def set_config(self, key, value):
        self.spec.config[key] = value
        return self

    def add_volume(self, local, remote, name='fs',
                   access_key='', user=''):
        raise Exception('deprecated, use .apply(mount_v3io())')

    def add_trigger(self, name, spec):
        if hasattr(spec, 'to_dict'):
            spec = spec.to_dict()
        self.spec.config['spec.triggers.{}'.format(name)] = spec
        return self

    def with_v3io(self, local='', remote=''):
        for key in ['V3IO_FRAMESD', 'V3IO_USERNAME',
                    'V3IO_ACCESS_KEY', 'V3IO_API']:
            if key in environ:
                self.set_env(key, environ[key])
        if local and remote:
            self.apply(mount_v3io(remote=remote, mount_path=local))
        return self

    def with_http(self, workers=8, port=0, host=None,
                  paths=None, canary=None, secret=None):
        self.add_trigger('http', nuclio.HttpTrigger(
            workers, port=port, host=host, paths=paths,
            canary=canary, secret=secret))
        return self

    def add_model(self, key, model):
        if model.startswith('v3io://'):
            model = '/User/' + '/'.join(model.split('/')[5:])
        self.set_env('SERVING_MODEL_{}'.format(key), model)
        return self

    def serving(
      self, models: dict = None, model_class='', protocol='', image='',
      endpoint='', explainer=False, workers=8, canary=None):

        if models:
            for k, v in models.items():
                self.set_env('SERVING_MODEL_{}'.format(k), v)

        if protocol:
            self.set_env('TRANSPORT_PROTOCOL', protocol)
        if model_class:
            self.set_env('MODEL_CLASS', model_class)
        self.set_env('ENABLE_EXPLAINER', str(explainer))
        self.with_http(workers, host=endpoint, canary=canary)
        self.spec.function_kind = 'serving'

        if image:
            config = nuclio.config.new_config()
            update_in(config, 'spec.handler',
                      self.spec.function_handler or 'main:{}'.format(
                          serving_handler))
            update_in(config, 'spec.image', image)
            update_in(config, 'spec.build.codeEntryType', 'image')
            self.spec.base_spec = config

        return self

    def deploy(self, dashboard='', project='', tag='', kind=None):

        def get_fullname(config, name, project, tag):
            if project:
                name = '{}-{}'.format(project, name)
            if tag:
                name = '{}-{}'.format(name, tag)
            update_in(config, 'metadata.name', name)
            return name

        self.set_config('metadata.labels.mlrun/class', self.kind)
        env_dict = {get_item_name(v): get_item_name(v, 'value')
                    for v in self.spec.env}
        spec = nuclio.ConfigSpec(env=env_dict, config=self.spec.config)
        spec.cmd = self.spec.build.commands or []
        project = project or self.metadata.project or 'default'
        handler = self.spec.function_handler
        if self.spec.no_cache:
            spec.set_config('spec.build.noCache', True)
        if self.spec.replicas:
            spec.set_config('spec.minReplicas', self.spec.replicas)
            spec.set_config('spec.maxReplicas', self.spec.replicas)
        else:
            spec.set_config('spec.minReplicas', self.spec.min_replicas)
            spec.set_config('spec.maxReplicas', self.spec.max_replicas)

        dashboard = get_auth_filled_platform_dashboard_url(dashboard)
        if self.spec.base_spec:
            if kind:
                raise ValueError('kind cannot be specified on built functions')
            config = nuclio.config.extend_config(
                self.spec.base_spec, spec, tag, self.spec.build.code_origin)
            update_in(config, 'metadata.name', self.metadata.name)
            update_in(config, 'spec.volumes', self.spec.to_nuclio_vol())
            base_image = get_in(config, 'spec.build.baseImage')
            if base_image:
                update_in(config, 'spec.build.baseImage', tag_image(base_image))

            logger.info('deploy started')
            name = get_fullname(config, self.metadata.name, project, tag)
            addr = nuclio.deploy.deploy_config(
                config, dashboard, name=name,
                project=project, tag=tag, verbose=self.verbose,
                create_new=True)
        else:

            kind = kind if kind is not None else self.spec.function_kind
            name, config, code = nuclio.build_file(self.spec.source,
                                                   name=self.metadata.name,
                                                   project=project,
                                                   handler=handler,
                                                   tag=tag, spec=spec,
                                                   kind=kind,
                                                   verbose=self.verbose)

            update_in(config, 'spec.volumes', self.spec.to_nuclio_vol())
            name = get_fullname(config, name, project, tag)
            addr = deploy_config(
                config, dashboard_url=dashboard, name=name, project=project,
                tag=tag, verbose=self.verbose, create_new=True)

        self.spec.command = 'http://{}'.format(addr)
        self.status.nuclio_name = name
        if addr:
            self.status.state = 'ready'
            self.status.address = addr
            self.save()
        return self.spec.command

    def deploy_step(self, dashboard='', project='', models=None, env=None,
                    tag=None, verbose=None):
        models = {} if models is None else models
        name = 'deploy_{}'.format(self.metadata.name or 'function')
        project = project or self.metadata.project
        dashboard = get_auth_filled_platform_dashboard_url(dashboard)
        return deploy_op(name, self, dashboard=dashboard,
                         project=project, models=models, env=env,
                         tag=tag, verbose=verbose)

    def _raise_mlrun(self):
        if self.spec.function_kind != 'mlrun':
            raise RunError('.run() can only be execute on "mlrun" kind'
                           ', recreate with function kind "mlrun"')

    def _run(self, runobj: RunObject, execution):
        self._raise_mlrun()
        self.store_run(runobj)
        if self._secrets:
            runobj.spec.secret_sources = self._secrets.to_serial()
        log_level = execution.log_level
        command = self.spec.command
        if runobj.spec.handler:
            command = '{}/{}'.format(command, runobj.spec.handler_name)
        headers = {'x-nuclio-log-level': log_level}
        try:
            resp = requests.put(
                command, json=runobj.to_dict(), headers=headers)
        except OSError as err:
            logger.error('error invoking function: {}'.format(err))
            raise OSError(
                'error: cannot run function at url {}'.format(command))

        if not resp.ok:
            logger.error('bad function resp!!\n{}'.format(resp.text))
            raise RunError('bad function response')

        logs = resp.headers.get('X-Nuclio-Logs')
        if logs:
            log_std(self._db_conn, runobj, parse_logs(logs))

        return self._update_state(resp.json())

    def _run_many(self, tasks, execution, runobj: RunObject):
        self._raise_mlrun()
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

    def _update_state(self, rundict: dict):
        last_state = get_in(rundict, 'status.state', '')
        if last_state != 'error':
            update_in(rundict, 'status.state', 'completed')
        self._store_run_dict(rundict)
        return rundict

    async def invoke_async(self, runs, url, headers, secrets):
        results = RunList()
        tasks = []

        async with ClientSession() as session:
            for run in runs:
                self.store_run(run)
                run.spec.secret_sources = secrets or []
                tasks.append(asyncio.ensure_future(
                    submit(session, url, run, headers),
                ))

            for status, resp, logs, run in await asyncio.gather(*tasks):

                if status != 200:
                    logger.error("failed to access {} - {}".format(url, resp))
                else:
                    results.append(self._update_state(json.loads(resp)))

                if logs:
                    log_std(self._db_conn, run, parse_logs(logs))

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
        line['time'] = datetime.fromtimestamp(
            float(line['time'])/1000).strftime('%Y-%m-%d %H:%M:%S.%f')
        lines += '{time}  {level:<6} {message}  {extra}\n'.format(**line)

    return lines


async def submit(session, url, run, headers=None):
    async with session.put(url, json=run.to_dict(), headers=headers) as response:
        text = await response.text()
        logs = response.headers.get('X-Nuclio-Logs', None)
        return response.status, text, logs, run


def fake_nuclio_context(body, headers=None):
    return nuclio.Context(), nuclio.Event(body=body, headers=headers)


def _fullname(project, name):
    if project:
        return '{}-{}'.format(project, name)
    return name


def get_auth_filled_platform_dashboard_url(dashboard: str) -> str:
    if not is_iguazio_endpoint(mlconf.dbpath):
        return dashboard or mlconf.nuclio_dashboard_url

    # todo: workaround for 2.8 use nuclio_dashboard_url for subdns name
    parsed_dbpath = urlparse(mlconf.dbpath)
    user, control_session = add_or_refresh_credentials(parsed_dbpath.hostname)
    return 'https://{}:{}@{}{}'.format(
        user,
        control_session,
        mlconf.nuclio_dashboard_url or 'nuclio-api-ext',
        parsed_dbpath.hostname[parsed_dbpath.hostname.find('.'):])
