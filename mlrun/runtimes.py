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
import logging
import os
import uuid
import inspect
from ast import literal_eval
from copy import deepcopy
from os import environ
from tempfile import mktemp

import requests
import yaml

from .kfp import write_kfpmeta
from .execution import MLClientCtx
from .rundb import get_run_db
from .secrets import SecretsStore
from sys import executable, stderr
from subprocess import run, PIPE


def get_or_create_ctx(name, uid='', event=None, spec=None, with_env=True, rundb=''):

    config = environ.get('MLRUN_EXEC_CONFIG')
    if with_env and config:
        spec = config

    if event:
        spec = event.body
        uid = uid or event.id

    if spec and not isinstance(spec, dict):
        spec = yaml.safe_load(spec)

    if spec and spec.get('spec'):
        uid = uid or spec['spec'].get('uid')
    uid = uid or uuid.uuid4().hex

    autocommit = False
    tmp = environ.get('MLRUN_META_TMPFILE')
    out = environ.get('MLRUN_META_DBPATH', rundb)
    if out:
        autocommit = True

    ctx = MLClientCtx(name, uid, rundb=out, autocommit=autocommit, tmp=tmp)
    if spec:
        ctx.from_dict(spec)
    return ctx


def run_start(struct, command='', args=[], runtime=None, rundb='',
              kfp=False, handler=None, hyperparams=None):

    if not runtime and handler:
        runtime = HandlerRuntime(handler=handler)
    else:
        if runtime:
            if isinstance(runtime, str):
                runtime = literal_eval(runtime)
            if not isinstance(runtime, dict):
                runtime = runtime.to_dict()

            if 'spec' not in struct.keys():
                struct['spec'] = {}
            struct['spec']['runtime'] = runtime

        if struct and 'spec' in struct.keys() and 'runtime' in struct['spec'].keys():
            kind = struct['spec']['runtime'].get('kind', '')
            command = struct['spec']['runtime'].get('command', '')
            if kind == 'remote' or (kind == '' and '://' in command):
                runtime = RemoteRuntime()
            elif kind in ['', 'local']:
                runtime = LocalRuntime()
            elif kind == 'mpijob':
                runtime = MpiRuntime()
            elif kind == 'dask':
                runtime = DaskRuntime()
            else:
                raise Exception('unsupported runtime - %s' % kind)

        elif command:
            if '://' in command:
                runtime = RemoteRuntime(command, args)
            else:
                runtime = LocalRuntime(command, args)

        else:
            raise Exception('runtime was not specified via struct or runtime or command!')

    runtime.rundb = rundb
    runtime.handler = handler
    runtime.process_struct(struct)
    if hyperparams:
        resp = runtime.run_many(hyperparams)
    else:
        resp = runtime.run()

    if not resp:
        return {}

    if isinstance(resp, str):
        resp = json.loads(resp)

    if kfp:
        write_kfpmeta(resp)
    return resp


def grid_to_list(params={}):
    arr = {}
    lastlen = 1
    for pk, pv in params.items():
        for p in arr.keys():
            arr[p] = arr[p] * len(pv)
        expanded = []
        for i in range(len(pv)):
            expanded += [pv[i]] * lastlen
        arr[pk] = expanded
        lastlen = lastlen * len(pv)

    return arr


def task_gen(struct, hyperparams):
    i = 0
    params = grid_to_list(hyperparams)
    max = len(next(iter(params.values())))
    base_uid = struct['spec'].get('uid', uuid.uuid4().hex)
    if not struct['spec'].get('parameters'):
        struct['spec']['parameters'] = {}

    while i < max:
        newstruct = deepcopy(struct)
        for key, values in params.items():
            newstruct['spec']['parameters'][key] = values[i]
        newstruct['spec']['uid'] = f'{base_uid}-{i}'
        i += 1
        yield newstruct


class MLRuntime:
    kind = ''
    def __init__(self, command='', args=[], handler=None):
        self.struct = None
        self.command = command
        self.args = args
        self.handler = handler
        self.rundb = ''
        self.hyperparams = None

    def process_struct(self, struct):
        self.struct = struct
        if 'spec' not in self.struct.keys():
            self.struct['spec'] = {}
        if 'runtime' not in self.struct['spec'].keys():
            self.struct['spec']['runtime'] = {}

        self.struct['spec']['runtime']['kind'] = self.kind
        if self.command:
            self.struct['spec']['runtime']['command'] = self.command
        else:
            self.command = self.struct['spec']['runtime'].get('command')
        if self.args:
            self.struct['spec']['runtime']['args'] = self.args
        else:
            self.args = self.struct['spec']['runtime'].get('args', [])

    def run(self):
        pass

    def run_many(self, hyperparams={}):
        base_struct = self.struct
        results = []
        for task in task_gen(base_struct, hyperparams):
            self.struct = task
            resp = self.run()
            results.append(resp)

        return results

    def _force_handler(self):
        if not self.handler:
            raise ValueError('handler must be provided for {} runtime'.format(self.kind))


class LocalRuntime(MLRuntime):
    kind = 'local'

    def run(self):
        environ['MLRUN_EXEC_CONFIG'] = json.dumps(self.struct)
        tmp = mktemp('.json')
        environ['MLRUN_META_TMPFILE'] = tmp
        if self.rundb:
            environ['MLRUN_META_DBPATH'] = self.rundb

        cmd = [executable, self.command]
        if self.args:
            cmd += self.args
        out = run(cmd, stdout=PIPE, stderr=PIPE)
        if out.returncode != 0:
            print(out.stderr.decode('utf-8'), file=stderr)
        print(out.stdout.decode('utf-8'))

        try:
            with open(tmp) as fp:
                resp = fp.read()
            os.remove(tmp)
            if resp:
                return json.loads(resp)
        except FileNotFoundError as err:
            print(err)


class RemoteRuntime(MLRuntime):
    kind = 'remote'

    def run(self):
        secrets = SecretsStore()
        secrets.from_dict(self.struct['spec'])
        self.struct['spec']['secret_sources'] = secrets.to_serial()
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

        if self.rundb:
            rundb = get_run_db(self.rundb)
            rundb.connect(secrets)
            rundb.store_run(resp.json(), commit=True)

        return resp.json()


class MpiRuntime(MLRuntime):
    kind = 'mpijob'

    def run(self):
        from .mpijob import MpiJob
        uid = self.struct['spec'].get('uid', uuid.uuid4().hex)
        self.struct['spec']['uid'] = uid
        runtime = self.struct['spec']['runtime']

        mpijob = MpiJob.from_dict(runtime.get('spec'))

        mpijob.env('MLRUN_EXEC_CONFIG', json.dumps(self.struct))
        if self.rundb:
            mpijob.env('MLRUN_META_DBPATH', self.rundb)

        mpijob.submit()

        if self.rundb:
            print(uid)

        return None


class HandlerRuntime(MLRuntime):
    kind = 'handler'

    def run(self):
        self._force_handler()
        if self.rundb:
            environ['MLRUN_META_DBPATH'] = self.rundb

        args = inspect.signature(self.handler).parameters
        if len(args) > 1 and args[0] == 'context':
            # its a nuclio function
            from .utils import fake_nuclio_context
            context, event = fake_nuclio_context(self.struct)
            out = self.handler(context, event)
        elif len(args) == 1:
            out = self.handler(self.struct)
        else:
            out = self.handler()
        if isinstance(out, dict):
            return out
        return json.loads(out)


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
                results.append(result)

        return results




