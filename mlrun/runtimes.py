import json
import os
import uuid
from os import environ
from tempfile import mktemp

import requests
import yaml

from .utils import run_keys
from .tomarkdown import to_markdown
from .execution import MLClientCtx
from .rundb import get_run_db
from .secrets import SecretsStore
from sys import executable, stderr
from subprocess import run, PIPE


def get_or_create_ctx(name, uid='', event=None, spec=None, with_env=True, rundb=''):

    if event:
        spec = event.body
        uid = uid or event.id

    config = environ.get('MLRUN_EXEC_CONFIG')
    if with_env and config:
        spec = config

    uid = uid or uuid.uuid4().hex
    if spec and not isinstance(spec, dict):
        spec = yaml.safe_load(spec)

    autocommit = False
    tmp = environ.get('MLRUN_META_TMPFILE')
    out = environ.get('MLRUN_META_DBPATH', rundb)
    if out:
        autocommit = True

    ctx = MLClientCtx(name, uid, rundb=out, autocommit=autocommit, tmp=tmp)
    if spec:
        ctx.from_dict(spec)
    return ctx


def run_start(struct, runtime=None, args=[], rundb='', kfp=False, handler=None):

    if isinstance(runtime, str):
        if '://' in runtime:
            runtime = RemoteRuntime(runtime, args)
        else:
            runtime = LocalRuntime(runtime, args)
    elif struct and 'spec' in struct.keys() and 'runtime' in struct['spec'].keys():
        kind = struct['spec']['runtime'].get('kind', '')
        if kind in ['', 'local']:
            runtime = LocalRuntime()
        elif kind == 'remote':
            runtime = RemoteRuntime()
        else:
            raise Exception('unsupported runtime - %s' % kind)

    runtime.rundb = rundb
    runtime.process_struct(struct)
    resp = runtime.run()

    # todo: add runtimes, e.g. Horovod, Pipelines workflow

    if not resp:
        return {}
    struct = json.loads(resp)

    if kfp:
        write_kfpmeta(struct)
    return struct


class MLRuntime:
    kind = ''
    def __init__(self, command='', args=[]):
        self.struct = None
        self.command = command
        self.args = args
        self.rundb = ''

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


class LocalRuntime(MLRuntime):

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
            return resp
        except FileNotFoundError as err:
            print(err)


class RemoteRuntime(MLRuntime):
    kind = 'remote'
    def run(self):
        secrets = SecretsStore()
        secrets.from_dict(self.struct['spec'])
        self.struct['spec']['secret_sources'] = secrets.to_serial()
        try:
            resp = requests.put(self.command, json=json.dumps(self.struct))
        except OSError as err:
            print('ERROR: %s', str(err))
            raise OSError('error: cannot run function at url {}'.format(self.command))

        if not resp.ok:
            print('bad resp!!')
            return None

        if self.rundb:
            rundb = get_run_db(self.rundb)
            rundb.connect(secrets)
            rundb.store_run(resp.json(), commit=True)

        return resp.json()


KFPMETA_DIR = '/'


def write_kfpmeta(struct):
    outputs = struct['status']['outputs']
    metrics = {'metrics':
                   [{'name': k, 'numberValue':v } for k, v in outputs.items() if isinstance(v, (int, float, complex))]}
    with open(KFPMETA_DIR + 'mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    text = yaml.dump(struct, default_flow_style=False, sort_keys=False)
    text = "# Run Report\n```yaml\n" + text + "```\n"

    metadata = {
        'outputs': [{
            'type': 'markdown',
            'storage': 'inline',
            'source': text
        }]
    }
    with open(KFPMETA_DIR + 'mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    for output in struct['status'][run_keys.output_artifacts]:
        try:
            key = output["key"]
            with open(f'/tmp/{key}', 'w') as fp:
                fp.write(output["target_path"])
        except:
            pass

