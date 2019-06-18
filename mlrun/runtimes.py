import json
import os
import uuid
from os import environ
from tempfile import mktemp

import requests
import yaml

from .tomd import to_markdown
from .execution import MLClientCtx
from .rundb import get_run_db
from .secrets import SecretsStore
from sys import executable, stderr
from subprocess import run, PIPE


def get_or_create_ctx(name, uid='', event=None, spec=None, with_env=True, save_to=''):

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
    out = environ.get('MLRUN_META_DBPATH', save_to)
    if out:
        autocommit = True

    ctx = MLClientCtx(name, uid, rundb=out, autocommit=autocommit)
    if spec:
        ctx.from_dict(spec)
    return ctx


def run_start(url, struct={}, save_to='', kfp=False):
    # todo: add runtimes, e.g. Horovod, Pipelines workflow
    if '://' in url:
        resp = remote_run(url, struct, save_to)
    else:
        resp = local_run(url, struct, save_to)
    if kfp:
        write_kfpmeta(resp)
    return resp



def local_run(url, struct={}, save_to=''):
    environ['MLRUN_EXEC_CONFIG'] = json.dumps(struct)
    if not save_to:
        save_to = mktemp('.json')
        is_tmp = True
    else:
        is_tmp = False
    environ['MLRUN_META_DBPATH'] = save_to

    cmd = [
        executable, url,
    ]
    out = run(cmd, stdout=PIPE, stderr=PIPE)
    if out.returncode != 0:
        print(out.stderr.decode('utf-8'), file=stderr)
    print(out.stdout.decode('utf-8'))

    try:
        with open(save_to) as fp:
            resp = fp.read()

        if is_tmp:
            os.remove(save_to)
        return resp
    except FileNotFoundError as err:
        print(err)


def remote_run(url, struct={}, save_to=''):
    secrets = SecretsStore()
    secrets.from_dict(struct['spec'])
    struct['spec']['secret_sources'] = secrets.to_serial()
    try:
        resp = requests.put(url, json=json.dumps(struct))
    except OSError as err:
        print('ERROR: %s', str(err))
        raise OSError('error: cannot run function at url {}'.format(url))

    if not resp.ok:
        print('bad resp!!')
        return None

    if save_to:
        rundb = get_run_db(save_to)
        rundb.connect(secrets)
        rundb.store_run(resp.json(), commit=True)

    return resp.json()


def write_kfpmeta(resp):
    if not resp:
        return
    struct = json.loads(resp)
    outputs = struct['status']['outputs']
    metrics = {'metrics':
                   [{'name': k, 'numberValue':v } for k, v in outputs.items() if isinstance(v, (int, float, complex))]}
    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    text = to_markdown(resp)
    print(text)
    metadata = {
        'outputs': [{
            'type': 'markdown',
            'storage': 'inline',
            'source': text
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
