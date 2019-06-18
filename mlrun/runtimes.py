import json
import os
import uuid
from os import environ
from tempfile import mktemp

import requests
import yaml
from py._builtin import execfile

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


def run_start(url, struct={}, save_to=''):
    # todo: add runtimes, e.g. Horovod, Pipelines workflow
    if '://' in url:
        remote_run(url, struct, save_to)
    else:
        local_run(url, struct, save_to)


def local_run(url, struct={}, save_to=''):
    environ['MLRUN_EXEC_CONFIG'] = json.dumps(struct)
    if not save_to:
        save_to = mktemp('.yaml')
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
        if is_tmp:
            with open(save_to) as fp:
                resp = fp.read()
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
