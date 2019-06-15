import json
import uuid
from os import environ

import requests
import yaml

from .execution import MLClientCtx
from .rundb import FileRunDB


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

    rundb = None
    autocommit = False
    out = environ.get('MLRUN_META_FILEPATH', save_to)
    if out:
        autocommit = True
    rundb = FileRunDB(fullpath=out)

    ctx = MLClientCtx(name, uid, rundb=rundb, autocommit=autocommit)
    if spec:
        ctx.from_dict(spec)
    return ctx


def remote_run(url, struct={}, save_to='', secrets=None):

    try:
        resp = requests.put(url, json=json.dumps(struct))
    except OSError as err:
        print('ERROR: %s', str(err))
        raise OSError('error: cannot run function at url {}'.format(url))

    if not resp.ok:
        print('bad resp!!')
        return None

    if save_to:
        rundb = FileRunDB(fullpath=save_to, secrets_func=secrets)
        rundb.store(resp.json(), commit=True)

    return resp.json()
