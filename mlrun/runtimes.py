import uuid
from os import environ

import requests
import yaml

from mlrun.execution import MLClientCtx


def get_or_create_ctx(name, uid='', event=None, spec=None, with_env=True):

    if event:
        spec = event.body
        uid = uid or event.id

    config = environ.get('MLRUN_EXEC_CONFIG')
    if with_env and config:
        spec = config

    uid = uid or uuid.uuid4().hex
    if spec and not isinstance(spec, dict):
        spec = yaml.safe_load(spec)

    ctx = MLClientCtx(name, uid)
    if spec:
        ctx.from_dict(spec)
    return ctx


def remote_run(url, spec={}):

    try:
        resp = requests.put(url, json=json.dumps(spec))
    except OSError as err:
        print('ERROR: %s', str(err))
        raise OSError('error: cannot run function at url {}'.format(verb, api_url))

    if not resp.ok:
        print('bad resp!!')

    print(resp.text)
