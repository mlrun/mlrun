import uuid
from os import environ

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