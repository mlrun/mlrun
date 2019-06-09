import json
import os
import uuid
import getpass
from os import environ

from mlrun.execution import KFPClientCtx


class LocalRuntime(KFPClientCtx):

    def __init__(self, name, uid='', parameters={}, artifacts={}):
        uid = uid or uuid.uuid4().hex
        KFPClientCtx.__init__(self, uid, name)
        self.parent_type = 'local'
        self.owner = getpass.getuser()

        config = environ.get('MLRUN_EXEC_CONFIG')
        if config:
            attrs = json.loads(config)
            self._set_from_dict(attrs)

        secrets = environ.get('MLRUN_EXEC_SECRETS')
        if secrets:
            self._secrets = json.loads(secrets)
            self._secrets_function = self._secrets.get

        self._parameters = {**parameters, **self._parameters}
