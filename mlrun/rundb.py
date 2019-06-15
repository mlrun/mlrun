import json
from os import path
from urllib.parse import urlparse

import yaml

from .datastore import StoreManager


def get_run_db(url='', secrets_func=None):
    p = urlparse(url)
    scheme = p.scheme.lower()
    if '://' not in url or scheme in ['file', 's3', 'v3io', 'http', 'https']:
        return FileRunDB(url, secrets_func)
    else:
        raise ValueError('unsupported run DB scheme ({})'.format(scheme))


class RunDBInterface:

    def store(self, execution, elements=[], commit=False):
        pass

    def read(self, uid):
        pass


class FileRunDB(RunDBInterface):

    def __init__(self, dirpath='', fullpath='', secrets_func=None):
        self.fullpath = fullpath
        self.dirpath = dirpath
        sm = StoreManager(secrets_func)
        self._datastore, self._subpath = sm.get_or_create_store(fullpath or dirpath)

    def store(self, execution, elements=[], commit=False):
        data = execution.to_yaml()
        filepath = self.fullpath or self._filepath(execution.uid)
        self._datastore.put(filepath, data)

    def read(self, uid):
        filepath = self.fullpath or self._filepath(uid)
        data = self._datastore.get(filepath)
        return yaml.loads(data)

    def _filepath(self, uid):
        return path.join(self.dirpath, 'mlrun-{}.yaml'.format(uid))
