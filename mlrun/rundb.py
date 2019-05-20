import json
from os import path
from urllib.parse import urlparse

from mlrun import stores


def get_run_db(url='', secrets_func=None):
    p = urlparse(url)
    scheme = p.scheme.lower()
    if '://' not in url or scheme in ['file', 's3', 'v3io', 'http', 'https']:
        return FileRunDB(url, secrets_func)
    else:
        raise ValueError('unsupported run DB scheme ({})'.format(scheme))


class RunDBInterface:

    def store(self, execution, elements=[]):
        pass

    def read(self, uid):
        pass


class FileRunDB(RunDBInterface):

    def __init__(self, dirpath, secrets_func=None):
        self.dirpath = dirpath
        self._secrets_func = secrets_func

    def store(self, execution, elements=[]):
        data = execution.to_json()
        filepath = path.join(self.dirpath, execution.uid)
        repo = stores.url2repo(filepath, self._secrets_func)
        repo.put(data)

    def read(self, uid):
        filepath = path.join(self.dirpath, uid)
        repo = stores.url2repo(filepath, self._secrets_func)
        data = repo.get()
        return json.loads(data)
