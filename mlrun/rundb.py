import json
from os import path
from urllib.parse import urlparse
from .datastore import StoreManager



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
        self._datastore = StoreManager(secrets_func).get_or_create_store(dirpath)

    def store(self, execution, elements=[]):
        data = execution.to_json()
        filepath = path.join(self.dirpath, execution.uid)
        self._datastore.put(data)

    def read(self, uid):
        filepath = path.join(self.dirpath, uid)
        data = self._datastore.get()
        return json.loads(data)
